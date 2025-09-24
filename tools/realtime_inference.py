#!/usr/bin/env python3
"""
Real-time Video Processing with Moondream Station

This script processes live camera feeds or video files using moondream-station
for real-time inference tasks like captioning, querying, detection, and pointing.

Author: anudit
"""

import sys
import json
import asyncio
import argparse
import cv2
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import base64
from datetime import datetime
import numpy as np
from threading import Thread, Lock
import queue

# Add the moondream-station package to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setup" / "moondream-station"))

try:
    from moondream_station.core.config import ConfigManager
    from moondream_station.core.manifest import ManifestManager
    from moondream_station.core.models import ModelManager
    from moondream_station.core.inference_service import InferenceService
    from moondream_station.core.analytics import Analytics
    from moondream_station.ui.display import Display
except ImportError as e:
    print(f"Error importing moondream-station modules: {e}")
    print("Make sure you're running this from the correct directory with moondream-station installed.")
    sys.exit(1)


class RealTimeProcessor:
    """Process live video streams using moondream-station inference."""
    
    def __init__(self, model_name: str = "moondream-3-preview"):
        self.model_name = model_name
        self.config = ConfigManager()
        self.analytics = Analytics(self.config)
        self.display = Display()
        self.manifest_manager = ManifestManager(self.config)
        self.model_manager = ModelManager(self.config, self.manifest_manager)
        self.inference_service = InferenceService(self.config, self.manifest_manager)
        
        # Video processing
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the moondream-station system."""
        print("Initializing moondream-station...")
        
        # Load manifest
        try:
            self.manifest_manager.load_manifest("production", self.analytics, self.display)
        except Exception as e:
            print(f"Warning: Could not load production manifest: {e}")
            # Try local manifest
            try:
                manifest_path = Path(__file__).parent / "setup" / "moondream-station" / "local_manifest.json"
                if manifest_path.exists():
                    self.manifest_manager.load_manifest(str(manifest_path), self.analytics, self.display)
                else:
                    print("No manifest found. Please ensure moondream-station is properly set up.")
                    sys.exit(1)
            except Exception as e2:
                print(f"Error loading manifest: {e2}")
                sys.exit(1)
        
        # Switch to the specified model
        if not self.model_manager.switch_model(self.model_name, self.display):
            available_models = self.model_manager.list_models()
            print(f"Model '{self.model_name}' not found. Available models: {available_models}")
            if available_models:
                self.model_name = available_models[0]
                print(f"Using model: {self.model_name}")
                if not self.model_manager.switch_model(self.model_name, self.display):
                    print("Failed to switch to any model.")
                    sys.exit(1)
            else:
                print("No models available.")
                sys.exit(1)
        
        # Start inference service
        if not self.inference_service.start(self.model_name):
            print("Failed to start inference service.")
            sys.exit(1)
        
        print(f"Initialized with model: {self.model_name}")
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode OpenCV frame to base64 data URL."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_data = buffer.tobytes()
            
            encoded = base64.b64encode(image_data).decode()
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            raise ValueError(f"Failed to encode frame: {e}")
    
    async def process_frame(self, frame: np.ndarray, function_name: str, **kwargs) -> Dict[str, Any]:
        """Process a single frame with the specified function."""
        try:
            # Encode the frame
            image_url = self._encode_frame(frame)
            
            # Prepare arguments
            inference_kwargs = {
                "image_url": image_url,
                "stream": False,
                **kwargs
            }
            
            # Execute the inference
            result = await self.inference_service.execute_function(
                function_name, **inference_kwargs
            )
            
            return {
                "function": function_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": "error" not in result
            }
            
        except Exception as e:
            return {
                "function": function_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def _capture_frames(self, source: Union[int, str], fps: int = 1):
        """Capture frames from camera or video file."""
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Set FPS for video files
        if isinstance(source, str):
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        frame_count = 0
        target_fps = fps
        frame_interval = max(1, int(30 / target_fps))  # Process every Nth frame
        
        print(f"Starting video capture from: {source}")
        print(f"Processing every {frame_interval} frames (target FPS: {target_fps})")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(source, str):
                    print("End of video file reached")
                    break
                else:
                    print("Failed to read from camera")
                    break
            
            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_interval == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                else:
                    # Remove old frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame.copy())
                    except queue.Empty:
                        pass
            
            # Update current frame for display
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        self.cap.release()
        print("Video capture stopped")
    
    def _process_frames_async(self, function_name: str, **kwargs):
        """Process frames asynchronously."""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Run async processing in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        self.process_frame(frame, function_name, **kwargs)
                    )
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                finally:
                    loop.close()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def _display_results(self, show_video: bool = True, save_results: bool = False, output_file: str = None):
        """Display video and results."""
        results = []
        
        while self.is_running:
            # Display current frame
            if show_video:
                with self.frame_lock:
                    if self.current_frame is not None:
                        display_frame = self.current_frame.copy()
                        
                        # Add text overlay
                        cv2.putText(display_frame, f"Model: {self.model_name}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, "Press 'q' to quit", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow('Moondream Station - Real-time Processing', display_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.is_running = False
                            break
            
            # Process results
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                
                if result["success"]:
                    print(f"[{result['timestamp']}] ✓ {result['function']}: {result['result']}")
                else:
                    print(f"[{result['timestamp']}] ✗ Error: {result.get('error', 'Unknown error')}")
                
            except queue.Empty:
                pass
        
        # Save results if requested
        if save_results and results and output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {output_path}")
        
        cv2.destroyAllWindows()
        return results
    
    def process_live(self, 
                    source: Union[int, str] = 0, 
                    function_name: str = "caption",
                    fps: int = 1,
                    show_video: bool = True,
                    save_results: bool = False,
                    output_file: str = None,
                    question: str = None,
                    object_name: str = None,
                    length: str = "normal") -> List[Dict[str, Any]]:
        """Process live video stream."""
        
        # Prepare function-specific arguments
        kwargs = {}
        if function_name == "query" and question:
            kwargs["question"] = question
        elif function_name in ["detect", "point"] and object_name:
            kwargs["object"] = object_name
        elif function_name == "caption":
            kwargs["length"] = length
        
        self.is_running = True
        
        # Start capture thread
        capture_thread = Thread(target=self._capture_frames, args=(source, fps))
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start processing thread
        process_thread = Thread(target=self._process_frames_async, args=(function_name,), kwargs=kwargs)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            # Display results
            results = self._display_results(show_video, save_results, output_file)
            return results
        
        except KeyboardInterrupt:
            print("\nStopping processing...")
            self.is_running = False
        
        finally:
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            process_thread.join(timeout=2)
            self.cleanup()
    
    def process_video_file(self, 
                          video_path: str,
                          function_name: str = "caption",
                          fps: int = 1,
                          show_video: bool = True,
                          save_results: bool = False,
                          output_file: str = None,
                          question: str = None,
                          object_name: str = None,
                          length: str = "normal") -> List[Dict[str, Any]]:
        """Process a video file."""
        if not Path(video_path).exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        return self.process_live(
            source=video_path,
            function_name=function_name,
            fps=fps,
            show_video=show_video,
            save_results=save_results,
            output_file=output_file,
            question=question,
            object_name=object_name,
            length=length
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.inference_service:
            asyncio.run(self.inference_service.stop())


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Real-time video processing with moondream-station")
    parser.add_argument("source", nargs='?', default=0, 
                       help="Video source: camera index (0, 1, 2...) or video file path (default: 0)")
    parser.add_argument("-f", "--function", 
                       choices=["caption", "query", "detect", "point"], 
                       default="caption",
                       help="Inference function to use (default: caption)")
    parser.add_argument("-o", "--output", help="Output JSON file to save results")
    parser.add_argument("-m", "--model", default="moondream-3-preview", 
                       help="Model to use (default: moondream-3-preview)")
    parser.add_argument("-q", "--question", help="Question for query function")
    parser.add_argument("-obj", "--object", help="Object name for detect/point functions")
    parser.add_argument("-l", "--length", choices=["normal", "short", "long"], 
                       default="normal", help="Caption length (default: normal)")
    parser.add_argument("--fps", type=int, default=1, 
                       help="Processing FPS (frames per second to process, default: 1)")
    parser.add_argument("--no-display", action="store_true", 
                       help="Don't show video display window")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.function == "query" and not args.question:
        print("Error: --question is required for query function")
        sys.exit(1)
    
    if args.function in ["detect", "point"] and not args.object:
        print(f"Error: --object is required for {args.function} function")
        sys.exit(1)
    
    # Determine source type
    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # Assume it's a file path
    
    # Process video
    processor = None
    try:
        processor = RealTimeProcessor(args.model)
        
        print(f"Starting real-time processing...")
        print(f"Source: {source}")
        print(f"Function: {args.function}")
        print(f"FPS: {args.fps}")
        print("Press 'q' in the video window to quit")
        
        results = processor.process_live(
            source=source,
            function_name=args.function,
            fps=args.fps,
            show_video=not args.no_display,
            save_results=args.save_results or bool(args.output),
            output_file=args.output,
            question=args.question,
            object_name=args.object,
            length=args.length
        )
        
        # Print summary
        if results:
            successful = sum(1 for r in results if r["success"])
            total = len(results)
            print(f"\nProcessing complete: {successful}/{total} frames processed successfully")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if processor:
            processor.cleanup()


if __name__ == "__main__":
    # Activate virtual environment if it exists
    venv_path = Path(__file__).resolve().parents[1] / "venv"
    if venv_path.exists():
        # Add venv to Python path
        venv_lib = venv_path / "lib" / "python3.8" / "site-packages"
        if venv_lib.exists():
            sys.path.insert(0, str(venv_lib))
        else:
            # Try other Python versions
            for py_ver in ["python3.9", "python3.10", "python3.11", "python3.12"]:
                venv_lib = venv_path / "lib" / py_ver / "site-packages"
                if venv_lib.exists():
                    sys.path.insert(0, str(venv_lib))
                    break
    
    main()
