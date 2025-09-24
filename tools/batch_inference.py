#!/usr/bin/env python3
"""
Batch Image Processing with Moondream Station

This script processes multiple images from a directory using moondream-station
for various inference tasks like captioning, querying, detection, and pointing.

Author: anudit
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from datetime import datetime

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


class BatchImageProcessor:
    """Process multiple images using moondream-station inference."""
    
    def __init__(self, model_name: str = "moondream-3-preview"):
        self.model_name = model_name
        self.config = ConfigManager()
        self.analytics = Analytics(self.config)
        self.display = Display()
        self.manifest_manager = ManifestManager(self.config)
        self.model_manager = ModelManager(self.config, self.manifest_manager)
        self.inference_service = InferenceService(self.config, self.manifest_manager)
        
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
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URL."""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            encoded = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    async def process_image(self, image_path: str, function_name: str, **kwargs) -> Dict[str, Any]:
        """Process a single image with the specified function."""
        try:
            # Encode the image
            image_url = self._encode_image(image_path)
            
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
                "image_path": image_path,
                "function": function_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": "error" not in result
            }
            
        except Exception as e:
            return {
                "image_path": image_path,
                "function": function_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def process_directory(self, 
                              input_dir: str, 
                              function_name: str = "caption",
                              output_file: Optional[str] = None,
                              question: Optional[str] = None,
                              object_name: Optional[str] = None,
                              length: str = "normal",
                              supported_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Process all images in a directory."""
        
        if supported_extensions is None:
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} image files to process")
        
        # Prepare function-specific arguments
        kwargs = {}
        if function_name == "query" and question:
            kwargs["question"] = question
        elif function_name in ["detect", "point"] and object_name:
            kwargs["object"] = object_name
        elif function_name == "caption":
            kwargs["length"] = length
        
        # Process images
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            result = await self.process_image(str(image_file), function_name, **kwargs)
            results.append(result)
            
            if result["success"]:
                print(f"  ✓ Success")
                if "result" in result and isinstance(result["result"], dict):
                    # Print a preview of the result
                    for key, value in result["result"].items():
                        if key != "error" and isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"    {key}: {preview}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {output_path}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.inference_service:
            asyncio.run(self.inference_service.stop())


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Batch process images with moondream-station")
    parser.add_argument("input_dir", help="Directory containing images to process")
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
    parser.add_argument("--extensions", nargs="+", 
                       default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
                       help="Supported image extensions")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.function == "query" and not args.question:
        print("Error: --question is required for query function")
        sys.exit(1)
    
    if args.function in ["detect", "point"] and not args.object:
        print(f"Error: --object is required for {args.function} function")
        sys.exit(1)
    
    # Process images
    processor = None
    try:
        processor = BatchImageProcessor(args.model)
        
        # Run the async processing
        results = asyncio.run(processor.process_directory(
            input_dir=args.input_dir,
            function_name=args.function,
            output_file=args.output,
            question=args.question,
            object_name=args.object,
            length=args.length,
            supported_extensions=args.extensions
        ))
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        print(f"\nProcessing complete: {successful}/{total} images processed successfully")
        
        if successful < total:
            print("Failed images:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['image_path']}: {result.get('error', 'Unknown error')}")
    
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
