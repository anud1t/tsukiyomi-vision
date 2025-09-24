#!/usr/bin/env python3
"""
Real-time Video Processing Examples with Moondream Station

This script demonstrates various ways to use the real-time processor
for live camera feeds and video files.
"""

import asyncio
from pathlib import Path
from realtime_inference import RealTimeProcessor


def example_live_camera_captioning():
    """Example: Live camera captioning."""
    print("=== Example: Live Camera Captioning ===")
    
    processor = RealTimeProcessor("moondream-3-preview")
    
    try:
        # Process live camera feed with captioning
        results = processor.process_live(
            source=0,  # Default camera
            function_name="caption",
            fps=1,  # Process 1 frame per second
            show_video=True,
            save_results=True,
            output_file="./results/live_captions.json",
            length="short"
        )
        
        print(f"Processed {len(results)} frames")
    
    finally:
        processor.cleanup()


def example_live_camera_queries():
    """Example: Live camera with questions."""
    print("\n=== Example: Live Camera Queries ===")
    
    processor = RealTimeProcessor("moondream-3-preview")
    
    try:
        # Ask questions about the live feed
        questions = [
            "What objects do you see?",
            "What color is the main subject?",
            "How many people are visible?"
        ]
        
        for question in questions:
            print(f"\nAsking: '{question}'")
            results = processor.process_live(
                source=0,
                function_name="query",
                fps=0.5,  # Slower processing for questions
                show_video=True,
                question=question,
                save_results=True,
                output_file=f"./results/query_{question.replace(' ', '_').replace('?', '')}.json"
            )
            
            print(f"Processed {len(results)} frames for question: {question}")
    
    finally:
        processor.cleanup()


def example_video_file_processing():
    """Example: Process a video file."""
    print("\n=== Example: Video File Processing ===")
    
    processor = RealTimeProcessor("moondream-3-preview")
    
    try:
        # Process a video file
        video_path = "./test_video.mp4"  # Change this to your video file
        
        if not Path(video_path).exists():
            print(f"Video file not found: {video_path}")
            print("Please provide a valid video file path")
            return
        
        results = processor.process_video_file(
            video_path=video_path,
            function_name="caption",
            fps=2,  # Process 2 frames per second
            show_video=True,
            save_results=True,
            output_file="./results/video_captions.json",
            length="normal"
        )
        
        print(f"Processed {len(results)} frames from video")
    
    finally:
        processor.cleanup()


def example_object_detection():
    """Example: Real-time object detection."""
    print("\n=== Example: Real-time Object Detection ===")
    
    processor = RealTimeProcessor("moondream-3-preview")
    
    try:
        # Detect specific objects in real-time
        objects_to_detect = ["person", "car", "dog", "bicycle"]
        
        for obj in objects_to_detect:
            print(f"\nDetecting: '{obj}'")
            results = processor.process_live(
                source=0,
                function_name="detect",
                fps=1,
                show_video=True,
                object_name=obj,
                save_results=True,
                output_file=f"./results/detect_{obj}.json"
            )
            
            print(f"Processed {len(results)} frames for {obj} detection")
    
    finally:
        processor.cleanup()


def example_headless_processing():
    """Example: Headless processing (no video display)."""
    print("\n=== Example: Headless Processing ===")
    
    processor = RealTimeProcessor("moondream-3-preview")
    
    try:
        # Process without showing video window
        results = processor.process_live(
            source=0,
            function_name="caption",
            fps=1,
            show_video=False,  # No video display
            save_results=True,
            output_file="./results/headless_captions.json",
            length="short"
        )
        
        print(f"Processed {len(results)} frames (headless mode)")
        
        # Print results
        for i, result in enumerate(results[-5:], 1):  # Show last 5 results
            if result["success"]:
                print(f"  {i}. {result['result']}")
            else:
                print(f"  {i}. Error: {result.get('error', 'Unknown')}")
    
    finally:
        processor.cleanup()


def main():
    """Run all examples."""
    print("Moondream Station Real-time Processing Examples")
    print("=" * 60)
    
    # Create results directory
    Path("./results").mkdir(exist_ok=True)
    
    print("\nChoose an example to run:")
    print("1. Live camera captioning")
    print("2. Live camera queries")
    print("3. Video file processing")
    print("4. Object detection")
    print("5. Headless processing")
    print("6. Run all examples")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        example_live_camera_captioning()
    elif choice == "2":
        example_live_camera_queries()
    elif choice == "3":
        example_video_file_processing()
    elif choice == "4":
        example_object_detection()
    elif choice == "5":
        example_headless_processing()
    elif choice == "6":
        print("Running all examples...")
        example_live_camera_captioning()
        example_live_camera_queries()
        example_object_detection()
        example_headless_processing()
    else:
        print("Invalid choice. Please run the script again.")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
