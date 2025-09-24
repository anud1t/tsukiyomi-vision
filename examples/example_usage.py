#!/usr/bin/env python3
"""
Example usage of the batch image processor with moondream-station.

This script demonstrates how to use the BatchImageProcessor class
to process images in different ways.
"""

import asyncio
from pathlib import Path
from batch_inference import BatchImageProcessor


async def example_caption_images():
    """Example: Caption all images in a directory."""
    print("=== Example: Captioning Images ===")
    
    processor = BatchImageProcessor("Moondream 3 Preview")
    
    try:
        # Process all images in the current directory
        results = await processor.process_directory(
            input_dir="./test_images",  # Change this to your image directory
            function_name="caption",
            output_file="./results/captions.json",
            length="normal"  # or "short" or "long"
        )
        
        print(f"Processed {len(results)} images")
        for result in results:
            if result["success"]:
                print(f"✓ {Path(result['image_path']).name}: {result['result']}")
            else:
                print(f"✗ {Path(result['image_path']).name}: {result.get('error', 'Unknown error')}")
    
    finally:
        processor.cleanup()


async def example_query_images():
    """Example: Ask questions about images."""
    print("\n=== Example: Querying Images ===")
    
    processor = BatchImageProcessor("Moondream 3 Preview")
    
    try:
        # Ask specific questions about images
        questions = [
            "What color is the car?",
            "How many people are in the image?",
            "What is the main subject of this image?"
        ]
        
        for question in questions:
            print(f"\nAsking: '{question}'")
            results = await processor.process_directory(
                input_dir="./test_images",
                function_name="query",
                output_file=f"./results/query_{question.replace(' ', '_').replace('?', '')}.json",
                question=question
            )
            
            for result in results:
                if result["success"]:
                    print(f"  {Path(result['image_path']).name}: {result['result']}")
                else:
                    print(f"  {Path(result['image_path']).name}: Error - {result.get('error', 'Unknown')}")
    
    finally:
        processor.cleanup()


async def example_detect_objects():
    """Example: Detect specific objects in images."""
    print("\n=== Example: Object Detection ===")
    
    processor = BatchImageProcessor("Moondream 3 Preview")
    
    try:
        # Detect specific objects
        objects_to_detect = ["car", "person", "dog", "building"]
        
        for obj in objects_to_detect:
            print(f"\nDetecting: '{obj}'")
            results = await processor.process_directory(
                input_dir="./test_images",
                function_name="detect",
                output_file=f"./results/detect_{obj}.json",
                object_name=obj
            )
            
            for result in results:
                if result["success"]:
                    print(f"  {Path(result['image_path']).name}: {result['result']}")
                else:
                    print(f"  {Path(result['image_path']).name}: Error - {result.get('error', 'Unknown')}")
    
    finally:
        processor.cleanup()


def main():
    """Run all examples."""
    print("Moondream Station Batch Processing Examples")
    print("=" * 50)
    
    # Create results directory
    Path("./results").mkdir(exist_ok=True)
    
    # Run examples
    asyncio.run(example_caption_images())
    asyncio.run(example_query_images())
    asyncio.run(example_detect_objects())
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
