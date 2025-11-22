"""
Simple example of using FloralArea package
This script shows basic usage for estimating floral area
"""

import json
import sys
from pathlib import Path

from floralarea.cv import yolov8, img_processing as ip, runTiles


def main():
    """Main example function"""
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_example.py <path_to_image>")
        print("Example: python simple_example.py test_flower.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Verify image exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    print("-" * 50)
    
    # Load configuration
    config_path = "../config/config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        print("Make sure you're running from the examples/ directory")
        sys.exit(1)
    
    # Initialize YOLO models
    print("Loading models...")
    model = yolov8.yolov8(
        config['flower_model'],
        config['reference_object_model']
    )
    print("✓ Models loaded successfully")
    
    # Preprocess image
    print("Preprocessing image...")
    ip.apply_preprocessing(image_path)
    print("✓ Preprocessing complete")
    
    # Parameters
    threshold = 0.5  # Confidence threshold
    num_tiles = 2    # Split image into 2x2 tiles
    
    # Estimate flower area
    print(f"Detecting flowers (threshold={threshold}, tiles={num_tiles}x{num_tiles})...")
    component_area, annotated_image = runTiles.runTilles(
        model, 
        image_path, 
        threshold, 
        num_tiles, 
        'flower'
    )
    print(f"✓ Flower pixel area: {component_area}")
    
    # Get reference object area
    print("Detecting reference object...")
    results = model.runInference(image_path, 'reference_object', threshold)
    mask = model.getMask(results)
    reference_pixel_area = ip.getPixelArea(mask)
    
    # Fallback to tiling if reference object not found
    if reference_pixel_area == 0:
        print("Reference object not found in full image, trying with tiles...")
        reference_pixel_area, _ = runTiles.runTilles(
            model, 
            image_path, 
            threshold, 
            num_tiles, 
            'reference_object'
        )
    
    print(f"✓ Reference pixel area: {reference_pixel_area}")
    
    # Calculate actual area
    if reference_pixel_area > 0:
        reference_actual_area = config['reference_object_area']  # cm²
        actual_area = (component_area / reference_pixel_area) * reference_actual_area
        
        print("-" * 50)
        print(f"RESULTS:")
        print(f"  Floral Area: {actual_area:.2f} cm²")
        print(f"  Reference Area: {reference_actual_area} cm²")
        print(f"  Scaling Factor: {reference_actual_area/reference_pixel_area:.6f} cm²/pixel")
        print("-" * 50)
    else:
        print("Error: Could not detect reference object")
        print("Make sure your reference object is visible in the image")
    
    print(f"\nAnnotated image saved to: {annotated_image}")


if __name__ == "__main__":
    main()
