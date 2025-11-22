"""
Batch processing example for FloralArea
Process multiple images and save results to CSV
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

from floralarea.cv import yolov8, img_processing as ip, runTiles


def process_image(model, image_path, config, threshold=0.5, num_tiles=2):
    """
    Process a single image and return results
    
    Returns:
        dict with results or None if processing failed
    """
    try:
        # Preprocess
        ip.apply_preprocessing(str(image_path))
        
        # Get flower area
        component_area, _ = runTiles.runTilles(
            model, str(image_path), threshold, num_tiles, 'flower'
        )
        
        # Get reference area
        results = model.runInference(str(image_path), 'reference_object', threshold)
        mask = model.getMask(results)
        reference_pixel_area = ip.getPixelArea(mask)
        
        if reference_pixel_area == 0:
            reference_pixel_area, _ = runTiles.runTilles(
                model, str(image_path), threshold, num_tiles, 'reference_object'
            )
        
        # Calculate actual area
        if reference_pixel_area > 0:
            actual_area = (component_area / reference_pixel_area) * config['reference_object_area']
            
            return {
                'filename': image_path.name,
                'floral_area_cm2': round(actual_area, 2),
                'flower_pixels': component_area,
                'reference_pixels': reference_pixel_area,
                'status': 'success'
            }
        else:
            return {
                'filename': image_path.name,
                'floral_area_cm2': None,
                'flower_pixels': component_area,
                'reference_pixels': 0,
                'status': 'no_reference_object'
            }
    
    except Exception as e:
        return {
            'filename': image_path.name,
            'floral_area_cm2': None,
            'flower_pixels': None,
            'reference_pixels': None,
            'status': f'error: {str(e)}'
        }


def main():
    """Batch process images from a directory"""
    
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <image_directory>")
        print("Example: python batch_process.py ../data/images/")
        sys.exit(1)
    
    image_dir = Path(sys.argv[1])
    
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"Error: Directory not found: {image_dir}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in image_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(images)} images to process")
    print("-" * 60)
    
    # Load configuration
    config_path = "../config/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    print("Loading models...")
    model = yolov8.yolov8(
        config['flower_model'],
        config['reference_object_model']
    )
    print("✓ Models loaded\n")
    
    # Process each image
    results = []
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing {image_path.name}...", end=' ')
        
        result = process_image(model, image_path, config)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✓ {result['floral_area_cm2']} cm²")
        else:
            print(f"✗ {result['status']}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"batch_results_{timestamp}.csv"
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['filename', 'floral_area_cm2', 'flower_pixels', 
                     'reference_pixels', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print("-" * 60)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\nSummary:")
    print(f"  Total images: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        avg_area = sum(r['floral_area_cm2'] for r in results 
                      if r['status'] == 'success') / successful
        print(f"  Average floral area: {avg_area:.2f} cm²")


if __name__ == "__main__":
    main()
