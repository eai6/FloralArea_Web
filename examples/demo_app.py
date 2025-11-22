"""
FloralArea Demo Application
A Gradio interface for estimating floral area from images
"""

import gradio as gr
import json
import os
import traceback

from floralarea.cv import yolov8, img_processing as ip, runTiles
from floralarea.utils import helpers


def load_config(config_path="config/config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found!")
        return {}


def estimateArea(file_location: str, component: str, threshold: float, num_tiles: int, 
                yolo_model, reference_area: float) -> tuple:
    """
    Estimate the area of either flower or leaf
    
    Args:
        file_location: Path to the image file
        component: Component to estimate area for ('flower' or 'leaf')
        threshold: Confidence threshold for the model
        num_tiles: Number of tiles to split the image into
        yolo_model: Initialized YOLO model instance
        reference_area: Known area of the reference object (in cm^2)
    
    Returns:
        tuple: (estimated_area, annotated_image)
    """
    # Get component area using tiling approach
    component_area, image = runTiles.runTilles(
        yolo_model, file_location, threshold, num_tiles, component
    )
    
    # Get reference object area
    results = yolo_model.runInference(file_location, 'reference_object', threshold)
    mask = yolo_model.getMask(results)
    reference_pixel_area = ip.getPixelArea(mask)
    
    # If reference object not found, use tiling approach
    if reference_pixel_area == 0:
        reference_pixel_area = runTiles.runTilles(
            yolo_model, file_location, threshold, num_tiles, 'reference_object'
        )[0]
    
    # Calculate actual area using reference object scaling
    area = (component_area / reference_pixel_area) * reference_area
    
    return area, image


def process_image(input_img, config, yolo_model):
    """
    Process uploaded image and estimate floral area
    
    Args:
        input_img: Uploaded image file path
        config: Configuration dictionary
        yolo_model: Initialized YOLO model
        
    Returns:
        tuple: (area_text, mask_image)
    """
    try:
        file_location = input_img
        
        print(f"Processing file at: {file_location}")
        
        # Apply preprocessing
        ip.apply_preprocessing(file_location)
        
        # Estimate area
        area, mask = estimateArea(
            file_location, 
            "flower", 
            0.5,  # threshold
            2,    # num_tiles
            yolo_model,
            config['reference_object_area']
        )
        
        return round(area, 2), mask
    
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return "Error: Could not process image", None


def create_demo(config_path="config/config.json"):
    """
    Create and launch the Gradio demo interface
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    if not config:
        raise ValueError("Failed to load configuration")
    
    # Initialize YOLO model
    yolo_model = yolov8.yolov8(
        config['flower_model'], 
        config['reference_object_model']
    )
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# FloralArea Demo")
        gr.Markdown(
            "Upload an image of a flowering plant with a reference object "
            "to estimate the floral area in cm²"
        )
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type='filepath', label='Upload Flowering Plant')
                process_button = gr.Button("Process", variant="primary")
            
            with gr.Column():
                text_output = gr.Textbox(label="Floral Area (cm²)")
                image_output = gr.Image(label="Flower Mask")
        
        # Connect the processing function
        process_button.click(
            fn=lambda img: process_image(img, config, yolo_model),
            inputs=image_input,
            outputs=[text_output, image_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
