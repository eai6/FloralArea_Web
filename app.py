#import numpy as np
import gradio as gr

from utils import helpers
from cv import yolov8
from cv import img_processing as ip
from cv import runTiles
import json
import os
import tempfile
import traceback

#from PIL import Image

# load the config json file
#config = json.load(open('config.json'))
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")  # Default to "config.json"

try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: {CONFIG_PATH} not found!")
    config = {}  # Provide default values if missing

# initialize the model
yolo = yolov8.yolov8(config['flower_model'], config['reference_object_model'])

reference_area = config['reference_object_area'] # area of the reference object in inches  

# def estimateArea(file_location:str, component:str, threshold:float, num_tiles):
#     '''
#     Estimate the area of either flower or leaf

#     - Input:
#     file_location: str: path to the image file
#     component: str: component to estimate area for
#     threshold: float: threshold for the model

#     - Output:
#     float: area of the component
#     '''
#     # run inference on the image
#     flower_results = yolo.runInference(file_location, component, threshold)
#     # get the mask
#     flower_mask = yolo.getMask(flower_results)
#     # get the area of the mask
#     component_area = ip.getPixelArea(flower_mask)

#     # # get area for the reference object
#     reference_results = yolo.runInference(file_location, 'reference_object', threshold)
#     reference_mask = yolo.getMask(reference_results)
#     reference_pixel_area = ip.getPixelArea(reference_mask)
#     if reference_pixel_area == 0:
#        #raise Exception("Reference object not found in the image")
#        reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, 3)

#     #component_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles, component)
#     #reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles+1, 'reference_object')

#     area = (component_area/reference_pixel_area) * reference_area
   
#     return area , flower_results[0].plot(conf=False, labels=True, boxes=True, masks=True)

def estimateArea2(file_location:str, component:str, threshold:float, num_tiles) -> float:
    '''
    Estimate the area of either flower or leaf

    - Input:
    file_location: str: path to the image file
    component: str: component to estimate area for
    threshold: float: threshold for the model

    - Output:
    float: area of the component
    '''
    # # run inference on the image
    # results = yolo.runInference(file_location, component, threshold)
    # # get the mask
    # mask = yolo.getMask(results)
    # # get the area of the mask
    # component_area = ip.getPixelArea(mask)
    
    component_area, image = runTiles.runTilles(yolo, file_location, threshold, num_tiles, component)

    # # get area for the reference object
    results = yolo.runInference(file_location, 'reference_object', threshold)
    mask = yolo.getMask(results)
    reference_pixel_area = ip.getPixelArea(mask)
    #savePlot(results, file_location, 'reference_object', threshold)
    if reference_pixel_area == 0:
       reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles, 'reference_object')

    area = (component_area/reference_pixel_area) * reference_area
   
    return area, image

# def getArea(input_img):
    
#     '''
#     Estimate the area of either flower or leaf
#     Takes in an image file and returns the area of the component with a  threshold
#     '''
#     try:
#         filename = input_img
#         file_location = filename #f"api/data/input/{filename}"
        
#         # save audio file temporally
#         #helpers.saveUploadfile(file_location, uploaded_file)

#         # apply preprocessing to the image
#         ip.apply_preprocessing(file_location)

#         #
#         #yolo.savePlot(file_location, component, threshold)

#         # run the model on the image
#         area, mask = estimateArea2(file_location, "flower", 0.5, 2)

#         #helpers.removefile(file_location)

#         return round(area,2), mask
#     except Exception as e:
#         print(e)

def getArea(input_img):
    try:
        # Save file in a temp directory
        #temp_dir = tempfile.gettempdir()  # Get system's temp folder
        #file_location = os.path.join(temp_dir, "input.jpg")
        file_location = input_img
        
        # Ensure image is saved correctly
        #input_img.save(file_location)

        # save audio file temporally
        #helpers.saveUploadfile(file_location, input_img)


        print(f"Processing file at: {file_location}")

        ip.apply_preprocessing(file_location)
        area, mask = estimateArea2(file_location, "flower", 0.5, 2)

        return round(area, 2), mask
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return "Error: Could not process image", None
    

#demo = gr.Interface(getArea, gr.Image(type='filepath'), "textbox")

with gr.Blocks() as demo:
    gr.Markdown("# FloralArea Demo")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='filepath', label='Upload Flowering Plant')
            process_button = gr.Button("Process")

        with gr.Column():
            text_output = gr.Textbox(label="Floral Area (cm^2)")
            image_output = gr.Image(label="Flower Mask")

    process_button.click(getArea, inputs=image_input, outputs=[text_output, image_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)