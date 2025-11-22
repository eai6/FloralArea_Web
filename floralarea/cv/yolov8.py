from ultralytics import YOLO
from PIL import Image
import numpy as np


class yolov8:
    def __init__(self, component_model_path = 'api/cv/component_model.pt', reference_model_path = 'api/cv/reference_model.pt'):
        self.component_model = YOLO(component_model_path)
        self.reference_model = YOLO(reference_model_path)
        
    def runInference(self, image, object, threshold):
        '''
        Run the model on the image with a threshold
        '''
        if object == 'flower':
            return self.component_model.predict(image, conf=threshold, classes=[2])
            #return self.component_model.predict(image, conf=threshold, classes=[2], imgsz=640)
            #return self.component_model.predict(image, classes=[2], imgsz=640)
            
        # elif object == 'leaf':
        #     return self.component_model.predict(image, conf=threshold, classes=[3], imgsz=640)
        #     #return self.component_model.predict(image, classes=[3], imgsz=640)
            
        elif object == 'reference_object':
            return self.reference_model.predict(image, conf=threshold, classes=[1])
            #return self.reference_model.predict(image, conf=threshold, classes=[1], imgsz=640)
            #return self.reference_model.predict(image, classes=[2], imgsz=640)
            
        else: # default to flower
            return self.component_model.predict(image, conf=threshold, classes=[2])
            #return self.component_model.predict(image, conf=threshold, classes=[2], imgsz=640)
            #return self.component_model.predict(image, classes=[2], imgsz=640)
        
    def getMask(self, results) -> Image:
        '''
        create prediction mask
        '''
        
        img = results[0].plot(conf=False, labels=False, boxes=False, masks=True)
        black_img = np.zeros_like(img) # create a black image for mask
        img = results[0].plot(conf=False, labels=False, boxes=False, masks=True, img=black_img)
        img = Image.fromarray(img[..., ::-1])

        img = np.array(img)
        img[img != 0] = 255
        img = Image.fromarray(img)
        return img
    
    def savePlot(self, path, component, threshold, conf=False, labels=False, boxes=True, masks=True):
        '''
        Save the plot of the results
        '''
        root = "api/data/output/"

        if component == "flower":
            results = self.runInference(path, "flower", threshold)
        elif component == "leaf":
            results = self.runInference(path, "leaf", threshold)

        img = results[0].plot(conf=conf, labels=labels, boxes=boxes, masks=masks)
        img = Image.fromarray(img[..., ::-1])
        filename = path.split("/")[-1]
        filename1 = f"api/data/output/flower_{component}_{threshold}_{filename}"
        img.save(filename1)
        #mask = self.getMask(results)
        #mask.save(f"api/data/output/flower_mask_{component}_{threshold}_{filename}")

      
        results = self.runInference(path, "reference_object", threshold)

        img = results[0].plot(conf=conf, labels=labels, boxes=boxes, masks=masks)
        img = Image.fromarray(img[..., ::-1])
        filename = path.split("/")[-1]
        filename1 = f"api/data/output/reference_object_{component}_{threshold}_{filename}"
        img.save(filename1)
        #mask = self.getMask(results)
        #mask.save(f"api/data/output/reference_object_mask_{component}_{threshold}_{filename}")

        #results[0].save(filename)


    def getPlot(self, path, component, threshold, conf=True, labels=True, boxes=True, masks=True):
        '''
        Save the plot of the results
        '''
        root = "api/data/output/"

        if component == "flower":
            results = self.runInference(path, "flower_and_object", threshold)
        elif component == "leaf":
            results = self.runInference(path, "leaf_and_object", threshold)

        img = results[0].plot(conf=conf, labels=labels, boxes=boxes, masks=masks)
        img = Image.fromarray(img[..., ::-1])
        #filename = path.split("/")[-1]
        #filename = f"api/data/output/{component}_{threshold}_{filename}"
        #img.save(filename)
        #results[0].save(filename)
        return img

        
        