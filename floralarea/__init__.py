"""
FloralArea - A package for estimating floral area using YOLOv8
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from floralarea.cv.yolov8 import yolov8
from floralarea.cv.img_processing import getPixelArea, apply_preprocessing
from floralarea.cv.runTiles import runTilles

__all__ = [
    'yolov8',
    'getPixelArea',
    'apply_preprocessing',
    'runTilles',
]
