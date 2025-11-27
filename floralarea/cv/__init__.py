"""
Computer Vision modules for FloralArea

Includes:
- SAM 3 segmentation (text-prompted, zero-shot)
- Distance-based measurement (LiDAR-ready)
"""

# SAM 3
try:
    from floralarea.cv.sam3_huggingface import SAM3HuggingFaceSegmenter
except ImportError:
    SAM3HuggingFaceSegmenter = None

# Distance measurement
try:
    from floralarea.cv.distance_measurement import (
        DistanceBasedMeasurement,
        estimate_area_from_distance,
        estimate_error_percentage
    )
except ImportError:
    DistanceBasedMeasurement = None
    estimate_area_from_distance = None
    estimate_error_percentage = None

__all__ = [
    'SAM3HuggingFaceSegmenter',
    'DistanceBasedMeasurement',
    'estimate_area_from_distance',
    'estimate_error_percentage',
]