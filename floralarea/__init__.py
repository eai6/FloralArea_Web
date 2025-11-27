"""
FloralArea - Floral Area Measurement using Computer Vision

Production-ready package for measuring floral area from images.

Main Features:
- Reference-based measurement (±2-5% accuracy)
- Distance-based measurement with LiDAR support (±3-10% accuracy)
- SAM 3 text-prompted segmentation (zero-shot, no training needed)
- Batch processing support
- Multiple camera models supported

Quick Start:
    >>> from floralarea import FloralAreaAnalyzer
    >>> 
    >>> # Initialize analyzer (loads models once)
    >>> analyzer = FloralAreaAnalyzer(camera_model='iphone_15_pro')
    >>> 
    >>> # Measure single image
    >>> result = analyzer.measure('flower.jpg')
    >>> print(f"Area: {result['area_cm2']:.2f} cm²")
    >>> 
    >>> # Batch processing
    >>> results = analyzer.measure_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

For more examples, see: examples/smart_measurement.py
"""

__version__ = "1.0.0"
__author__ = "FloralArea Team"
__license__ = "MIT"

# Core functionality
try:
    from floralarea.core.image_analyzer import FloralAreaAnalyzer, measure_floral_area
    _HAS_CORE = True
except ImportError as e:
    _HAS_CORE = False
    FloralAreaAnalyzer = None
    measure_floral_area = None
    _CORE_IMPORT_ERROR = str(e)

# SAM 3 segmentation (HuggingFace implementation)
try:
    from floralarea.cv.sam3_huggingface import SAM3HuggingFaceSegmenter
    _HAS_SAM3 = True
except ImportError as e:
    _HAS_SAM3 = False
    SAM3HuggingFaceSegmenter = None
    _SAM3_IMPORT_ERROR = str(e)

# Distance-based measurement utilities
try:
    from floralarea.cv.distance_measurement import (
        DistanceBasedMeasurement,
        estimate_area_from_distance,
        estimate_error_percentage
    )
    _HAS_DISTANCE = True
except ImportError as e:
    _HAS_DISTANCE = False
    DistanceBasedMeasurement = None
    estimate_area_from_distance = None
    estimate_error_percentage = None
    _DISTANCE_IMPORT_ERROR = str(e)


# Public API
__all__ = [
    # Main classes
    'FloralAreaAnalyzer',
    'SAM3HuggingFaceSegmenter',
    'DistanceBasedMeasurement',
    
    # Convenience functions
    'measure_floral_area',
    'estimate_area_from_distance',
    'estimate_error_percentage',
    
    # System info
    'check_dependencies',
    'print_system_info',
    'get_version',
    'get_supported_cameras',
]


def check_dependencies():
    """
    Check if all required dependencies are installed
    
    Returns:
        dict: Status of each component
    """
    status = {
        'core': _HAS_CORE,
        'sam3': _HAS_SAM3,
        'distance': _HAS_DISTANCE,
    }
    
    return status


def print_system_info():
    """
    Print detailed system information about FloralArea installation
    """
    print("="*70)
    print(f"FloralArea v{__version__}")
    print("="*70)
    
    status = check_dependencies()
    
    print("\n📦 Component Status:")
    print("-"*70)
    
    # Core
    if status['core']:
        print("  ✅ Core (FloralAreaAnalyzer)")
    else:
        print("  ❌ Core (FloralAreaAnalyzer)")
        print(f"     Error: {_CORE_IMPORT_ERROR}")
    
    # SAM 3
    if status['sam3']:
        print("  ✅ SAM 3 Segmentation (HuggingFace)")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"     Device: {device}")
        except:
            pass
    else:
        print("  ❌ SAM 3 Segmentation")
        print(f"     Error: {_SAM3_IMPORT_ERROR}")
        print("     Install: pip install transformers torch pillow")
    
    # Distance measurement
    if status['distance']:
        print("  ✅ Distance-based Measurement (LiDAR-ready)")
    else:
        print("  ❌ Distance-based Measurement")
        print(f"     Error: {_DISTANCE_IMPORT_ERROR}")
    
    print("-"*70)
    
    # Overall status
    all_ready = all(status.values())
    
    if all_ready:
        print("\n✅ FloralArea is ready to use!")
        print("\nQuick start:")
        print("  >>> from floralarea import FloralAreaAnalyzer")
        print("  >>> analyzer = FloralAreaAnalyzer()")
        print("  >>> result = analyzer.measure('flower.jpg')")
        print("  >>> print(f\"Area: {result['area_cm2']:.2f} cm²\")")
    else:
        print("\n❌ FloralArea is not fully installed")
        print("\nInstall missing dependencies:")
        print("  pip install transformers torch torchvision pillow numpy opencv-python matplotlib")
    
    print("="*70)


def get_version():
    """Get FloralArea version"""
    return __version__


def get_supported_cameras():
    """
    Get list of supported camera models
    
    Returns:
        dict: Camera model specifications
    """
    if not _HAS_DISTANCE:
        return {}
    
    return DistanceBasedMeasurement.CAMERA_SPECS