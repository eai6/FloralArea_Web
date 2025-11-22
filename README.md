# FloralArea

A Python package for estimating floral area using YOLOv8 computer vision models.

## Overview

FloralArea uses state-of-the-art YOLOv8 object detection and segmentation models to automatically measure the area of flowers in images. The package includes a reference object calibration system to provide accurate real-world measurements.

## Features

- 🌸 Automatic flower detection and segmentation
- 📏 Reference object-based calibration for accurate measurements
- 🔲 Image tiling for processing large images
- 🖼️ Gradio web interface for easy interaction
- 📦 Clean, modular package structure
- 🔧 Configurable models and parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository (or download the package)
cd floralarea

# Install in development mode (recommended for research)
pip install -e .

# Or install normally
pip install .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using the Package Programmatically

```python
from floralarea.cv import yolov8, img_processing as ip, runTiles
import json

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Initialize YOLO model
model = yolov8.yolov8(
    config['flower_model'],
    config['reference_object_model']
)

# Preprocess image
ip.apply_preprocessing('path/to/your/image.jpg')

# Estimate floral area
component_area, image = runTiles.runTilles(
    model, 
    'path/to/your/image.jpg', 
    threshold=0.5, 
    num_tiles=2, 
    component='flower'
)

# Get reference object area for scaling
results = model.runInference('path/to/your/image.jpg', 'reference_object', 0.5)
mask = model.getMask(results)
reference_pixel_area = ip.getPixelArea(mask)

# Calculate actual area
actual_area = (component_area / reference_pixel_area) * config['reference_object_area']
print(f"Floral area: {actual_area:.2f} cm²")
```

### Using the Gradio Demo

```bash
# Run the demo application
cd examples
python demo_app.py
```

Then open your browser to `http://localhost:7860`

## Configuration

Edit `config/config.json` to customize your setup:

```json
{
    "flower_model": "models/flower_model.pt",
    "reference_object_model": "models/reference_object_model.pt",
    "reference_object_area": 58.0
}
```

- `flower_model`: Path to your trained YOLOv8 flower detection model
- `reference_object_model`: Path to your trained YOLOv8 reference object model
- `reference_object_area`: Known area of your reference object in cm²

## Project Structure

```
floralarea/
├── floralarea/           # Main package
│   ├── cv/              # Computer vision modules
│   │   ├── yolov8.py           # YOLO model wrapper
│   │   ├── img_processing.py  # Image preprocessing
│   │   └── runTiles.py         # Tiling for large images
│   └── utils/           # Utility functions
│       └── helpers.py
├── config/              # Configuration files
│   └── config.json
├── models/              # Place your trained models here
├── examples/            # Usage examples
│   └── demo_app.py     # Gradio demo application
├── setup.py            # Installation script
├── pyproject.toml      # Modern Python packaging
└── requirements.txt    # Dependencies
```

## Model Training

You'll need to provide your own trained YOLOv8 models:

1. **Flower Model**: Trained to detect and segment flowers (class 2)
2. **Reference Object Model**: Trained to detect your reference object (class 1)

Place your `.pt` model files in the `models/` directory and update `config/config.json` accordingly.

## How It Works

1. **Image Preprocessing**: Images are resized to 640x640 and contrast-enhanced
2. **Tiling**: Large images are split into tiles for better detection
3. **Detection**: YOLOv8 models detect and segment flowers and reference objects
4. **Pixel Counting**: Segmentation masks are counted to get pixel areas
5. **Calibration**: Reference object provides scale for real-world measurements
6. **Area Calculation**: Flower area = (flower_pixels / reference_pixels) × reference_area

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black floralarea/
```

### Type Checking

```bash
mypy floralarea/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{floralarea2024,
  author = {Your Name},
  title = {FloralArea: Automated Floral Area Estimation},
  year = {2024},
  url = {https://github.com/yourusername/floralarea}
}
```

## Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Web interface powered by [Gradio](https://gradio.app/)

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/yourusername/floralarea/issues).
