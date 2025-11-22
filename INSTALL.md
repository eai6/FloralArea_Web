# Installation Guide for macOS

This guide will walk you through installing the FloralArea package on your Mac.

## Prerequisites

### 1. Check Python Installation

Open Terminal and check your Python version:

```bash
python3 --version
```

You should see Python 3.8 or higher. If not, install Python from [python.org](https://www.python.org/downloads/).

### 2. Install pip (if not already installed)

```bash
python3 -m ensurepip --upgrade
```

## Installation Steps

### Step 1: Navigate to the Package Directory

```bash
cd /path/to/floralarea
```

Replace `/path/to/floralarea` with the actual path where you saved the package.

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the beginning.

### Step 3: Install the Package

#### Option A: Development Installation (Recommended for Research)

This allows you to edit the code and see changes immediately:

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode.

#### Option B: Regular Installation

```bash
pip install .
```

### Step 4: Verify Installation

Test that the package is installed correctly:

```bash
python3 -c "import floralarea; print(floralarea.__version__)"
```

You should see: `0.1.0`

## Setting Up Your Models

### 1. Place Model Files

Copy your trained YOLOv8 model files into the `models/` directory:

```bash
# Example structure
models/
├── flower_model.pt
└── reference_object_model.pt
```

### 2. Update Configuration

Edit `config/config.json` to match your model file names:

```json
{
    "flower_model": "models/flower_model.pt",
    "reference_object_model": "models/reference_object_model.pt",
    "reference_object_area": 58.0
}
```

Update `reference_object_area` to match your reference object's actual area in cm².

## Running the Demo Application

### Start the Gradio Interface

```bash
cd examples
python demo_app.py
```

Open your browser to `http://localhost:7860`

## Using the Package in Your Code

Create a new Python file (e.g., `my_analysis.py`):

```python
from floralarea.cv import yolov8, img_processing as ip
import json

# Load config
with open('config/config.json') as f:
    config = json.load(f)

# Initialize model
model = yolov8.yolov8(
    config['flower_model'],
    config['reference_object_model']
)

# Process an image
image_path = 'path/to/your/image.jpg'
ip.apply_preprocessing(image_path)

# Run inference
results = model.runInference(image_path, 'flower', 0.5)
mask = model.getMask(results)
area = ip.getPixelArea(mask)

print(f"Flower pixel area: {area}")
```

## Troubleshooting

### Issue: "No module named 'floralarea'"

**Solution**: Make sure you've installed the package and activated your virtual environment:

```bash
source venv/bin/activate
pip install -e .
```

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install PyTorch for your system. For Mac with Apple Silicon:

```bash
pip install torch torchvision
```

For Intel Macs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Model files not found

**Solution**: Check that your model files are in the correct location and that `config/config.json` has the right paths.

```bash
ls -la models/
cat config/config.json
```

### Issue: Permission denied

**Solution**: You might need to adjust file permissions:

```bash
chmod +x examples/demo_app.py
```

## Updating the Package

If you make changes to the code and installed in development mode (`-e`), changes take effect immediately. No reinstallation needed!

If you installed normally, reinstall after changes:

```bash
pip install --upgrade .
```

## Uninstalling

To remove the package:

```bash
pip uninstall floralarea
```

To remove the virtual environment:

```bash
deactivate  # Exit the virtual environment
rm -rf venv  # Delete the virtual environment directory
```

## Next Steps

1. **Test with sample images**: Try processing some test images
2. **Adjust parameters**: Experiment with confidence thresholds
3. **Batch processing**: Write scripts to process multiple images
4. **Integration**: Integrate into your research workflow

## Getting Help

- Check the main README.md for API documentation
- Review the example code in `examples/demo_app.py`
- Look at the source code in `floralarea/cv/` for implementation details

## Best Practices for Research

1. **Version control**: Use git to track your changes
2. **Document parameters**: Keep notes on threshold values and settings
3. **Validate results**: Compare automated measurements with manual measurements
4. **Keep backups**: Save your trained models in multiple locations
5. **Log experiments**: Record which parameters work best for your data
