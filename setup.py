"""
Setup script for FloralArea package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name="floralarea",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for estimating floral area using YOLOv8 computer vision",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/floralarea",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gradio>=4.0.0",
        "ultralytics>=8.0.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'floralarea-demo=examples.demo_app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'floralarea': ['config/*.json'],
    },
)
