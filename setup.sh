#!/bin/bash

echo "Setting up Object Detection for Jetson Nano"
echo "=========================================="

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install opencv-python numpy argparse

# Create models directory
mkdir -p models
cd models

# Download YOLOv5s ONNX model (optimized for Jetson Nano)
echo "Downloading YOLOv5s ONNX model..."
if [ ! -f "yolov5s.onnx" ]; then
    wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
    if [ $? -eq 0 ]; then
        echo "YOLOv5s model downloaded successfully"
        # Move model to parent directory for easy access
        mv yolov5s.onnx ../
    else
        echo "Failed to download YOLOv5s model. Please download manually from:"
        echo "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"
    fi
else
    echo "YOLOv5s model already exists"
fi

cd ..

# Make the Python script executable
chmod +x object_detection.py

echo ""
echo "Setup complete!"
echo ""
echo "Usage examples:"
echo "1. Use webcam (camera 0): python3 object_detection.py --input 0"
echo "2. Use video file: python3 object_detection.py --input /path/to/video.mp4"
echo "3. Save output: python3 object_detection.py --input 0 --output output.mp4"
echo "4. Adjust confidence: python3 object_detection.py --input 0 --conf 0.6"
echo ""
echo "The system will detect: Humans, Cars, Trucks, Buses, Motorcycles, Aircraft/Drones"
echo "Press 'q' to quit during detection"