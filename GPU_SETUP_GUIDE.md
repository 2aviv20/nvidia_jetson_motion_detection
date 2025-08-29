# GPU Setup Guide for Jetson Nano

## Current Status
- CUDA 10.2 is installed but OpenCV was not compiled with CUDA support
- Currently running on optimized CPU backend

## To Enable GPU Acceleration

### Option 1: Install OpenCV with CUDA Support (Recommended)

```bash
# Remove current OpenCV
pip3 uninstall opencv-python opencv-contrib-python

# Install OpenCV compiled with CUDA for Jetson
# This requires building from source or using pre-built wheels
sudo apt update
sudo apt install -y libopencv-dev libopencv-contrib-dev

# Or install from NVIDIA's pre-built packages
sudo apt install -y nvidia-opencv
```

### Option 2: Build OpenCV from Source (Advanced)

```bash
# Install dependencies
sudo apt install -y build-essential cmake git pkg-config
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y python3-dev python3-numpy
sudo apt install -y libtbb2 libtbb-dev

# Clone OpenCV
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Build with CUDA
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=5.3 \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF ..

make -j4
sudo make install
```

### Option 3: Use TensorRT (Highest Performance)

```bash
# TensorRT is already installed on Jetson Nano
# Convert ONNX model to TensorRT engine
python3 -c "
import tensorrt as trt
# Add TensorRT optimization code here
"
```

## Verification

After installation, verify GPU support:

```bash
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Performance Comparison

| Backend | Expected FPS on Jetson Nano |
|---------|------------------------------|
| CPU (current) | 8-12 FPS |
| CUDA | 15-25 FPS |
| TensorRT | 25-35 FPS |

## Current Optimizations Applied

1. **ARM NEON optimizations** - Enabled for ARM CPU
2. **Reduced input size** - Using 416x416 instead of 640x640
3. **Vectorized processing** - NumPy operations for faster post-processing
4. **Frame skipping** - Process every 3rd frame
5. **High confidence threshold** - Reduce false positives
6. **Target classes only** - Filter specific object types