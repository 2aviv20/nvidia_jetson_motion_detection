# Jetson Nano GPU Object Detection

PyTorch-powered real-time object detection optimized for NVIDIA Jetson Nano with CUDA GPU acceleration, specifically designed to detect:
- **Humans** (people)
- **Vehicles** (cars, trucks, buses, motorcycles)  
- **Aircraft/Drones** (airplanes and drone-like objects)

## Features

- **Full GPU Acceleration**: PyTorch with CUDA support for maximum performance
- **Real-time Performance**: 15-25 FPS on Jetson Nano 4GB
- **FP16 Optimization**: Half-precision for faster inference
- **Live Video Display**: Real-time visualization with bounding boxes
- **Terminal Mode**: Headless operation for SSH/remote use
- **RTSP Streaming**: Hardware-accelerated H.264 streaming to mobile devices and network clients

## Prerequisites

Make sure you have PyTorch with CUDA support installed:
```bash
pip3 install ultralytics torch torchvision
```

## How to Run

The main script is `pytorch_gpu_detection.py` which provides direct YOLOv5 detection on every frame.

### Basic Usage

**Live video detection with camera display:**
```bash
python3 pytorch_gpu_detection.py -i 0
```

**Headless mode (no display, perfect for SSH):**
```bash
python3 pytorch_gpu_detection.py -i 0 --no-display
```

**With RTSP streaming for remote viewing:**
```bash
python3 pytorch_gpu_detection.py -i 0 --rtsp
```

**Headless with RTSP streaming:**
```bash
python3 pytorch_gpu_detection.py -i 0 --rtsp --no-display
```

### Advanced Options

**Use different YOLOv5 models:**
```bash
# Fast model (default)
python3 pytorch_gpu_detection.py -i 0 --model yolov5s

# Better accuracy, slower
python3 pytorch_gpu_detection.py -i 0 --model yolov5m

# Best accuracy, slowest
python3 pytorch_gpu_detection.py -i 0 --model yolov5l
```

**Custom confidence threshold:**
```bash
python3 pytorch_gpu_detection.py -i 0 --conf 0.6
```

**Use video file instead of camera:**
```bash
python3 pytorch_gpu_detection.py -i /path/to/video.mp4
```

**Custom RTSP port:**
```bash
python3 pytorch_gpu_detection.py -i 0 --rtsp --rtsp-port 5000
```

### Complete Command Line Options

```bash
python3 pytorch_gpu_detection.py [OPTIONS]

Options:
  -i, --input          Input source (0 for camera, path for video file)
  --model              YOLOv5 model (yolov5s, yolov5m, yolov5l)
  --conf               Confidence threshold (0.0-1.0, default: 0.5)
  --no-display         Run without video display (headless mode)
  --rtsp               Enable RTSP streaming
  --rtsp-port          RTSP streaming port (default: 8554)
```

### Performance

- **FPS**: 15-25 FPS on Jetson Nano 4GB
- **Efficiency**: Full-frame object detection with GPU acceleration
- **Memory Usage**: ~2GB GPU memory
- **Models**: Automatic download of YOLOv5 models

## Controls

- **Live mode**: Press 'q' to quit
- **Terminal mode**: Press Ctrl+C to stop
- Real-time FPS and detection statistics displayed

## GPU Optimization Tips

1. **Maximum Performance Mode**: `sudo nvpmodel -m 0`
2. **Increase GPU Memory**: `sudo systemctl disable nvzramconfig`
3. **Cool the Jetson**: Use active cooling for sustained performance
4. **Close Other Apps**: Free GPU memory for detection

## RTSP Streaming

The script supports hardware-accelerated RTSP streaming for remote viewing on mobile devices and network clients.

### Enable RTSP Streaming
```bash
# Basic RTSP streaming
python3 pytorch_gpu_detection.py -i 0 --rtsp

# Custom port
python3 pytorch_gpu_detection.py -i 0 --rtsp --rtsp-port 5000

# Headless with RTSP
python3 pytorch_gpu_detection.py -i 0 --rtsp --no-display
```

### View RTSP Stream
Find your Jetson's IP address:
```bash
hostname -I
```

**VLC (iOS/Android/Desktop):**
- Open VLC → Network Stream
- Enter: `udp://YOUR_JETSON_IP:8554`
- Replace `YOUR_JETSON_IP` with actual IP

**GStreamer Client:**
```bash
gst-launch-1.0 udpsrc multicast-group=224.1.1.1 port=8554 ! application/x-rtp ! rtph264depay ! h264parse ! omxh264dec ! autovideosink
```

**FFplay:**
```bash
ffplay udp://224.1.1.1:8554
```

### RTSP Features
- Hardware H.264 encoding (minimal CPU impact)
- Real-time streaming with detection overlays
- Works in both display and headless modes
- Compatible with mobile VLC apps
- Configurable streaming port

## Advanced Options

### Command Line Options Summary
```bash
--model yolov5s          # Fast model (default)
--model yolov5m          # Balanced speed/accuracy  
--model yolov5l          # Best accuracy
--conf 0.5               # Confidence threshold (default: 0.5)
--no-display             # Headless mode
--rtsp                   # Enable RTSP streaming
--rtsp-port 8554         # Set streaming port (default: 8554)
```

## Detection Classes

Optimized for security and surveillance applications:
- **Human** (class 0) - Green boxes
- **Car** (class 2) - Blue boxes  
- **Motorcycle** (class 3) - Yellow boxes
- **Aircraft** (class 4) - Magenta boxes (includes drones)
- **Bus** (class 5) - Cyan boxes
- **Truck** (class 7) - Purple boxes

## Technical Specifications

- **Hardware**: NVIDIA Jetson Nano 4GB with Tegra X1 GPU
- **Framework**: PyTorch with CUDA acceleration
- **Model**: YOLOv5 with FP16 optimization
- **Memory Usage**: ~2GB GPU memory

## Troubleshooting

1. **CUDA not available**: Ensure PyTorch was compiled with CUDA support
2. **Out of memory**: Close other applications or use a smaller model (yolov5s)
3. **Low FPS**: Check GPU temperature and use `nvpmodel -m 0`
4. **Model download fails**: Check internet connection, models download automatically
5. **Camera not found**: Check camera connection and permissions, try different input numbers