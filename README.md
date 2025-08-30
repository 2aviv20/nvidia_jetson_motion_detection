# Jetson Nano GPU Object Detection

PyTorch-powered real-time object detection optimized for NVIDIA Jetson Nano with CUDA GPU acceleration, specifically designed to detect:
- **Humans** (people)
- **Vehicles** (cars, trucks, buses, motorcycles)  
- **Aircraft/Drones** (airplanes and drone-like objects)

## Features

- **Full GPU Acceleration**: PyTorch with CUDA support for maximum performance
- **Motion Detection**: Advanced GPU-accelerated motion detection with background subtraction
- **Smart Processing**: Object detection only runs in motion areas for efficiency
- **Real-time Performance**: 15-35 FPS on Jetson Nano 4GB
- **FP16 Optimization**: Half-precision for faster inference
- **Live Video Display**: Real-time visualization with bounding boxes
- **Terminal Mode**: Headless operation for SSH/remote use
- **RTSP Streaming**: Hardware-accelerated H.264 streaming to mobile devices and network clients

## Setup

Install PyTorch dependencies:
```bash
pip3 install ultralytics seaborn
```

## Main Scripts

### 1. GPU Motion Detection (Recommended)
**File**: `fast_gpu_motion.py`

Advanced motion detection + object recognition with maximum efficiency:

```bash
# Live video with motion detection
python3 fast_gpu_motion.py -i 0

# Terminal mode (faster, no display)
python3 fast_gpu_motion.py -i 0 --no-display

# With RTSP streaming for mobile devices
python3 fast_gpu_motion.py -i 0 --rtsp

# Headless with RTSP streaming
python3 fast_gpu_motion.py -i 0 --rtsp --no-display

# Adjust motion sensitivity
python3 fast_gpu_motion.py -i 0 --motion-threshold 30 --min-area 1000
```

**Features:**
- GPU motion detection at 20-35 FPS
- Object detection only in motion areas (10x efficiency gain)
- Background subtraction with adaptive learning
- Morphological filtering for noise reduction
- Real-time motion visualization

### 2. Pure GPU Object Detection
**File**: `pytorch_gpu_detection.py`

Direct YOLOv5 detection on every frame:

```bash
# Live video detection
python3 pytorch_gpu_detection.py -i 0

# Terminal mode
python3 pytorch_gpu_detection.py -i 0 --no-display

# With RTSP streaming
python3 pytorch_gpu_detection.py -i 0 --rtsp

# Use larger model for better accuracy
python3 pytorch_gpu_detection.py -i 0 --model yolov5m --rtsp
```

**Features:**
- Full-frame object detection at 15-25 FPS
- Direct PyTorch YOLOv5 inference
- FP16 optimization for speed
- Automatic model download

## Performance Comparison

| Method | FPS | Efficiency | Use Case |
|--------|-----|------------|-----------|
| GPU Motion Detection | 25-35 | Highest | Security, monitoring |
| Pure GPU Detection | 15-25 | High | General detection |
| CPU OpenCV (old) | 0.8-2 | Low | Not recommended |

## Controls

- **Live mode**: Press 'q' to quit
- **Terminal mode**: Press Ctrl+C to stop
- Real-time FPS and detection statistics displayed

## GPU Optimization Tips

1. **Maximum Performance Mode**: `sudo nvpmodel -m 0`
2. **Increase GPU Memory**: `sudo systemctl disable nvzramconfig`
3. **Cool the Jetson**: Use active cooling for sustained performance
4. **Close Other Apps**: Free GPU memory for detection
5. **Use Motion Detection**: 10x more efficient than full-frame detection

## RTSP Streaming

Both scripts support hardware-accelerated RTSP streaming for remote viewing on mobile devices and network clients.

### Enable RTSP Streaming
```bash
# Motion detection with RTSP
python3 fast_gpu_motion.py -i 0 --rtsp

# Object detection with RTSP  
python3 pytorch_gpu_detection.py -i 0 --rtsp

# Custom port
python3 fast_gpu_motion.py -i 0 --rtsp --rtsp-port 5000
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

### Motion Detection Parameters
```bash
--motion-threshold 25     # Motion sensitivity (lower = more sensitive)
--min-area 500           # Minimum motion area to consider
```

### Model Options
```bash
--model yolov5s          # Fast, lower accuracy
--model yolov5m          # Balanced speed/accuracy
--model yolov5l          # Slow, higher accuracy
```

### RTSP Options
```bash
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
- **Model**: YOLOv5s with FP16 optimization
- **Motion Algorithm**: GPU background subtraction with morphological filtering
- **Memory Usage**: ~2GB GPU memory for motion detection mode

## Troubleshooting

1. **CUDA not available**: Ensure PyTorch was compiled with CUDA support
2. **Out of memory**: Use motion detection mode or close other applications  
3. **Low FPS**: Check GPU temperature and use `nvpmodel -m 0`
4. **Model download fails**: Check internet connection, model downloads automatically