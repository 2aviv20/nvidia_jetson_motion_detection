#!/bin/bash

# Object Detection Runner Script
# Author: Claude Code Assistant

echo "🚀 Object Detection Application Launcher"
echo "========================================"
echo ""

# Function to display header
show_header() {
    echo "Available Detection Interfaces:"
    echo ""
    echo "1. 🖥️  GUI Interface (Fast ImGui Style)"
    echo "   - Touch-friendly controls"
    echo "   - Real-time video display" 
    echo "   - Performance statistics"
    echo "   - RTSP streaming with copy URL"
    echo ""
    echo "2. 🖨️  Console Interface (OpenCV)"
    echo "   - Maximum performance"
    echo "   - Terminal output"
    echo "   - Keyboard controls"
    echo "   - RTSP streaming support"
    echo ""
    echo "3. 🌐 Web Interface (Streamlit)"
    echo "   - Browser-based GUI"
    echo "   - Remote access capable"
    echo "   - Modern web dashboard"
    echo "   - Mobile friendly"
    echo ""
    echo "4. ⚙️  Advanced Options"
    echo ""
}

# Function to run GUI interface
run_gui() {
    echo "🎮 Starting GUI Interface..."
    echo "Controls:"
    echo "  - F11 or F: Toggle fullscreen"
    echo "  - ESC: Exit fullscreen / Quit"
    echo "  - Click buttons to interact"
    echo ""
    
    # Check if user wants fullscreen
    read -p "Start in fullscreen mode? [y/N]: " fullscreen
    
    if [[ $fullscreen =~ ^[Yy]$ ]]; then
        echo "🖥️ Starting in fullscreen mode..."
        python3 fast_imgui_detection.py --fullscreen
    else
        echo "🪟 Starting in windowed mode..."
        python3 fast_imgui_detection.py
    fi
}

# Function to run console interface
run_console() {
    echo "🖨️ Starting Console Interface..."
    echo "Available options:"
    echo ""
    echo "a) Basic detection (camera 0, default settings)"
    echo "b) Custom settings (model, confidence, device)"
    echo "c) RTSP streaming enabled"
    echo "d) No display mode (terminal only)"
    echo ""
    
    read -p "Select option [a]: " console_option
    console_option=${console_option:-a}
    
    case $console_option in
        a|A)
            echo "🚀 Starting basic detection..."
            echo "Press 'q' to quit"
            python3 pytorch_gpu_detection.py
            ;;
        b|B)
            echo "⚙️ Custom Settings:"
            echo "Available models: yolov5s (fast), yolov5m (balanced), yolov5l (accurate)"
            read -p "Model [yolov5s]: " model
            model=${model:-yolov5s}
            
            read -p "Confidence threshold [0.5]: " conf
            conf=${conf:-0.5}
            
            read -p "Device [cuda]: " device
            device=${device:-cuda}
            
            echo "🚀 Starting with custom settings..."
            python3 pytorch_gpu_detection.py --model "$model" --conf "$conf" --device "$device"
            ;;
        c|C)
            echo "📡 Starting with RTSP streaming..."
            read -p "RTSP port [8554]: " port
            port=${port:-8554}
            
            # Get local IP for display
            LOCAL_IP=$(python3 -c "
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        print(s.getsockname()[0])
except:
    print('192.168.1.100')
")
            
            echo "📱 RTSP Stream will be available at:"
            echo "   http://$LOCAL_IP:$port/stream.mjpg"
            echo ""
            echo "🚀 Starting detection with RTSP..."
            python3 pytorch_gpu_detection.py --rtsp --rtsp-port "$port"
            ;;
        d|D)
            echo "🖥️ Starting terminal-only mode..."
            python3 pytorch_gpu_detection.py --no-display
            ;;
        *)
            echo "❌ Invalid option, starting basic detection..."
            python3 pytorch_gpu_detection.py
            ;;
    esac
}

# Function to run web interface
run_web() {
    echo "🌐 Starting Web Interface..."
    echo ""
    echo "Choose web interface:"
    echo "a) Full Dashboard (with charts)"
    echo "b) Simple Interface (lightweight)"
    echo ""
    
    read -p "Select option [a]: " web_option
    web_option=${web_option:-a}
    
    # Get local IP
    LOCAL_IP=$(python3 -c "
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        print(s.getsockname()[0])
except:
    print('192.168.1.100')
")
    
    echo ""
    echo "🌐 Web Interface URLs:"
    echo "   Local:    http://localhost:8501"
    echo "   Network:  http://$LOCAL_IP:8501"
    echo ""
    echo "📱 Access from any device on your network!"
    echo "⏹️ Press Ctrl+C to stop the server"
    echo ""
    
    case $web_option in
        a|A)
            echo "🚀 Starting full dashboard..."
            streamlit run streamlit_detection_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
            ;;
        b|B)
            echo "🚀 Starting simple interface..."
            streamlit run streamlit_simple_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
            ;;
        *)
            echo "❌ Invalid option, starting full dashboard..."
            streamlit run streamlit_detection_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
            ;;
    esac
}

# Function for advanced options
advanced_options() {
    echo "⚙️ Advanced Options:"
    echo ""
    echo "a) OpenCV GUI (original with toggle buttons)"
    echo "b) Pygame ImGui (styled interface)"  
    echo "c) List available cameras"
    echo "d) Test GPU/CUDA availability"
    echo "e) Install missing dependencies"
    echo "f) View system information"
    echo ""
    
    read -p "Select option: " advanced_option
    
    case $advanced_option in
        a|A)
            echo "🎮 Starting OpenCV GUI..."
            python3 opencv_gui_detection.py
            ;;
        b|B)
            echo "🎨 Starting Pygame ImGui interface..."
            python3 pygame_imgui_detection.py
            ;;
        c|C)
            echo "📹 Available cameras:"
            python3 -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        break
"
            ;;
        d|D)
            echo "🔍 Testing GPU/CUDA availability..."
            python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - will use CPU')
"
            ;;
        e|E)
            echo "📦 Installing missing dependencies..."
            echo "This may take several minutes..."
            pip3 install --upgrade -r requirements_tensorrt.txt 2>/dev/null || \
            pip3 install torch torchvision opencv-python numpy streamlit plotly
            echo "✅ Dependencies installation complete"
            ;;
        f|F)
            echo "💻 System Information:"
            echo "OS: $(uname -a)"
            echo "Python: $(python3 --version)"
            echo "GPU Memory:"
            free -h
            echo "Disk Space:"
            df -h .
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
}

# Main menu loop
while true; do
    show_header
    
    read -p "Select interface [1]: " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            run_gui
            ;;
        2)
            run_console
            ;;
        3)
            run_web
            ;;
        4)
            advanced_options
            read -p "Press Enter to return to main menu..." 
            continue
            ;;
        q|Q|quit|exit)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid selection. Please choose 1, 2, 3, or 4"
            read -p "Press Enter to continue..."
            continue
            ;;
    esac
    
    # Ask if user wants to run something else
    echo ""
    read -p "Run another interface? [y/N]: " again
    if [[ ! $again =~ ^[Yy]$ ]]; then
        echo "👋 Goodbye!"
        break
    fi
    echo ""
done