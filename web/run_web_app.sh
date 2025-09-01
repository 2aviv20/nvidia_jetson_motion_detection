#!/bin/bash
echo "🚀 Starting Object Detection Web Interface..."
echo "📱 Access from any device on your network!"
echo ""

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

echo "🌐 Web Interface URLs:"
echo "   Local:    http://localhost:8501"
echo "   Network:  http://$LOCAL_IP:8501"
echo ""
echo "🎮 Usage:"
echo "   1. Open the URL in any web browser"
echo "   2. Use the sidebar to configure detection"
echo "   3. Click 'Initialize Detector' then 'Start Detection'"
echo "   4. View live feed and performance analytics"
echo ""
echo "📱 Mobile/Tablet: Works on any device with a web browser!"
echo "⏹️  Press Ctrl+C to stop the server"
echo "=" * 60

# Run Streamlit
streamlit run streamlit_detection_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true