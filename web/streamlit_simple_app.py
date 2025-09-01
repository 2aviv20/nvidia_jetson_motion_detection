#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from PIL import Image
import pandas as pd
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for touch-friendly interface
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 60px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .start-button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .stop-button {
        background-color: #f44336 !important;
        color: white !important;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-stopped {
        color: #f44336;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

class SimpleStreamlitApp:
    def __init__(self):
        # Initialize session state
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'cap' not in st.session_state:
            st.session_state.cap = None
        if 'current_frame' not in st.session_state:
            st.session_state.current_frame = None
        if 'current_fps' not in st.session_state:
            st.session_state.current_fps = 0
        if 'current_detections' not in st.session_state:
            st.session_state.current_detections = 0
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
        if 'detection_thread' not in st.session_state:
            st.session_state.detection_thread = None
        
    def initialize_detector(self, model_name, conf_threshold, device, enable_rtsp):
        """Initialize detector"""
        try:
            st.session_state.detector = PyTorchGPUDetector(
                model_name=model_name,
                conf_threshold=conf_threshold,
                device=device,
                enable_rtsp=enable_rtsp,
                rtsp_port=8554
            )
            return True
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return False
    
    def detection_loop(self):
        """Background detection"""
        while st.session_state.is_running:
            try:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    continue
                
                # Detection
                detections, inf_time = st.session_state.detector.detect_objects(frame)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Draw detections
                if detections:
                    frame = st.session_state.detector.draw_detections(frame, detections)
                
                # Update state
                st.session_state.current_frame = frame
                st.session_state.current_fps = fps
                st.session_state.current_detections = len(detections)
                st.session_state.total_detections += len(detections)
                
                time.sleep(0.1)  # 10 FPS for web
                
            except Exception as e:
                st.error(f"Detection error: {e}")
                break
    
    def start_detection(self):
        """Start detection"""
        try:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if st.session_state.cap.isOpened():
                st.session_state.is_running = True
                st.session_state.total_detections = 0
                st.session_state.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                st.session_state.detection_thread.start()
                return True
            return False
        except Exception:
            return False
    
    def stop_detection(self):
        """Stop detection"""
        st.session_state.is_running = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
    
    def render_sidebar(self):
        """Render controls"""
        with st.sidebar:
            st.markdown("# 🎯 Controls")
            
            # Model
            model = st.selectbox("🤖 Model", ["yolov5s", "yolov5m", "yolov5l"])
            
            # Confidence
            conf = st.slider("🎯 Confidence", 0.1, 0.9, 0.5, 0.1)
            
            # Device
            device = st.radio("⚡ Device", ["cuda", "cpu"])
            
            # RTSP
            rtsp = st.checkbox("📡 RTSP Stream")
            
            st.markdown("---")
            
            # Initialize
            if st.button("🔄 Initialize", use_container_width=True):
                with st.spinner("Initializing..."):
                    if self.initialize_detector(model, conf, device, rtsp):
                        st.success("✅ Ready!")
                        st.rerun()
            
            # Start/Stop
            if st.session_state.detector:
                if not st.session_state.is_running:
                    if st.button("🚀 START", use_container_width=True, type="primary"):
                        if self.start_detection():
                            st.success("🎬 Started!")
                            st.rerun()
                else:
                    if st.button("⏹️ STOP", use_container_width=True):
                        self.stop_detection()
                        st.success("⏹️ Stopped!")
                        st.rerun()
            
            st.markdown("---")
            
            # Status
            if st.session_state.is_running:
                st.markdown('<p class="status-running">🟢 RUNNING</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-stopped">🔴 STOPPED</p>', unsafe_allow_html=True)
            
            # RTSP URL
            if st.session_state.detector and st.session_state.detector.enable_rtsp:
                local_ip = get_local_ip()
                url = f"http://{local_ip}:8554/stream.mjpg"
                st.markdown("**📡 Stream:**")
                st.code(url)
    
    def render_main(self):
        """Render main content"""
        # Title
        st.title("🚀 Object Detection Dashboard")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚡ FPS", f"{st.session_state.current_fps:.1f}")
        with col2:
            st.metric("🎯 Objects", st.session_state.current_detections)
        with col3:
            st.metric("📊 Total", st.session_state.total_detections)
        
        # Video feed
        if st.session_state.current_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_column_width=True)
        else:
            st.markdown("### 📹 Camera Feed")
            st.info("Initialize detector and click START to begin")
        
        # Quick controls (touch-friendly)
        if st.session_state.detector:
            st.markdown("### 🎮 Quick Controls")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🎯 High Accuracy", use_container_width=True):
                    st.session_state.detector.conf_threshold = 0.8
                    st.session_state.detector.model.conf = 0.8
                    st.rerun()
            
            with col2:
                if st.button("⚡ Fast Detection", use_container_width=True):
                    st.session_state.detector.conf_threshold = 0.3
                    st.session_state.detector.model.conf = 0.3
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset Stats", use_container_width=True):
                    st.session_state.total_detections = 0
                    st.rerun()
            
            with col4:
                if st.button("📱 Fullscreen", use_container_width=True):
                    st.markdown("""
                    <script>
                    document.documentElement.requestFullscreen();
                    </script>
                    """, unsafe_allow_html=True)
    
    def run(self):
        """Main app"""
        self.render_sidebar()
        self.render_main()
        
        # Auto-refresh when running
        if st.session_state.is_running:
            time.sleep(0.5)
            st.rerun()

def main():
    app = SimpleStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()