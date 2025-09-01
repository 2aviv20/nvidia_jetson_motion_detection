#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ultralytics/yolov5',
        'Report a bug': None,
        'About': "Real-time Object Detection with YOLOv5"
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
        padding: 0.5rem 1rem;
        margin: 0.2rem 0;
    }
    .start-button {
        background-color: #4CAF50;
        color: white;
    }
    .stop-button {
        background-color: #f44336;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .detection-counter {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .fps-counter {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-stopped {
        color: #f44336;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDetectionApp:
    def __init__(self):
        # Initialize session state
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'cap' not in st.session_state:
            st.session_state.cap = None
        if 'frame_queue' not in st.session_state:
            st.session_state.frame_queue = queue.Queue(maxsize=2)
        if 'stats_history' not in st.session_state:
            st.session_state.stats_history = {
                'timestamps': [],
                'gpu_fps': [],
                'detections': [],
                'confidence_avg': []
            }
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
        if 'detection_thread' not in st.session_state:
            st.session_state.detection_thread = None
        
        # Configuration
        self.max_history = 50
        
    def initialize_detector(self, model_name, conf_threshold, device, enable_rtsp):
        """Initialize or update the detector"""
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
            st.error(f"❌ Failed to initialize detector: {e}")
            return False
    
    def detection_loop(self):
        """Background detection loop"""
        while st.session_state.is_running:
            try:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    continue
                
                # Run detection
                detections, inf_time = st.session_state.detector.detect_objects(frame)
                inference_fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Calculate average confidence
                avg_conf = np.mean([det[4] for det in detections]) if detections else 0
                
                # Update statistics
                current_time = time.time()
                st.session_state.stats_history['timestamps'].append(current_time)
                st.session_state.stats_history['gpu_fps'].append(inference_fps)
                st.session_state.stats_history['detections'].append(len(detections))
                st.session_state.stats_history['confidence_avg'].append(avg_conf)
                
                # Keep history manageable
                if len(st.session_state.stats_history['timestamps']) > self.max_history:
                    for key in st.session_state.stats_history:
                        st.session_state.stats_history[key] = st.session_state.stats_history[key][-self.max_history:]
                
                st.session_state.total_detections += len(detections)
                
                # Draw detections on frame
                if detections:
                    frame = st.session_state.detector.draw_detections(frame, detections)
                
                # Add frame to queue (non-blocking)
                try:
                    st.session_state.frame_queue.put((frame, inference_fps, len(detections), avg_conf), block=False)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        st.session_state.frame_queue.get_nowait()
                        st.session_state.frame_queue.put((frame, inference_fps, len(detections), avg_conf), block=False)
                    except queue.Empty:
                        pass
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                st.error(f"Detection error: {e}")
                break
    
    def start_detection(self):
        """Start camera and detection"""
        try:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if st.session_state.cap.isOpened():
                st.session_state.is_running = True
                st.session_state.total_detections = 0
                
                # Clear history
                for key in st.session_state.stats_history:
                    st.session_state.stats_history[key] = []
                
                # Start detection thread
                st.session_state.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                st.session_state.detection_thread.start()
                
                st.success("🎬 Detection started!")
                return True
            else:
                st.error("❌ Cannot open camera")
                return False
        except Exception as e:
            st.error(f"❌ Start error: {e}")
            return False
    
    def stop_detection(self):
        """Stop detection and camera"""
        st.session_state.is_running = False
        
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        
        st.success("⏹️ Detection stopped!")
    
    def create_performance_chart(self):
        """Create real-time performance charts"""
        if not st.session_state.stats_history['timestamps']:
            return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('GPU FPS', 'Detections Per Frame', 'Average Confidence', 'Detection Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        times = st.session_state.stats_history['timestamps']
        relative_times = [(t - times[0]) for t in times] if times else []
        
        # GPU FPS
        fig.add_trace(
            go.Scatter(x=relative_times, y=st.session_state.stats_history['gpu_fps'],
                      mode='lines+markers', name='GPU FPS', line=dict(color='#2ca02c')),
            row=1, col=1
        )
        
        # Detections per frame
        fig.add_trace(
            go.Scatter(x=relative_times, y=st.session_state.stats_history['detections'],
                      mode='lines+markers', name='Objects', line=dict(color='#1f77b4')),
            row=1, col=2
        )
        
        # Average confidence
        fig.add_trace(
            go.Scatter(x=relative_times, y=st.session_state.stats_history['confidence_avg'],
                      mode='lines+markers', name='Avg Confidence', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        # Detection timeline (cumulative)
        cumulative_detections = np.cumsum(st.session_state.stats_history['detections'])
        fig.add_trace(
            go.Scatter(x=relative_times, y=cumulative_detections,
                      mode='lines', name='Total Detections', line=dict(color='#d62728')),
            row=2, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        fig.update_xaxes(title_text="Time (seconds)")
        
        return fig
    
    def render_sidebar(self):
        """Render the control sidebar"""
        with st.sidebar:
            st.markdown("# 🎯 Detection Controls")
            
            # Model selection
            model_name = st.selectbox(
                "🤖 Model",
                options=["yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                index=0,
                help="Choose YOLOv5 model size (s=small/fast, x=large/accurate)"
            )
            
            # Confidence threshold
            conf_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Minimum confidence for detections"
            )
            
            # Device selection
            device = st.radio(
                "⚡ Processing Device",
                options=["cuda", "cpu"],
                index=0,
                help="Use GPU (cuda) for faster processing"
            )
            
            # RTSP streaming
            enable_rtsp = st.checkbox(
                "📡 Enable RTSP Stream",
                help="Enable HTTP streaming for remote access"
            )
            
            st.markdown("---")
            
            # Initialize detector button
            if st.button("🔄 Initialize Detector", use_container_width=True):
                with st.spinner("Initializing detector..."):
                    if self.initialize_detector(model_name, conf_threshold, device, enable_rtsp):
                        st.success("✅ Detector ready!")
            
            # Start/Stop buttons
            if st.session_state.detector is None:
                st.warning("⚠️ Initialize detector first")
            else:
                if not st.session_state.is_running:
                    if st.button("🚀 Start Detection", use_container_width=True, type="primary"):
                        self.start_detection()
                        st.rerun()
                else:
                    if st.button("⏹️ Stop Detection", use_container_width=True):
                        self.stop_detection()
                        st.rerun()
            
            st.markdown("---")
            
            # Status
            status_text = "🟢 **Running**" if st.session_state.is_running else "🔴 **Stopped**"
            st.markdown(f"**Status:** {status_text}")
            
            # RTSP URL
            if st.session_state.detector and st.session_state.detector.enable_rtsp:
                local_ip = get_local_ip()
                stream_url = f"http://{local_ip}:8554/stream.mjpg"
                st.markdown("**📡 Stream URL:**")
                st.code(stream_url, language=None)
                st.markdown(f"[🔗 Open Stream]({stream_url})")
    
    def render_main_content(self):
        """Render the main content area"""
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("🚀 Real-Time Object Detection")
        with col2:
            st.metric("🔍 Total Detections", st.session_state.total_detections)
        with col3:
            current_fps = st.session_state.stats_history['gpu_fps'][-1] if st.session_state.stats_history['gpu_fps'] else 0
            st.metric("⚡ Current FPS", f"{current_fps:.1f}")
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📹 Live Feed", "📊 Performance", "⚙️ Settings"])
        
        with tab1:
            self.render_video_feed()
        
        with tab2:
            self.render_performance_dashboard()
        
        with tab3:
            self.render_settings()
    
    def render_video_feed(self):
        """Render the live video feed"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_placeholder = st.empty()
            
            # Display current frame
            if st.session_state.is_running:
                try:
                    frame_data = st.session_state.frame_queue.get_nowait()
                    frame, fps, detections, avg_conf = frame_data
                    
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                except queue.Empty:
                    video_placeholder.markdown("📹 **Waiting for frames...**")
            else:
                video_placeholder.markdown("📹 **Camera stopped** - Click 'Start Detection' to begin")
        
        with col2:
            if st.session_state.is_running and st.session_state.stats_history['timestamps']:
                st.markdown("### 📈 Live Stats")
                
                # Current stats
                current_stats = {
                    'GPU FPS': st.session_state.stats_history['gpu_fps'][-1],
                    'Objects': st.session_state.stats_history['detections'][-1],
                    'Confidence': st.session_state.stats_history['confidence_avg'][-1]
                }
                
                for label, value in current_stats.items():
                    if label == 'Confidence':
                        st.metric(label, f"{value:.3f}")
                    else:
                        st.metric(label, f"{value:.1f}")
                
                # Object detection breakdown
                if st.session_state.detector and hasattr(st.session_state.detector, 'class_names'):
                    st.markdown("### 🏷️ Detected Classes")
                    for class_id, class_name in st.session_state.detector.class_names.items():
                        count = sum(1 for det in (st.session_state.stats_history.get('recent_detections', []) or []) 
                                  if det[5] == class_id)
                        if count > 0:
                            st.text(f"{class_name}: {count}")
    
    def render_performance_dashboard(self):
        """Render performance analytics"""
        if not st.session_state.stats_history['timestamps']:
            st.info("📊 Start detection to see performance analytics")
            return
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fps = np.mean(st.session_state.stats_history['gpu_fps']) if st.session_state.stats_history['gpu_fps'] else 0
            st.metric("📈 Avg GPU FPS", f"{avg_fps:.1f}")
        
        with col2:
            max_fps = max(st.session_state.stats_history['gpu_fps']) if st.session_state.stats_history['gpu_fps'] else 0
            st.metric("🚀 Peak FPS", f"{max_fps:.1f}")
        
        with col3:
            avg_detections = np.mean(st.session_state.stats_history['detections']) if st.session_state.stats_history['detections'] else 0
            st.metric("🎯 Avg Objects/Frame", f"{avg_detections:.1f}")
        
        with col4:
            avg_confidence = np.mean(st.session_state.stats_history['confidence_avg']) if st.session_state.stats_history['confidence_avg'] else 0
            st.metric("✅ Avg Confidence", f"{avg_confidence:.3f}")
        
        # Performance charts
        st.markdown("### 📊 Real-Time Performance")
        chart = self.create_performance_chart()
        st.plotly_chart(chart, use_container_width=True)
        
        # Export data
        if st.button("💾 Export Performance Data"):
            df = pd.DataFrame(st.session_state.stats_history)
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name=f"detection_performance_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    def render_settings(self):
        """Render advanced settings"""
        st.markdown("### ⚙️ Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎥 Camera Settings")
            camera_resolution = st.selectbox(
                "Resolution",
                options=["640x480", "1280x720", "1920x1080"],
                index=0
            )
            
            camera_fps = st.slider("Target FPS", 10, 60, 30)
            
            st.markdown("#### 🖼️ Display Settings")
            show_fps = st.checkbox("Show FPS Overlay", value=True)
            show_confidence = st.checkbox("Show Confidence Scores", value=True)
        
        with col2:
            st.markdown("#### 🔧 Detection Settings")
            iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45, help="Intersection over Union threshold for NMS")
            
            max_detections = st.slider("Max Detections", 10, 1000, 300)
            
            st.markdown("#### 📊 Performance Settings")
            history_length = st.slider("Stats History Length", 10, 200, 50)
            
            update_interval = st.slider("Update Interval (ms)", 50, 1000, 100)
        
        # Apply settings
        if st.button("✅ Apply Settings", use_container_width=True):
            st.success("Settings applied!")
            if st.session_state.detector:
                st.session_state.detector.nms_threshold = iou_threshold
    
    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_content()
        
        # Auto-refresh when running
        if st.session_state.is_running:
            time.sleep(0.1)
            st.rerun()

def main():
    app = StreamlitDetectionApp()
    app.run()

if __name__ == "__main__":
    main()