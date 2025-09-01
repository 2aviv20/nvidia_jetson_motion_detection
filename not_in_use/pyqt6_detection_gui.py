#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QSlider, QFrame,
                            QGroupBox, QGridLayout, QCheckBox, QSpinBox, QComboBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

class TouchFriendlyButton(QPushButton):
    """Custom button optimized for touch screens"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(120, 60)
        self.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #2196F3;
                border-radius: 8px;
                background-color: #E3F2FD;
                color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
                color: white;
            }
        """)

class VideoDisplayWidget(QLabel):
    """Custom widget for displaying OpenCV video frames"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #666; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Video Feed\nWaiting for camera...")
        self.setScaledContents(True)

class PyQt5DetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # GUI state
        self.is_running = False
        self.show_gui_info = True
        self.show_detections = True
        self.fullscreen_mode = False
        
        self.setupUI()
        self.setupDetector()
        
    def setupUI(self):
        """Setup the main GUI layout"""
        self.setWindowTitle("PyTorch GPU Object Detection - PyQt6 Touch Interface")
        self.setMinimumSize(1024, 768)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Video display
        video_layout = QVBoxLayout()
        self.video_widget = VideoDisplayWidget()
        video_layout.addWidget(self.video_widget)
        
        # Video controls
        video_controls = QHBoxLayout()
        self.fullscreen_btn = TouchFriendlyButton("Fullscreen")
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        video_controls.addWidget(self.fullscreen_btn)
        video_controls.addStretch()
        
        video_layout.addLayout(video_controls)
        main_layout.addLayout(video_layout, 3)
        
        # Right side - Control panel
        self.control_panel = self.create_control_panel()
        main_layout.addWidget(self.control_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click Start to begin detection")
        
    def create_control_panel(self):
        """Create the touch-friendly control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        
        # Main controls
        main_group = QGroupBox("Detection Controls")
        main_layout = QVBoxLayout(main_group)
        
        # Start/Stop button
        self.start_btn = TouchFriendlyButton("Start Detection")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_detection)
        main_layout.addWidget(self.start_btn)
        
        # Display toggles
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.gui_toggle = QCheckBox("Show GUI Info")
        self.gui_toggle.setChecked(True)
        self.gui_toggle.toggled.connect(self.toggle_gui_info)
        display_layout.addWidget(self.gui_toggle)
        
        self.detection_toggle = QCheckBox("Show Detection Boxes")
        self.detection_toggle.setChecked(True)
        self.detection_toggle.toggled.connect(self.toggle_detections)
        display_layout.addWidget(self.detection_toggle)
        
        main_layout.addWidget(display_group)
        
        # Settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Confidence threshold
        settings_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        settings_layout.addWidget(self.conf_slider, 0, 1)
        self.conf_label = QLabel("0.50")
        settings_layout.addWidget(self.conf_label, 0, 2)
        
        # Model selection
        settings_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov5s", "yolov5m", "yolov5l", "yolov5x"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        settings_layout.addWidget(self.model_combo, 1, 1, 1, 2)
        
        main_layout.addWidget(settings_group)
        
        # Statistics
        self.stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(self.stats_group)
        
        self.fps_label = QLabel("GPU FPS: 0.0")
        self.real_fps_label = QLabel("Real FPS: 0.0")
        self.objects_label = QLabel("Objects: 0")
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.real_fps_label)
        stats_layout.addWidget(self.objects_label)
        
        main_layout.addWidget(self.stats_group)
        
        # RTSP Streaming
        rtsp_group = QGroupBox("RTSP Streaming")
        rtsp_layout = QVBoxLayout(rtsp_group)
        
        self.rtsp_toggle = QCheckBox("Enable RTSP Stream")
        self.rtsp_toggle.toggled.connect(self.toggle_rtsp)
        rtsp_layout.addWidget(self.rtsp_toggle)
        
        self.stream_label = QLabel("Stream URL will appear here")
        self.stream_label.setWordWrap(True)
        self.stream_label.setStyleSheet("font-size: 10px; color: blue;")
        rtsp_layout.addWidget(self.stream_label)
        
        main_layout.addWidget(rtsp_group)
        
        layout.addWidget(main_group)
        layout.addStretch()
        
        return panel
        
    def setupDetector(self):
        """Initialize the PyTorch detector"""
        try:
            self.detector = PyTorchGPUDetector(
                model_name="yolov5s",
                conf_threshold=0.5,
                device="cuda",
                enable_rtsp=False
            )
            self.statusBar().showMessage("Detector initialized successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Error initializing detector: {e}")
            
    def toggle_detection(self):
        """Start/stop detection"""
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        """Start camera and detection"""
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            if self.cap.isOpened():
                self.is_running = True
                self.start_btn.setText("Stop Detection")
                self.start_btn.setChecked(True)
                self.timer.start(33)  # ~30 FPS
                self.statusBar().showMessage("Detection running...")
            else:
                self.statusBar().showMessage("Error: Cannot open camera")
                
        except Exception as e:
            self.statusBar().showMessage(f"Error starting detection: {e}")
            
    def stop_detection(self):
        """Stop detection and camera"""
        self.is_running = False
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.start_btn.setText("Start Detection")
        self.start_btn.setChecked(False)
        self.video_widget.setText("Video Feed\nStopped")
        self.statusBar().showMessage("Detection stopped")
        
    def update_frame(self):
        """Update video frame with detection results"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        try:
            # Run detection
            detections, inf_time = self.detector.detect_objects(frame)
            
            # Calculate FPS
            inference_fps = 1.0 / inf_time if inf_time > 0 else 0
            
            # Draw detections if enabled
            if self.show_detections:
                frame = self.detector.draw_detections(frame, detections)
                
            # Add GUI info if enabled
            if self.show_gui_info:
                cv2.putText(frame, f"GPU FPS: {inference_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Objects: {len(detections)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
            # Update statistics
            self.fps_label.setText(f"GPU FPS: {inference_fps:.1f}")
            self.objects_label.setText(f"Objects: {len(detections)}")
            
            # Convert to Qt format and display
            self.display_frame(frame)
            
        except Exception as e:
            self.statusBar().showMessage(f"Detection error: {e}")
            
    def display_frame(self, frame):
        """Convert OpenCV frame to Qt format and display"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_widget.setPixmap(pixmap)
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.fullscreen_mode:
            self.showNormal()
            self.control_panel.show()
            self.fullscreen_btn.setText("Fullscreen")
            self.fullscreen_mode = False
        else:
            self.showFullScreen()
            self.control_panel.hide()
            self.fullscreen_btn.setText("Exit Fullscreen")
            self.fullscreen_mode = True
            
    def toggle_gui_info(self, checked):
        """Toggle GUI info display"""
        self.show_gui_info = checked
        
    def toggle_detections(self, checked):
        """Toggle detection boxes display"""
        self.show_detections = checked
        
    def update_confidence(self, value):
        """Update confidence threshold"""
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        if self.detector:
            self.detector.conf_threshold = conf
            self.detector.model.conf = conf
            
    def change_model(self, model_name):
        """Change detection model"""
        if self.detector:
            try:
                self.detector.load_model(model_name)
                self.statusBar().showMessage(f"Model changed to {model_name}")
            except Exception as e:
                self.statusBar().showMessage(f"Error changing model: {e}")
                
    def toggle_rtsp(self, enabled):
        """Toggle RTSP streaming"""
        if enabled:
            # Reinitialize detector with RTSP
            try:
                self.detector = PyTorchGPUDetector(
                    model_name=self.model_combo.currentText(),
                    conf_threshold=self.conf_slider.value() / 100.0,
                    device="cuda",
                    enable_rtsp=True,
                    rtsp_port=8554
                )
                local_ip = get_local_ip()
                stream_url = f"http://{local_ip}:8554/stream.mjpg"
                self.stream_label.setText(f"Stream: {stream_url}")
                self.statusBar().showMessage("RTSP streaming enabled")
            except Exception as e:
                self.rtsp_toggle.setChecked(False)
                self.statusBar().showMessage(f"RTSP error: {e}")
        else:
            self.stream_label.setText("RTSP streaming disabled")
            
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_detection()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='PyQt6 Object Detection GUI')
    parser.add_argument('--touch', action='store_true', help='Optimize for touch screens')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set touch-friendly styling if requested
    if args.touch:
        app.setStyleSheet("""
            QWidget { font-size: 14px; }
            QPushButton { min-height: 50px; }
            QCheckBox { min-height: 30px; }
            QSlider::handle:horizontal { width: 30px; height: 30px; }
        """)
    
    window = PyQt5DetectionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()