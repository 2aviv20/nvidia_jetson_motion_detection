#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
import argparse
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

try:
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    import OpenGL.GL as gl
    import glfw
    IMGUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ImGui not available: {e}")
    print("💡 Run: pip3 install imgui[glfw,opengl]")
    IMGUI_AVAILABLE = False

class ImGuiDetectionApp:
    def __init__(self):
        if not IMGUI_AVAILABLE:
            raise ImportError("ImGui dependencies not available")
        
        # Initialize GLFW and OpenGL
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Create window
        self.window = glfw.create_window(1200, 800, "Object Detection - Dear ImGui", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync
        
        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        
        # Style
        self.setup_style()
        
        # Detection system
        self.detector = None
        self.cap = None
        self.is_running = False
        
        # Video handling
        self.current_frame = None
        self.frame_texture = None
        self.frame_width = 640
        self.frame_height = 480
        self.frame_lock = threading.Lock()
        
        # UI state
        self.show_demo = False
        self.show_metrics = True
        self.show_controls = True
        self.model_names = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
        self.current_model = 0
        self.confidence_threshold = 0.5
        self.enable_rtsp = False
        self.show_detections = True
        self.show_gui_info = True
        
        # Performance tracking
        self.fps_history = []
        self.detection_history = []
        self.max_history = 100
        
        # Statistics
        self.current_fps = 0.0
        self.current_objects = 0
        self.total_detections = 0
        
        print("✅ Dear ImGui initialized successfully!")
    
    def setup_style(self):
        """Configure Dear ImGui style for touch-friendly interface"""
        style = imgui.get_style()
        
        # Make everything larger for touch
        style.frame_padding = (12, 8)
        style.item_spacing = (12, 8)
        style.window_padding = (16, 16)
        style.grab_min_size = 20
        style.frame_border_size = 1
        
        # Colors (Dark theme with blue accents)
        colors = style.colors
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.06, 0.94)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.16, 0.29, 0.48, 0.54)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.26, 0.59, 0.98, 0.40)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.26, 0.59, 0.98, 0.67)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.16, 0.29, 0.48, 1.00)
        colors[imgui.COLOR_CHECK_MARK] = (0.26, 0.59, 0.98, 1.00)
        colors[imgui.COLOR_SLIDER_GRAB] = (0.24, 0.52, 0.88, 1.00)
        colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.26, 0.59, 0.98, 1.00)
        colors[imgui.COLOR_BUTTON] = (0.26, 0.59, 0.98, 0.40)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.26, 0.59, 0.98, 1.00)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.06, 0.53, 0.98, 1.00)
        colors[imgui.COLOR_TEXT] = (1.00, 1.00, 1.00, 1.00)
    
    def initialize_detector(self):
        """Initialize the PyTorch detector"""
        try:
            model_name = self.model_names[self.current_model]
            self.detector = PyTorchGPUDetector(
                model_name=model_name,
                conf_threshold=self.confidence_threshold,
                device="cuda",
                enable_rtsp=self.enable_rtsp,
                rtsp_port=8554
            )
            print(f"✅ Detector initialized: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Detector error: {e}")
            return False
    
    def create_texture_from_cv_image(self, cv_image):
        """Create OpenGL texture from OpenCV image"""
        if cv_image is None:
            return None
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        
        # Create OpenGL texture
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, 
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_image)
        
        return texture_id
    
    def detection_loop(self):
        """Background detection thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Run detection
                detections, inf_time = self.detector.detect_objects(frame)
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
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_fps = inference_fps
                    self.current_objects = len(detections)
                    self.total_detections += len(detections)
                    
                    # Update history
                    self.fps_history.append(inference_fps)
                    self.detection_history.append(len(detections))
                    
                    # Limit history size
                    if len(self.fps_history) > self.max_history:
                        self.fps_history.pop(0)
                        self.detection_history.pop(0)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def start_detection(self):
        """Start camera and detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if self.cap.isOpened():
                self.is_running = True
                self.total_detections = 0
                self.fps_history.clear()
                self.detection_history.clear()
                
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                return True
            return False
        except Exception as e:
            print(f"Start error: {e}")
            return False
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def render_control_panel(self):
        """Render the main control panel"""
        if not self.show_controls:
            return
        
        imgui.begin("🎯 Detection Controls", True)
        
        # Model selection
        imgui.text("🤖 Model Selection")
        changed, self.current_model = imgui.combo("##model", self.current_model, self.model_names)
        if changed and not self.is_running:
            self.initialize_detector()
        
        imgui.spacing()
        
        # Confidence threshold
        imgui.text("🎯 Confidence Threshold")
        changed, self.confidence_threshold = imgui.slider_float("##confidence", self.confidence_threshold, 0.1, 0.9, "%.2f")
        if changed and self.detector:
            self.detector.conf_threshold = self.confidence_threshold
            self.detector.model.conf = self.confidence_threshold
        
        imgui.spacing()
        
        # Checkboxes
        _, self.enable_rtsp = imgui.checkbox("📡 Enable RTSP Stream", self.enable_rtsp)
        _, self.show_detections = imgui.checkbox("🎯 Show Detection Boxes", self.show_detections)
        _, self.show_gui_info = imgui.checkbox("📊 Show GUI Info", self.show_gui_info)
        
        imgui.spacing()
        
        # Control buttons
        if not self.detector:
            if imgui.button("🔄 Initialize Detector", width=200, height=40):
                self.initialize_detector()
        else:
            if not self.is_running:
                if imgui.button("🚀 Start Detection", width=200, height=40):
                    self.start_detection()
            else:
                if imgui.button("⏹️ Stop Detection", width=200, height=40):
                    self.stop_detection()
        
        imgui.spacing()
        
        # Status
        status_color = (0.2, 0.8, 0.2) if self.is_running else (0.8, 0.2, 0.2)
        status_text = "🟢 RUNNING" if self.is_running else "🔴 STOPPED"
        imgui.text_colored(status_text, *status_color)
        
        # RTSP URL
        if self.detector and self.detector.enable_rtsp:
            imgui.spacing()
            imgui.text("📡 Stream URL:")
            local_ip = get_local_ip()
            url = f"http://{local_ip}:8554/stream.mjpg"
            imgui.text(url)
        
        imgui.end()
    
    def render_metrics_panel(self):
        """Render performance metrics"""
        if not self.show_metrics:
            return
        
        imgui.begin("📊 Performance Metrics", True)
        
        # Current stats
        imgui.text(f"⚡ Current FPS: {self.current_fps:.1f}")
        imgui.text(f"🎯 Current Objects: {self.current_objects}")
        imgui.text(f"📈 Total Detections: {self.total_detections}")
        
        imgui.spacing()
        
        # Average stats
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            max_fps = max(self.fps_history)
            avg_objects = sum(self.detection_history) / len(self.detection_history)
            
            imgui.text(f"📊 Average FPS: {avg_fps:.1f}")
            imgui.text(f"🚀 Peak FPS: {max_fps:.1f}")
            imgui.text(f"🎯 Avg Objects/Frame: {avg_objects:.1f}")
        
        imgui.spacing()
        
        # Simple FPS graph
        if self.fps_history and len(self.fps_history) > 1:
            imgui.text("📈 FPS History")
            imgui.plot_lines("##fps", self.fps_history, scale_min=0.0, scale_max=max(self.fps_history) + 1.0, graph_size=(250, 100))
        
        # Reset button
        if imgui.button("🔄 Reset Stats", width=120):
            self.total_detections = 0
            self.fps_history.clear()
            self.detection_history.clear()
        
        imgui.end()
    
    def render_video_window(self):
        """Render video display window"""
        imgui.begin("📹 Live Video Feed", True)
        
        if self.current_frame is not None:
            with self.frame_lock:
                frame_copy = self.current_frame.copy()
            
            # Update texture
            if self.frame_texture:
                gl.glDeleteTextures([self.frame_texture])
            self.frame_texture = self.create_texture_from_cv_image(frame_copy)
            
            if self.frame_texture:
                # Get window size
                window_width = imgui.get_window_width() - 20
                window_height = imgui.get_window_height() - 60
                
                # Calculate aspect ratio
                aspect_ratio = self.frame_width / self.frame_height
                
                if window_width / window_height > aspect_ratio:
                    # Window is wider than video
                    display_height = window_height
                    display_width = display_height * aspect_ratio
                else:
                    # Window is taller than video
                    display_width = window_width
                    display_height = display_width / aspect_ratio
                
                # Display image
                imgui.image(self.frame_texture, display_width, display_height)
        else:
            imgui.text("📹 Camera not active")
            imgui.text("Initialize detector and start detection to see video feed")
        
        imgui.end()
    
    def render_menu_bar(self):
        """Render main menu bar"""
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("View"):
                _, self.show_controls = imgui.menu_item("Controls", selected=self.show_controls)
                _, self.show_metrics = imgui.menu_item("Metrics", selected=self.show_metrics)
                _, self.show_demo = imgui.menu_item("ImGui Demo", selected=self.show_demo)
                imgui.end_menu()
            
            # FPS display in menu bar
            imgui.same_line(imgui.get_window_width() - 100)
            imgui.text(f"FPS: {self.current_fps:.1f}")
            
            imgui.end_main_menu_bar()
    
    def run(self):
        """Main application loop"""
        print("🎮 Dear ImGui Controls:")
        print("  - Use mouse or touch to interact with controls")
        print("  - View menu to show/hide panels")
        print("  - ESC or close window to exit")
        print("=" * 50)
        
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            
            imgui.new_frame()
            
            # Render UI
            self.render_menu_bar()
            self.render_control_panel()
            self.render_metrics_panel()
            self.render_video_window()
            
            # Show ImGui demo if requested
            if self.show_demo:
                self.show_demo = imgui.show_demo_window(self.show_demo)
            
            # Rendering
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)
        
        # Cleanup
        self.stop_detection()
        if self.frame_texture:
            gl.glDeleteTextures([self.frame_texture])
        
        self.impl.shutdown()
        glfw.terminate()

def main():
    parser = argparse.ArgumentParser(description='Dear ImGui Object Detection')
    parser.add_argument('--demo', action='store_true', help='Show ImGui demo window')
    args = parser.parse_args()
    
    print("🚀 Dear ImGui Object Detection Interface")
    print("⚡ Hardware-accelerated OpenGL rendering")
    print("=" * 60)
    
    try:
        app = ImGuiDetectionApp()
        if args.demo:
            app.show_demo = True
        app.run()
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip3 install imgui[glfw,opengl]")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()