#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

class OpenCVGUIDetector:
    def __init__(self, model_name="yolov5s", conf_threshold=0.5, device="cuda", enable_rtsp=False):
        self.detector = PyTorchGPUDetector(model_name, conf_threshold, device, enable_rtsp)
        
        # GUI state
        self.show_gui = True
        self.show_detections = True
        self.show_controls = True
        self.fullscreen = False
        
        # GUI elements positions and sizes
        self.button_height = 40
        self.button_width = 120
        self.margin = 10
        
        # Colors
        self.bg_color = (40, 40, 40)
        self.button_color = (100, 100, 100)
        self.button_active = (0, 255, 0)
        self.button_inactive = (0, 0, 255)
        self.text_color = (255, 255, 255)
        
        # Mouse tracking
        self.mouse_x = 0
        self.mouse_y = 0
        
    def draw_button(self, frame, x, y, w, h, text, active=True, clicked=False):
        """Draw a touch-friendly button"""
        color = self.button_active if active else self.button_inactive
        if clicked:
            color = tuple(max(0, c - 50) for c in color)  # Darker when clicked
            
        # Button background
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Button text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
        
        return (x, y, w, h)
    
    def draw_slider(self, frame, x, y, w, h, value, min_val, max_val, label):
        """Draw a slider control"""
        # Slider background
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.button_color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Slider handle
        handle_pos = int(x + (value - min_val) / (max_val - min_val) * w)
        cv2.circle(frame, (handle_pos, y + h // 2), h // 2 - 2, (255, 255, 255), -1)
        
        # Label
        cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return (x, y, w, h)
    
    def draw_control_panel(self, frame):
        """Draw the control panel"""
        if not self.show_controls:
            return []
        
        h, w = frame.shape[:2]
        panel_width = 200
        panel_x = w - panel_width
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        buttons = []
        y_pos = self.margin
        
        # GUI Toggle Button
        btn_rect = self.draw_button(frame, panel_x + self.margin, y_pos, 
                                   self.button_width, self.button_height, 
                                   "GUI", self.show_gui)
        buttons.append(("gui_toggle", btn_rect))
        y_pos += self.button_height + self.margin
        
        # Detection Toggle Button
        btn_rect = self.draw_button(frame, panel_x + self.margin, y_pos, 
                                   self.button_width, self.button_height, 
                                   "DETECT", self.show_detections)
        buttons.append(("det_toggle", btn_rect))
        y_pos += self.button_height + self.margin
        
        # Fullscreen Button
        btn_rect = self.draw_button(frame, panel_x + self.margin, y_pos, 
                                   self.button_width, self.button_height, 
                                   "FULL", self.fullscreen)
        buttons.append(("fullscreen", btn_rect))
        y_pos += self.button_height + self.margin
        
        # Confidence Slider
        slider_rect = self.draw_slider(frame, panel_x + self.margin, y_pos + 30, 
                                     self.button_width, 20, 
                                     self.detector.conf_threshold, 0.1, 0.9, "Conf")
        buttons.append(("conf_slider", slider_rect))
        y_pos += 60
        
        # Hide Controls Button (small X)
        btn_rect = self.draw_button(frame, w - 30, 10, 20, 20, "X", True)
        buttons.append(("hide_controls", btn_rect))
        
        return buttons
    
    def draw_show_controls_button(self, frame):
        """Draw button to show controls when hidden"""
        if self.show_controls:
            return []
        
        h, w = frame.shape[:2]
        btn_rect = self.draw_button(frame, w - 50, 10, 40, 30, "MENU", True)
        return [("show_controls", btn_rect)]
    
    def handle_click(self, x, y, buttons):
        """Handle mouse click on buttons"""
        for button_id, (bx, by, bw, bh) in buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if button_id == "gui_toggle":
                    self.show_gui = not self.show_gui
                elif button_id == "det_toggle":
                    self.show_detections = not self.show_detections
                elif button_id == "fullscreen":
                    self.toggle_fullscreen()
                elif button_id == "hide_controls":
                    self.show_controls = False
                elif button_id == "show_controls":
                    self.show_controls = True
                elif button_id == "conf_slider":
                    # Handle slider
                    new_val = 0.1 + (x - bx) / bw * 0.8
                    new_val = max(0.1, min(0.9, new_val))
                    self.detector.conf_threshold = new_val
                    self.detector.model.conf = new_val
                return True
        return False
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty('OpenCV GUI Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('OpenCV GUI Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x, self.mouse_y = x, y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check control panel buttons
            if hasattr(self, 'current_buttons'):
                if self.handle_click(x, y, self.current_buttons):
                    return
                    
            # Check show controls button
            if hasattr(self, 'show_button'):
                self.handle_click(x, y, self.show_button)
    
    def run_detection(self, video_source=0):
        """Run detection with OpenCV GUI"""
        # Open video source
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Cannot open video source: {video_source}")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create window
        cv2.namedWindow('OpenCV GUI Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('OpenCV GUI Detection', self.mouse_callback)
        
        print("🎮 Controls:")
        print("  - Click GUI button to toggle info overlay")
        print("  - Click DETECT button to toggle detection boxes")
        print("  - Click FULL button for fullscreen")
        print("  - Click and drag confidence slider")
        print("  - Press 'q' to quit")
        print("  - Press 'h' to hide/show controls")
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                detections, inf_time = self.detector.detect_objects(frame)
                inference_fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Draw detections if enabled
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, detections)
                
                # Add GUI info if enabled
                if self.show_gui:
                    elapsed = time.time() - start_time
                    real_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    cv2.putText(frame, f"GPU FPS: {inference_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Real FPS: {real_fps:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.detector.enable_rtsp:
                        local_ip = get_local_ip()
                        stream_url = f"http://{local_ip}:{self.detector.rtsp_port}/stream.mjpg"
                        cv2.putText(frame, f"Stream: {stream_url}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Draw control panel
                self.current_buttons = self.draw_control_panel(frame)
                self.show_button = self.draw_show_controls_button(frame)
                
                # Push to RTSP if enabled
                if self.detector.enable_rtsp:
                    self.detector._push_frame_to_rtsp(frame)
                
                # Display
                cv2.imshow('OpenCV GUI Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    self.show_controls = not self.show_controls
                elif key == ord('g'):
                    self.show_gui = not self.show_gui
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
                elif key == ord('f'):
                    self.toggle_fullscreen()
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if self.detector.enable_rtsp:
                self.detector._stop_rtsp_streaming()

def main():
    parser = argparse.ArgumentParser(description='OpenCV GUI Object Detection')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or file)')
    parser.add_argument('--model', '-m', default='yolov5s', help='YOLOv5 model')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--rtsp', action='store_true', help='Enable RTSP streaming')
    parser.add_argument('--rtsp-port', type=int, default=8554, help='RTSP port')
    
    args = parser.parse_args()
    
    print("🚀 OpenCV GUI Object Detection")
    print(f"⚡ Device: {args.device.upper()}")
    print("=" * 50)
    
    try:
        detector = OpenCVGUIDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            device=args.device,
            enable_rtsp=args.rtsp
        )
        
        detector.run_detection(args.input)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()