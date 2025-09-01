#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import threading
import time
import argparse
import math
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

class ImGuiStyleRenderer:
    """Custom renderer that mimics Dear ImGui styling with Pygame"""
    
    def __init__(self, screen):
        self.screen = screen
        pygame.font.init()
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Dear ImGui color scheme
        self.colors = {
            'window_bg': (15, 15, 15, 240),
            'frame_bg': (41, 74, 122, 138),
            'frame_bg_hovered': (66, 150, 250, 102),
            'frame_bg_active': (66, 150, 250, 171),
            'title_bg': (10, 10, 10, 255),
            'title_bg_active': (41, 74, 122, 255),
            'button': (66, 150, 250, 102),
            'button_hovered': (66, 150, 250, 255),
            'button_active': (15, 135, 250, 255),
            'text': (255, 255, 255, 255),
            'text_disabled': (128, 128, 128, 255),
            'border': (110, 110, 128, 128),
            'check_mark': (66, 150, 250, 255),
            'slider_grab': (61, 133, 224, 255),
            'success': (76, 175, 80, 255),
            'error': (244, 67, 54, 255),
            'warning': (255, 152, 0, 255)
        }
        
        # Layout
        self.padding = 16
        self.item_spacing = 8
        self.frame_padding = 8
        self.button_height = 50  # Extra large for touch
        self.rounding = 4
        
        # State tracking
        self.mouse_pos = (0, 0)
        self.mouse_pressed = False
        self.widgets = []
        
    def begin_window(self, title, x, y, width, height):
        """Begin a window"""
        # Window background
        window_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        window_surf.fill(self.colors['window_bg'])
        self.screen.blit(window_surf, (x, y))
        
        # Window border
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, y, width, height), 2, self.rounding)
        
        # Title bar
        title_height = 35
        title_surf = pygame.Surface((width, title_height), pygame.SRCALPHA)
        title_surf.fill(self.colors['title_bg_active'])
        self.screen.blit(title_surf, (x, y))
        
        # Title text
        title_text = self.font_large.render(title, True, self.colors['text'][:3])
        self.screen.blit(title_text, (x + self.padding, y + 8))
        
        return (x + self.padding, y + title_height + self.padding, width - 2 * self.padding, height - title_height - 2 * self.padding)
    
    def button(self, text, x, y, width=None, height=None, style='normal'):
        """Render a button"""
        if width is None:
            text_size = self.font.size(text)
            width = text_size[0] + 2 * self.frame_padding + 20
        if height is None:
            height = self.button_height
        
        # Check interaction
        mouse_x, mouse_y = pygame.mouse.get_pos()
        hovered = x <= mouse_x <= x + width and y <= mouse_y <= y + height
        clicked = hovered and pygame.mouse.get_pressed()[0]
        
        # Button color
        if style == 'success':
            base_color = self.colors['success']
        elif style == 'error':
            base_color = self.colors['error']
        elif style == 'warning':
            base_color = self.colors['warning']
        else:
            base_color = self.colors['button']
        
        if clicked:
            color = tuple(max(0, c - 30) for c in base_color[:3]) + (base_color[3],)
        elif hovered:
            color = self.colors['button_hovered'] if style == 'normal' else tuple(min(255, c + 30) for c in base_color[:3]) + (base_color[3],)
        else:
            color = base_color
        
        # Draw button
        button_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        button_surf.fill(color)
        self.screen.blit(button_surf, (x, y))
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, y, width, height), 2, self.rounding)
        
        # Button text
        text_surf = self.font.render(text, True, self.colors['text'][:3])
        text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text_surf, text_rect)
        
        # Return if clicked (single click detection)
        return hovered and hasattr(self, 'parent_app') and self.parent_app.mouse_clicked
    
    def checkbox(self, text, checked, x, y):
        """Render a checkbox"""
        box_size = 24
        
        # Check interaction
        mouse_x, mouse_y = pygame.mouse.get_pos()
        text_width = self.font.size(text)[0]
        total_width = box_size + 8 + text_width
        hovered = x <= mouse_x <= x + total_width and y <= mouse_y <= y + box_size
        
        # Checkbox background
        color = self.colors['button_active'] if checked else self.colors['frame_bg']
        checkbox_surf = pygame.Surface((box_size, box_size), pygame.SRCALPHA)
        checkbox_surf.fill(color)
        self.screen.blit(checkbox_surf, (x, y))
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, y, box_size, box_size), 2)
        
        # Checkmark
        if checked:
            points = [(x + 5, y + 12), (x + 10, y + 17), (x + 19, y + 8)]
            pygame.draw.lines(self.screen, self.colors['check_mark'][:3], False, points, 3)
        
        # Label
        text_surf = self.font.render(text, True, self.colors['text'][:3])
        self.screen.blit(text_surf, (x + box_size + 8, y + 2))
        
        return hovered and hasattr(self, 'parent_app') and self.parent_app.mouse_clicked
    
    def slider_float(self, label, value, min_val, max_val, x, y, width=200):
        """Render a float slider"""
        # Label
        label_surf = self.font.render(f"{label}: {value:.2f}", True, self.colors['text'][:3])
        self.screen.blit(label_surf, (x, y))
        
        # Slider background
        slider_y = y + 25
        slider_height = 20
        
        slider_surf = pygame.Surface((width, slider_height), pygame.SRCALPHA)
        slider_surf.fill(self.colors['frame_bg'])
        self.screen.blit(slider_surf, (x, slider_y))
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, slider_y, width, slider_height), 1)
        
        # Slider handle
        handle_pos = int(x + (value - min_val) / (max_val - min_val) * width)
        handle_rect = pygame.Rect(handle_pos - 8, slider_y, 16, slider_height)
        
        # Check interaction
        mouse_x, mouse_y = pygame.mouse.get_pos()
        slider_rect = pygame.Rect(x, slider_y, width, slider_height)
        hovered = slider_rect.collidepoint((mouse_x, mouse_y))
        
        handle_color = self.colors['button_hovered'] if hovered else self.colors['slider_grab']
        handle_surf = pygame.Surface((handle_rect.width, handle_rect.height), pygame.SRCALPHA)
        handle_surf.fill(handle_color)
        self.screen.blit(handle_surf, handle_rect)
        
        # Handle dragging
        new_value = value
        if pygame.mouse.get_pressed()[0] and hovered:
            relative_pos = (mouse_x - x) / width
            new_value = min_val + relative_pos * (max_val - min_val)
            new_value = max(min_val, min(max_val, new_value))
        
        return new_value
    
    def combo(self, label, current_item, items, x, y, width=200):
        """Render a combo box (simplified)"""
        # Label
        label_surf = self.font.render(label, True, self.colors['text'][:3])
        self.screen.blit(label_surf, (x, y))
        
        # Combo box
        combo_y = y + 25
        combo_height = 35
        
        # Background
        combo_surf = pygame.Surface((width, combo_height), pygame.SRCALPHA)
        combo_surf.fill(self.colors['frame_bg'])
        self.screen.blit(combo_surf, (x, combo_y))
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, combo_y, width, combo_height), 2)
        
        # Current item text
        current_text = items[current_item] if 0 <= current_item < len(items) else "None"
        text_surf = self.font.render(current_text, True, self.colors['text'][:3])
        self.screen.blit(text_surf, (x + 8, combo_y + 8))
        
        # Arrow
        arrow_points = [(x + width - 20, combo_y + 12), (x + width - 10, combo_y + 22), (x + width - 20, combo_y + 22)]
        pygame.draw.polygon(self.screen, self.colors['text'][:3], arrow_points)
        
        # Simple click handling (cycle through items)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        clicked = (x <= mouse_x <= x + width and combo_y <= mouse_y <= combo_y + combo_height and 
                  pygame.mouse.get_pressed()[0])
        
        if clicked:
            return (current_item + 1) % len(items)
        
        return current_item
    
    def text(self, text, x, y, color='text'):
        """Render text"""
        text_surf = self.font.render(text, True, self.colors[color][:3])
        self.screen.blit(text_surf, (x, y))
        return self.font.size(text)
    
    def text_colored(self, text, x, y, color):
        """Render colored text"""
        text_surf = self.font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))
        return self.font.size(text)
    
    def separator(self, x, y, width):
        """Draw a separator line"""
        pygame.draw.line(self.screen, self.colors['border'][:3], (x, y), (x + width, y), 1)
    
    def progress_bar(self, value, x, y, width, height=10):
        """Draw a progress bar"""
        # Background
        bg_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_surf.fill(self.colors['frame_bg'])
        self.screen.blit(bg_surf, (x, y))
        
        # Progress
        progress_width = int(width * value)
        if progress_width > 0:
            progress_surf = pygame.Surface((progress_width, height), pygame.SRCALPHA)
            progress_surf.fill(self.colors['button'])
            self.screen.blit(progress_surf, (x, y))
        
        # Border
        pygame.draw.rect(self.screen, self.colors['border'][:3], (x, y, width, height), 1)

class ImGuiStyledDetector:
    """Object detection app with Dear ImGui styling using Pygame"""
    
    def __init__(self):
        pygame.init()
        
        # Display
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Object Detection - Dear ImGui Style")
        
        # Renderer
        self.renderer = ImGuiStyleRenderer(self.screen)
        self.renderer.parent_app = self  # Give renderer access to app state
        
        # Detection system
        self.detector = None
        self.cap = None
        self.is_running = False
        
        # Video
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # UI state
        self.model_names = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
        self.current_model = 0
        self.confidence_threshold = 0.5
        self.enable_rtsp = False
        self.show_detections = True
        self.show_gui_info = True
        
        # Stats
        self.current_fps = 0.0
        self.current_objects = 0
        self.total_detections = 0
        self.fps_history = []
        
        # Button states (for single-click detection)
        self.button_states = {}
        self.mouse_clicked = False
        self.last_mouse_state = False
        
        print("✅ ImGui-styled detector initialized!")
    
    def initialize_detector(self):
        """Initialize detector"""
        try:
            model_name = self.model_names[self.current_model]
            self.detector = PyTorchGPUDetector(
                model_name=model_name,
                conf_threshold=self.confidence_threshold,
                device="cuda",
                enable_rtsp=self.enable_rtsp
            )
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def detection_loop(self):
        """Background detection"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                detections, inf_time = self.detector.detect_objects(frame)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, detections)
                
                if self.show_gui_info:
                    cv2.putText(frame, f"GPU FPS: {fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_fps = fps
                    self.current_objects = len(detections)
                    self.total_detections += len(detections)
                    
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 50:
                        self.fps_history.pop(0)
                
                time.sleep(0.033)
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def start_detection(self):
        """Start detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.cap.isOpened():
                self.is_running = True
                self.total_detections = 0
                self.fps_history.clear()
                
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                return True
            return False
        except:
            return False
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def was_clicked(self):
        """Check if mouse was just clicked this frame"""
        return self.mouse_clicked
    
    def render_controls(self):
        """Render control panel"""
        x, y, w, h = self.renderer.begin_window("🎯 Detection Controls", 20, 20, 350, 700)
        
        current_y = y
        
        # Model selection
        self.renderer.text("🤖 Model Selection", x, current_y)
        current_y += 30
        new_model = self.renderer.combo("", self.current_model, self.model_names, x, current_y, 300)
        if new_model != self.current_model and not self.is_running:
            self.current_model = new_model
            if self.detector:
                self.initialize_detector()
        current_y += 70
        
        # Confidence slider
        new_conf = self.renderer.slider_float("🎯 Confidence", self.confidence_threshold, 0.1, 0.9, x, current_y, 300)
        if abs(new_conf - self.confidence_threshold) > 0.01:
            self.confidence_threshold = new_conf
            if self.detector:
                self.detector.conf_threshold = new_conf
                self.detector.model.conf = new_conf
        current_y += 70
        
        # Checkboxes
        if self.renderer.checkbox("📡 Enable RTSP Stream", self.enable_rtsp, x, current_y):
            self.enable_rtsp = not self.enable_rtsp
        current_y += 40
        
        if self.renderer.checkbox("🎯 Show Detection Boxes", self.show_detections, x, current_y):
            self.show_detections = not self.show_detections
        current_y += 40
        
        if self.renderer.checkbox("📊 Show GUI Info", self.show_gui_info, x, current_y):
            self.show_gui_info = not self.show_gui_info
        current_y += 50
        
        # Separator
        self.renderer.separator(x, current_y, 300)
        current_y += 20
        
        # Initialize button
        if not self.detector:
            if self.renderer.button("🔄 Initialize Detector", x, current_y, 300, style='warning'):
                print("🔄 Initializing detector...")
                self.initialize_detector()
        else:
            # Start/Stop button
            if not self.is_running:
                if self.renderer.button("🚀 Start Detection", x, current_y, 300, style='success'):
                    print("🚀 Starting detection...")
                    self.start_detection()
            else:
                if self.renderer.button("⏹️ Stop Detection", x, current_y, 300, style='error'):
                    print("⏹️ Stopping detection...")
                    self.stop_detection()
        
        current_y += 70
        
        # Status
        status_text = "🟢 RUNNING" if self.is_running else "🔴 STOPPED"
        status_color = (76, 175, 80) if self.is_running else (244, 67, 54)
        self.renderer.text_colored(status_text, x, current_y, status_color)
        current_y += 30
        
        # RTSP URL
        if self.detector and self.detector.enable_rtsp:
            self.renderer.text("📡 Stream URL:", x, current_y)
            current_y += 25
            local_ip = get_local_ip()
            url = f"http://{local_ip}:8554/stream.mjpg"
            # Split URL for display
            url_parts = [url[i:i+35] for i in range(0, len(url), 35)]
            for part in url_parts:
                self.renderer.text_colored(part, x, current_y, (100, 200, 255))
                current_y += 20
    
    def render_stats(self):
        """Render statistics panel"""
        x, y, w, h = self.renderer.begin_window("📊 Performance Stats", 390, 20, 300, 400)
        
        current_y = y
        
        # Current stats
        self.renderer.text(f"⚡ Current FPS: {self.current_fps:.1f}", x, current_y)
        current_y += 25
        
        self.renderer.text(f"🎯 Current Objects: {self.current_objects}", x, current_y)
        current_y += 25
        
        self.renderer.text(f"📈 Total Detections: {self.total_detections}", x, current_y)
        current_y += 35
        
        # Average stats
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            max_fps = max(self.fps_history)
            
            self.renderer.text(f"📊 Average FPS: {avg_fps:.1f}", x, current_y)
            current_y += 25
            
            self.renderer.text(f"🚀 Peak FPS: {max_fps:.1f}", x, current_y)
            current_y += 35
        
        # FPS progress bar
        if self.current_fps > 0:
            self.renderer.text("FPS Level:", x, current_y)
            current_y += 20
            fps_ratio = min(1.0, self.current_fps / 10.0)  # Scale to 10 FPS max
            self.renderer.progress_bar(fps_ratio, x, current_y, 250, 15)
            current_y += 35
        
        # Reset button
        if self.renderer.button("🔄 Reset Stats", x, current_y, 150):
            print("🔄 Resetting stats...")
            self.total_detections = 0
            self.fps_history.clear()
    
    def render_video(self):
        """Render video display"""
        video_x = 710
        video_y = 20
        video_w = self.screen_width - video_x - 20
        video_h = self.screen_height - 40
        
        x, y, w, h = self.renderer.begin_window("📹 Live Video Feed", video_x, video_y, video_w, video_h)
        
        if self.current_frame is not None:
            with self.frame_lock:
                frame_copy = self.current_frame.copy()
            
            # Convert to pygame surface
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Scale to fit
            available_w = w - 20
            available_h = h - 20
            
            frame_w, frame_h = frame_surface.get_size()
            scale_x = available_w / frame_w
            scale_y = available_h / frame_h
            scale = min(scale_x, scale_y)
            
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            scaled_surface = pygame.transform.scale(frame_surface, (new_w, new_h))
            
            # Center the video
            center_x = x + (available_w - new_w) // 2
            center_y = y + (available_h - new_h) // 2
            
            self.screen.blit(scaled_surface, (center_x, center_y))
        else:
            self.renderer.text("📹 Camera not active", x + 20, y + 50)
            self.renderer.text("Initialize detector and start detection", x + 20, y + 80)
            self.renderer.text("to see the live video feed", x + 20, y + 110)
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("🎮 Dear ImGui-styled Controls:")
        print("  - Touch-friendly interface with large buttons")
        print("  - Real-time video display and statistics")
        print("  - ESC or close window to exit")
        print("=" * 50)
        
        while running:
            # Update click detection
            current_mouse_state = pygame.mouse.get_pressed()[0]
            self.mouse_clicked = current_mouse_state and not self.last_mouse_state
            self.last_mouse_state = current_mouse_state
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width, self.screen_height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                    self.renderer.screen = self.screen
            
            # Clear screen
            self.screen.fill((20, 20, 20))
            
            # Render UI
            self.render_controls()
            self.render_stats()
            self.render_video()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)
        
        # Cleanup
        self.stop_detection()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='ImGui-styled Object Detection')
    args = parser.parse_args()
    
    print("🚀 Dear ImGui-styled Object Detection Interface")
    print("🎨 Native Pygame rendering with ImGui styling")
    print("=" * 60)
    
    try:
        app = ImGuiStyledDetector()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()