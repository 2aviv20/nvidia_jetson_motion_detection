#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import threading
import time
import argparse
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

class ImGuiStyle:
    """Dear ImGui style colors and settings"""
    def __init__(self):
        # Colors (RGBA format for pygame)
        self.window_bg = (15, 15, 15, 240)
        self.frame_bg = (41, 74, 122, 138)
        self.frame_bg_hovered = (66, 150, 250, 102)
        self.frame_bg_active = (66, 150, 250, 171)
        self.title_bg = (10, 10, 10, 255)
        self.title_bg_active = (41, 74, 122, 255)
        self.button = (66, 150, 250, 102)
        self.button_hovered = (66, 150, 250, 255)
        self.button_active = (15, 135, 250, 255)
        self.text = (255, 255, 255, 255)
        self.text_disabled = (128, 128, 128, 255)
        self.border = (110, 110, 128, 128)
        
        # Sizes
        self.window_padding = 8
        self.frame_padding = 4
        self.item_spacing = 4
        self.grab_min_size = 10
        self.button_height = 40
        self.slider_height = 20

class PygameImGuiDetector:
    def __init__(self):
        pygame.init()
        
        # Display settings
        self.screen_width = 1024
        self.screen_height = 768
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Object Detection - ImGui Style")
        
        # Fonts
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        # Style
        self.style = ImGuiStyle()
        
        # Detection system
        self.detector = None
        self.cap = None
        self.is_running = False
        
        # Video handling
        self.current_frame = None
        self.frame_surface = None
        self.frame_lock = threading.Lock()
        
        # UI state
        self.show_controls = True
        self.show_gui_info = True
        self.show_detections = True
        self.show_demo = False
        self.is_fullscreen = False
        
        # Performance stats
        self.fps_stats = []
        self.inference_fps = 0
        self.display_fps = 0
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_pressed = [False, False, False]
        
        self.setup_detector()
    
    def setup_detector(self):
        """Initialize the PyTorch detector"""
        try:
            self.detector = PyTorchGPUDetector(
                model_name="yolov5s",
                conf_threshold=0.5,
                device="cuda",
                enable_rtsp=False
            )
            print("✅ Detector initialized")
        except Exception as e:
            print(f"❌ Detector error: {e}")
    
    def cv2_to_pygame(self, cv_image):
        """Convert OpenCV image to Pygame surface"""
        if cv_image is None:
            return None
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Rotate 90 degrees and flip for correct orientation
        rgb_image = np.rot90(rgb_image)
        rgb_image = np.flipud(rgb_image)
        
        return pygame.surfarray.make_surface(rgb_image)
    
    def draw_button(self, surface, rect, text, hovered=False, active=False, enabled=True):
        """Draw ImGui-style button"""
        if not enabled:
            color = self.style.frame_bg
            text_color = self.style.text_disabled
        elif active:
            color = self.style.button_active
            text_color = self.style.text
        elif hovered:
            color = self.style.button_hovered
            text_color = self.style.text
        else:
            color = self.style.button
            text_color = self.style.text
        
        # Draw button background
        button_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        button_surf.fill(color)
        surface.blit(button_surf, rect)
        
        # Draw border
        pygame.draw.rect(surface, self.style.border[:3], rect, 1)
        
        # Draw text
        text_surf = self.font.render(text, True, text_color[:3])
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)
        
        return rect.collidepoint(self.mouse_pos)
    
    def draw_slider(self, surface, rect, value, min_val, max_val, label=""):
        """Draw ImGui-style slider"""
        # Draw background
        bg_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        bg_surf.fill(self.style.frame_bg)
        surface.blit(bg_surf, rect)
        pygame.draw.rect(surface, self.style.border[:3], rect, 1)
        
        # Draw handle
        handle_pos = int(rect.x + (value - min_val) / (max_val - min_val) * rect.width)
        handle_rect = pygame.Rect(handle_pos - 8, rect.y, 16, rect.height)
        
        # Handle interaction
        hovered = handle_rect.collidepoint(self.mouse_pos)
        handle_color = self.style.button_hovered if hovered else self.style.button
        
        handle_surf = pygame.Surface((handle_rect.width, handle_rect.height), pygame.SRCALPHA)
        handle_surf.fill(handle_color)
        surface.blit(handle_surf, handle_rect)
        
        # Draw label
        if label:
            label_text = f"{label}: {value:.2f}"
            text_surf = self.small_font.render(label_text, True, self.style.text[:3])
            surface.blit(text_surf, (rect.x, rect.y - 20))
        
        # Handle dragging
        if self.mouse_pressed[0] and hovered:
            new_val = min_val + (self.mouse_pos[0] - rect.x) / rect.width * (max_val - min_val)
            return max(min_val, min(max_val, new_val))
        
        return value
    
    def draw_checkbox(self, surface, pos, checked, label):
        """Draw ImGui-style checkbox"""
        box_size = 20
        box_rect = pygame.Rect(pos[0], pos[1], box_size, box_size)
        
        # Draw checkbox background
        color = self.style.button_active if checked else self.style.frame_bg
        checkbox_surf = pygame.Surface((box_size, box_size), pygame.SRCALPHA)
        checkbox_surf.fill(color)
        surface.blit(checkbox_surf, box_rect)
        pygame.draw.rect(surface, self.style.border[:3], box_rect, 1)
        
        # Draw checkmark
        if checked:
            pygame.draw.lines(surface, self.style.text[:3], False, 
                            [(pos[0] + 4, pos[1] + 10), (pos[0] + 8, pos[1] + 14), (pos[0] + 16, pos[1] + 6)], 2)
        
        # Draw label
        text_surf = self.font.render(label, True, self.style.text[:3])
        surface.blit(text_surf, (pos[0] + box_size + 8, pos[1]))
        
        # Return if clicked
        label_rect = pygame.Rect(pos[0], pos[1], box_size + text_surf.get_width() + 8, box_size)
        return label_rect.collidepoint(self.mouse_pos)
    
    def draw_controls_window(self, surface):
        """Draw the main controls window"""
        if not self.show_controls:
            return
        
        # Window setup
        window_width = 300
        window_height = 500
        window_x = self.screen_width - window_width - 10
        window_y = 10
        
        # Window background
        window_surf = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        window_surf.fill(self.style.window_bg)
        surface.blit(window_surf, (window_x, window_y))
        pygame.draw.rect(surface, self.style.border[:3], (window_x, window_y, window_width, window_height), 1)
        
        # Title bar
        title_rect = pygame.Rect(window_x, window_y, window_width, 30)
        title_surf = pygame.Surface((window_width, 30), pygame.SRCALPHA)
        title_surf.fill(self.style.title_bg_active)
        surface.blit(title_surf, title_rect)
        
        title_text = self.font.render("Detection Controls", True, self.style.text[:3])
        surface.blit(title_text, (window_x + 10, window_y + 5))
        
        # Controls
        y_pos = window_y + 40
        x_pos = window_x + 10
        spacing = 50
        
        # Start/Stop button
        start_text = "Stop Detection" if self.is_running else "Start Detection"
        start_rect = pygame.Rect(x_pos, y_pos, window_width - 20, self.style.button_height)
        if self.draw_button(surface, start_rect, start_text):
            if pygame.mouse.get_pressed()[0]:
                self.toggle_detection()
        y_pos += spacing
        
        # Checkboxes
        if self.draw_checkbox(surface, (x_pos, y_pos), self.show_gui_info, "Show GUI Info"):
            if pygame.mouse.get_pressed()[0]:
                self.show_gui_info = not self.show_gui_info
        y_pos += 30
        
        if self.draw_checkbox(surface, (x_pos, y_pos), self.show_detections, "Show Detections"):
            if pygame.mouse.get_pressed()[0]:
                self.show_detections = not self.show_detections
        y_pos += 40
        
        # Confidence slider
        if self.detector:
            slider_rect = pygame.Rect(x_pos, y_pos + 20, window_width - 40, self.style.slider_height)
            new_conf = self.draw_slider(surface, slider_rect, self.detector.conf_threshold, 0.1, 0.9, "Confidence")
            if new_conf != self.detector.conf_threshold:
                self.detector.conf_threshold = new_conf
                self.detector.model.conf = new_conf
        y_pos += 60
        
        # Performance stats
        stats_y = y_pos + 20
        stats_text = [
            f"GPU FPS: {self.inference_fps:.1f}",
            f"Display FPS: {self.display_fps:.1f}",
            f"Camera: {'Active' if self.is_running else 'Inactive'}"
        ]
        
        for i, stat in enumerate(stats_text):
            text_surf = self.small_font.render(stat, True, self.style.text[:3])
            surface.blit(text_surf, (x_pos, stats_y + i * 20))
        
        # RTSP info
        if self.detector and self.detector.enable_rtsp:
            rtsp_y = stats_y + 80
            local_ip = get_local_ip()
            url = f"http://{local_ip}:8554/stream.mjpg"
            url_text = self.small_font.render("Stream URL:", True, self.style.text[:3])
            surface.blit(url_text, (x_pos, rtsp_y))
            
            # Split URL to fit
            url_parts = [url[i:i+30] for i in range(0, len(url), 30)]
            for i, part in enumerate(url_parts):
                part_surf = self.small_font.render(part, True, (100, 200, 255))
                surface.blit(part_surf, (x_pos, rtsp_y + 20 + i * 15))
    
    def toggle_detection(self):
        """Start/stop detection"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start camera and detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.cap.isOpened():
                self.is_running = True
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                print("🎬 Detection started")
            else:
                print("❌ Cannot open camera")
        except Exception as e:
            print(f"❌ Start error: {e}")
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("⏹️ Detection stopped")
    
    def detection_loop(self):
        """Detection loop running in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Run detection
                detections, inf_time = self.detector.detect_objects(frame)
                self.inference_fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Draw detections if enabled
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, detections)
                
                # Add GUI overlay if enabled
                if self.show_gui_info:
                    cv2.putText(frame, f"GPU FPS: {self.inference_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update frame surface
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        running = True
        fps_counter = 0
        fps_timer = time.time()
        
        print("🎮 Controls:")
        print("  - Right panel: Detection controls")
        print("  - H key: Hide/show controls")
        print("  - F key: Toggle fullscreen")
        print("  - Q/ESC: Quit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        self.show_controls = not self.show_controls
                    elif event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_g:
                        self.show_gui_info = not self.show_gui_info
                    elif event.key == pygame.K_d:
                        self.show_detections = not self.show_detections
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button <= 3:
                        self.mouse_pressed[event.button - 1] = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button <= 3:
                        self.mouse_pressed[event.button - 1] = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width, self.screen_height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
            
            # Update mouse position
            self.mouse_pos = pygame.mouse.get_pos()
            
            # Clear screen
            self.screen.fill((20, 20, 20))
            
            # Draw video frame
            if self.current_frame is not None:
                with self.frame_lock:
                    frame_surface = self.cv2_to_pygame(self.current_frame)
                
                if frame_surface:
                    # Scale to fit screen while maintaining aspect ratio
                    frame_rect = frame_surface.get_rect()
                    
                    # Calculate available space (leaving room for controls)
                    available_width = self.screen_width - (320 if self.show_controls else 20)
                    available_height = self.screen_height - 20
                    
                    # Scale factor
                    scale_x = available_width / frame_rect.width
                    scale_y = available_height / frame_rect.height
                    scale = min(scale_x, scale_y)
                    
                    new_width = int(frame_rect.width * scale)
                    new_height = int(frame_rect.height * scale)
                    
                    scaled_surface = pygame.transform.scale(frame_surface, (new_width, new_height))
                    
                    # Center the video
                    video_x = (available_width - new_width) // 2 + 10
                    video_y = (self.screen_height - new_height) // 2
                    
                    self.screen.blit(scaled_surface, (video_x, video_y))
            
            # Draw controls
            self.draw_controls_window(self.screen)
            
            # Show controls button when hidden
            if not self.show_controls:
                button_rect = pygame.Rect(self.screen_width - 80, 10, 70, 30)
                if self.draw_button(self.screen, button_rect, "SHOW"):
                    if pygame.mouse.get_pressed()[0]:
                        self.show_controls = True
            
            # Update display
            pygame.display.flip()
            
            # Calculate display FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.display_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Limit FPS
            clock.tick(60)
        
        # Cleanup
        self.stop_detection()
        pygame.quit()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_width, self.screen_height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((1024, 768), pygame.RESIZABLE)
            self.screen_width, self.screen_height = 1024, 768

def main():
    parser = argparse.ArgumentParser(description='Pygame ImGui-style Object Detection')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen')
    args = parser.parse_args()
    
    print("🚀 Pygame ImGui-Style Object Detection")
    print("⚡ High-performance graphics with Dear ImGui styling")
    print("=" * 60)
    
    try:
        app = PygameImGuiDetector()
        if args.fullscreen:
            app.toggle_fullscreen()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")

if __name__ == "__main__":
    main()