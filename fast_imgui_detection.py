#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import threading
import time
import argparse
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip, get_stream_url

class FastImGuiDetector:
    """Optimized ImGui-style detector with minimal overhead"""
    
    def __init__(self):
        pygame.init()
        
        # Display settings
        self.screen_width = 1200
        self.screen_height = 800
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Fast Object Detection - ImGui Style")
        
        # Simple font
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Colors (simplified)
        self.bg_dark = (25, 25, 25)
        self.panel_bg = (40, 40, 40)
        self.button_bg = (60, 120, 200)
        self.button_hover = (80, 140, 220)
        self.text_color = (255, 255, 255)
        self.green = (50, 200, 50)
        self.red = (200, 50, 50)
        self.yellow = (200, 200, 50)
        
        # Detection system
        self.detector = None
        self.cap = None
        self.is_running = False
        
        # Video (direct surface, no conversion)
        self.current_frame = None
        self.frame_surface = None
        self.frame_lock = threading.Lock()
        
        # UI state
        self.model_index = 0
        self.models = ["yolov5s", "yolov5m", "yolov5l"]
        self.confidence = 0.5
        self.show_detections = True
        self.show_gui_info = True
        self.enable_rtsp = False
        
        # Loading state
        self.is_initializing = True
        self.initialization_complete = False
        self.init_thread = None
        self.loading_dots = 0
        self.loading_timer = 0
        
        # Stats
        self.fps = 0.0
        self.objects = 0
        self.total_objects = 0
        
        # Mouse
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        self.last_click = False
        
        # Pre-render surfaces (cache for performance)
        self.button_cache = {}
        
        print("✅ Fast ImGui detector ready!")
        
        # Start automatic initialization
        self.start_initialization()
    
    def draw_button(self, x, y, w, h, text, active=True, style='normal'):
        """Optimized button drawing"""
        # Cache key
        cache_key = f"{text}_{w}_{h}_{active}_{style}"
        
        mouse_x, mouse_y = self.mouse_pos
        hovered = x <= mouse_x <= x + w and y <= mouse_y <= y + h
        
        # Choose color
        if style == 'success':
            color = self.green
        elif style == 'error':
            color = self.red
        elif style == 'warning':
            color = self.yellow
        else:
            color = self.button_hover if hovered else self.button_bg
        
        # Draw button (no fancy effects for speed)
        pygame.draw.rect(self.screen, color, (x, y, w, h))
        pygame.draw.rect(self.screen, self.text_color, (x, y, w, h), 1)
        
        # Text (cached)
        if cache_key not in self.button_cache:
            text_surface = self.font.render(text, True, self.text_color)
            self.button_cache[cache_key] = text_surface
        
        text_surf = self.button_cache[cache_key]
        text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
        self.screen.blit(text_surf, text_rect)
        
        return hovered and self.mouse_clicked
    
    def toggle_fullscreen(self):
        """Toggle between windowed and fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            # Save current window size
            self.windowed_width = self.screen_width
            self.windowed_height = self.screen_height
            
            # Switch to fullscreen
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_width, self.screen_height = self.screen.get_size()
            print(f"🖥️ Fullscreen: {self.screen_width}x{self.screen_height}")
        else:
            # Switch back to windowed
            self.screen_width = self.windowed_width
            self.screen_height = self.windowed_height
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            print(f"🪟 Windowed: {self.screen_width}x{self.screen_height}")
    
    def draw_text(self, text, x, y, color=None):
        """Fast text drawing"""
        if color is None:
            color = self.text_color
        text_surf = self.font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))
        return text_surf.get_size()
    
    def draw_slider(self, x, y, w, value, min_val, max_val, label):
        """Simple slider"""
        # Label
        self.draw_text(f"{label}: {value:.2f}", x, y - 20)
        
        # Slider track
        track_y = y + 5
        track_h = 10
        pygame.draw.rect(self.screen, self.panel_bg, (x, track_y, w, track_h))
        pygame.draw.rect(self.screen, self.text_color, (x, track_y, w, track_h), 1)
        
        # Handle
        handle_pos = int(x + (value - min_val) / (max_val - min_val) * w)
        pygame.draw.circle(self.screen, self.button_bg, (handle_pos, track_y + track_h//2), 8)
        
        # Handle dragging
        mouse_x, mouse_y = self.mouse_pos
        if (pygame.mouse.get_pressed()[0] and 
            x <= mouse_x <= x + w and track_y <= mouse_y <= track_y + track_h):
            new_value = min_val + (mouse_x - x) / w * (max_val - min_val)
            return max(min_val, min(max_val, new_value))
        
        return value
    
    def render_loading_screen(self):
        """Render loading screen during initialization"""
        # Full screen loading overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Loading box
        box_w = 400
        box_h = 200
        box_x = (self.screen_width - box_w) // 2
        box_y = (self.screen_height - box_h) // 2
        
        # Loading box background
        pygame.draw.rect(self.screen, self.panel_bg, (box_x, box_y, box_w, box_h))
        pygame.draw.rect(self.screen, self.button_bg, (box_x, box_y, box_w, box_h), 3)
        
        # Title
        title_font = pygame.font.Font(None, 36)
        title_text = title_font.render("🚀 Initializing Detector", True, self.text_color)
        title_rect = title_text.get_rect(center=(box_x + box_w//2, box_y + 40))
        self.screen.blit(title_text, title_rect)
        
        # Model name
        model_text = f"Loading {self.models[self.model_index]} model..."
        model_surface = self.font_large.render(model_text, True, self.yellow)
        model_rect = model_surface.get_rect(center=(box_x + box_w//2, box_y + 80))
        self.screen.blit(model_surface, model_rect)
        
        # Animated loading dots
        dots = "." * (self.loading_dots + 1)
        loading_text = f"Please wait{dots}"
        loading_surface = self.font.render(loading_text, True, self.text_color)
        loading_rect = loading_surface.get_rect(center=(box_x + box_w//2, box_y + 110))
        self.screen.blit(loading_surface, loading_rect)
        
        # Progress bar (fake progress for visual feedback)
        progress_w = 300
        progress_h = 10
        progress_x = box_x + (box_w - progress_w) // 2
        progress_y = box_y + 140
        
        # Progress background
        pygame.draw.rect(self.screen, (60, 60, 60), (progress_x, progress_y, progress_w, progress_h))
        
        # Progress fill (animated)
        progress_fill = int((time.time() % 3) / 3 * progress_w)
        pygame.draw.rect(self.screen, self.button_bg, (progress_x, progress_y, progress_fill, progress_h))
        
        # Instructions
        instr_text = "The model is being downloaded and loaded..."
        instr_surface = self.font.render(instr_text, True, (180, 180, 180))
        instr_rect = instr_surface.get_rect(center=(box_x + box_w//2, box_y + 170))
        self.screen.blit(instr_surface, instr_rect)
    
    def start_initialization(self):
        """Start background initialization"""
        self.is_initializing = True
        self.initialization_complete = False
        self.init_thread = threading.Thread(target=self.background_initialization, daemon=True)
        self.init_thread.start()
        print("🔄 Starting background initialization...")
    
    def background_initialization(self):
        """Initialize detector in background thread"""
        try:
            print(f"📦 Loading {self.models[self.model_index]} model...")
            model = self.models[self.model_index]
            self.detector = PyTorchGPUDetector(
                model_name=model,
                conf_threshold=self.confidence,
                device="cuda",
                enable_rtsp=self.enable_rtsp
            )
            self.initialization_complete = True
            print(f"✅ {model} initialized successfully!")
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            self.initialization_complete = False
        finally:
            self.is_initializing = False
    
    def initialize_detector(self):
        """Initialize detector"""
        try:
            model = self.models[self.model_index]
            self.detector = PyTorchGPUDetector(
                model_name=model,
                conf_threshold=self.confidence,
                device="cuda",
                enable_rtsp=self.enable_rtsp
            )
            print(f"✅ Detector: {model}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def detection_loop(self):
        """Minimal overhead detection loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Detection
                detections, inf_time = self.detector.detect_objects(frame)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Draw detections
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, detections)
                
                # Add minimal GUI info
                if self.show_gui_info:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(frame, f"Objects: {len(detections)}", (10, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Update (minimal locking)
                with self.frame_lock:
                    self.current_frame = frame
                    self.fps = fps
                    self.objects = len(detections)
                    self.total_objects += len(detections)
                
                # Faster loop
                time.sleep(0.01)  # 100 Hz
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def start_detection(self):
        """Start detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            # Reduced resolution for speed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
            
            if self.cap.isOpened():
                self.is_running = True
                self.total_objects = 0
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
    
    def update_video_surface(self):
        """Fast video surface update"""
        if self.current_frame is None:
            return
            
        with self.frame_lock:
            frame = self.current_frame.copy()
        
        # Direct conversion (fastest method)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rotated = np.rot90(frame_rgb)
        frame_flipped = np.flipud(frame_rotated)
        
        self.frame_surface = pygame.surfarray.make_surface(frame_flipped)
    
    def render_controls(self):
        """Minimal control panel"""
        panel_x = 10
        panel_y = 10
        panel_w = 280
        # Dynamic height based on content
        base_height = 350
        url_height = 60 if (self.enable_rtsp and self.detector and self.detector.enable_rtsp) else 0
        panel_h = base_height + url_height
        
        # Panel background
        pygame.draw.rect(self.screen, self.panel_bg, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.text_color, (panel_x, panel_y, panel_w, panel_h), 1)
        
        x = panel_x + 10
        y = panel_y + 10
        
        # Title
        self.draw_text("🎯 Controls", x, y, self.text_color)
        y += 30
        
        # Model selector (simple)
        if self.draw_button(x, y, 120, 25, f"Model: {self.models[self.model_index]}", style='normal'):
            self.model_index = (self.model_index + 1) % len(self.models)
            if not self.is_running and self.detector:
                self.initialize_detector()
        y += 35
        
        # Confidence slider
        new_conf = self.draw_slider(x, y + 20, 200, self.confidence, 0.1, 0.9, "Confidence")
        if abs(new_conf - self.confidence) > 0.01:
            self.confidence = new_conf
            if self.detector:
                self.detector.conf_threshold = new_conf
                self.detector.model.conf = new_conf
        y += 50
        
        # Toggle buttons (checkboxes as buttons)
        det_color = 'success' if self.show_detections else 'normal'
        if self.draw_button(x, y, 80, 25, "Detections", style=det_color):
            self.show_detections = not self.show_detections
        
        gui_color = 'success' if self.show_gui_info else 'normal'
        if self.draw_button(x + 90, y, 60, 25, "GUI", style=gui_color):
            self.show_gui_info = not self.show_gui_info
        y += 35
        
        rtsp_color = 'success' if self.enable_rtsp else 'normal'
        if self.draw_button(x, y, 80, 25, "RTSP", style=rtsp_color):
            old_enable_rtsp = self.enable_rtsp
            self.enable_rtsp = not self.enable_rtsp
            print(f"🔄 RTSP: {'ON' if self.enable_rtsp else 'OFF'}")
            
            # Reinitialize detector with new RTSP setting if detector exists and not running
            if self.detector and not self.is_running and not self.is_initializing:
                print("🔄 Reinitializing detector with new RTSP setting...")
                self.start_initialization()
            elif not self.detector and self.initialization_complete:
                # If no detector but init was complete, restart initialization
                print("🔄 Restarting initialization with RTSP...")
                self.start_initialization()
        
        # Fullscreen button
        fs_text = "Exit FS" if self.is_fullscreen else "Fullscreen"
        if self.draw_button(x + 90, y, 80, 25, fs_text, style='normal'):
            self.toggle_fullscreen()
        y += 35
        
        # Main buttons
        if self.is_initializing:
            # Show disabled button during initialization
            disabled_color = (100, 100, 100)
            pygame.draw.rect(self.screen, disabled_color, (x, y, 200, 30))
            pygame.draw.rect(self.screen, self.text_color, (x, y, 200, 30), 1)
            init_text = self.font.render("🔄 Initializing...", True, self.text_color)
            init_rect = init_text.get_rect(center=(x + 100, y + 15))
            self.screen.blit(init_text, init_rect)
        elif not self.detector or not self.initialization_complete:
            # Show retry button if initialization failed
            if self.draw_button(x, y, 200, 30, "🔄 Retry Initialize", style='warning'):
                print("Retrying initialization...")
                self.start_initialization()
        else:
            # Normal start/stop buttons
            if not self.is_running:
                if self.draw_button(x, y, 200, 30, "🚀 Start Detection", style='success'):
                    print("Starting...")
                    self.start_detection()
            else:
                if self.draw_button(x, y, 200, 30, "⏹️ Stop Detection", style='error'):
                    print("Stopping...")
                    self.stop_detection()
        y += 40
        
        # HTTP Stream URL (above status)
        if self.enable_rtsp and self.detector and hasattr(self.detector, 'enable_rtsp') and self.detector.enable_rtsp:
            stream_url = get_stream_url(8554)
            self.draw_text("📡 Stream:", x, y, self.yellow)
            y += 15
            
            # Split URL if too long for display
            if len(stream_url) > 28:
                # Split at a logical point
                if '://' in stream_url:
                    protocol_end = stream_url.find('://') + 3
                    part1 = stream_url[:protocol_end]
                    part2 = stream_url[protocol_end:]
                    
                    self.draw_text(part1, x, y, (100, 200, 255))
                    y += 12
                    self.draw_text(part2, x, y, (100, 200, 255))
                    y += 15
                else:
                    # Fallback split
                    mid = len(stream_url) // 2
                    self.draw_text(stream_url[:mid], x, y, (100, 200, 255))
                    y += 12
                    self.draw_text(stream_url[mid:], x, y, (100, 200, 255))
                    y += 15
            else:
                self.draw_text(stream_url, x, y, (100, 200, 255))
                y += 18
        
        # Stats
        status = "🟢 RUNNING" if self.is_running else "🔴 STOPPED"
        self.draw_text(status, x, y)
        y += 20
        self.draw_text(f"FPS: {self.fps:.1f}", x, y)
        y += 15
        self.draw_text(f"Objects: {self.objects}", x, y)
        y += 15
        self.draw_text(f"Total: {self.total_objects}", x, y)
        y += 20
        
        # RTSP URL display
        if self.enable_rtsp and self.detector and hasattr(self.detector, 'enable_rtsp') and self.detector.enable_rtsp:
            self.draw_text("📡 Stream URL:", x, y, self.yellow)
            y += 18
            url = get_stream_url(8554)
            
            # Create clickable URL box
            url_box_width = 260
            url_box_height = 40
            url_box_rect = (x, y, url_box_width, url_box_height)
            
            # Check if mouse is over URL box
            mouse_x, mouse_y = self.mouse_pos
            url_hovered = (x <= mouse_x <= x + url_box_width and 
                          y <= mouse_y <= y + url_box_height)
            
            # Draw URL box
            box_color = (80, 80, 120) if url_hovered else (60, 60, 80)
            pygame.draw.rect(self.screen, box_color, url_box_rect)
            pygame.draw.rect(self.screen, (100, 200, 255), url_box_rect, 2)
            
            # URL text (split if too long)
            if len(url) > 35:
                # Split into two lines
                mid_point = len(url) // 2
                # Find a good split point (prefer splitting at //)
                if '//' in url:
                    split_pos = url.find('//') + 2
                    url_part1 = url[:split_pos]
                    url_part2 = url[split_pos:]
                else:
                    url_part1 = url[:mid_point]
                    url_part2 = url[mid_point:]
                
                self.draw_text(url_part1, x + 5, y + 5, (200, 220, 255))
                self.draw_text(url_part2, x + 5, y + 20, (200, 220, 255))
            else:
                text_y = y + (url_box_height - 15) // 2
                self.draw_text(url, x + 5, text_y, (200, 220, 255))
            
            # Click to copy functionality
            if url_hovered and self.mouse_clicked:
                # Try multiple clipboard methods
                copied = False
                
                # Method 1: Try pyperclip
                try:
                    import pyperclip
                    pyperclip.copy(url)
                    copied = True
                    print(f"📋 Copied to clipboard: {url}")
                except ImportError:
                    pass
                except Exception:
                    pass
                
                # Method 2: Try using xclip (Linux)
                if not copied:
                    try:
                        import subprocess
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                        process.communicate(url.encode())
                        copied = True
                        print(f"📋 Copied to clipboard (xclip): {url}")
                    except Exception:
                        pass
                
                # Method 3: Fallback - print to console for manual copy
                if not copied:
                    print("=" * 50)
                    print(f"📋 RTSP STREAM URL (copy manually):")
                    print(f"{url}")
                    print("=" * 50)
                    print("💡 To enable automatic copying:")
                    print("   sudo apt install xclip")
                    print("   pip3 install pyperclip")
                
                # Show feedback regardless
                self.show_copy_feedback = True
                self.copy_feedback_timer = time.time()
            
            # Copy instruction
            instruction_color = (150, 150, 150) if not url_hovered else (200, 200, 200)
            self.draw_text("Click to copy", x + 5, y + url_box_height + 2, instruction_color)
            
            y += url_box_height + 20
            
        # Copy feedback
        if hasattr(self, 'show_copy_feedback') and self.show_copy_feedback:
            if time.time() - self.copy_feedback_timer < 2.0:  # Show for 2 seconds
                feedback_text = "✅ Copied!"
                feedback_surface = self.font.render(feedback_text, True, self.green)
                feedback_rect = feedback_surface.get_rect(center=(x + 130, y - 30))
                self.screen.blit(feedback_surface, feedback_rect)
            else:
                self.show_copy_feedback = False
    
    def render_video(self):
        """Fast video rendering with large display"""
        # Video area - most of the screen
        video_x = 300  # Leave space for control panel
        video_y = 10
        available_width = self.screen_width - video_x - 10
        available_height = self.screen_height - video_y - 10
        
        if self.frame_surface is None:
            # Placeholder - show large area
            pygame.draw.rect(self.screen, self.panel_bg, (video_x, video_y, available_width, available_height))
            pygame.draw.rect(self.screen, self.text_color, (video_x, video_y, available_width, available_height), 2)
            
            # Center text in large area
            text_x = video_x + available_width // 2 - 60
            text_y = video_y + available_height // 2
            self.draw_text("📹 No Video Feed", text_x, text_y)
            self.draw_text("Click 'Initialize' then 'Start'", text_x - 40, text_y + 20)
            return
        
        # Get frame size
        frame_w, frame_h = self.frame_surface.get_size()
        
        # Calculate scaling to fill most of the available space
        scale_x = available_width / frame_w
        scale_y = available_height / frame_h
        
        # Use the larger scale to fill more space (allow slight upscaling for better viewing)
        scale = min(scale_x, scale_y)
        
        # Allow upscaling up to 2x for better visibility on large screens
        if scale < 2.0:
            scale = min(scale, 2.0)
        
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        # Center the video in the available space
        center_x = video_x + (available_width - new_w) // 2
        center_y = video_y + (available_height - new_h) // 2
        
        # Scale the surface
        if abs(scale - 1.0) > 0.01:  # Only scale if significantly different
            scaled_surface = pygame.transform.scale(self.frame_surface, (new_w, new_h))
        else:
            scaled_surface = self.frame_surface
        
        # Draw video frame
        self.screen.blit(scaled_surface, (center_x, center_y))
        
        # Optional: Draw border around video for clarity
        border_color = (100, 100, 100)
        pygame.draw.rect(self.screen, border_color, (center_x-1, center_y-1, new_w+2, new_h+2), 1)
    
    def run(self):
        """Optimized main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("🎮 Fast ImGui Interface:")
        print("  - Optimized for maximum FPS")
        print("  - Minimal GUI overhead")
        print("  - F11 or F key: Toggle fullscreen")
        print("  - ESC: Exit fullscreen / Quit")
        print("  - Click 'Fullscreen' button to toggle")
        print("=" * 40)
        
        frame_count = 0
        last_video_update = 0
        
        while running:
            current_time = time.time()
            
            # Update loading animation
            if current_time - self.loading_timer > 0.5:  # Update every 500ms
                self.loading_dots = (self.loading_dots + 1) % 3
                self.loading_timer = current_time
            
            # Mouse state
            self.mouse_pos = pygame.mouse.get_pos()
            current_click = pygame.mouse.get_pressed()[0]
            self.mouse_clicked = current_click and not self.last_click
            self.last_click = current_click
            
            # Events (minimal)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.is_initializing:
                            continue  # Don't allow exit during initialization
                        elif self.is_fullscreen:
                            self.toggle_fullscreen()  # Exit fullscreen first
                        else:
                            running = False
                    elif event.key == pygame.K_F11:
                        if not self.is_initializing:
                            self.toggle_fullscreen()
                    elif event.key == pygame.K_f:
                        if not self.is_initializing:
                            self.toggle_fullscreen()
            
            # Clear screen
            self.screen.fill(self.bg_dark)
            
            if self.is_initializing:
                # Show loading screen
                self.render_loading_screen()
            else:
                # Normal UI
                # Update video surface (throttled to 15 FPS for GUI)
                if current_time - last_video_update > 0.066:  # ~15 FPS GUI updates
                    self.update_video_surface()
                    last_video_update = current_time
                
                # Render UI (fast)
                self.render_controls()
                self.render_video()
            
            # Update display
            pygame.display.flip()
            
            # Target 30 FPS for GUI (detection runs faster)
            clock.tick(30)
            frame_count += 1
        
        # Cleanup
        self.stop_detection()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Fast Object Detection Interface')
    parser.add_argument('--fullscreen', '-f', action='store_true', help='Start in fullscreen mode')
    args = parser.parse_args()
    
    print("🚀 Fast Object Detection Interface")
    print("⚡ Optimized for maximum FPS performance")
    if args.fullscreen:
        print("🖥️ Starting in fullscreen mode")
    print("=" * 50)
    
    try:
        app = FastImGuiDetector()
        if args.fullscreen:
            app.toggle_fullscreen()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()