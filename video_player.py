#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import json
import sys
import os
from datetime import datetime
from pathlib import Path
import argparse

class DetectionVideoPlayer:
    """Video player with synchronized detection data display"""
    
    def __init__(self, video_path=None):
        pygame.init()
        
        # Display settings
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Detection Video Player")
        
        # Fonts
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 16)
        
        # Colors
        self.bg_dark = (25, 25, 25)
        self.panel_bg = (40, 40, 40)
        self.button_bg = (60, 120, 200)
        self.button_hover = (80, 140, 220)
        self.text_color = (255, 255, 255)
        self.green = (50, 200, 50)
        self.red = (200, 50, 50)
        self.yellow = (200, 200, 50)
        self.blue = (50, 150, 255)
        
        # Video playback
        self.video_path = video_path
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.frame_surface = None
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = 30.0
        self.frame_time = 1.0 / self.fps
        self.last_frame_time = 0
        self.video_duration = 0
        
        # Detection data
        self.detection_data = None
        self.detection_file = None
        self.show_detections = True
        self.current_detections = []
        
        # UI state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        self.last_click = False
        self.dragging_progress = False
        
        # UI layout
        self.video_x = 10
        self.video_y = 10
        self.video_w = 900
        self.video_h = 600
        
        self.controls_x = 10
        self.controls_y = self.video_y + self.video_h + 10
        self.controls_w = self.video_w
        self.controls_h = 60
        
        self.info_x = self.video_x + self.video_w + 10
        self.info_y = 10
        self.info_w = self.screen_width - self.info_x - 10
        self.info_h = self.screen_height - 20
        
        # Load video if provided
        if video_path:
            self.load_video(video_path)
    
    def load_video(self, video_path):
        """Load video file and associated detection data"""
        try:
            video_path = Path(video_path)
            print(f"🎬 Attempting to load video: {video_path}")
            
            if not video_path.exists():
                print(f"❌ Video file not found: {video_path}")
                return False
            
            # Check file size
            file_size = video_path.stat().st_size
            print(f"📄 File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
            
            if file_size < 1000:  # Less than 1KB is likely corrupted
                print(f"❌ Video file too small, likely corrupted: {file_size} bytes")
                return False
            
            # Release previous video
            if self.cap:
                self.cap.release()
            
            # Load new video
            self.video_path = str(video_path)
            print(f"🔧 Opening video with OpenCV...")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                print(f"💡 This could be due to:")
                print(f"   - Unsupported codec")
                print(f"   - Corrupted file")
                print(f"   - Missing OpenCV codecs")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_time = 1.0 / self.fps
            self.video_duration = self.total_frames / self.fps
            self.current_frame_num = 0
            
            # Get video resolution
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📊 Video properties: {self.total_frames} frames at {self.fps:.1f} fps")
            print(f"📐 Video resolution: {video_width}x{video_height}")
            
            if self.total_frames <= 0:
                print(f"⚠️ Warning: Video has no frames or frame count unknown")
                print(f"💡 This could be:")
                print(f"   - Recording in progress")
                print(f"   - Corrupted file")
                print(f"   - Unsupported format")
            
            # Try to load detection data (handle both new and legacy naming)
            file_stem = str(video_path).replace('.mp4', '').replace('.avi', '')
            detection_path = f"{file_stem}_detections.json"
            self.load_detection_data(detection_path)
            
            print(f"✅ Video loaded: {video_path.name}")
            print(f"📊 Frames: {self.total_frames}, FPS: {self.fps:.1f}, Duration: {self.video_duration:.1f}s")
            
            # Load first frame
            if self.total_frames > 0:
                self.seek_to_frame(0)
            else:
                # Try to read first frame manually for videos with unknown frame count
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.update_frame_surface()
                    # Reset to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("✅ Successfully read first frame despite unknown frame count")
                else:
                    print("⚠️ Cannot read any frames from video")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading video: {e}")
            return False
    
    def load_detection_data(self, detection_path):
        """Load detection data from JSON file"""
        try:
            detection_path = Path(detection_path)
            if detection_path.exists():
                with open(detection_path, 'r') as f:
                    self.detection_data = json.load(f)
                    self.detection_file = str(detection_path)
                
                detections = self.detection_data.get('detections', [])
                print(f"📊 Detection data loaded: {len(detections)} detections")
                return True
            else:
                print(f"ℹ️ No detection data found: {detection_path.name}")
                self.detection_data = None
                self.detection_file = None
                return False
                
        except Exception as e:
            print(f"❌ Error loading detection data: {e}")
            self.detection_data = None
            self.detection_file = None
            return False
    
    def seek_to_frame(self, frame_num):
        """Seek to specific frame"""
        if not self.cap:
            return
        
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.current_frame_num = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.update_frame_surface()
            self.update_current_detections()
    
    def seek_to_time(self, time_seconds):
        """Seek to specific time"""
        if not self.cap:
            return
        
        frame_num = int(time_seconds * self.fps)
        self.seek_to_frame(frame_num)
    
    def play_pause(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_frame_time = pygame.time.get_ticks() / 1000.0
    
    def update_playback(self):
        """Update video playback"""
        if not self.is_playing or not self.cap:
            return
        
        current_time = pygame.time.get_ticks() / 1000.0
        time_since_last = current_time - self.last_frame_time
        
        # Only advance if enough time has passed
        if time_since_last >= self.frame_time:
            self.last_frame_time = current_time
            
            # Calculate how many frames to advance (for smooth playback)
            frames_to_advance = max(1, int(time_since_last / self.frame_time))
            next_frame = self.current_frame_num + frames_to_advance
            
            if next_frame >= self.total_frames:
                self.is_playing = False  # End of video
                return
            
            self.seek_to_frame(next_frame)
    
    def update_frame_surface(self):
        """Convert current frame to pygame surface"""
        if self.current_frame is None:
            return
        
        # Use original frame for better performance
        frame = self.current_frame
        
        # Draw detections if enabled (only copy if needed)
        if self.show_detections and self.current_detections:
            frame = self.draw_detections_on_frame(frame.copy())
        
        # Fast conversion to pygame surface
        try:
            # Convert colorspace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create surface directly from array for better performance
            frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
            self.frame_surface = pygame.surfarray.make_surface(frame_transposed)
        except Exception as e:
            print(f"⚠️ Error converting frame: {e}")
            # Fallback to original method
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rotated = np.rot90(frame_rgb)
            frame_flipped = np.flipud(frame_rotated)
            self.frame_surface = pygame.surfarray.make_surface(frame_flipped)
    
    def draw_detections_on_frame(self, frame):
        """Draw detection boxes on frame"""
        if not self.current_detections:
            return frame
            
        for detection in self.current_detections:
            try:
                # Get coordinates - ensure they're integers
                x1_raw = int(detection.get('x1', 0))
                y1_raw = int(detection.get('y1', 0)) 
                x2_raw = int(detection.get('x2', 0))
                y2_raw = int(detection.get('y2', 0))
                confidence = float(detection.get('confidence', 0))
                obj_type_raw = str(detection.get('object_type', 'unknown'))
                
                # Map numeric object types to human-readable names (COCO classes)
                coco_classes = {
                    '0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle', '4': 'airplane',
                    '5': 'bus', '6': 'train', '7': 'truck', '8': 'boat', '9': 'traffic light',
                    '10': 'fire hydrant', '11': 'stop sign', '12': 'parking meter', '13': 'bench',
                    '14': 'bird', '15': 'cat', '16': 'dog', '17': 'horse', '18': 'sheep', '19': 'cow',
                    '20': 'elephant', '21': 'bear', '22': 'zebra', '23': 'giraffe', '24': 'backpack',
                    '25': 'umbrella', '26': 'handbag', '27': 'tie', '28': 'suitcase', '29': 'frisbee',
                    '30': 'skis', '31': 'snowboard', '32': 'sports ball', '33': 'kite', '34': 'baseball bat',
                    '35': 'baseball glove', '36': 'skateboard', '37': 'surfboard', '38': 'tennis racket',
                    '39': 'bottle', '40': 'wine glass', '41': 'cup', '42': 'fork', '43': 'knife',
                    '44': 'spoon', '45': 'bowl', '46': 'banana', '47': 'apple', '48': 'sandwich',
                    '49': 'orange', '50': 'broccoli', '51': 'carrot', '52': 'hot dog', '53': 'pizza',
                    '54': 'donut', '55': 'cake', '56': 'chair', '57': 'couch', '58': 'potted plant',
                    '59': 'bed', '60': 'dining table', '61': 'toilet', '62': 'tv', '63': 'laptop',
                    '64': 'mouse', '65': 'remote', '66': 'keyboard', '67': 'cell phone', '68': 'microwave',
                    '69': 'oven', '70': 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book',
                    '74': 'clock', '75': 'vase', '76': 'scissors', '77': 'teddy bear', '78': 'hair drier',
                    '79': 'toothbrush'
                }
                
                # Get human-readable object type
                obj_type = coco_classes.get(obj_type_raw, obj_type_raw)
                
                # Fix coordinate order - ensure x1 < x2 and y1 < y2
                x1 = min(x1_raw, x2_raw)
                x2 = max(x1_raw, x2_raw)
                y1 = min(y1_raw, y2_raw)
                y2 = max(y1_raw, y2_raw)
                
                # Validate coordinates within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w-1))
                x2 = max(0, min(x2, frame_w-1))
                y1 = max(0, min(y1, frame_h-1))
                y2 = max(0, min(y2, frame_h-1))
                
                # Ensure valid box (minimum size of 1 pixel)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Color based on object type with transparency
                colors = {
                    'person': (0, 200, 0),      # Green (slightly dimmed)
                    'car': (200, 0, 0),         # Blue (slightly dimmed)
                    'truck': (0, 0, 200),       # Red (slightly dimmed)
                    'bicycle': (200, 200, 0),   # Cyan (slightly dimmed)
                    'motorcycle': (200, 0, 200), # Magenta (slightly dimmed)
                }
                color = colors.get(obj_type.lower(), (0, 200, 200))  # Default yellow (dimmed)
                
                # Create overlay for transparency effect
                overlay = frame.copy()
                
                # Draw bounding box with thinner line (thickness 1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                
                # Draw label with background
                label = f"{obj_type}: {confidence:.2f}"
                font_scale = 0.6
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Label background with transparency
                label_y = y1 - 10 if y1 > 30 else y2 + 30
                cv2.rectangle(overlay, (x1, label_y - label_h - 5), (x1 + label_w + 10, label_y + 5), color, -1)
                
                # Apply transparency (alpha blend)
                alpha = 0.7  # 70% opacity, 30% transparency
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Label text (opaque for readability)
                cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
            except Exception as e:
                print(f"⚠️ Error drawing detection: {e}")
                continue
        
        return frame
    
    def update_current_detections(self):
        """Update detections for current frame"""
        self.current_detections = []
        
        if not self.detection_data:
            return
        
        current_time = self.current_frame_num / self.fps
        
        # Find detections for current frame (with small tolerance)
        tolerance = self.frame_time / 2
        
        for detection in self.detection_data.get('detections', []):
            detection_time = detection.get('timestamp', 0)
            if abs(detection_time - current_time) <= tolerance:
                self.current_detections.append(detection)
    
    def draw_button(self, x, y, w, h, text, active=True, style='normal'):
        """Draw button"""
        mouse_x, mouse_y = self.mouse_pos
        hovered = x <= mouse_x <= x + w and y <= mouse_y <= y + h and active
        
        # Choose color
        if style == 'success':
            color = self.green
        elif style == 'error':
            color = self.red
        elif style == 'warning':
            color = self.yellow
        else:
            color = self.button_hover if hovered else self.button_bg
        
        if not active:
            color = (80, 80, 80)
        
        # Draw button
        pygame.draw.rect(self.screen, color, (x, y, w, h))
        pygame.draw.rect(self.screen, self.text_color, (x, y, w, h), 1)
        
        # Text
        text_color = self.text_color if active else (150, 150, 150)
        text_surface = self.font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=(x + w//2, y + h//2))
        self.screen.blit(text_surface, text_rect)
        
        return hovered and self.mouse_clicked and active
    
    def draw_text(self, text, x, y, color=None, font=None):
        """Draw text"""
        if color is None:
            color = self.text_color
        if font is None:
            font = self.font
        
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))
        return text_surf.get_size()
    
    def draw_progress_bar(self, x, y, w, h):
        """Draw progress bar with seeking capability"""
        # Background
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, w, h))
        pygame.draw.rect(self.screen, self.text_color, (x, y, w, h), 1)
        
        if self.total_frames > 0:
            # Progress fill
            progress = self.current_frame_num / self.total_frames
            fill_w = int(w * progress)
            pygame.draw.rect(self.screen, self.blue, (x, y, fill_w, h))
            
            # Handle clicking for seeking
            mouse_x, mouse_y = self.mouse_pos
            if (x <= mouse_x <= x + w and y <= mouse_y <= y + h):
                if self.mouse_clicked or self.dragging_progress:
                    self.dragging_progress = True
                    # Calculate new position
                    relative_x = mouse_x - x
                    new_progress = relative_x / w
                    new_frame = int(new_progress * self.total_frames)
                    self.seek_to_frame(new_frame)
                    
                    if not pygame.mouse.get_pressed()[0]:
                        self.dragging_progress = False
        
        # Time display
        current_time = self.current_frame_num / self.fps if self.fps > 0 else 0
        time_text = f"{self.format_time(current_time)} / {self.format_time(self.video_duration)}"
        self.draw_text(time_text, x + w + 10, y + (h - 20) // 2, font=self.font_small)
    
    def format_time(self, seconds):
        """Format time as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def render_video(self):
        """Render video area"""
        # Video background
        pygame.draw.rect(self.screen, (20, 20, 20), (self.video_x, self.video_y, self.video_w, self.video_h))
        pygame.draw.rect(self.screen, self.text_color, (self.video_x, self.video_y, self.video_w, self.video_h), 2)
        
        if self.frame_surface is None:
            # No video message
            self.draw_text("No video loaded", self.video_x + self.video_w // 2 - 60, 
                          self.video_y + self.video_h // 2, font=self.font_large)
            return
        
        # Scale frame to fit video area
        frame_w, frame_h = self.frame_surface.get_size()
        scale_x = self.video_w / frame_w
        scale_y = self.video_h / frame_h
        scale = min(scale_x, scale_y)
        
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        # Center video
        video_x = self.video_x + (self.video_w - new_w) // 2
        video_y = self.video_y + (self.video_h - new_h) // 2
        
        # Scale and draw
        if abs(scale - 1.0) > 0.01:
            scaled_surface = pygame.transform.scale(self.frame_surface, (new_w, new_h))
        else:
            scaled_surface = self.frame_surface
        
        self.screen.blit(scaled_surface, (video_x, video_y))
    
    def render_controls(self):
        """Render playback controls"""
        # Controls background
        pygame.draw.rect(self.screen, self.panel_bg, (self.controls_x, self.controls_y, self.controls_w, self.controls_h))
        pygame.draw.rect(self.screen, self.text_color, (self.controls_x, self.controls_y, self.controls_w, self.controls_h), 1)
        
        x = self.controls_x + 10
        y = self.controls_y + 10
        
        # Play/Pause button
        play_text = "⏸️ Pause" if self.is_playing else "▶️ Play"
        if self.draw_button(x, y, 80, 30, play_text, active=self.cap is not None):
            self.play_pause()
        x += 90
        
        # Stop button
        if self.draw_button(x, y, 60, 30, "⏹️ Stop", active=self.cap is not None):
            self.is_playing = False
            self.seek_to_frame(0)
        x += 70
        
        # Frame step buttons
        if self.draw_button(x, y, 40, 30, "⏮️", active=self.cap is not None):
            self.seek_to_frame(self.current_frame_num - 1)
        x += 45
        
        if self.draw_button(x, y, 40, 30, "⏭️", active=self.cap is not None):
            self.seek_to_frame(self.current_frame_num + 1)
        x += 50
        
        # Detection toggle
        detection_style = 'success' if self.show_detections else 'normal'
        detection_text = "🎯 Hide" if self.show_detections else "🎯 Show"
        if self.draw_button(x, y, 90, 30, detection_text, 
                           active=self.detection_data is not None, style=detection_style):
            self.show_detections = not self.show_detections
            self.update_frame_surface()
        x += 100
        
        # Load file button
        if self.draw_button(x, y, 100, 30, "📁 Load Video"):
            print("💡 To load a video, run: python3 video_player.py <video_file>")
        
        # Progress bar
        progress_y = y + 35
        self.draw_progress_bar(self.controls_x + 10, progress_y, self.controls_w - 20, 15)
    
    def render_info_panel(self):
        """Render information panel"""
        # Info panel background
        pygame.draw.rect(self.screen, self.panel_bg, (self.info_x, self.info_y, self.info_w, self.info_h))
        pygame.draw.rect(self.screen, self.text_color, (self.info_x, self.info_y, self.info_w, self.info_h), 1)
        
        x = self.info_x + 10
        y = self.info_y + 10
        
        # Title
        self.draw_text("📊 Video Information", x, y, font=self.font_large)
        y += 35
        
        if self.video_path:
            video_name = Path(self.video_path).name
            self.draw_text(f"📄 File: {video_name}", x, y)
            y += 25
            
            self.draw_text(f"🎬 Frames: {self.total_frames}", x, y)
            y += 20
            
            self.draw_text(f"⚡ FPS: {self.fps:.1f}", x, y)
            y += 20
            
            self.draw_text(f"⏱️ Duration: {self.format_time(self.video_duration)}", x, y)
            y += 20
            
            current_time = self.current_frame_num / self.fps if self.fps > 0 else 0
            self.draw_text(f"⏰ Current: {self.format_time(current_time)}", x, y)
            y += 30
        
        # Detection information
        self.draw_text("🎯 Detection Data", x, y, font=self.font_large)
        y += 35
        
        if self.detection_data:
            total_detections = len(self.detection_data.get('detections', []))
            self.draw_text(f"📊 Total detections: {total_detections}", x, y)
            y += 20
            
            self.draw_text(f"📄 File: {Path(self.detection_file).name if self.detection_file else 'None'}", x, y)
            y += 20
            
            current_count = len(self.current_detections)
            self.draw_text(f"👁️ Current frame: {current_count} detections", x, y, 
                          self.green if current_count > 0 else self.text_color)
            y += 30
            
            # Show current detections
            if self.current_detections:
                self.draw_text("Current Detections:", x, y, self.yellow)
                y += 25
                
                for i, detection in enumerate(self.current_detections[:10]):  # Show max 10
                    obj_type = detection.get('object_type', 'Unknown')
                    confidence = detection.get('confidence', 0)
                    x1, y1 = detection.get('x1', 0), detection.get('y1', 0)
                    
                    det_text = f"• {obj_type}: {confidence:.2f} at ({x1},{y1})"
                    self.draw_text(det_text, x + 10, y, font=self.font_small)
                    y += 18
                    
                    if y > self.info_y + self.info_h - 50:  # Don't overflow
                        break
        else:
            self.draw_text("No detection data loaded", x, y, (150, 150, 150))
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("🎬 Detection Video Player")
        print("Controls:")
        print("  - Space: Play/Pause")
        print("  - Left/Right arrows: Frame step")
        print("  - Click progress bar: Seek")
        print("  - ESC: Quit")
        print("=" * 50)
        
        while running:
            # Mouse state
            self.mouse_pos = pygame.mouse.get_pos()
            current_click = pygame.mouse.get_pressed()[0]
            self.mouse_clicked = current_click and not self.last_click
            self.last_click = current_click
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width = event.w
                    self.screen_height = event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                    # Update layout based on new size
                    self.video_w = min(900, self.screen_width - 520)  # Leave space for info panel
                    self.video_h = min(600, self.screen_height - 90)  # Leave space for controls
                    self.info_x = self.video_x + self.video_w + 10
                    self.info_w = self.screen_width - self.info_x - 10
                    self.info_h = self.screen_height - 20
                    self.controls_w = self.video_w
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.cap:
                            self.play_pause()
                    elif event.key == pygame.K_LEFT:
                        if self.cap:
                            self.seek_to_frame(self.current_frame_num - 1)
                    elif event.key == pygame.K_RIGHT:
                        if self.cap:
                            self.seek_to_frame(self.current_frame_num + 1)
                    elif event.key == pygame.K_d:
                        if self.detection_data:
                            self.show_detections = not self.show_detections
                            self.update_frame_surface()
            
            # Update playback
            self.update_playback()
            
            # Render
            self.screen.fill(self.bg_dark)
            self.render_video()
            self.render_controls()
            self.render_info_panel()
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS for smooth UI
        
        # Cleanup
        if self.cap:
            self.cap.release()
        pygame.quit()

def test_video_only(video_path):
    """Test video loading without GUI"""
    print(f"🧪 Testing video file: {video_path}")
    
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        return False
    
    # Test OpenCV loading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ OpenCV cannot open video")
        return False
    
    # Get properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video properties:")
    print(f"   📐 Resolution: {width}x{height}")
    print(f"   🎬 Frames: {total_frames}")
    print(f"   ⚡ FPS: {fps}")
    
    # Test reading first few frames
    for i in range(min(5, total_frames)):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i}")
            cap.release()
            return False
        print(f"✅ Frame {i}: {frame.shape}")
    
    cap.release()
    print("✅ Video test completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Video Player with Detection Data')
    parser.add_argument('video_file', nargs='?', help='Path to video file')
    parser.add_argument('--test', action='store_true', help='Test mode (no GUI)')
    args = parser.parse_args()
    
    if args.test and args.video_file:
        test_video_only(args.video_file)
        return
    
    try:
        player = DetectionVideoPlayer(args.video_file)
        player.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()