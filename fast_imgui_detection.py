#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import threading
import time
import argparse
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip, get_stream_url

class FastImGuiDetector:
    """Optimized ImGui-style detector with minimal overhead"""
    
    def __init__(self):
        pygame.init()
        
        # Display settings
        self.screen_width = 1200
        self.screen_height = 800
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
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
        
        # Recording functionality - frame sharing approach
        self.is_recording = False
        self.recording_thread = None
        self.recording_start_time = None
        self.detection_log_file = None
        self.detection_data = []
        self.recording_frame_count = 0
        self.recording_should_stop = False
        self.shared_recording_frame = None
        self.recording_frame_lock = threading.Lock()
        
        # Snapshot feedback
        self.show_snapshot_feedback = False
        self.snapshot_feedback_timer = 0
        
        # Tab system
        self.current_tab = "monitor"  # "monitor" or "gallery"
        self.tab_height = 40
        
        # Gallery
        self.gallery_items = []
        self.gallery_folders = []
        self.current_folder = None  # None means showing folder list
        self.gallery_scroll = 0
        self.gallery_item_height = 120
        self.gallery_cols = 4
        self.thumbnail_cache = {}
        self.selected_media = None  # For viewing full media
        
        # Mouse
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        self.last_click = False
        
        # Pre-render surfaces (cache for performance)
        self.button_cache = {}
        
        print("✅ Fast ImGui detector ready!")
        
        # Create recordings directory structure
        self.setup_recording_directories()
        
        # Load gallery items
        self.refresh_gallery()
        
        # Start automatic initialization
        self.start_initialization()
    
    def setup_recording_directories(self):
        """Create directory structure for recordings"""
        base_dir = Path("recordings")
        base_dir.mkdir(exist_ok=True)
        
        # Create date-based subdirectory
        today = datetime.now().strftime("%Y-%m-%d")
        self.today_dir = base_dir / today
        self.today_dir.mkdir(exist_ok=True)
        
        print(f"📁 Recording directory: {self.today_dir}")
    
    def get_timestamp_filename(self, file_type, extension):
        """Generate filename with format: [type]_[date]_[time]_[timestamp]"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")  # 2025-09-05
        time_str = now.strftime("%H-%M-%S")  # 14-30-22
        timestamp = int(now.timestamp())
        return f"{file_type}_{date_str}_{time_str}_{timestamp}.{extension}"
    
    def start_recording(self):
        """Start completely independent video recording thread"""
        if not self.is_recording:
            # Ensure directory exists for today
            today = datetime.now().strftime("%Y-%m-%d")
            today_dir = Path("recordings") / today
            today_dir.mkdir(exist_ok=True)
            
            # Generate filename with new format
            video_filename = self.get_timestamp_filename("video", "avi")
            base_name = video_filename.replace(".avi", "")
            
            # File paths
            video_filepath = today_dir / video_filename
            detection_filepath = today_dir / f"{base_name}_detections.json"
            
            # Initialize recording state
            self.detection_log_file = str(detection_filepath)
            self.detection_data = []
            self.recording_frame_count = 0
            self.recording_should_stop = False
            self.is_recording = True
            self.recording_start_time = time.time()
            
            # Start completely independent recording thread
            self.recording_thread = threading.Thread(
                target=self.independent_recording_worker, 
                args=(video_filepath,), 
                daemon=True
            )
            self.recording_thread.start()
            
            print(f"🎥 Started independent recording: {video_filepath}")
            print(f"📊 Detection logging: {detection_filepath}")
            return True
        return False
    
    def independent_recording_worker(self, video_filepath):
        """Recording thread using shared frames to avoid camera conflicts"""
        print("🎬 Frame-sharing recording thread started")
        
        video_writer = None
        frame_count = 0
        last_frame_time = time.time()
        target_fps = 20.0
        frame_interval = 1.0 / target_fps
        
        try:
            # Wait for first shared frame
            while self.is_recording and self.shared_recording_frame is None:
                time.sleep(0.01)
            
            if not self.is_recording:
                return
            
            # Get frame dimensions from first shared frame
            with self.recording_frame_lock:
                if self.shared_recording_frame is not None:
                    h, w = self.shared_recording_frame.shape[:2]
                else:
                    print("❌ No shared frame available")
                    return
            
            # Try different codecs with better debugging
            codecs_to_try = [
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
                ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
                ('I420', cv2.VideoWriter_fourcc(*'I420')),
                ('raw ', cv2.VideoWriter_fourcc(*'raw ')),
            ]
            
            print(f"🎥 Trying to create video writer for {w}x{h} at {target_fps}fps")
            
            for codec_name, fourcc in codecs_to_try:
                print(f"🔧 Trying codec: {codec_name}")
                video_writer = cv2.VideoWriter(str(video_filepath), fourcc, target_fps, (w, h))
                
                if video_writer.isOpened():
                    print(f"✅ Video writer opened with codec: {codec_name}")
                    # Test write a dummy frame
                    test_frame = self.shared_recording_frame.copy()
                    test_result = video_writer.write(test_frame)
                    if test_result:
                        print(f"✅ Test frame write successful with {codec_name}")
                        break
                    else:
                        print(f"❌ Test frame write failed with {codec_name}")
                        video_writer.release()
                        video_writer = None
                else:
                    print(f"❌ Failed to open video writer with {codec_name}")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
            
            if not video_writer or not video_writer.isOpened():
                print("❌ All video codecs failed, trying image sequence fallback...")
                # Create directory for image sequence
                import os
                img_seq_dir = str(video_filepath).replace('.avi', '_frames')
                os.makedirs(img_seq_dir, exist_ok=True)
                print(f"📁 Saving image sequence to: {img_seq_dir}")
                video_writer = "image_sequence"  # Special marker
            else:
                img_seq_dir = None
            
            print(f"📐 Recording dimensions: {w}x{h}")
            
            # Recording loop using shared frames
            while self.is_recording and not self.recording_should_stop:
                current_time = time.time()
                
                # Maintain target FPS
                if current_time - last_frame_time >= frame_interval:
                    # Get shared frame
                    frame_to_record = None
                    with self.recording_frame_lock:
                        if self.shared_recording_frame is not None:
                            frame_to_record = self.shared_recording_frame.copy()
                    
                    if frame_to_record is not None:
                        # Validate frame before writing
                        if frame_to_record.shape[:2] == (h, w) and len(frame_to_record.shape) == 3:
                            # Ensure frame is in correct format (BGR)
                            if frame_to_record.dtype != np.uint8:
                                frame_to_record = frame_to_record.astype(np.uint8)
                            
                            success = False
                            
                            if video_writer == "image_sequence":
                                # Save as image sequence
                                img_filename = f"frame_{frame_count:06d}.jpg"
                                img_path = os.path.join(img_seq_dir, img_filename)
                                success = cv2.imwrite(img_path, frame_to_record)
                                if not success:
                                    print(f"⚠️ Failed to save image {img_filename}")
                            else:
                                # Write to video file
                                success = video_writer.write(frame_to_record)
                                
                                # If video writing fails, switch to image sequence
                                if not success and frame_count == 0:
                                    print(f"❌ Video writing failed, switching to image sequence mode")
                                    # Close current video writer
                                    video_writer.release()
                                    
                                    # Setup image sequence
                                    import os
                                    img_seq_dir = str(video_filepath).replace('.avi', '_frames')
                                    os.makedirs(img_seq_dir, exist_ok=True)
                                    video_writer = "image_sequence"
                                    print(f"📁 Switched to image sequence: {img_seq_dir}")
                                    
                                    # Save current frame as first image
                                    img_filename = f"frame_{frame_count:06d}.jpg"
                                    img_path = os.path.join(img_seq_dir, img_filename)
                                    success = cv2.imwrite(img_path, frame_to_record)
                            
                            if success:
                                frame_count += 1
                                self.recording_frame_count = frame_count
                                last_frame_time = current_time
                                
                                # Progress every 60 frames (3 seconds at 20fps)
                                if frame_count % 60 == 0:
                                    method = "image sequence" if video_writer == "image_sequence" else "video file"
                                    print(f"📹 Recorded {frame_count} frames via {method}")
                            else:
                                method_info = f"image sequence to {img_seq_dir}" if video_writer == "image_sequence" else f"video file - format: {frame_to_record.shape}, dtype: {frame_to_record.dtype}"
                                print(f"⚠️ Failed to write frame {frame_count} - {method_info}")
                        else:
                            print(f"⚠️ Invalid frame dimensions: got {frame_to_record.shape}, expected ({h}, {w}, 3)")
                    else:
                        # No frame available, wait
                        time.sleep(0.001)
                else:
                    # Sleep until next frame time
                    sleep_time = frame_interval - (current_time - last_frame_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            print(f"🎬 Frame-sharing recording completed: {frame_count} frames")
            
        except Exception as e:
            print(f"❌ Frame-sharing recording error: {e}")
        
        finally:
            # Cleanup
            if video_writer and video_writer != "image_sequence":
                video_writer.release()
            
            # Create video from image sequence if needed
            if video_writer == "image_sequence" and frame_count > 0:
                print(f"🎬 Converting {frame_count} images to video...")
                try:
                    self.convert_image_sequence_to_video(img_seq_dir, video_filepath, target_fps)
                except Exception as e:
                    print(f"⚠️ Failed to convert image sequence: {e}")
                    print(f"📁 Raw frames saved in: {img_seq_dir}")
            
            print("🎬 Frame-sharing recording thread stopped")
    
    def convert_image_sequence_to_video(self, img_seq_dir, output_video_path, fps):
        """Convert image sequence to video using ffmpeg or OpenCV"""
        try:
            import subprocess
            import glob
            
            # Try using ffmpeg first (more reliable)
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite
                '-framerate', str(fps),
                '-i', f'{img_seq_dir}/frame_%06d.jpg',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                str(output_video_path)
            ]
            
            print(f"🎬 Converting with ffmpeg: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Successfully converted to video: {output_video_path}")
                # Clean up image sequence
                import shutil
                shutil.rmtree(img_seq_dir)
                print(f"🗑️ Cleaned up temporary frames")
                return True
            else:
                print(f"❌ ffmpeg failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"❌ ffmpeg conversion failed: {e}")
        
        # Fallback: leave image sequence
        print(f"📁 Image sequence preserved at: {img_seq_dir}")
        print(f"💡 To convert manually: ffmpeg -framerate {fps} -i {img_seq_dir}/frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")
        return False
    
    def stop_recording(self):
        """Stop independent video recording and save detection data"""
        if self.is_recording:
            print("⏹️ Stopping independent recording...")
            self.recording_should_stop = True
            self.is_recording = False
            
            # Wait for independent recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                print("⏳ Waiting for recording thread to finish...")
                self.recording_thread.join(timeout=3.0)  # Wait max 3 seconds
            
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            # Save detection data to JSON file
            if self.detection_log_file and self.detection_data:
                try:
                    detection_summary = {
                        "video_info": {
                            "duration_seconds": duration,
                            "fps": 20.0,  # Recording FPS
                            "total_frames": len([d for d in self.detection_data if d.get('frame_number')]),
                            "recording_start": datetime.fromtimestamp(self.recording_start_time).isoformat(),
                            "recording_end": datetime.now().isoformat()
                        },
                        "detections": self.detection_data
                    }
                    
                    with open(self.detection_log_file, 'w') as f:
                        json.dump(detection_summary, f, indent=2)
                    
                    print(f"📊 Detection data saved: {self.detection_log_file}")
                    print(f"📈 Total detections logged: {len(self.detection_data)}")
                    
                    # Also create CSV version for easy analysis
                    csv_file = self.detection_log_file.replace('.json', '.csv')
                    self.export_detections_to_csv(csv_file)
                    
                except Exception as e:
                    print(f"❌ Error saving detection data: {e}")
            
            self.recording_start_time = None
            self.detection_log_file = None
            self.detection_data = []
            self.recording_frame_count = 0
            self.recording_should_stop = False
            self.recording_thread = None
            
            # Clean up shared frame
            with self.recording_frame_lock:
                self.shared_recording_frame = None
            
            print(f"⏹️ Recording stopped (duration: {duration:.1f}s)")
            
            # Refresh gallery if we're in gallery tab
            if self.current_tab == 'gallery':
                self.refresh_gallery()
    
    def export_detections_to_csv(self, csv_file):
        """Export detection data to CSV format"""
        try:
            with open(csv_file, 'w', newline='') as f:
                if not self.detection_data:
                    return
                
                # Get fieldnames from first detection
                fieldnames = ['timestamp', 'frame_number', 'object_type', 'confidence', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'width', 'height']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for detection in self.detection_data:
                    writer.writerow(detection)
                
            print(f"📊 CSV exported: {csv_file}")
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
    
    def take_snapshot(self):
        """Take a snapshot image"""
        if self.current_frame is not None:
            # Ensure directory exists for today
            today = datetime.now().strftime("%Y-%m-%d")
            today_dir = Path("recordings") / today
            today_dir.mkdir(exist_ok=True)
            
            # Generate filename with new format
            filename = self.get_timestamp_filename("image", "jpg")
            filepath = today_dir / filename
            
            # Save current frame
            success = cv2.imwrite(str(filepath), self.current_frame)
            if success:
                print(f"📸 Snapshot saved: {filepath}")
                # Show visual feedback
                self.show_snapshot_feedback = True
                self.snapshot_feedback_timer = time.time()
                # Refresh gallery if we're in gallery tab
                if self.current_tab == 'gallery':
                    self.refresh_gallery()
                return True
            else:
                print("❌ Failed to save snapshot")
                return False
        return False
    
    def refresh_gallery(self):
        """Scan recordings directory and load folders and media files"""
        self.gallery_items = []
        self.gallery_folders = []
        recordings_dir = Path("recordings")
        
        if not recordings_dir.exists():
            return
        
        # Scan all date directories for folder list
        for date_dir in sorted(recordings_dir.glob("*"), reverse=True):  # Most recent first
            if date_dir.is_dir():
                # Count media files in this folder (both new and legacy formats)
                media_count = 0
                media_count += len(list(date_dir.glob("image_*.jpg")))      # New format
                media_count += len(list(date_dir.glob("video_*.mp4")))      # New format
                media_count += len(list(date_dir.glob("video_*.avi")))      # New format
                media_count += len(list(date_dir.glob("recording_*.mp4")))  # Legacy
                media_count += len(list(date_dir.glob("recording_*.avi")))  # Legacy
                
                if media_count > 0:
                    folder_info = {
                        'path': date_dir,
                        'name': date_dir.name,
                        'count': media_count,
                        'mtime': date_dir.stat().st_mtime
                    }
                    self.gallery_folders.append(folder_info)
        
        # If we're viewing a specific folder, load its media files
        if self.current_folder:
            folder_path = Path(self.current_folder)
            if folder_path.exists():
                media_files = []
                
                # Add images (all formats: new with dashes, legacy)
                for img_file in folder_path.glob("image_*.jpg"):  # All image files
                    media_files.append(img_file)
                
                # Add videos (both old and new format)
                for video_file in folder_path.glob("video_*.mp4"):  # New format: video_YYYYMMDD_HHMMSS_timestamp.mp4
                    media_files.append(video_file)
                for video_file in folder_path.glob("video_*.avi"):  # New format: video_YYYYMMDD_HHMMSS_timestamp.avi
                    media_files.append(video_file)
                
                # Legacy formats
                for recording_file in folder_path.glob("recording_*.mp4"):  # Legacy
                    media_files.append(recording_file)
                for recording_file in folder_path.glob("recording_*.avi"):  # Legacy
                    media_files.append(recording_file)
                
                # Sort files by modification time (newest first)
                media_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for file_path in media_files:
                    item = {
                        'path': file_path,
                        'name': file_path.name,
                        'type': 'image' if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png'] else 'video',
                        'size': file_path.stat().st_size,
                        'mtime': file_path.stat().st_mtime
                    }
                    self.gallery_items.append(item)
        
        if self.current_folder:
            print(f"📁 Folder loaded: {len(self.gallery_items)} items in {self.current_folder}")
        else:
            print(f"📁 Gallery loaded: {len(self.gallery_folders)} folders")
    
    def generate_thumbnail(self, file_path, size=(80, 80)):
        """Generate thumbnail for image or video file"""
        try:
            cache_key = f"{file_path}_{size[0]}x{size[1]}"
            
            if cache_key in self.thumbnail_cache:
                return self.thumbnail_cache[cache_key]
            
            # Limit cache size to prevent memory issues
            if len(self.thumbnail_cache) > 100:
                # Remove oldest entries
                old_keys = list(self.thumbnail_cache.keys())[:20]
                for key in old_keys:
                    del self.thumbnail_cache[key]
        
        except Exception as e:
            print(f"❌ Error managing thumbnail cache: {e}")
            return None
        
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Image thumbnail
                img = cv2.imread(str(file_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    
                    # Calculate aspect ratio preserving dimensions
                    aspect = w / h
                    if aspect > 1:
                        new_w, new_h = size[0], int(size[0] / aspect)
                    else:
                        new_w, new_h = int(size[1] * aspect), size[1]
                    
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                    
                    # Create centered thumbnail with black background
                    thumbnail = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    y_offset = (size[1] - new_h) // 2
                    x_offset = (size[0] - new_w) // 2
                    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
                    
                    # Convert to pygame surface with safety checks
                    if thumbnail is not None and thumbnail.size > 0:
                        thumbnail_rotated = np.rot90(thumbnail)
                        thumbnail_flipped = np.flipud(thumbnail_rotated)
                        
                        # Ensure array is contiguous and proper format
                        if not thumbnail_flipped.flags['C_CONTIGUOUS']:
                            thumbnail_flipped = np.ascontiguousarray(thumbnail_flipped)
                        
                        surface = pygame.surfarray.make_surface(thumbnail_flipped)
                    else:
                        return None
                    
                    self.thumbnail_cache[cache_key] = surface
                    return surface
            
            elif file_path.suffix.lower() in ['.mp4', '.avi']:
                # Video thumbnail - get first frame
                cap = cv2.VideoCapture(str(file_path))
                try:
                    if not cap.isOpened():
                        print(f"❌ Cannot open video for thumbnail: {file_path}")
                        return None
                    
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"❌ Cannot read frame for thumbnail: {file_path}")
                        return None
                finally:
                    cap.release()
                
                if ret and frame is not None:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    
                    # Calculate aspect ratio preserving dimensions
                    aspect = w / h
                    if aspect > 1:
                        new_w, new_h = size[0], int(size[0] / aspect)
                    else:
                        new_w, new_h = int(size[1] * aspect), size[1]
                    
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                    
                    # Create centered thumbnail with black background
                    thumbnail = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    y_offset = (size[1] - new_h) // 2
                    x_offset = (size[0] - new_w) // 2
                    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
                    
                    # Convert to pygame surface with safety checks
                    if thumbnail is not None and thumbnail.size > 0:
                        thumbnail_rotated = np.rot90(thumbnail)
                        thumbnail_flipped = np.flipud(thumbnail_rotated)
                        
                        # Ensure array is contiguous and proper format
                        if not thumbnail_flipped.flags['C_CONTIGUOUS']:
                            thumbnail_flipped = np.ascontiguousarray(thumbnail_flipped)
                        
                        surface = pygame.surfarray.make_surface(thumbnail_flipped)
                    else:
                        return None
                    
                    self.thumbnail_cache[cache_key] = surface
                    return surface
                    
        except Exception as e:
            print(f"❌ Error generating thumbnail for {file_path}: {e}")
        
        # Return default thumbnail on error
        return self.create_default_thumbnail(size)
    
    def create_default_thumbnail(self, size=(80, 80)):
        """Create a default thumbnail surface"""
        surface = pygame.Surface(size)
        surface.fill((60, 60, 60))
        pygame.draw.rect(surface, (100, 100, 100), (0, 0, size[0], size[1]), 2)
        return surface
    
    def render_media_viewer(self):
        """Render full media viewer for selected image or video"""
        if not self.selected_media:
            return
        
        # Overlay background
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Media viewer area
        viewer_margin = 50
        viewer_w = self.screen_width - 2 * viewer_margin
        viewer_h = self.screen_height - 2 * viewer_margin
        viewer_x = viewer_margin
        viewer_y = viewer_margin
        
        # Background
        pygame.draw.rect(self.screen, self.panel_bg, (viewer_x, viewer_y, viewer_w, viewer_h))
        pygame.draw.rect(self.screen, self.text_color, (viewer_x, viewer_y, viewer_w, viewer_h), 2)
        
        # Title bar
        title_h = 40
        pygame.draw.rect(self.screen, (50, 50, 50), (viewer_x, viewer_y, viewer_w, title_h))
        self.draw_text(f"📄 {self.selected_media['name']}", viewer_x + 10, viewer_y + 10, self.text_color)
        
        # Close button
        close_x = viewer_x + viewer_w - 80
        close_y = viewer_y + 5
        if self.draw_button(close_x, close_y, 70, 30, "✕ Close", style='error'):
            self.selected_media = None
            return
        
        # Media display area
        media_y = viewer_y + title_h
        media_h = viewer_h - title_h - 80
        media_w = viewer_w
        
        try:
            file_path = self.selected_media['path']
            
            if self.selected_media['type'] == 'image':
                # Display image
                img = cv2.imread(str(file_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    
                    # Scale to fit viewer area
                    scale_x = media_w / w
                    scale_y = media_h / h
                    scale = min(scale_x, scale_y, 1.0)  # Don't upscale
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Center the image
                    img_x = viewer_x + (media_w - new_w) // 2
                    img_y = media_y + (media_h - new_h) // 2
                    
                    # Resize and convert to pygame surface
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                    img_rotated = np.rot90(img_resized)
                    img_flipped = np.flipud(img_rotated)
                    img_surface = pygame.surfarray.make_surface(img_flipped)
                    
                    self.screen.blit(img_surface, (img_x, img_y))
                    
            elif self.selected_media['type'] == 'video':
                # Display video info and first frame as preview
                cap = cv2.VideoCapture(str(file_path))
                try:
                    if not cap.isOpened():
                        print(f"❌ Cannot open video for preview: {file_path}")
                        return
                    
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"❌ Cannot read frame for preview: {file_path}")
                        return
                finally:
                    cap.release()
                
                if ret and frame is not None:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    
                    # Scale to fit viewer area
                    scale_x = media_w / w
                    scale_y = media_h / h
                    scale = min(scale_x, scale_y, 1.0)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Center the frame
                    img_x = viewer_x + (media_w - new_w) // 2
                    img_y = media_y + (media_h - new_h) // 2
                    
                    # Resize and convert to pygame surface
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                    img_rotated = np.rot90(img_resized)
                    img_flipped = np.flipud(img_rotated)
                    img_surface = pygame.surfarray.make_surface(img_flipped)
                    
                    self.screen.blit(img_surface, (img_x, img_y))
                    
                    # Video play icon overlay
                    icon_size = 60
                    icon_x = img_x + (new_w - icon_size) // 2
                    icon_y = img_y + (new_h - icon_size) // 2
                    
                    # Play button background
                    play_surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
                    pygame.draw.circle(play_surface, (0, 0, 0, 120), (icon_size//2, icon_size//2), icon_size//2)
                    pygame.draw.circle(play_surface, (255, 255, 255, 200), (icon_size//2, icon_size//2), icon_size//2, 3)
                    
                    # Play triangle
                    triangle_points = [
                        (icon_size//2 - 10, icon_size//2 - 15),
                        (icon_size//2 - 10, icon_size//2 + 15),
                        (icon_size//2 + 15, icon_size//2)
                    ]
                    pygame.draw.polygon(play_surface, (255, 255, 255), triangle_points)
                    
                    self.screen.blit(play_surface, (icon_x, icon_y))
                
        except Exception as e:
            # Error display
            error_text = f"Error loading media: {e}"
            self.draw_text(error_text, viewer_x + 10, media_y + 10, self.red)
        
        # Info panel at bottom
        info_y = viewer_y + viewer_h - 70
        info_h = 60
        pygame.draw.rect(self.screen, (40, 40, 40), (viewer_x, info_y, viewer_w, info_h))
        
        # File info
        size_mb = self.selected_media['size'] / (1024 * 1024)
        info_text = f"📄 Size: {size_mb:.1f}MB  |  📅 Modified: {datetime.fromtimestamp(self.selected_media['mtime']).strftime('%Y-%m-%d %H:%M:%S')}"
        self.draw_text(info_text, viewer_x + 10, info_y + 10, (200, 200, 200))
        
        # Action buttons
        button_y = info_y + 30
        if self.selected_media['type'] == 'video':
            if self.draw_button(viewer_x + 10, button_y, 120, 25, "🎬 Play in Player", style='normal'):
                # Try to open with custom detection video player
                try:
                    if self.selected_media and self.selected_media.get('path'):
                        import subprocess
                        player_script = Path(__file__).parent / "video_player.py"
                        video_path = str(self.selected_media['path'])
                        
                        if Path(video_path).exists():
                            subprocess.Popen(['python3', str(player_script), video_path], start_new_session=True)
                            print(f"🎬 Opening video in detection player: {self.selected_media.get('name', 'unknown')}")
                        else:
                            print(f"❌ Video file not found: {video_path}")
                    else:
                        print("❌ No media selected or invalid path")
                        
                except Exception as e:
                    print(f"❌ Error opening video player: {e}")
                    # Fallback to system player
                    try:
                        if self.selected_media and self.selected_media.get('path'):
                            subprocess.Popen(['xdg-open', str(self.selected_media['path'])], start_new_session=True)
                    except Exception as e2:
                        print(f"❌ Error opening with system player: {e2}")
            
            # Check if this is a recording with detection data
            file_path = self.selected_media['path']
            
            # Handle both new and legacy detection file naming
            file_stem = str(file_path).replace('.mp4', '').replace('.avi', '')
            detection_file = f"{file_stem}_detections.json"
            if Path(detection_file).exists():
                if self.draw_button(viewer_x + 140, button_y, 130, 25, "📊 View Detections", style='success'):
                    try:
                        import subprocess
                        # Open detection JSON file with default editor
                        subprocess.Popen(['xdg-open', detection_file], start_new_session=True)
                    except Exception as e:
                        print(f"❌ Error opening detections: {e}")
                
                # Show detection summary
                try:
                    with open(detection_file, 'r') as f:
                        detection_data = json.load(f)
                    
                    total_detections = len(detection_data.get('detections', []))
                    duration = detection_data.get('video_info', {}).get('duration_seconds', 0)
                    
                    summary_text = f"📊 Detections: {total_detections} | Duration: {duration:.1f}s"
                    self.draw_text(summary_text, viewer_x + 280, button_y + 5, self.yellow)
                except:
                    pass
        
        # Delete button (with confirmation needed)
        if self.draw_button(viewer_x + viewer_w - 120, button_y, 110, 25, "🗑️ Delete File", style='error'):
            print(f"⚠️ Delete functionality not implemented for safety")
    
    def render_tabs(self):
        """Render tab interface"""
        tab_x = 10
        tab_y = 10
        tab_width = 100
        
        # Monitor tab
        monitor_style = 'success' if self.current_tab == 'monitor' else 'normal'
        if self.draw_button(tab_x, tab_y, tab_width, self.tab_height, "📹 Monitor", style=monitor_style):
            self.current_tab = 'monitor'
        
        # Gallery tab
        gallery_style = 'success' if self.current_tab == 'gallery' else 'normal'  
        if self.draw_button(tab_x + tab_width + 5, tab_y, tab_width, self.tab_height, "🖼️ Gallery", style=gallery_style):
            self.current_tab = 'gallery'
            self.refresh_gallery()  # Refresh when switching to gallery
    
    def render_gallery(self):
        """Render gallery view with folder browsing and thumbnail grid"""
        # Gallery area
        gallery_x = 10
        gallery_y = 60  # Below tabs
        gallery_w = self.screen_width - 20
        gallery_h = self.screen_height - gallery_y - 10
        
        # Background
        pygame.draw.rect(self.screen, self.panel_bg, (gallery_x, gallery_y, gallery_w, gallery_h))
        pygame.draw.rect(self.screen, self.text_color, (gallery_x, gallery_y, gallery_w, gallery_h), 1)
        
        # Navigation bar
        nav_h = 40
        pygame.draw.rect(self.screen, (50, 50, 50), (gallery_x, gallery_y, gallery_w, nav_h))
        
        nav_y = gallery_y + 10
        
        # Back button (if in folder view)
        if self.current_folder:
            if self.draw_button(gallery_x + 10, nav_y, 80, 25, "← Back", style='normal'):
                self.current_folder = None
                self.gallery_scroll = 0
                self.refresh_gallery()
            
            # Current folder name
            folder_name = Path(self.current_folder).name
            self.draw_text(f"📁 {folder_name} ({len(self.gallery_items)} items)", 
                         gallery_x + 100, nav_y + 3, self.text_color)
        else:
            # Folder list title
            self.draw_text(f"📂 Folders ({len(self.gallery_folders)} folders)", 
                         gallery_x + 10, nav_y + 3, self.text_color)
        
        # Content area
        content_y = gallery_y + nav_h
        content_h = gallery_h - nav_h
        
        if self.current_folder is None:
            # Show folder list
            if len(self.gallery_folders) == 0:
                no_folders_text = "No recordings found. Start detection and take snapshots or record videos!"
                self.draw_text(no_folders_text, gallery_x + 10, content_y + 40, (150, 150, 150))
                return
            
            # Folder list
            y = content_y + 10
            visible_folders = 0
            max_visible = int(content_h / 35)
            
            for i, folder in enumerate(self.gallery_folders):
                if i < self.gallery_scroll:
                    continue
                
                if visible_folders >= max_visible:
                    break
                
                # Folder item background
                item_bg = (45, 45, 45) if i % 2 == 0 else (50, 50, 50)
                folder_rect = (gallery_x + 10, y, gallery_w - 20, 30)
                
                # Check if clicked
                mouse_x, mouse_y = self.mouse_pos
                hovered = (gallery_x + 10 <= mouse_x <= gallery_x + gallery_w - 10 and 
                          y <= mouse_y <= y + 30)
                
                if hovered:
                    item_bg = (70, 70, 70)
                
                pygame.draw.rect(self.screen, item_bg, folder_rect)
                
                # Folder icon and name
                self.draw_text("📁", gallery_x + 20, y + 5, self.text_color)
                self.draw_text(folder['name'], gallery_x + 50, y + 5, self.text_color)
                
                # File count
                count_text = f"{folder['count']} files"
                self.draw_text(count_text, gallery_x + gallery_w - 120, y + 5, (150, 150, 150))
                
                # Click to open folder
                if hovered and self.mouse_clicked:
                    self.current_folder = str(folder['path'])
                    self.gallery_scroll = 0
                    self.refresh_gallery()
                
                y += 35
                visible_folders += 1
                
        else:
            # Show media thumbnails in current folder
            if len(self.gallery_items) == 0:
                no_items_text = "No media files in this folder."
                self.draw_text(no_items_text, gallery_x + 10, content_y + 40, (150, 150, 150))
                return
            
            # Calculate thumbnail grid
            thumbnail_size = 100
            thumbnail_spacing = 10
            name_height = 20
            item_height = thumbnail_size + name_height + thumbnail_spacing
            
            cols = max(1, (gallery_w - 20) // (thumbnail_size + thumbnail_spacing))
            rows_visible = max(1, (content_h - 20) // item_height)
            
            # Thumbnail grid
            start_index = self.gallery_scroll * cols
            
            for i in range(rows_visible * cols):
                item_index = start_index + i
                
                if item_index >= len(self.gallery_items):
                    break
                
                item = self.gallery_items[item_index]
                
                # Calculate position
                row = i // cols
                col = i % cols
                
                x = gallery_x + 10 + col * (thumbnail_size + thumbnail_spacing)
                y = content_y + 10 + row * item_height
                
                # Thumbnail background
                thumb_rect = (x, y, thumbnail_size, thumbnail_size)
                pygame.draw.rect(self.screen, (60, 60, 60), thumb_rect)
                pygame.draw.rect(self.screen, (100, 100, 100), thumb_rect, 1)
                
                # Generate and display thumbnail
                try:
                    if item and item.get('path'):
                        thumbnail = self.generate_thumbnail(item['path'], (thumbnail_size-4, thumbnail_size-4))
                        if thumbnail:
                            thumb_x = x + 2
                            thumb_y = y + 2
                            self.screen.blit(thumbnail, (thumb_x, thumb_y))
                        else:
                            raise Exception("Thumbnail generation failed")
                    else:
                        raise Exception("Invalid item or path")
                        
                except Exception as e:
                    # Fallback icon
                    try:
                        icon = "🎥" if item.get('type') == 'video' else "📸"
                        icon_surface = self.font_large.render(icon, True, self.text_color)
                        icon_rect = icon_surface.get_rect(center=(x + thumbnail_size//2, y + thumbnail_size//2))
                        self.screen.blit(icon_surface, icon_rect)
                    except:
                        # Ultimate fallback - draw a rectangle
                        pygame.draw.rect(self.screen, (100, 100, 100), (x+10, y+10, thumbnail_size-20, thumbnail_size-20))
                
                # File name (truncated)
                name_y = y + thumbnail_size + 2
                name_text = item['name']
                if len(name_text) > 15:
                    name_text = name_text[:12] + "..."
                
                name_surface = pygame.font.Font(None, 16).render(name_text, True, self.text_color)
                name_rect = name_surface.get_rect(center=(x + thumbnail_size//2, name_y + 8))
                self.screen.blit(name_surface, name_rect)
                
                # Click detection
                mouse_x, mouse_y = self.mouse_pos
                if (x <= mouse_x <= x + thumbnail_size and 
                    y <= mouse_y <= y + thumbnail_size + name_height):
                    
                    # Highlight on hover
                    pygame.draw.rect(self.screen, (100, 150, 200), thumb_rect, 2)
                    
                    # Click to view
                    if self.mouse_clicked:
                        self.selected_media = item
        
        # Scroll indicators
        if self.current_folder is None:
            max_scroll = max(0, len(self.gallery_folders) - max_visible)
        else:
            # Recalculate grid parameters for scroll indicators
            thumbnail_size = 100
            thumbnail_spacing = 10
            items_per_row = max(1, (gallery_w - 20) // (thumbnail_size + thumbnail_spacing))
            item_height = thumbnail_size + 20 + thumbnail_spacing
            rows_visible = (content_h - 20) // item_height
            total_rows = (len(self.gallery_items) + items_per_row - 1) // items_per_row
            max_scroll = max(0, total_rows - rows_visible)
        
        if self.gallery_scroll > 0:
            self.draw_text("↑ Scroll up (Arrow keys)", gallery_x + 10, content_y - 15, self.yellow)
        
        if self.gallery_scroll < max_scroll:
            self.draw_text("↓ More items below (Arrow keys)", gallery_x + 10, gallery_y + gallery_h - 25, self.yellow)
    
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
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
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
        frame_number = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                
                # Validate frame data
                if frame.size == 0:
                    continue
                
                # Share original frame for recording (minimal overhead)
                if self.is_recording:
                    with self.recording_frame_lock:
                        self.shared_recording_frame = frame.copy()
                
                # Detection
                detections, inf_time = self.detector.detect_objects(frame)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Log detection data if recording (minimal impact)
                if self.is_recording and detections:
                    current_time = time.time()
                    relative_time = current_time - self.recording_start_time
                    
                    for detection in detections:
                        try:
                            # Handle different detection formats
                            if isinstance(detection, dict):
                                # Dictionary format
                                if 'bbox' in detection:
                                    x1, y1, x2, y2 = map(int, detection['bbox'])
                                else:
                                    # Try direct coordinates
                                    x1, y1, x2, y2 = int(detection.get('x1', 0)), int(detection.get('y1', 0)), int(detection.get('x2', 0)), int(detection.get('y2', 0))
                                
                                obj_class = detection.get('class', detection.get('name', 'unknown'))
                                confidence = float(detection.get('confidence', detection.get('conf', 0.0)))
                            else:
                                # List/tuple format [x1, y1, x2, y2, conf, class_id]
                                if len(detection) >= 4:
                                    x1, y1, x2, y2 = map(int, detection[:4])
                                    confidence = float(detection[4]) if len(detection) > 4 else 0.0
                                    obj_class = str(detection[5]) if len(detection) > 5 else 'unknown'
                                else:
                                    continue  # Skip invalid detections
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            detection_entry = {
                                'timestamp': relative_time,
                                'frame_number': frame_number,
                                'object_type': obj_class,
                                'confidence': confidence,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': width,
                                'height': height
                            }
                            self.detection_data.append(detection_entry)
                        except Exception as e:
                            print(f"⚠️ Error processing detection: {e}")
                            continue
                
                # Draw detections for display (not for recording)
                if self.show_detections and detections:
                    display_frame = self.detector.draw_detections(frame.copy(), detections)
                else:
                    display_frame = frame  # Use original frame if no detections to show
                
                # Add minimal GUI info to display frame only
                if self.show_gui_info:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Objects: {len(detections)}", (10, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    if self.is_recording:
                        cv2.putText(display_frame, f"REC", (10, 65), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Update (minimal locking) - separate display and recording
                with self.frame_lock:
                    self.current_frame = display_frame  # Display frame with detections
                    self.fps = fps
                    self.objects = len(detections)
                    self.total_objects += len(detections)
                
                # No video frame handling in main loop - recording is completely independent
                frame_number += 1
                
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
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
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
        """Minimal control panel - only shown in monitor tab"""
        panel_x = 10
        panel_y = 60  # Below tabs
        panel_w = 280
        # Dynamic height based on content
        base_height = 350
        url_height = 60 if (self.enable_rtsp and self.detector and self.detector.enable_rtsp) else 0
        recording_height = 35 if self.is_running else 0  # Add space for recording buttons
        panel_h = base_height + url_height + recording_height
        
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
        
        # Recording controls (only show when running)
        if self.is_running:
            # Video recording toggle button
            if not self.is_recording:
                if self.draw_button(x, y, 95, 25, "🎥 Record", style='normal'):
                    self.start_recording()
            else:
                recording_duration = time.time() - self.recording_start_time if self.recording_start_time else 0
                button_text = f"⏹️ {recording_duration:.0f}s"
                if self.draw_button(x, y, 95, 25, button_text, style='error'):
                    self.stop_recording()
            
            # Snapshot button with feedback
            snap_button_text = "📸 Snap"
            snap_style = 'normal'
            
            # Show different text/style when feedback is active
            if self.show_snapshot_feedback and time.time() - self.snapshot_feedback_timer < 0.5:
                snap_button_text = "✅ Saved!"
                snap_style = 'success'
            
            if self.draw_button(x + 105, y, 95, 25, snap_button_text, style=snap_style):
                self.take_snapshot()
            y += 35
        
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
        y += 15
        
        # Independent recording status
        if self.is_recording:
            logged_count = len(self.detection_data)
            self.draw_text(f"Logged: {logged_count}", x, y, self.green)
            y += 15
            
            # Independent recording frame count
            recorded_frames = getattr(self, 'recording_frame_count', 0)
            self.draw_text(f"Recorded: {recorded_frames}", x, y, self.green)
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
        
        # Snapshot feedback (in control panel)
        if self.show_snapshot_feedback:
            if time.time() - self.snapshot_feedback_timer < 2.0:  # Show for 2 seconds
                feedback_text = "📸 Snapshot Saved!"
                feedback_surface = self.font.render(feedback_text, True, self.green)
                feedback_rect = feedback_surface.get_rect(center=(x + 100, y + 10))
                self.screen.blit(feedback_surface, feedback_rect)
            else:
                self.show_snapshot_feedback = False
    
    def render_video(self):
        """Fast video rendering with large display"""
        # Video area - most of the screen
        video_x = 300  # Leave space for control panel
        video_y = 60  # Below tabs
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
        
        # Snapshot feedback overlay on video
        if self.show_snapshot_feedback:
            if time.time() - self.snapshot_feedback_timer < 2.0:
                # Large, prominent feedback on video area
                feedback_text = "📸 SNAPSHOT SAVED!"
                feedback_surface = self.font_large.render(feedback_text, True, self.green)
                feedback_bg = pygame.Surface((feedback_surface.get_width() + 20, feedback_surface.get_height() + 10))
                feedback_bg.fill((0, 0, 0))
                feedback_bg.set_alpha(180)
                
                # Position at top of video area
                feedback_x = center_x + (new_w - feedback_surface.get_width()) // 2
                feedback_y = center_y + 20
                
                # Draw background and text
                self.screen.blit(feedback_bg, (feedback_x - 10, feedback_y - 5))
                self.screen.blit(feedback_surface, (feedback_x, feedback_y))
    
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
                elif event.type == pygame.VIDEORESIZE:
                    if not self.is_fullscreen:
                        self.screen_width = event.w
                        self.screen_height = event.h
                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                        print(f"🪟 Window resized to: {self.screen_width}x{self.screen_height}")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.is_initializing:
                            continue  # Don't allow exit during initialization
                        elif self.selected_media:
                            self.selected_media = None  # Close media viewer
                        elif self.current_folder and self.current_tab == 'gallery':
                            self.current_folder = None  # Go back to folder list
                            self.gallery_scroll = 0
                            self.refresh_gallery()
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
                    elif event.key == pygame.K_UP:
                        # Scroll up in gallery
                        if self.current_tab == 'gallery' and self.gallery_scroll > 0:
                            self.gallery_scroll -= 1
                    elif event.key == pygame.K_DOWN:
                        # Scroll down in gallery
                        if self.current_tab == 'gallery':
                            # Calculate proper max scroll based on current view
                            if self.current_folder is None:
                                # Folder view
                                content_h = self.screen_height - 60 - 40  # Screen minus tabs minus nav
                                max_visible = int(content_h / 35)
                                max_scroll = max(0, len(self.gallery_folders) - max_visible)
                            else:
                                # Thumbnail view
                                gallery_w = self.screen_width - 20
                                thumbnail_size = 100
                                thumbnail_spacing = 10
                                item_height = thumbnail_size + 20 + thumbnail_spacing
                                
                                cols = max(1, (gallery_w - 20) // (thumbnail_size + thumbnail_spacing))
                                content_h = self.screen_height - 60 - 40
                                rows_visible = (content_h - 20) // item_height
                                total_rows = (len(self.gallery_items) + cols - 1) // cols
                                max_scroll = max(0, total_rows - rows_visible)
                            
                            if self.gallery_scroll < max_scroll:
                                self.gallery_scroll += 1
            
            # Clear screen
            self.screen.fill(self.bg_dark)
            
            if self.is_initializing:
                # Show loading screen
                self.render_loading_screen()
            else:
                # Normal UI
                # Always render tabs
                self.render_tabs()
                
                if self.current_tab == 'monitor':
                    # Monitor tab - show detection interface
                    # Update video surface (throttled to 15 FPS for GUI)
                    if current_time - last_video_update > 0.066:  # ~15 FPS GUI updates
                        self.update_video_surface()
                        last_video_update = current_time
                    
                    # Render UI (fast)
                    self.render_controls()
                    self.render_video()
                    
                elif self.current_tab == 'gallery':
                    # Gallery tab - show media files
                    self.render_gallery()
            
            # Media viewer overlay (renders on top of everything)
            if self.selected_media:
                self.render_media_viewer()
            
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