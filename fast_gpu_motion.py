#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import time

class FastGPUMotionDetector:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Motion detection parameters (optimized for speed)
        self.motion_threshold = 25    # Minimum difference to consider as motion
        self.min_area = 800          # Minimum area for motion blob
        self.learning_rate = 0.02    # Background learning rate
        self.merge_distance = 50     # Distance to merge rectangles (pixels)
        
        # GPU tensors for motion detection
        self.background_model = None
        self.frame_counter = 0
        
        print(f"🚀 Fast GPU Motion Detection on {self.device}")
        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def preprocess_frame_for_motion(self, frame):
        """Convert frame to GPU tensor for motion detection"""
        # Convert to grayscale and apply light blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)  # Light blur for speed
        
        # Convert to GPU tensor
        frame_tensor = torch.from_numpy(blurred.astype(np.float32)).to(self.device)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return frame_tensor
    
    def detect_motion_gpu(self, current_frame_tensor):
        """Ultra-fast GPU motion detection"""
        self.frame_counter += 1
        
        if self.background_model is None:
            # Initialize background model with first frame
            self.background_model = current_frame_tensor.clone()
            empty_mask = np.zeros((current_frame_tensor.shape[2], current_frame_tensor.shape[3]), dtype=np.uint8)
            return [], empty_mask
        
        # Simple background subtraction
        diff = torch.abs(current_frame_tensor - self.background_model)
        motion_mask = (diff > self.motion_threshold).float()
        
        # Convert to CPU only once
        motion_mask_cpu = motion_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Find contours for motion areas
        contours, _ = cv2.findContours(motion_mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h, area))
        
        # Update background model (running average)
        self.background_model = (1 - self.learning_rate) * self.background_model + \
                               self.learning_rate * current_frame_tensor
        
        # Merge overlapping/nearby motion areas
        merged_areas = self.merge_motion_areas(motion_areas)
        
        # Remove contained rectangles (smaller ones inside larger ones)
        filtered_areas = self.remove_contained_rectangles(merged_areas)
        
        return filtered_areas, motion_mask_cpu
    
    def merge_motion_areas(self, motion_areas):
        """Merge nearby motion rectangles into single rectangles per object"""
        if len(motion_areas) <= 1:
            return motion_areas
        
        # Convert to format: [x1, y1, x2, y2, area]
        rectangles = []
        for x, y, w, h, area in motion_areas:
            rectangles.append([x, y, x + w, y + h, area])
        
        merged = []
        used = [False] * len(rectangles)
        
        for i, rect1 in enumerate(rectangles):
            if used[i]:
                continue
            
            # Start with current rectangle
            merged_rect = rect1.copy()
            merged_area = rect1[4]
            used[i] = True
            
            # Find overlapping or nearby rectangles
            for j, rect2 in enumerate(rectangles):
                if used[j] or i == j:
                    continue
                
                # Check if rectangles should be merged
                if self.should_merge_rectangles(merged_rect, rect2):
                    # Merge rectangles by taking bounding box
                    merged_rect[0] = min(merged_rect[0], rect2[0])  # min x1
                    merged_rect[1] = min(merged_rect[1], rect2[1])  # min y1
                    merged_rect[2] = max(merged_rect[2], rect2[2])  # max x2
                    merged_rect[3] = max(merged_rect[3], rect2[3])  # max y2
                    merged_area += rect2[4]
                    used[j] = True
            
            # Convert back to (x, y, w, h, area) format
            x = merged_rect[0]
            y = merged_rect[1]
            w = merged_rect[2] - merged_rect[0]
            h = merged_rect[3] - merged_rect[1]
            merged.append((x, y, w, h, merged_area))
        
        return merged
    
    def should_merge_rectangles(self, rect1, rect2):
        """Check if two rectangles should be merged"""
        # rect format: [x1, y1, x2, y2, area]
        
        # Check if rectangles overlap or are close to each other
        x1_1, y1_1, x2_1, y2_1 = rect1[:4]
        x1_2, y1_2, x2_2, y2_2 = rect2[:4]
        
        # Expand rectangles by merge_distance
        expanded_rect1 = [
            x1_1 - self.merge_distance, y1_1 - self.merge_distance,
            x2_1 + self.merge_distance, y2_1 + self.merge_distance
        ]
        
        # Check if expanded rect1 overlaps with rect2
        return not (expanded_rect1[2] < x1_2 or  # rect1 is left of rect2
                   expanded_rect1[0] > x2_2 or   # rect1 is right of rect2
                   expanded_rect1[3] < y1_2 or   # rect1 is above rect2
                   expanded_rect1[1] > y2_2)     # rect1 is below rect2
    
    def remove_contained_rectangles(self, motion_areas):
        """Remove rectangles that are completely contained within larger ones"""
        if len(motion_areas) <= 1:
            return motion_areas
        
        # Sort by area (largest first)
        sorted_areas = sorted(motion_areas, key=lambda x: x[4], reverse=True)
        
        filtered = []
        
        for i, (x1, y1, w1, h1, area1) in enumerate(sorted_areas):
            is_contained = False
            
            # Check if this rectangle is contained in any larger rectangle
            for j in range(i):  # Only check against larger rectangles
                x2, y2, w2, h2, area2 = sorted_areas[j]
                
                # Check if rect1 is completely inside rect2
                if (x2 <= x1 and y2 <= y1 and 
                    x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1):
                    is_contained = True
                    break
            
            # Only keep rectangles that are not contained in others
            if not is_contained:
                filtered.append((x1, y1, w1, h1, area1))
        
        return filtered
    
    def draw_motion_areas(self, frame, motion_areas, show_overlay=False):
        """Draw motion rectangles on original video"""
        result_frame = frame.copy()
        
        # Draw motion areas as red rectangles on original video
        for i, (x, y, w, h, area) in enumerate(motion_areas):
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_frame, f"Motion {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result_frame
    
    def run_motion_detection(self, video_source=0, show_video=True, show_overlay=True):
        """Run ultra-fast GPU motion detection"""
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
        
        # Get actual properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📹 Camera: {w}x{h} @ {fps_cam} FPS")
        print(f"🏃 Motion detection only (no object recognition)")
        print(f"🎨 Motion overlay: {'ON' if show_overlay else 'OFF'}")
        if show_video:
            print("🎮 Press 'q' to quit")
        else:
            print("🎮 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Performance tracking
        frame_count = 0
        total_motion_time = 0
        total_motion_areas = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # GPU motion detection
                motion_start = time.time()
                frame_tensor = self.preprocess_frame_for_motion(frame)
                motion_areas, motion_mask = self.detect_motion_gpu(frame_tensor)
                motion_time = time.time() - motion_start
                total_motion_time += motion_time
                total_motion_areas += len(motion_areas)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                motion_fps = 1.0 / motion_time if motion_time > 0 else 0
                
                if show_video:
                    # Draw motion rectangles on original video (clean view)
                    result_frame = self.draw_motion_areas(frame, motion_areas, show_overlay)
                    
                    # Add performance info
                    info_y = 30
                    cv2.putText(result_frame, f"Motion FPS: {motion_fps:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Real FPS: {fps_actual:.1f}", 
                               (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Motion Areas: {len(motion_areas)}", 
                               (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display
                    cv2.imshow('Fast GPU Motion Detection', result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Terminal output
                    if motion_areas:
                        areas_info = [f"Area{i+1}({int(area)}px)" for i, (_, _, _, _, area) in enumerate(motion_areas)]
                        print(f"Frame {frame_count:3d} [Motion FPS: {motion_fps:4.1f}, Real FPS: {fps_actual:4.1f}]: " + 
                              f"{len(motion_areas)} areas - {', '.join(areas_info)}")
                    
                    # Periodic stats
                    if frame_count % 30 == 0:
                        avg_motion_fps = frame_count / total_motion_time if total_motion_time > 0 else 0
                        print(f"📊 Stats: {frame_count} frames, {total_motion_areas} motion areas, " + 
                              f"{avg_motion_fps:.1f} avg motion FPS")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
            
            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_motion_fps = frame_count / total_motion_time if total_motion_time > 0 else 0
            
            print(f"\n✅ Final Results:")
            print(f"   - Total frames: {frame_count}")
            print(f"   - Total motion areas detected: {total_motion_areas}")
            print(f"   - Average overall FPS: {avg_fps:.1f}")
            print(f"   - Average motion detection FPS: {avg_motion_fps:.1f}")
            print(f"   - Performance: {avg_motion_fps/avg_fps:.1f}x faster than real-time")

def main():
    parser = argparse.ArgumentParser(description='Fast GPU Motion Detection for Jetson Nano')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or file)')
    parser.add_argument('--motion-threshold', type=int, default=25, help='Motion detection threshold')
    parser.add_argument('--min-area', type=int, default=800, help='Minimum motion area')
    parser.add_argument('--merge-distance', type=int, default=50, help='Distance to merge rectangles (pixels)')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--no-display', action='store_true', help='Run in terminal mode')
    parser.add_argument('--no-overlay', action='store_true', help='Disable motion overlay for more speed')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("🚀 Fast GPU Motion Detection for Jetson Nano")
    print(f"⚡ Device: {args.device.upper()}")
    print(f"🎯 Motion Threshold: {args.motion_threshold}")
    print(f"📏 Minimum Area: {args.min_area}")
    if args.device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    try:
        detector = FastGPUMotionDetector(device=args.device)
        
        # Set motion detection parameters
        detector.motion_threshold = args.motion_threshold
        detector.min_area = args.min_area
        detector.merge_distance = args.merge_distance
        
        detector.run_motion_detection(args.input, not args.no_display, not args.no_overlay)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()