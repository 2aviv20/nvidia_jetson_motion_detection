#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import time
from collections import deque

class GPUMotionDetector:
    def __init__(self, model_name="yolov5s", conf_threshold=0.5, device="cuda"):
        self.conf_threshold = conf_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Target classes: person, car, motorcycle, airplane, bus, truck
        self.target_classes = [0, 2, 3, 4, 5, 7]
        self.class_names = {
            0: "Human", 2: "Car", 3: "Motorcycle", 
            4: "Aircraft", 5: "Bus", 7: "Truck"
        }
        
        # Colors (BGR)
        self.colors = {
            0: (0, 255, 0),    # Green for humans
            2: (255, 0, 0),    # Blue for cars
            3: (0, 255, 255),  # Yellow for motorcycles
            4: (255, 0, 255),  # Magenta for aircraft
            5: (255, 255, 0),  # Cyan for buses
            7: (128, 0, 128)   # Purple for trucks
        }
        
        # Motion detection parameters (optimized for speed)
        self.motion_threshold = 30    # Higher threshold for speed
        self.min_area = 1000         # Larger min area to reduce small detections
        self.background_history = 100 # Shorter history for faster adaptation
        self.learning_rate = 0.05    # Faster learning rate
        self.skip_motion_frames = 2  # Process motion every 3rd frame
        
        # GPU tensors for motion detection
        self.background_model = None
        self.frame_buffer = deque(maxlen=3)  # Store last 3 frames for temporal analysis
        self.motion_frame_counter = 0
        self.last_motion_areas = []  # Cache last motion areas
        
        print(f"🚀 Initializing GPU Motion Detection on {self.device}")
        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.load_model(model_name)
        self.init_motion_detection()
    
    def load_model(self, model_name):
        """Load YOLOv5 model with GPU acceleration"""
        print(f"📥 Loading {model_name} model...")
        
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', model_name, 
                                      pretrained=True, verbose=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if self.device == "cuda":
                self.model.half()  # Use FP16 for speed on GPU
                print("✅ Model loaded on GPU with FP16 optimization")
            else:
                print("✅ Model loaded on CPU")
                
            # Configure model
            self.model.conf = self.conf_threshold
            self.model.classes = self.target_classes
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def init_motion_detection(self):
        """Initialize GPU-based motion detection"""
        print("🎯 Initializing GPU motion detection algorithms...")
        
        # Will be initialized with first frame
        self.background_model = None
        self.frame_count = 0
        
        # GPU kernels for morphological operations
        self.morph_kernel = torch.ones((5, 5), device=self.device, dtype=torch.float32)
        
    def preprocess_frame_for_motion(self, frame):
        """Convert frame to GPU tensor for motion detection"""
        # Convert to grayscale and apply blur on CPU (more compatible)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Convert to GPU tensor
        frame_tensor = torch.from_numpy(blurred.astype(np.float32)).to(self.device)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return frame_tensor
    
    def detect_motion_gpu(self, current_frame_tensor):
        """Optimized GPU motion detection with frame skipping"""
        # Skip motion detection on some frames for speed
        self.motion_frame_counter += 1
        
        if self.background_model is None:
            # Initialize background model with first frame
            self.background_model = current_frame_tensor.clone()
            empty_mask = np.zeros((current_frame_tensor.shape[2], current_frame_tensor.shape[3]), dtype=np.uint8)
            return [], empty_mask
        
        # Skip motion detection every N frames, reuse last result
        if self.motion_frame_counter % (self.skip_motion_frames + 1) != 0:
            # Return cached motion areas for skipped frames
            empty_mask = np.zeros((current_frame_tensor.shape[2], current_frame_tensor.shape[3]), dtype=np.uint8)
            return self.last_motion_areas, empty_mask
        
        # Simple and fast motion detection
        diff = torch.abs(current_frame_tensor - self.background_model)
        motion_mask = (diff > self.motion_threshold).float()
        
        # Skip morphological operations for speed - just use raw mask
        motion_mask_cpu = motion_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Find contours (simplified)
        contours, _ = cv2.findContours(motion_mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h, area))
        
        # Cache for skipped frames
        self.last_motion_areas = motion_areas
        
        # Update background model less frequently for speed
        if self.motion_frame_counter % 5 == 0:
            self.background_model = (1 - self.learning_rate) * self.background_model + \
                                   self.learning_rate * current_frame_tensor
        
        return motion_areas, motion_mask_cpu
    
    def detect_objects_in_motion_areas(self, frame, motion_areas):
        """Run object detection only in motion areas for efficiency"""
        detections = []
        
        if not motion_areas:
            return detections
        
        # Extract regions of interest (ROIs) from motion areas
        rois = []
        roi_coords = []
        
        h_orig, w_orig = frame.shape[:2]
        
        for x, y, w, h, area in motion_areas:
            # Expand ROI slightly for better detection
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_orig, x + w + padding)
            y2 = min(h_orig, y + h + padding)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(roi)
                roi_coords.append((x1, y1, x2, y2))
        
        # Run detection on each ROI
        for roi, (x1, y1, x2, y2) in zip(rois, roi_coords):
            try:
                # Convert BGR to RGB for model
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Run inference
                with torch.no_grad():
                    results = self.model(rgb_roi)
                
                # Process results and adjust coordinates to original frame
                if len(results.xyxy[0]) > 0:
                    for detection in results.xyxy[0].cpu().numpy():
                        det_x1, det_y1, det_x2, det_y2, conf, class_id = detection
                        
                        # Adjust coordinates to original frame
                        abs_x1 = int(x1 + det_x1)
                        abs_y1 = int(y1 + det_y1)
                        abs_x2 = int(x1 + det_x2)
                        abs_y2 = int(y1 + det_y2)
                        
                        w = abs_x2 - abs_x1
                        h = abs_y2 - abs_y1
                        
                        detections.append([abs_x1, abs_y1, w, h, float(conf), int(class_id)])
                        
            except Exception as e:
                print(f"⚠️ Error processing ROI: {e}")
                continue
        
        return detections
    
    def draw_motion_and_detections(self, frame, motion_areas, motion_mask, detections):
        """Draw motion areas and object detections"""
        result_frame = frame.copy()
        
        # Draw motion areas in blue
        for x, y, w, h, area in motion_areas:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(result_frame, f"Motion: {int(area)}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw object detections with colors
        for detection in detections:
            x, y, w, h, conf, class_id = detection
            
            color = self.colors.get(class_id, (0, 255, 0))
            name = self.class_names.get(class_id, f"Class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(result_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create motion overlay
        motion_overlay = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
        result_frame = cv2.addWeighted(result_frame, 0.8, motion_overlay, 0.2, 0)
        
        return result_frame
    
    def run_motion_detection(self, video_source=0, show_video=True):
        """Run GPU-accelerated motion detection with object recognition"""
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
        print(f"🎯 Target classes: {list(self.class_names.values())}")
        print(f"🏃 Motion detection + Object recognition")
        if show_video:
            print("🎮 Press 'q' to quit")
        else:
            print("🎮 Press Ctrl+C to stop")
        print("-" * 60)
        
        # Performance tracking
        frame_count = 0
        total_motion_time = 0
        total_detection_time = 0
        total_detections = 0
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
                
                # Object detection in motion areas only
                detection_start = time.time()
                detections = self.detect_objects_in_motion_areas(frame, motion_areas)
                detection_time = time.time() - detection_start
                total_detection_time += detection_time
                total_detections += len(detections)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                motion_fps = 1.0 / motion_time if motion_time > 0 else 0
                detection_fps = 1.0 / detection_time if detection_time > 0 and detections else 0
                
                if show_video:
                    # Draw results
                    result_frame = self.draw_motion_and_detections(frame, motion_areas, motion_mask, detections)
                    
                    # Add performance info
                    info_y = 30
                    cv2.putText(result_frame, f"Motion FPS: {motion_fps:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Detection FPS: {detection_fps:.1f}", 
                               (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Real FPS: {fps_actual:.1f}", 
                               (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Motion Areas: {len(motion_areas)}", 
                               (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Objects: {len(detections)}", 
                               (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display
                    cv2.imshow('GPU Motion Detection + Object Recognition', result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Terminal output
                    if motion_areas or detections:
                        motion_info = f"{len(motion_areas)} motion areas" if motion_areas else ""
                        objects = []
                        for det in detections:
                            _, _, _, _, conf, class_id = det
                            name = self.class_names.get(class_id, f"Class_{class_id}")
                            objects.append(f"{name}({conf:.2f})")
                        
                        object_info = f", {len(detections)} objects: {', '.join(objects)}" if objects else ""
                        
                        print(f"Frame {frame_count:3d} [Motion: {motion_fps:4.1f} FPS, Det: {detection_fps:4.1f} FPS]: " + 
                              f"{motion_info}{object_info}")
                    
                    # Periodic stats
                    if frame_count % 50 == 0:
                        avg_motion_fps = frame_count / total_motion_time if total_motion_time > 0 else 0
                        avg_detection_fps = frame_count / total_detection_time if total_detection_time > 0 else 0
                        print(f"📊 Stats: {frame_count} frames, {total_motion_areas} motion areas, " + 
                              f"{total_detections} detections, {avg_motion_fps:.1f} motion FPS")
        
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
            avg_detection_fps = frame_count / total_detection_time if total_detection_time > 0 else 0
            
            print(f"\n✅ Final Results:")
            print(f"   - Total frames: {frame_count}")
            print(f"   - Total motion areas detected: {total_motion_areas}")
            print(f"   - Total object detections: {total_detections}")
            print(f"   - Average overall FPS: {avg_fps:.1f}")
            print(f"   - Average motion detection FPS: {avg_motion_fps:.1f}")
            print(f"   - Average object detection FPS: {avg_detection_fps:.1f}")
            print(f"   - Efficiency gain: Only detecting objects in {total_motion_areas} motion areas")

def main():
    parser = argparse.ArgumentParser(description='GPU Motion Detection + Object Recognition for Jetson Nano')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or file)')
    parser.add_argument('--model', '-m', default='yolov5s', help='YOLOv5 model (yolov5s, yolov5m, yolov5l)')
    parser.add_argument('--conf', '-c', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--motion-threshold', type=int, default=25, help='Motion detection threshold')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum motion area')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--no-display', action='store_true', help='Run in terminal mode')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("🚀 GPU Motion Detection + Object Recognition for Jetson Nano")
    print(f"⚡ Device: {args.device.upper()}")
    print(f"🎯 Motion Threshold: {args.motion_threshold}")
    print(f"📏 Minimum Area: {args.min_area}")
    if args.device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    try:
        detector = GPUMotionDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        
        # Set motion detection parameters
        detector.motion_threshold = args.motion_threshold
        detector.min_area = args.min_area
        
        detector.run_motion_detection(args.input, not args.no_display)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()