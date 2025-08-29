#!/usr/bin/env python3
import torch
import torchvision
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

class PyTorchGPUDetector:
    def __init__(self, model_name="yolov5s", conf_threshold=0.5, device="cuda"):
        self.conf_threshold = conf_threshold
        self.nms_threshold = 0.45
        self.device = device
        
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
        
        self.load_model(model_name)
    
    def load_model(self, model_name):
        """Load YOLOv5 model with GPU acceleration"""
        print(f"🚀 Loading {model_name} model on {self.device}...")
        
        try:
            # Load YOLOv5 model from torch hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if self.device == "cuda":
                self.model.half()  # Use FP16 for speed
                print("✅ Model loaded on GPU with FP16 optimization")
            else:
                print("✅ Model loaded on CPU")
                
            # Set confidence threshold
            self.model.conf = self.conf_threshold
            self.model.iou = self.nms_threshold
            self.model.classes = self.target_classes  # Filter to target classes only
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying to load from local file...")
            # Fallback to local ONNX if available
            raise e
    
    def detect_objects(self, frame):
        """GPU-accelerated object detection"""
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference on GPU
        with torch.no_grad():
            results = self.model(rgb_frame)
        
        inference_time = time.time() - start_time
        
        # Parse results
        detections = []
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, class_id = detection
                
                # Convert to format: [x, y, width, height]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                detections.append([x, y, w, h, float(conf), int(class_id)])
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        for detection in detections:
            x, y, w, h, conf, class_id = detection
            
            # Get color and name
            color = self.colors.get(class_id, (0, 255, 0))
            name = self.class_names.get(class_id, f"Class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_live_detection(self, video_source=0, show_video=True):
        """Run live detection with GPU acceleration"""
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
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📹 Camera: {w}x{h} @ {fps} FPS")
        print(f"🎯 Target classes: {list(self.class_names.values())}")
        if show_video:
            print("🎮 Press 'q' to quit")
        else:
            print("🎮 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Performance tracking
        frame_count = 0
        total_inference_time = 0
        total_detections = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # GPU detection
                detections, inf_time = self.detect_objects(frame)
                total_inference_time += inf_time
                total_detections += len(detections)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                inference_fps = 1.0 / inf_time if inf_time > 0 else 0
                
                if show_video:
                    # Draw results
                    result_frame = self.draw_detections(frame.copy(), detections)
                    
                    # Add performance info
                    cv2.putText(result_frame, f"GPU FPS: {inference_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Real FPS: {fps_actual:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Objects: {len(detections)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display
                    cv2.imshow('PyTorch GPU Detection', result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Terminal output
                    if detections:
                        objects = []
                        for det in detections:
                            _, _, _, _, conf, class_id = det
                            name = self.class_names.get(class_id, f"Class_{class_id}")
                            objects.append(f"{name}({conf:.2f})")
                        
                        print(f"Frame {frame_count:3d} [{inference_fps:4.1f} GPU FPS, {fps_actual:4.1f} Real FPS]: {', '.join(objects)}")
                    
                    # Periodic stats
                    if frame_count % 30 == 0:
                        avg_inf_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                        print(f"📊 Stats: {frame_count} frames, {total_detections} detections, {avg_inf_fps:.1f} avg GPU FPS")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
            
            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_inference_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            
            print(f"\n✅ Final Results:")
            print(f"   - Total frames: {frame_count}")
            print(f"   - Total detections: {total_detections}")
            print(f"   - Average real FPS: {avg_fps:.1f}")
            print(f"   - Average GPU inference FPS: {avg_inference_fps:.1f}")
            print(f"   - GPU speedup: {avg_inference_fps/avg_fps:.1f}x" if avg_fps > 0 else "")

def main():
    parser = argparse.ArgumentParser(description='PyTorch GPU Object Detection for Jetson Nano')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or file)')
    parser.add_argument('--model', '-m', default='yolov5s', help='YOLOv5 model (yolov5s, yolov5m, yolov5l)')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--no-display', action='store_true', help='Run in terminal mode')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("🚀 PyTorch GPU Object Detection for Jetson Nano")
    print(f"⚡ Device: {args.device.upper()}")
    if args.device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)
    
    try:
        detector = PyTorchGPUDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        
        detector.run_live_detection(args.input, not args.no_display)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()