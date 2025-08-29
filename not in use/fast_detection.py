#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time

class FastJetsonDetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.6, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Target classes: person, car, motorcycle, airplane, bus, truck
        self.target_classes = [0, 2, 3, 4, 5, 7]
        self.class_names = ['Human', 'Car', 'Motorcycle', 'Aircraft', 'Bus', 'Truck']
        
        # Colors (BGR)
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 128)]
        
        # Use model's expected input size
        self.input_width = 640  # YOLOv5 expects 640x640
        self.input_height = 640
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load and optimize model"""
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        # Try CUDA first
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("🚀 Using CUDA backend")
            else:
                # Optimized CPU settings
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("⚡ Using optimized CPU backend")
        except:
            print("⚡ Using CPU backend")
    
    def detect_frame(self, frame):
        """Fast detection with optimizations"""
        h, w = frame.shape[:2]
        
        # Smaller blob for speed
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Inference
        start = time.time()
        outputs = self.net.forward()
        inference_time = time.time() - start
        
        # Fast post-processing
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold and class_id in self.target_classes:
                    cx, cy, bw, bh = detection[0:4]
                    x = int((cx - bw/2) * w)
                    y = int((cy - bh/2) * h) 
                    width = int(bw * w)
                    height = int(bh * h)
                    
                    detections.append([x, y, width, height, confidence, class_id])
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections):
        """Fast drawing"""
        for det in detections:
            x, y, w, h, conf, class_id = det
            
            # Get color and name
            idx = self.target_classes.index(class_id)
            color = self.colors[idx % len(self.colors)]
            name = self.class_names[idx]
            
            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name}: {conf:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run(self, video_source=0, show_video=True):
        """Run fast detection"""
        # Convert string to int for camera index
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open {video_source}")
            return
        
        # Set resolution - try lower resolution first for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual resolution
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📹 Resolution: {w}x{h} @ {fps} FPS")
        print("🎯 Press 'q' to quit")
        print("-" * 40)
        
        # Performance tracking
        frame_count = 0
        total_time = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detection
                detections, inf_time = self.detect_frame(frame)
                total_time += inf_time
                detection_count += len(detections)
                
                # Calculate FPS
                avg_fps = frame_count / total_time if total_time > 0 else 0
                
                if show_video:
                    # Draw results
                    result_frame = self.draw_detections(frame.copy(), detections)
                    
                    # Add FPS counter
                    cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Objects: {len(detections)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display
                    cv2.imshow('Fast Detection', result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Terminal output only
                    if len(detections) > 0:
                        objects = []
                        for det in detections:
                            _, _, _, _, conf, class_id = det
                            idx = self.target_classes.index(class_id)
                            name = self.class_names[idx]
                            objects.append(f"{name}({conf:.2f})")
                        print(f"Frame {frame_count:3d} [{avg_fps:4.1f} FPS]: {', '.join(objects)}")
                    
                    if frame_count % 30 == 0:
                        print(f"📊 {frame_count} frames, {detection_count} total detections, {avg_fps:.1f} avg FPS")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped")
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
            
            print(f"✅ Final stats: {frame_count} frames, {avg_fps:.1f} avg FPS, {detection_count} total detections")

def main():
    parser = argparse.ArgumentParser(description='Fast Object Detection for Jetson Nano')
    parser.add_argument('--input', '-i', default=0, help='Video input')
    parser.add_argument('--model', '-m', default='yolov5s.onnx', help='Model path')
    parser.add_argument('--conf', '-c', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Terminal mode only')
    
    args = parser.parse_args()
    
    print("⚡ Fast Jetson Nano Object Detection")
    print("🎯 Optimized for speed!")
    print("=" * 40)
    
    detector = FastJetsonDetector(args.model, args.conf)
    detector.run(args.input, not args.no_display)

if __name__ == "__main__":
    main()