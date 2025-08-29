#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time

class SpeedOptimizedDetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.65):
        self.conf_threshold = conf_threshold
        self.nms_threshold = 0.5  # Higher for speed
        
        # Target classes
        self.target_classes = [0, 2, 3, 4, 5, 7]  # person, car, motorcycle, airplane, bus, truck
        self.class_names = {
            0: "Human", 2: "Car", 3: "Motorcycle", 
            4: "Aircraft", 5: "Bus", 7: "Truck"
        }
        
        # Load model
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        # Optimized CPU settings for Jetson Nano ARM processor
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Try to enable ARM NEON optimizations if available
        import os
        os.environ['OPENCV_DNN_OPENCL'] = '0'  # Disable OpenCL to use ARM optimizations
        
        print("⚡ Using optimized CPU backend for ARM (NEON optimizations)")
        
        # Skip frames for speed
        self.skip_frames = 2  # Process every 3rd frame
        self.frame_counter = 0
        self.last_detections = []
        
        # Pre-allocate arrays for better performance
        self.reusable_blob = None
    
    def detect_objects(self, frame):
        """Highly optimized detection for ARM CPU"""
        h, w = frame.shape[:2]
        
        # Use original model input size to avoid reshape errors
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Inference
        start = time.time()
        outputs = self.net.forward()
        inf_time = time.time() - start
        
        # Optimized processing - vectorized operations
        detections = []
        
        # Process outputs more efficiently
        for output in outputs:
            # Vectorized confidence filtering
            scores = output[:, 5:]
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # Filter by confidence and target classes
            valid_mask = (max_scores > self.conf_threshold)
            target_mask = np.isin(class_ids, self.target_classes)
            combined_mask = valid_mask & target_mask
            
            if np.any(combined_mask):
                valid_detections = output[combined_mask]
                valid_confidences = max_scores[combined_mask]
                valid_class_ids = class_ids[combined_mask]
                
                # Convert coordinates efficiently
                centers_x = valid_detections[:, 0] * w
                centers_y = valid_detections[:, 1] * h
                widths = valid_detections[:, 2] * w
                heights = valid_detections[:, 3] * h
                
                x_coords = (centers_x - widths/2).astype(int)
                y_coords = (centers_y - heights/2).astype(int)
                w_coords = widths.astype(int)
                h_coords = heights.astype(int)
                
                for i in range(len(valid_detections)):
                    detections.append([
                        [x_coords[i], y_coords[i], w_coords[i], h_coords[i]],
                        float(valid_confidences[i]),
                        int(valid_class_ids[i])
                    ])
        
        # Fast NMS if we have detections
        if detections:
            boxes = [det[0] for det in detections]
            confidences = [det[1] for det in detections]
            
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            final_detections = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append((detections[i][0], detections[i][1], detections[i][2]))
                    
            return final_detections, inf_time
        
        return [], inf_time
    
    def run_detection(self, video_source=0):
        """Run with frame skipping for speed"""
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Cannot open {video_source}")
            return
        
        # Lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Resolution: {w}x{h}")
        print("🚀 Frame skipping enabled for speed")
        print("🎯 Press Ctrl+C to stop")
        print("-" * 40)
        
        total_frames = 0
        processed_frames = 0
        total_detections = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                self.frame_counter += 1
                
                # Skip frames for speed
                if self.frame_counter % (self.skip_frames + 1) == 0:
                    processed_frames += 1
                    detections, inf_time = self.detect_objects(frame)
                    self.last_detections = detections
                    
                    # Calculate speeds
                    elapsed = time.time() - start_time
                    processing_fps = processed_frames / elapsed if elapsed > 0 else 0
                    real_fps = total_frames / elapsed if elapsed > 0 else 0
                    
                    if detections:
                        total_detections += len(detections)
                        objects = []
                        for (box, conf, class_id) in detections:
                            name = self.class_names[class_id]
                            objects.append(f"{name}({conf:.2f})")
                        
                        print(f"Frame {total_frames:3d} [Proc: {processing_fps:.1f} FPS, Real: {real_fps:.1f} FPS]: {', '.join(objects)}")
                    
                    # Progress update
                    if processed_frames % 10 == 0:
                        print(f"📊 Processed {processed_frames}/{total_frames} frames, {total_detections} detections")
                        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            elapsed = time.time() - start_time
            processing_fps = processed_frames / elapsed if elapsed > 0 else 0
            real_fps = total_frames / elapsed if elapsed > 0 else 0
            
            print(f"\n✅ Final Results:")
            print(f"   - Total frames: {total_frames}")
            print(f"   - Processed frames: {processed_frames}")
            print(f"   - Processing FPS: {processing_fps:.1f}")
            print(f"   - Real-time FPS: {real_fps:.1f}")
            print(f"   - Total detections: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description='Speed Optimized Detection')
    parser.add_argument('--input', '-i', default='0', help='Video input')
    parser.add_argument('--model', '-m', default='yolov5s.onnx', help='Model path')
    parser.add_argument('--conf', '-c', type=float, default=0.65, help='Confidence threshold')
    parser.add_argument('--skip', '-s', type=int, default=2, help='Skip frames (process every N+1 frames)')
    
    args = parser.parse_args()
    
    print("🚀 Speed Optimized Jetson Detection")
    print("⚡ Frame skipping + High confidence threshold")
    print("=" * 50)
    
    detector = SpeedOptimizedDetector(args.model, args.conf)
    detector.skip_frames = args.skip
    detector.run_detection(args.input)

if __name__ == "__main__":
    main()