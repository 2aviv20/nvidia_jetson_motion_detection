#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

class HeadlessObjectDetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # COCO class names - focusing on humans (0), cars (2), and drones/aircraft (4)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Target classes: person, car, truck, bus, motorcycle, airplane
        self.target_classes = [0, 2, 3, 5, 7, 4]  # person, car, motorcycle, bus, truck, airplane
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            0: (0, 255, 0),    # Green for humans
            2: (255, 0, 0),    # Blue for cars
            3: (0, 255, 255),  # Yellow for motorcycles
            5: (255, 255, 0),  # Cyan for buses
            7: (128, 0, 128),  # Purple for trucks
            4: (255, 0, 255)   # Magenta for aircraft/drones
        }
        
        self.model_path = model_path
        self.net = None
        self.input_width = 640
        self.input_height = 640
        
        self.load_model()
    
    def load_model(self):
        """Load ONNX model optimized for Jetson Nano"""
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            
            # Try CUDA first, fall back to CPU
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✅ Using CUDA backend for inference")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("✅ Using CPU backend for inference")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_objects(self, frame):
        """Perform object detection on a single frame"""
        height, width = frame.shape[:2]
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        start_time = time.time()
        outputs = self.net.forward()
        inference_time = time.time() - start_time
        
        # Post-process
        boxes = []
        confidences = []
        class_ids = []
        
        # YOLOv5 output format: [batch_size, num_detections, 85]
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter by confidence and target classes
                if confidence > self.conf_threshold and class_id in self.target_classes:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        # Draw results
        result_frame = frame.copy()
        for box, confidence, class_id in zip(final_boxes, final_confidences, final_class_ids):
            x, y, w, h = box
            
            # Get class name and color
            class_name = self.class_names[class_id]
            if class_id == 0:
                display_name = "Human"
            elif class_id in [2, 7]:  # car, truck
                display_name = "Vehicle"
            elif class_id == 4:  # airplane
                display_name = "Aircraft/Drone"
            else:
                display_name = class_name.capitalize()
            
            color = self.colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{display_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(result_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add FPS and detection count
        fps = 1.0 / inference_time if inference_time > 0 else 0
        detection_count = len(final_boxes)
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Detections: {detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame, detection_count, fps, final_boxes, final_confidences, final_class_ids
    
    def process_video(self, video_source, output_path, max_frames=300):
        """Process video input in headless mode"""
        # Open video source
        if isinstance(video_source, int) or video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
            print(f"📹 Opening camera {video_source}")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"📁 Opening video file: {video_source}")
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video source {video_source}")
            return
        
        # Set camera resolution to 576p if using camera
        if isinstance(video_source, int) or video_source.isdigit():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        
        # Get actual video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📺 Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Error: Cannot create output video file {output_path}")
            cap.release()
            return
        
        try:
            frame_count = 0
            total_detections = 0
            start_time = time.time()
            
            print(f"🎬 Starting processing (max {max_frames} frames)...")
            print("🎯 Detecting: Humans, Cars, Trucks, Buses, Motorcycles, Aircraft/Drones")
            print("-" * 60)
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video stream")
                    break
                
                frame_count += 1
                
                # Process frame
                result_frame, detection_count, current_fps, boxes, confidences, class_ids = self.detect_objects(frame)
                total_detections += detection_count
                
                # Save frame
                out.write(result_frame)
                
                # Print detection details
                if detection_count > 0:
                    detected_objects = []
                    for class_id, confidence in zip(class_ids, confidences):
                        class_name = self.class_names[class_id]
                        if class_id == 0:
                            display_name = "Human"
                        elif class_id in [2, 7]:
                            display_name = "Vehicle"
                        elif class_id == 4:
                            display_name = "Aircraft/Drone"
                        else:
                            display_name = class_name.capitalize()
                        detected_objects.append(f"{display_name}({confidence:.2f})")
                    
                    print(f"Frame {frame_count:3d}: {detection_count} objects - {', '.join(detected_objects)}")
                
                # Print periodic updates
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"📊 Progress: {frame_count}/{max_frames} frames, Avg FPS: {avg_fps:.1f}, Total detections: {total_detections}")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print("-" * 60)
            print(f"✅ Processing complete!")
            print(f"📹 Output saved to: {output_path}")
            print(f"📊 Stats:")
            print(f"   - Total frames processed: {frame_count}")
            print(f"   - Total detections: {total_detections}")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Processing time: {elapsed:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Headless Object Detection on Jetson Nano')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or video file path)')
    parser.add_argument('--model', '-m', default='yolov5s.onnx', help='Path to ONNX model file')
    parser.add_argument('--output', '-o', default='detected_output.mp4', help='Output video file path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', '-n', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--frames', '-f', type=int, default=300, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    print("🚀 Jetson Nano Headless Object Detection System")
    print("🎯 Target classes: Humans, Cars, Trucks, Buses, Motorcycles, Aircraft/Drones")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = HeadlessObjectDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            nms_threshold=args.nms
        )
        
        # Process video
        detector.process_video(args.input, args.output, args.frames)
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()