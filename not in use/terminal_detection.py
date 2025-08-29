#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time

class TerminalObjectDetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # COCO class names
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
        self.target_classes = [0, 2, 3, 5, 7, 4]
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load ONNX model"""
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✅ Using CUDA backend")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("✅ Using CPU backend")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_objects(self, frame):
        """Perform object detection"""
        height, width = frame.shape[:2]
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Inference
        start_time = time.time()
        outputs = self.net.forward()
        inference_time = time.time() - start_time
        
        # Post-process
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold and class_id in self.target_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        fps = 1.0 / inference_time if inference_time > 0 else 0
        return final_boxes, final_confidences, final_class_ids, fps
    
    def run_detection(self, video_source, max_frames=300):
        """Run detection and print results"""
        # Open video source
        if isinstance(video_source, int) or video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
            print(f"📹 Opening camera {video_source}")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"📁 Opening video file: {video_source}")
        
        if not cap.isOpened():
            print(f"❌ Cannot open video source {video_source}")
            return
        
        # Set 576p resolution for camera
        if isinstance(video_source, int) or video_source.isdigit():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📺 Video: {width}x{height} @ {fps} FPS")
        print("🎯 Detecting: Humans, Cars, Trucks, Buses, Motorcycles, Aircraft/Drones")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            frame_count = 0
            total_detections = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                boxes, confidences, class_ids, current_fps = self.detect_objects(frame)
                detection_count = len(boxes)
                total_detections += detection_count
                
                if detection_count > 0:
                    detected_objects = []
                    for class_id, confidence in zip(class_ids, confidences):
                        if class_id == 0:
                            name = "Human"
                        elif class_id in [2, 7]:
                            name = "Vehicle"
                        elif class_id == 4:
                            name = "Aircraft/Drone"
                        else:
                            name = self.class_names[class_id].capitalize()
                        detected_objects.append(f"{name}({confidence:.2f})")
                    
                    print(f"Frame {frame_count:3d} [{current_fps:4.1f} FPS]: {detection_count} objects - {', '.join(detected_objects)}")
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Progress: {frame_count} frames, {total_detections} total detections")
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            cap.release()
            print(f"✅ Processed {frame_count} frames, found {total_detections} total objects")

def main():
    parser = argparse.ArgumentParser(description='Terminal Object Detection')
    parser.add_argument('--input', '-i', default='0', help='Video input')
    parser.add_argument('--model', '-m', default='yolov5s.onnx', help='Model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--frames', '-f', type=int, default=300, help='Max frames')
    
    args = parser.parse_args()
    
    print("🚀 Jetson Nano Object Detection (Terminal Mode)")
    print("=" * 50)
    
    detector = TerminalObjectDetector(args.model, args.conf)
    detector.run_detection(args.input, args.frames)

if __name__ == "__main__":
    main()