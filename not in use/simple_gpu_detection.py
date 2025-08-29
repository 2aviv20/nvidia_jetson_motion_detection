#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import time
import onnxruntime as ort

class SimpleGPUDetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.nms_threshold = 0.45
        
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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load ONNX model with GPU acceleration"""
        try:
            # Configure ONNX Runtime for GPU
            providers = []
            if self.device == "cuda":
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"✅ Model loaded with providers: {self.session.get_providers()}")
            print(f"📋 Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def preprocess_frame(self, frame):
        """Preprocess frame for ONNX model"""
        # Resize to model input size
        input_h, input_w = 640, 640
        resized = cv2.resize(frame, (input_w, input_h))
        
        # Convert BGR to RGB and normalize
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        return input_batch
    
    def postprocess_outputs(self, outputs, frame_shape):
        """Process ONNX model outputs"""
        h_orig, w_orig = frame_shape[:2]
        
        detections = []
        output = outputs[0][0]  # Remove batch dimension
        
        # Process each detection
        for detection in output:
            # Extract values
            x_center, y_center, width, height = detection[:4]
            confidence_scores = detection[5:]
            
            # Get class with highest confidence
            class_id = np.argmax(confidence_scores)
            confidence = confidence_scores[class_id]
            
            # Filter by confidence and target classes
            if confidence > self.conf_threshold and class_id in self.target_classes:
                # Convert to original frame coordinates
                x_center *= w_orig / 640
                y_center *= h_orig / 640
                width *= w_orig / 640
                height *= h_orig / 640
                
                # Convert to top-left coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                w = int(width)
                h = int(height)
                
                detections.append([x1, y1, w, h, float(confidence), int(class_id)])
        
        return detections
    
    def detect_objects(self, frame):
        """Run GPU-accelerated inference"""
        start_time = time.time()
        
        # Preprocess
        input_batch = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_batch})
        
        # Postprocess
        detections = self.postprocess_outputs(outputs, frame.shape)
        
        inference_time = time.time() - start_time
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
        """Run live detection"""
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
        
        print(f"📹 Camera: {w}x{h}")
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
                    cv2.putText(result_frame, f"Inference FPS: {inference_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Real FPS: {fps_actual:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Objects: {len(detections)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Device: {self.device.upper()}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display
                    cv2.imshow('GPU Detection (ONNX Runtime)', result_frame)
                    
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
                        
                        print(f"Frame {frame_count:3d} [{inference_fps:4.1f} inf FPS, {fps_actual:4.1f} real FPS]: {', '.join(objects)}")
                    
                    # Periodic stats
                    if frame_count % 30 == 0:
                        avg_inf_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                        print(f"📊 Stats: {frame_count} frames, {total_detections} detections, {avg_inf_fps:.1f} avg inference FPS")
        
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
            print(f"   - Average inference FPS: {avg_inference_fps:.1f}")
            print(f"   - Speedup vs CPU OpenCV: ~{avg_inference_fps/1.0:.1f}x")

def main():
    parser = argparse.ArgumentParser(description='Simple GPU Object Detection for Jetson Nano')
    parser.add_argument('--input', '-i', default='0', help='Video input (camera index or file)')
    parser.add_argument('--model', '-m', default='yolov5s.onnx', help='ONNX model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Run in terminal mode')
    
    args = parser.parse_args()
    
    print("🚀 Simple GPU Object Detection for Jetson Nano")
    print(f"⚡ Using ONNX Runtime with GPU acceleration")
    print("=" * 60)
    
    try:
        detector = SimpleGPUDetector(
            model_path=args.model,
            conf_threshold=args.conf
        )
        
        detector.run_live_detection(args.input, not args.no_display)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()