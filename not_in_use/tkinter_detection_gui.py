#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from pytorch_gpu_detection import PyTorchGPUDetector, get_local_ip

class TkinterDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Object Detection - Touch Interface")
        self.root.geometry("1024x768")
        
        # Initialize detector
        self.detector = None
        self.cap = None
        self.is_running = False
        self.show_gui_info = True
        self.show_detections = True
        
        # Threading
        self.detection_thread = None
        self.stop_thread = False
        
        # Video scaling
        self.current_frame = None
        self.video_width = 640
        self.video_height = 480
        
        # Performance optimization - cache scaling parameters
        self.cached_label_size = (0, 0)
        self.cached_scale_params = None
        
        self.setup_ui()
        self.setup_detector()
        
        # Bind keyboard shortcuts
        self.root.bind('<KeyPress-q>', lambda e: self.on_closing())
        self.root.bind('<KeyPress-Q>', lambda e: self.on_closing())
        self.root.focus_set()  # Make sure window can receive keyboard events
        
    def setup_ui(self):
        """Setup the GUI layout"""
        # Configure touch-friendly styling
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 14), padding=10)
        style.configure("Large.TLabel", font=("Arial", 12))
        style.configure("Large.TCheckbutton", font=("Arial", 12))
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video display
        video_frame = tk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video label
        self.video_label = tk.Label(video_frame, text="Video Feed\nWaiting for camera...", 
                                   bg="black", fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event to scale video
        self.video_label.bind('<Configure>', self.on_video_resize)
        
        # Video controls
        video_controls = tk.Frame(video_frame)
        video_controls.pack(fill=tk.X, pady=(10, 0))
        
        self.fullscreen_btn = tk.Button(video_controls, text="Fullscreen", 
                                       font=("Arial", 12), bg="#4CAF50", fg="white",
                                       command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side - Control panel
        control_frame = tk.Frame(main_frame, bg="#f0f0f0", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # Main controls
        controls_label = tk.Label(control_frame, text="Detection Controls", 
                                 font=("Arial", 14, "bold"), bg="#f0f0f0")
        controls_label.pack(pady=10)
        
        # Start/Stop button
        self.start_btn = tk.Button(control_frame, text="Start Detection", 
                                  font=("Arial", 14), bg="#2196F3", fg="white",
                                  height=2, command=self.toggle_detection)
        self.start_btn.pack(fill=tk.X, padx=20, pady=10)
        
        # Display options
        display_frame = tk.LabelFrame(control_frame, text="Display Options", 
                                     font=("Arial", 12), bg="#f0f0f0")
        display_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.gui_var = tk.BooleanVar(value=True)
        self.gui_check = tk.Checkbutton(display_frame, text="Show GUI Info", 
                                       variable=self.gui_var, font=("Arial", 11),
                                       bg="#f0f0f0", command=self.toggle_gui_info)
        self.gui_check.pack(anchor=tk.W, padx=10, pady=5)
        
        self.det_var = tk.BooleanVar(value=True)
        self.det_check = tk.Checkbutton(display_frame, text="Show Detections", 
                                       variable=self.det_var, font=("Arial", 11),
                                       bg="#f0f0f0", command=self.toggle_detections)
        self.det_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Settings
        settings_frame = tk.LabelFrame(control_frame, text="Settings", 
                                      font=("Arial", 12), bg="#f0f0f0")
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame, bg="#f0f0f0")
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(conf_frame, text="Confidence:", font=("Arial", 11), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = tk.Scale(conf_frame, from_=0.1, to=0.9, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.conf_var,
                                  command=self.update_confidence, bg="#f0f0f0")
        self.conf_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Statistics
        stats_frame = tk.LabelFrame(control_frame, text="Statistics", 
                                   font=("Arial", 12), bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.fps_label = tk.Label(stats_frame, text="GPU FPS: 0.0", 
                                 font=("Arial", 11), bg="#f0f0f0")
        self.fps_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.objects_label = tk.Label(stats_frame, text="Objects: 0", 
                                     font=("Arial", 11), bg="#f0f0f0")
        self.objects_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # RTSP
        rtsp_frame = tk.LabelFrame(control_frame, text="RTSP Stream", 
                                  font=("Arial", 12), bg="#f0f0f0")
        rtsp_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.rtsp_var = tk.BooleanVar()
        self.rtsp_check = tk.Checkbutton(rtsp_frame, text="Enable RTSP", 
                                        variable=self.rtsp_var, font=("Arial", 11),
                                        bg="#f0f0f0", command=self.toggle_rtsp)
        self.rtsp_check.pack(anchor=tk.W, padx=10, pady=5)
        
        self.stream_label = tk.Label(rtsp_frame, text="Stream URL will appear here", 
                                    font=("Arial", 9), bg="#f0f0f0", fg="blue",
                                    wraplength=250, justify=tk.LEFT)
        self.stream_label.pack(anchor=tk.W, padx=10, pady=5)
        
    def setup_detector(self):
        """Initialize the detector"""
        try:
            self.detector = PyTorchGPUDetector(
                model_name="yolov5s",
                conf_threshold=0.5,
                device="cuda",
                enable_rtsp=False
            )
            messagebox.showinfo("Success", "Detector initialized successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector: {e}")
            
    def toggle_detection(self):
        """Start/stop detection"""
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        """Start camera and detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.cap.isOpened():
                self.is_running = True
                self.stop_thread = False
                self.start_btn.config(text="Stop Detection", bg="#f44336")
                
                # Start detection in separate thread
                self.detection_thread = threading.Thread(target=self.detection_loop)
                self.detection_thread.daemon = True
                self.detection_thread.start()
            else:
                messagebox.showerror("Error", "Cannot open camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error starting detection: {e}")
            
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        self.stop_thread = True
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.start_btn.config(text="Start Detection", bg="#2196F3")
        self.video_label.config(image="", text="Video Feed\nStopped")
        
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_running and not self.stop_thread:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Run detection
                detections, inf_time = self.detector.detect_objects(frame)
                inference_fps = 1.0 / inf_time if inf_time > 0 else 0
                
                # Draw detections if enabled
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, detections)
                    
                # Add GUI info if enabled
                if self.show_gui_info:
                    cv2.putText(frame, f"GPU FPS: {inference_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Store current frame for scaling
                self.current_frame = frame
                
                # Update GUI in main thread
                self.root.after(0, self.update_display, frame, inference_fps, len(detections))
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
                
    def update_display(self, frame, fps, obj_count):
        """Update display in main thread - optimized version"""
        try:
            # Convert to PIL format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get current label size
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # Only recalculate scaling if label size changed
            current_label_size = (label_width, label_height)
            if current_label_size != self.cached_label_size and label_width > 1 and label_height > 1:
                img_width, img_height = pil_image.size
                
                # Calculate scale factors
                scale_x = label_width / img_width
                scale_y = label_height / img_height
                scale = min(scale_x, scale_y)  # Use min to fit within frame (no cropping)
                
                # Calculate new size
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Cache the scaling parameters
                self.cached_label_size = current_label_size
                self.cached_scale_params = (new_width, new_height)
            
            # Apply cached scaling if available
            if self.cached_scale_params and label_width > 1 and label_height > 1:
                new_width, new_height = self.cached_scale_params
                # Use faster NEAREST resampling for better performance
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
            
            # Convert to Tkinter format
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep reference
            
            # Update stats
            self.fps_label.config(text=f"GPU FPS: {fps:.1f}")
            self.objects_label.config(text=f"Objects: {obj_count}")
            
        except Exception as e:
            print(f"Display update error: {e}")
            
    def on_video_resize(self, event):
        """Handle video label resize - invalidate cache"""
        # Just invalidate the cache, don't process frame again
        self.cached_label_size = (0, 0)
        self.cached_scale_params = None
            
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.root.attributes('-fullscreen'):
            self.root.attributes('-fullscreen', False)
            self.fullscreen_btn.config(text="Fullscreen")
        else:
            self.root.attributes('-fullscreen', True)
            self.fullscreen_btn.config(text="Exit Fullscreen")
            
    def toggle_gui_info(self):
        """Toggle GUI info display"""
        self.show_gui_info = self.gui_var.get()
        
    def toggle_detections(self):
        """Toggle detection display"""
        self.show_detections = self.det_var.get()
        
    def update_confidence(self, value):
        """Update confidence threshold"""
        if self.detector:
            conf = float(value)
            self.detector.conf_threshold = conf
            self.detector.model.conf = conf
            
    def toggle_rtsp(self):
        """Toggle RTSP streaming"""
        if self.rtsp_var.get():
            try:
                # Reinitialize detector with RTSP
                self.detector = PyTorchGPUDetector(
                    model_name="yolov5s",
                    conf_threshold=self.conf_var.get(),
                    device="cuda",
                    enable_rtsp=True,
                    rtsp_port=8554
                )
                local_ip = get_local_ip()
                stream_url = f"http://{local_ip}:8554/stream.mjpg"
                self.stream_label.config(text=f"Stream: {stream_url}")
            except Exception as e:
                self.rtsp_var.set(False)
                messagebox.showerror("Error", f"RTSP error: {e}")
        else:
            self.stream_label.config(text="RTSP streaming disabled")
            
    def on_closing(self):
        """Handle window close"""
        print("Closing application...")
        self.stop_detection()
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.stop_thread = True
            self.detection_thread.join(timeout=2)
        
        # Force close camera
        if self.cap:
            self.cap.release()
            
        self.root.quit()  # Exit mainloop
        self.root.destroy()  # Destroy window
        
        # Force exit if needed
        import sys
        sys.exit(0)
        
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    try:
        gui = TkinterDetectionGUI()
        gui.run()
    except Exception as e:
        print(f"GUI Error: {e}")

if __name__ == "__main__":
    main()