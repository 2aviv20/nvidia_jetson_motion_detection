#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

def test_video_recording():
    """Test video recording with different codecs"""
    print("🧪 Testing video recording capabilities...")
    
    # Create test directory
    test_dir = Path("test_recordings")
    test_dir.mkdir(exist_ok=True)
    
    # Test parameters
    width, height = 640, 480
    fps = 20.0
    duration_seconds = 5
    total_frames = int(fps * duration_seconds)
    
    # Codecs to test
    codecs_to_test = [
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
    ]
    
    for codec_name, fourcc in codecs_to_test:
        print(f"\n🔧 Testing codec: {codec_name}")
        
        # Output file
        output_file = test_dir / f"test_{codec_name.lower()}.avi"
        
        # Create video writer
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"❌ Failed to open writer with {codec_name}")
            continue
        
        print(f"✅ Writer opened with {codec_name}")
        
        # Generate test frames
        frames_written = 0
        for frame_num in range(total_frames):
            # Create a test frame with moving rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Background color based on frame
            bg_color = (frame_num * 5 % 255, 100, 150)
            frame[:] = bg_color
            
            # Moving rectangle
            rect_x = (frame_num * 10) % (width - 100)
            rect_y = (frame_num * 5) % (height - 50)
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 50), (255, 255, 255), -1)
            
            # Frame number text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Debug frame before writing
            if frame_num == 0:
                print(f"🔍 Frame info: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min()}-{frame.max()}]")
                
            # Ensure frame is contiguous and correct format
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Write frame
            success = writer.write(frame)
            if success:
                frames_written += 1
                if frame_num == 0:
                    print(f"✅ First frame written successfully")
            else:
                print(f"❌ Failed to write frame {frame_num}")
                if frame_num == 0:
                    print(f"🔍 OpenCV error - trying alternative method")
                break
        
        writer.release()
        
        # Check result
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"📄 Created file: {output_file.name} ({file_size} bytes)")
            print(f"📊 Frames written: {frames_written}/{total_frames}")
            
            # Test reading back
            test_cap = cv2.VideoCapture(str(output_file))
            if test_cap.isOpened():
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Playback test: {test_frames} frames at {test_fps} fps")
                
                # Try to read first frame
                ret, first_frame = test_cap.read()
                if ret:
                    print(f"✅ Can read frames: {first_frame.shape}")
                else:
                    print(f"❌ Cannot read frames")
                test_cap.release()
            else:
                print(f"❌ Cannot open for playback")
        else:
            print(f"❌ File not created")

if __name__ == "__main__":
    test_video_recording()