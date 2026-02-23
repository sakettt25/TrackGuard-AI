"""
Create a sample test video for TrackGuard AI
"""
import cv2
import numpy as np
import os

def create_sample_video():
    """Create a simple test video with moving objects"""
    output_path = os.path.join('data', 'test2.mp4')
    os.makedirs('data', exist_ok=True)
    
    # Video parameters
    width, height = 1280, 720
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating sample video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Duration: {duration}s")
    
    for frame_num in range(total_frames):
        # Create a frame with a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            color_value = int((y / height) * 100)
            frame[y, :] = [color_value, color_value + 20, color_value + 40]
        
        # Add moving rectangles to simulate people/objects
        # Person 1 - moving left to right
        x1 = int((frame_num / total_frames) * (width - 100))
        cv2.rectangle(frame, (x1, 300), (x1 + 80, 500), (0, 255, 0), -1)
        cv2.putText(frame, "Person", (x1 + 10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Person 2 - moving right to left
        x2 = width - int((frame_num / total_frames) * (width - 100))
        cv2.rectangle(frame, (x2, 200), (x2 + 60, 400), (255, 165, 0), -1)  # Orange - Fire Personnel
        cv2.putText(frame, "Person", (x2 + 5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Car moving
        car_x = int((frame_num / total_frames) * (width - 200))
        cv2.rectangle(frame, (car_x, 450), (car_x + 150, 550), (200, 200, 200), -1)
        cv2.rectangle(frame, (car_x + 20, 490), (car_x + 60, 530), (50, 50, 50), -1)  # Window
        cv2.rectangle(frame, (car_x + 90, 490), (car_x + 130, 530), (50, 50, 50), -1)  # Window
        cv2.circle(frame, (car_x + 30, 550), 20, (0, 0, 0), -1)  # Wheel
        cv2.circle(frame, (car_x + 120, 550), 20, (0, 0, 0), -1)  # Wheel
        
        # Add text overlay
        cv2.putText(frame, f"TrackGuard AI Test - Frame {frame_num + 1}/{total_frames}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add railway platform indicator
        cv2.rectangle(frame, (0, height - 100), (width, height), (50, 50, 50), -1)
        cv2.putText(frame, "Platform 1 - Railway Station", 
                   (width // 2 - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        if (frame_num + 1) % 30 == 0:
            print(f"  Generated {frame_num + 1}/{total_frames} frames...")
    
    out.release()
    print(f"✓ Sample video created successfully: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    create_sample_video()
