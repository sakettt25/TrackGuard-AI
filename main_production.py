import os
import sys
import random
import numpy as np
from deep_sort1.deep_sort import DeepSort
from deep_sort1.sort.tracker import Tracker
import cv2
from ultralytics import YOLO
import threading
import time
import datetime
import requests

# Configuration
CONFIG = {
    'video_path': os.path.join('.', 'data', 'test2.mp4'),
    'video_out_path': os.path.join('.', 'out.mp4'),
    'use_webcam': False,  # Set to True to use webcam instead of video file
    'webcam_id': 0,
    'detection_threshold': 0.5,
    'fire_conf_threshold': 0.20,
    'max_age': 30,
}

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import cv2
        import numpy
        import torch
        from ultralytics import YOLO
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def download_sample_video():
    """Download a sample video for testing"""
    print("Downloading sample video...")
    sample_url = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
    video_path = CONFIG['video_path']
    
    try:
        response = requests.get(sample_url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Sample video downloaded to {video_path}")
        return True
    except Exception as e:
        print(f"✗ Could not download sample video: {e}")
        return False

def setup_video_source():
    """Set up video source (file or webcam)"""
    if CONFIG['use_webcam']:
        print(f"Using webcam {CONFIG['webcam_id']}")
        cap = cv2.VideoCapture(CONFIG['webcam_id'])
    else:
        video_path = CONFIG['video_path']
        if not os.path.exists(video_path):
            print(f"✗ Video file not found: {video_path}")
            print("Options:")
            print("1. Place your video file at:", video_path)
            print("2. Set 'use_webcam': True in CONFIG to use webcam")
            print("3. Create a sample video")
            
            choice = input("Create a blank sample video for testing? (y/n): ").lower()
            if choice == 'y':
                create_sample_video(video_path)
            else:
                return None, None, None
        
        print(f"Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("✗ Could not open video source")
        return None, None, None
    
    ret, frame = cap.read()
    if not ret:
        print("✗ Could not read first frame")
        return None, None, None
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"✓ Video source opened: {frame_width}x{frame_height}")
    
    return cap, frame, (frame_width, frame_height)

def create_sample_video(output_path):
    """Create a simple test video"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    print("Creating sample video...")
    for i in range(100):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Sample Frame {i}", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add a moving rectangle to simulate motion
        x = (i * 5) % 600
        cv2.rectangle(frame, (x, 200), (x+40, 280), (0, 255, 0), -1)
        out.write(frame)
    
    out.release()
    print(f"✓ Sample video created: {output_path}")

def setup_models():
    """Load YOLO models"""
    print("Loading models...")
    
    try:
        # YOLO COCO model - will auto-download if not present
        print("Loading YOLOv8 COCO model...")
        model = YOLO("yolov8n.pt")
        print("✓ YOLOv8 COCO model loaded")
    except Exception as e:
        print(f"✗ Could not load YOLO COCO model: {e}")
        return None, None
    
    # Fire detection model
    fire_model_path = 'weights/best.pt'
    if not os.path.exists(fire_model_path):
        print(f"⚠ Fire detection model not found: {fire_model_path}")
        print("  Fire detection will be disabled")
        print("  To enable: Place your trained fire detection model at", fire_model_path)
        modelf = None
    else:
        try:
            print("Loading fire detection model...")
            modelf = YOLO(fire_model_path)
            print("✓ Fire detection model loaded")
        except Exception as e:
            print(f"✗ Could not load fire detection model: {e}")
            modelf = None
    
    return model, modelf

def setup_tracker():
    """Initialize DeepSort tracker"""
    print("Initializing tracker...")
    try:
        deep_sort_weight = 'deep_sort1/deep/checkpoint/ckpt.t7'
        tracker = DeepSort(model_path=deep_sort_weight, max_age=CONFIG['max_age'])
        print("✓ Tracker initialized")
        return tracker
    except Exception as e:
        print(f"✗ Could not initialize tracker: {e}")
        return None

def main():
    """Main function"""
    print("=" * 60)
    print("TrackGuard AI - CCTV guided Railway Management System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup video source
    cap, first_frame, dimensions = setup_video_source()
    if cap is None:
        return
    
    frame_width, frame_height = dimensions
    
    # Setup output video - use AVI with XVID for broad compatibility
    video_out_path = CONFIG['video_out_path']
    # Change extension to .avi for XVID codec compatibility
    video_out_path_avi = video_out_path.replace('.mp4', '.avi')
    cap_out = cv2.VideoWriter(
        video_out_path_avi,
        cv2.VideoWriter_fourcc(*'XVID'),
        cap.get(cv2.CAP_PROP_FPS),
        (first_frame.shape[1], first_frame.shape[0])
    )
    video_out_path = video_out_path_avi
    print(f"✓ Output video: {video_out_path}")
    
    # Load models
    model, modelf = setup_models()
    if model is None:
        cap.release()
        cap_out.release()
        return
    
    # Setup tracker
    tracker = setup_tracker()
    if tracker is None:
        cap.release()
        cap_out.release()
        return
    
    # Initialize variables
    frame_data_list = []
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    detection_threshold = CONFIG['detection_threshold']
    
    # Class mapping for COCO dataset
    integer_keys = list(range(80))
    values = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    class_map = {key: value for key, value in zip(integer_keys, values)}
    
    # Color to personnel label mapping
    color_to_label = {
        (0, 165, 255): 'Fire Personnel',
        (128, 0, 128): 'Station Personnel',
        (31, 31, 31): 'Cleaning Personnel',
        (47, 47, 47): 'Security Personnel'
    }
    
    print("\n" + "=" * 60)
    print("Processing video...")
    print("=" * 60)
    
    frame_count = 0
    ret = True
    frame = first_frame
    
    try:
        while ret:
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")
            
            frame_data = {}
            
            # Fire detection
            if modelf is not None:
                try:
                    resultsfire = modelf.predict(source=frame, conf=CONFIG['fire_conf_threshold'], verbose=False)
                    bounding_boxes = resultsfire[0].boxes.xyxy
                    confidences = resultsfire[0].boxes.conf
                    class_labels = resultsfire[0].boxes.cls
                    
                    frame_data["fire_detected"] = len(bounding_boxes) > 0
                    
                    for box, confidence, class_label in zip(bounding_boxes, confidences, class_labels):
                        x_min, y_min, x_max, y_max = map(int, box.tolist())
                        confidence = confidence.item()
                        class_label = int(class_label.item())
                        
                        frame_data["fire_confidence"] = confidence
                        
                        color = (255, 0, 255) if class_label == 0 else (255, 0, 0)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                except Exception as e:
                    print(f"Error in fire detection: {e}")
                    frame_data["fire_detected"] = False
            else:
                frame_data["fire_detected"] = False
            
            # Timestamp
            timestamp = time.time()
            timestamp_datetime = datetime.datetime.fromtimestamp(timestamp)
            formatted_timestamp = timestamp_datetime.strftime("%A, %d %B %Y %I:%M:%S %p")
            frame_data["timestamp"] = formatted_timestamp
            
            # Object detection with YOLO
            try:
                results = model(frame, verbose=False)
                
                num_keys = 80
                dict_counts = {key: 0 for key in range(num_keys)}
                
                for result in results:
                    bboxes_xywh = []
                    confidence = []
                    
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = r
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        w = x2 - x1
                        h = y2 - y1
                        bbox_xywh = [x1, y1, w, h]
                        bboxes_xywh.append(bbox_xywh)
                        confidence.append(score)
                        class_id = int(class_id)
                        dict_counts[class_id] += 1
                    
                    filtered_dict = {class_map[key]: value for key, value in dict_counts.items() if value != 0}
                    
                    # Update tracker
                    tracks = tracker.update(bboxes_xywh, confidence, frame)
                    personnel_detected = []
                    
                    for track in tracker.tracker.tracks:
                        track_id = track.track_id
                        bbox_xywh = np.array(track.to_tlwh())
                        
                        x, y, w, h = bbox_xywh
                        
                        shift_per = 0.5
                        y_shift = int(h * shift_per)
                        x_shift = int(w * shift_per)
                        y += y_shift
                        x += x_shift
                        
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
                                    (colors[track_id % len(colors)]), 3)
                        
                        # Center coordinates
                        x_center = int(x + w / 2)
                        y_center = int(y + h / 2)
                        
                        if 0 <= x_center < frame.shape[1] and 0 <= y_center < frame.shape[0]:
                            center_color = tuple(map(int, frame[y_center, x_center]))
                            
                            if center_color in color_to_label:
                                label = color_to_label[center_color]
                                personnel_detected.append(label)
                            
                            cv2.circle(frame, (x_center, y_center - 5), 10, center_color, -1)
                    
                    frame_data["personnel_detected"] = personnel_detected
                    
                    # Draw annotations
                    text_annotations = [(key, value) for key, value in filtered_dict.items()]
                    cv2.rectangle(frame, (0, 0), (250, 180), (222, 49, 99), -1)
                    y_text = 60
                    frame_data["class_counts"] = filtered_dict
                    
                    for key, value in text_annotations:
                        cv2.putText(frame, f"{key}: {value}", (25, y_text),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        y_text += 30
                    
                    frame_data_list.append(frame_data)
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            # Write frame to output
            cap_out.write(frame)
            
            # Read next frame
            ret, frame = cap.read()
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cap_out.release()
        cv2.destroyAllWindows()
        
        # Write frame data to file
        output_file_path = "frame_data.txt"
        try:
            with open(output_file_path, "w") as file:
                for frame_data in frame_data_list:
                    file.write(f"Frame Data:\n{frame_data}\n\n")
            print(f"✓ Frame data written to {output_file_path}")
        except Exception as e:
            print(f"✗ Could not write frame data: {e}")
        
        print(f"✓ Output video saved to {video_out_path}")
        print(f"✓ Processed {frame_count} frames")
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)

if __name__ == "__main__":
    main()
