# TrackGuard AI - Production Ready Setup
CCTV guided Railway Management System

## Overview
TrackGuard AI is an AI-powered video analytics system designed for railway management. It provides:
- **Fire & Smoke Detection**: Real-time detection of fire and smoke
- **People Tracking**: Multi-object tracking using DeepSort
- **Personnel Recognition**: Identification of different personnel types by uniform color
- **Crowd Detection**: Monitoring crowd density
- **Object Detection**: Detection of various objects using YOLOv8

## Features
- Real-time video processing
- Fire and smoke detection with confidence scores
- Object detection and classification (80 COCO classes)
- Multi-object tracking with DeepSort
- Personnel classification by uniform color:
  - Orange: Fire Personnel
  - Purple: Station Personnel
  - Dark Gray: Cleaning Personnel
  - Gray: Security Personnel
- Frame-by-frame data logging
- Video output with annotations

## System Requirements
- Python 3.8 or higher
- Windows 10/11 (PowerShell scripts provided)
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU recommended for better performance (optional)

## Quick Start

### 1. Setup
Run the setup script to install dependencies and create directories:
```powershell
.\setup.ps1
```

This will:
- Check Python installation
- Create necessary directories
- Install all required packages
- Verify setup

### 2. Prepare Video (Optional)
Place your video file at:
```
data/test2.mp4
```

Or edit `main_production.py` to use webcam:
```python
CONFIG = {
    'use_webcam': True,
    'webcam_id': 0,
    ...
}
```

### 3. Add Fire Detection Model (Optional)
If you have a trained fire detection model, place it at:
```
weights/best.pt
```

If not provided, fire detection will be disabled but the system will still work for object detection and tracking.

### 4. Run
Execute the run script:
```powershell
.\run.ps1
```

Or run directly:
```powershell
python main_production.py
```

## Manual Installation

If you prefer manual setup:

1. **Create directories:**
```powershell
New-Item -ItemType Directory -Force -Path data, weights, output
```

2. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

3. **Run the application:**
```powershell
python main_production.py
```

## Configuration

Edit `main_production.py` to customize settings:

```python
CONFIG = {
    'video_path': os.path.join('.', 'data', 'test2.mp4'),
    'video_out_path': os.path.join('.', 'out.mp4'),
    'use_webcam': False,  # Set to True for webcam
    'webcam_id': 0,
    'detection_threshold': 0.5,
    'fire_conf_threshold': 0.20,
    'max_age': 30,
}
```

## Output

The system generates:
1. **Video Output**: `out.mp4` - Processed video with annotations
2. **Frame Data**: `frame_data.txt` - Detailed frame-by-frame analysis including:
   - Timestamp
   - Fire detection status and confidence
   - Object counts by class
   - Detected personnel types

## Project Structure
```
TrackGuard-AI/
├── main_production.py      # Production-ready main script
├── main.py                  # Original main script
├── edgedet.py              # Edge detection for railway tracks
├── Inference_video.py      # FastSAM inference script
├── requirements.txt        # Python dependencies
├── setup.ps1               # Automated setup script
├── run.ps1                 # Run script
├── README_PRODUCTION.md    # This file
├── data/                   # Input videos
├── weights/                # Model weights
│   └── best.pt            # Fire detection model (optional)
├── output/                 # Additional outputs
└── deep_sort1/            # DeepSort tracker implementation
    ├── deep_sort.py
    └── sort/
        └── tracker.py
```

## Dependencies
- opencv-python: Video processing
- numpy: Numerical operations
- torch: Deep learning framework
- torchvision: Computer vision utilities
- ultralytics: YOLOv8 implementation
- deep-sort-realtime: Object tracking
- scipy: Scientific computing
- requests: HTTP requests

## Models

### YOLOv8 COCO Model
- **File**: `yolov8n.pt`
- **Auto-download**: Yes
- **Purpose**: General object detection (80 classes)

### Fire Detection Model
- **File**: `weights/best.pt`
- **Auto-download**: No
- **Purpose**: Fire and smoke detection
- **Status**: Optional

## Troubleshooting

### "No video file found"
- Place video at `data/test2.mp4`
- Or enable webcam mode in CONFIG
- Or the script will create a sample video

### "Could not load fire detection model"
- Fire detection model is optional
- System will work without it for object detection
- To enable: Place trained model at `weights/best.pt`

### "Missing dependency"
- Run `.\setup.ps1` again
- Or manually: `pip install -r requirements.txt`

### Poor Performance
- Reduce video resolution
- Use GPU if available (CUDA-enabled PyTorch)
- Reduce `detection_threshold` for faster processing

### GPU Support
To enable CUDA support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Known Limitations
1. Fire detection requires a trained model (not included)
2. Personnel detection relies on specific uniform colors
3. Processing speed depends on hardware
4. Best results with good lighting conditions

## Advanced Usage

### Custom Video Source
```python
CONFIG['video_path'] = 'path/to/your/video.mp4'
```

### Webcam Mode
```python
CONFIG['use_webcam'] = True
CONFIG['webcam_id'] = 0  # Change for different cameras
```

### Adjust Detection Sensitivity
```python
CONFIG['detection_threshold'] = 0.3  # Lower = more detections
CONFIG['fire_conf_threshold'] = 0.15  # Lower = more sensitive
```

## Support
For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure video source is accessible
4. Check Python version (3.8+)

## License
See original project repository for license information.

## Credits
- YOLOv8: Ultralytics
- DeepSort: Deep SORT implementation
- OpenCV: Computer vision library
