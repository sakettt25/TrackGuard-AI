# TrackGuard AI 🚆

**AI-Powered CCTV Railway Management System**

TrackGuard AI is a real-time video analytics system designed for railway station surveillance. It combines **YOLOv8 object detection**, **DeepSort multi-object tracking**, and **fire/smoke detection** to monitor platforms, detect hazards, and classify station personnel — all from standard CCTV feeds.

---

## Features

| Feature | Description |
|---|---|
| **Object Detection** | Detects 80 COCO classes (person, train, car, handbag, etc.) using YOLOv8 |
| **Multi-Object Tracking** | Tracks individuals across frames with DeepSort (unique IDs per person) |
| **Fire & Smoke Detection** | Real-time fire/smoke alerts with confidence scores (requires custom model) |
| **Personnel Classification** | Identifies staff by uniform color — Fire, Station, Cleaning, Security |
| **Crowd Monitoring** | Counts people per frame for crowd density analysis |
| **Frame-by-Frame Logging** | Timestamped JSON-style logs of all detections per frame |
| **Annotated Output Video** | Bounding boxes, tracking IDs, and class overlays rendered on output |

---

## Quick Start

### Prerequisites
- **Python 3.8+**
- **Windows 10/11** (PowerShell scripts provided)
- 4 GB RAM minimum (8 GB recommended)
- NVIDIA GPU optional (CPU works fine)

### 1. Clone & Setup

```powershell
git clone https://github.com/arnvsnigi/TrackGuard-AI.git
cd TrackGuard-AI
.\install_venv.ps1
```

This creates a virtual environment and installs all dependencies.

### 2. Add Input Video

Place your video file at:
```
data/test2.mp4
```

Or switch to webcam mode by editing `main_production.py`:
```python
CONFIG = {
    'use_webcam': True,
    'webcam_id': 0,
}
```

### 3. Run

```powershell
.\run.ps1
```

Or manually:
```powershell
.\.venv\Scripts\Activate.ps1
python main_production.py
```

---

## Output

The system generates two outputs:

### `out.avi` — Annotated Video
- Bounding boxes around detected objects
- Colored tracking IDs following each person
- Top-left overlay showing real-time class counts (e.g., `person: 10`, `train: 1`)
- Fire/smoke bounding boxes (if fire model is loaded)

### `frame_data.txt` — Detection Log
Each frame is logged with:
```python
{
    'fire_detected': False,
    'timestamp': 'Monday, 23 February 2026 03:18:05 PM',
    'personnel_detected': [],
    'class_counts': {'person': 8, 'train': 1, 'handbag': 1, 'clock': 1}
}
```

---

## Configuration

All settings are in `main_production.py`:

```python
CONFIG = {
    'video_path': os.path.join('.', 'data', 'test2.mp4'),
    'video_out_path': os.path.join('.', 'out.mp4'),
    'use_webcam': False,
    'webcam_id': 0,
    'detection_threshold': 0.5,       # YOLO confidence threshold
    'fire_conf_threshold': 0.20,      # Fire model confidence threshold
    'max_age': 30,                    # DeepSort max frames to keep lost tracks
}
```

### Personnel Color Mapping

| Uniform Color | Label |
|---|---|
| Orange `(0, 165, 255)` | Fire Personnel |
| Purple `(128, 0, 128)` | Station Personnel |
| Dark Gray `(31, 31, 31)` | Cleaning Personnel |
| Gray `(47, 47, 47)` | Security Personnel |

---

## Project Structure

```
TrackGuard-AI/
├── main_production.py       # Main entry point (production-ready)
├── main.py                  # Original development script
├── requirements.txt         # Python dependencies
├── yolov8n.pt              # YOLOv8 nano model (auto-downloads)
├── install_venv.ps1         # Setup script (venv + dependencies)
├── run.ps1                  # Run script
├── setup.ps1                # Alternative setup script
├── create_sample_video.py   # Generate test video
├── edgedet.py               # Edge detection for track boundaries
├── Inference_video.py       # FastSAM inference script
├── data/
│   └── test2.mp4            # Input video
├── weights/
│   └── best.pt              # Fire detection model (optional)
├── output/                  # Additional outputs
├── deep_sort1/              # DeepSort tracker wrapper
│   ├── deep_sort.py         # DeepSort ↔ deep-sort-realtime bridge
│   ├── sort/
│   │   └── tracker.py       # Tracker interface
│   └── deep/
│       └── checkpoint/      # Embedder weights (auto-managed)
├── out.avi                  # Generated output video
└── frame_data.txt           # Generated detection log
```

---

## Models

| Model | File | Purpose | Required |
|---|---|---|---|
| YOLOv8n | `yolov8n.pt` | General object detection (80 COCO classes) | Yes (auto-downloads) |
| Fire/Smoke | `weights/best.pt` | Fire & smoke detection | No (optional) |
| MobileNet Embedder | Managed by `deep-sort-realtime` | Appearance features for tracking | Yes (auto-downloads) |

---

## Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
deep-sort-realtime>=1.3.2
scipy>=1.10.0
requests>=2.31.0
```

PyTorch is installed automatically as a dependency of `ultralytics`.

### GPU Support (Optional)

For CUDA-accelerated inference:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `Video file not found` | Place video at `data/test2.mp4` or enable webcam mode |
| `Fire model not found` | Optional — system works without it |
| `Missing dependency` | Run `.\install_venv.ps1` or `pip install -r requirements.txt` |
| `Output video won't play` | Output uses XVID/AVI — use VLC or Windows Media Player |
| `Slow processing` | Use a lower resolution video or enable GPU |
| `torch DLL error` | Ensure you're using the venv: `.\.venv\Scripts\Activate.ps1` |

---

## How It Works

1. **Frame Capture** — Reads frames from video file or webcam
2. **Fire Detection** — Runs fire/smoke YOLO model (if available)
3. **Object Detection** — YOLOv8 detects all objects in frame
4. **Tracking** — DeepSort assigns persistent IDs to tracked objects
5. **Personnel Classification** — Center pixel color of each tracked person is matched to uniform colors
6. **Annotation** — Bounding boxes, IDs, and class counts are drawn on the frame
7. **Logging** — Per-frame detection data is saved to `frame_data.txt`
8. **Output** — Annotated frames are written to `out.avi`

---

## Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) — Multi-object tracking
- [OpenCV](https://opencv.org/) — Video processing & rendering

## License

See the original project repository for license information.
