# TrackGuard AI — Technical Deep Dive

> A comprehensive breakdown of the machine learning models, algorithms, and technologies powering TrackGuard AI.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [YOLOv8 — Object Detection](#yolov8--object-detection)
3. [DeepSort — Multi-Object Tracking](#deepsort--multi-object-tracking)
4. [Fire & Smoke Detection](#fire--smoke-detection)
5. [Personnel Classification via Color Analysis](#personnel-classification-via-color-analysis)
6. [Technology Stack](#technology-stack)
7. [ML Pipeline — Frame-by-Frame](#ml-pipeline--frame-by-frame)
8. [Use Cases](#use-cases)
9. [Model Summary](#model-summary)
10. [References](#references)

---

## System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Video Input │────▶│  YOLOv8      │────▶│  DeepSort        │────▶│  Annotated    │
│  (CCTV/File) │     │  Detection   │     │  Tracking        │     │  Output Video │
└──────────────┘     └──────┬───────┘     └──────────────────┘     └───────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────────┐
                     │  Fire Model  │     │  Personnel Color │
                     │  (Custom)    │     │  Classification  │
                     └──────────────┘     └──────────────────┘
                            │                      │
                     ┌──────▼──────────────────────▼──┐
                     │   Streamlit Web Dashboard       │
                     │   (Live Feed, Charts, Logs)     │
                     └─────────────────────────────────┘
```

---

## YOLOv8 — Object Detection

### What is YOLO?

**YOLO (You Only Look Once)** is a family of real-time object detection models. Unlike traditional two-stage detectors (R-CNN), YOLO processes the entire image in a **single forward pass** through the network, making it extremely fast.

### How YOLOv8 Works

YOLOv8 (by [Ultralytics](https://docs.ultralytics.com/)) is the latest iteration of YOLO and introduces several improvements:

| Component | Description |
|---|---|
| **Backbone** | CSPDarknet53 — extracts hierarchical features from the input image using cross-stage partial connections for efficient gradient flow |
| **Neck** | PANet (Path Aggregation Network) — fuses multi-scale feature maps so the model detects both small and large objects |
| **Head** | Anchor-free decoupled head — predicts bounding boxes, objectness scores, and class probabilities separately, improving accuracy |
| **Loss Function** | CIoU (Complete Intersection over Union) for box regression + Binary Cross-Entropy for classification |

### YOLOv8n (Nano) — Our Choice

We use **YOLOv8n**, the smallest and fastest variant:

| Property | Value |
|---|---|
| Parameters | 3.2 million |
| FLOPs | 8.7B |
| mAP@50 (COCO) | 37.3% |
| Speed (CPU) | ~80ms/frame |
| Input Size | 640 × 640 |
| Classes | 80 (COCO dataset) |

**Why YOLOv8n?** It offers the best trade-off between speed and accuracy for real-time CCTV applications. It runs comfortably on CPU, making it deployable on edge devices without a GPU.

### COCO Dataset Classes

The model detects **80 object classes** from the Microsoft COCO dataset, including those most relevant to railway stations:

- **People & Vehicles**: person, bicycle, car, motorcycle, bus, train, truck
- **Personal Items**: backpack, umbrella, handbag, suitcase
- **Station Objects**: bench, chair, clock, bottle, cell phone
- **Animals**: bird, cat, dog (stray animal detection)

---

## DeepSort — Multi-Object Tracking

### The Tracking Problem

Detection alone tells you *what* is in each frame, but not *who*. If 15 people appear across 100 frames, detection sees 1,500 independent boxes. **Tracking** links these detections across frames, assigning persistent IDs to each individual.

### How DeepSort Works

**DeepSort (Deep Simple Online and Realtime Tracking)** extends the classic SORT algorithm with a deep learning appearance model. It operates in four stages:

#### 1. Kalman Filter — Motion Prediction
Each tracked object has a **Kalman filter** that maintains a state vector:

$$\mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$

Where:
- $(u, v)$ = bounding box center
- $s$ = scale (area)
- $r$ = aspect ratio
- $\dot{u}, \dot{v}, \dot{s}$ = respective velocities

The Kalman filter **predicts** where each object will be in the next frame, even before seeing the new detections.

#### 2. Hungarian Algorithm — Assignment
New detections are matched to existing tracks using the **Hungarian algorithm** (Kuhn-Munkres), which solves the optimal assignment problem by minimizing a cost matrix built from:

- **Mahalanobis distance** (motion) — How far the detection is from the predicted position
- **Cosine distance** (appearance) — How similar the detection looks to the tracked object

$$c_{i,j} = \lambda \cdot d_{\text{Mahalanobis}}(i,j) + (1 - \lambda) \cdot d_{\text{cosine}}(i,j)$$

#### 3. Deep Appearance Descriptor
A **MobileNet CNN** (pre-trained) extracts a 128-dimensional feature vector from each detected bounding box. This embedding captures *appearance* — clothing color, shape, texture — enabling re-identification even after brief occlusions.

#### 4. Track Lifecycle Management

| State | Condition |
|---|---|
| **Tentative** | New detection, not yet confirmed |
| **Confirmed** | Matched for `n_init` consecutive frames (default: 3) |
| **Deleted** | Not matched for `max_age` frames (default: 30) |

### Key Parameters in TrackGuard AI

```python
DeepSort(
    max_age=30,           # Delete track after 30 missed frames
    n_init=3,             # Confirm after 3 consecutive matches
    max_cosine_distance=0.2,  # Appearance matching threshold
    nms_max_overlap=1.0,  # Non-max suppression overlap
    embedder="mobilenet"  # CNN for appearance features
)
```

---

## Fire & Smoke Detection

### Custom YOLOv8 Model

A **separate YOLOv8 model** (`weights/best.pt`) is trained specifically for fire and smoke detection. This is a custom-trained model (not included by default) that runs in parallel with the COCO detection model.

### Training Approach

| Aspect | Detail |
|---|---|
| **Base Model** | YOLOv8n (transfer learning) |
| **Dataset** | Fire/smoke images from open datasets |
| **Classes** | 2 — Fire, Smoke |
| **Method** | Fine-tuning: freeze backbone, train head on fire data |
| **Confidence Threshold** | 0.20 (lower than general detection to prioritize recall over precision — missing a fire is worse than a false alarm) |

### Graceful Degradation

If the fire model weights are not found at `weights/best.pt`, the system continues operating with all other features — fire detection is simply disabled. No crash, no error.

---

## Personnel Classification via Color Analysis

### How It Works

After tracking a person, the system samples the **pixel color at the center of their bounding box** and matches it against a predefined lookup table:

```python
COLOR_TO_LABEL = {
    (0, 165, 255):  'Fire Personnel',      # Orange uniform
    (128, 0, 128):  'Station Personnel',    # Purple uniform
    (31, 31, 31):   'Cleaning Personnel',   # Dark gray uniform
    (47, 47, 47):   'Security Personnel'    # Gray uniform
}
```

### Method

This is **not** a deep learning approach — it's a deterministic **color-matching heuristic**:

1. Identify tracked person's bounding box
2. Compute center point $(x_c, y_c)$
3. Read BGR pixel value at that coordinate
4. Match against known uniform colors (exact tuple match)

### Limitations & Trade-offs

- Works best when uniforms are standardized and clearly visible
- Sensitive to lighting conditions and occlusion
- Exact color match is strict — could be extended with tolerance ranges or a small classifier
- Designed for controlled environments (railway stations with known uniform colors)

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Object Detection** | YOLOv8 (Ultralytics) | Detect 80 object classes per frame |
| **Object Tracking** | DeepSort (deep-sort-realtime) | Assign persistent IDs across frames |
| **Appearance Model** | MobileNet (in DeepSort) | 128-D embeddings for re-identification |
| **Motion Model** | Kalman Filter (in DeepSort) | Predict object positions between frames |
| **Assignment** | Hungarian Algorithm | Optimal detection-to-track matching |
| **Fire Detection** | Custom YOLOv8 | Detect fire/smoke in frames |
| **Video I/O** | OpenCV | Read, process, and write video frames |
| **Deep Learning** | PyTorch | Backend for YOLO and DeepSort models |
| **Web Interface** | Streamlit | Interactive dashboard for presentation |
| **Numerical Ops** | NumPy, SciPy | Array operations, distance calculations |

---

## ML Pipeline — Frame-by-Frame

For every video frame, the following steps execute sequentially:

```
Frame N
  │
  ├─▶ [1] Fire Model Inference
  │     └─ YOLOv8 (custom) → fire/smoke bounding boxes + confidence
  │
  ├─▶ [2] Object Detection  
  │     └─ YOLOv8n (COCO) → 80-class bounding boxes + confidence
  │
  ├─▶ [3] DeepSort Update
  │     ├─ Kalman Filter predict → expected positions
  │     ├─ MobileNet → appearance embeddings
  │     ├─ Cost matrix (Mahalanobis + Cosine)
  │     ├─ Hungarian matching → assign detections to tracks
  │     └─ Update track states (confirm / delete / create)
  │
  ├─▶ [4] Personnel Color Check
  │     └─ Sample center pixel → match uniform color
  │
  ├─▶ [5] Annotation & Overlay
  │     ├─ Draw bounding boxes with track IDs
  │     ├─ Overlay class counts panel
  │     └─ Mark fire regions (if detected)
  │
  └─▶ [6] Logging
        └─ Timestamp, class_counts, fire_status, personnel → frame_data.txt / JSON
```

---

## Use Cases

### 1. Railway Station Surveillance
> Monitor platforms, tracks, and concourses in real time.

- **Crowd density monitoring** — Count people per frame to detect overcrowding on platforms
- **Unauthorized area intrusion** — Detect persons near tracks or restricted zones
- **Abandoned luggage** — Detect suitcases/backpacks not associated with a tracked person
- **Train arrival/departure** — Detect train objects entering and leaving the frame

### 2. Fire & Hazard Detection
> Early warning system for fire and smoke on station premises.

- **Smoke detection** — Identify smoke before flames appear, enabling faster evacuation
- **Fire localization** — Bounding boxes pinpoint fire location for rapid response
- **24/7 automated monitoring** — No human operator fatigue; constant vigilance

### 3. Personnel Management
> Track and identify station staff by uniform color.

- **Attendance verification** — Confirm that required personnel (security, cleaning, fire) are present in designated areas
- **Response time tracking** — Measure how quickly personnel respond to incidents
- **Shift coverage analysis** — Ensure adequate staffing across platform zones

### 4. Crowd Analytics & Planning
> Data-driven insights for station management.

- **Peak hour analysis** — Historical person count data reveals rush hour patterns
- **Flow optimization** — Understand pedestrian movement to improve signage and barriers
- **Capacity planning** — Use detection logs to plan platform expansions or schedule changes

### 5. Safety & Compliance Auditing
> Automated safety reporting for regulatory compliance.

- **Frame-by-frame audit trail** — Every detection is logged with timestamp, class, and confidence
- **Incident review** — Download annotated video to review specific events
- **JSON export** — Machine-readable logs for integration with station management systems

### 6. Smart City Integration
> TrackGuard AI can be deployed beyond railways.

- **Bus terminals** — Same detection stack applies to bus stations
- **Metro/subway** — Underground platform monitoring
- **Public spaces** — Parks, malls, airports — any CCTV-equipped area
- **Edge deployment** — YOLOv8n runs on CPU, suitable for embedded devices (Jetson Nano, Raspberry Pi with accelerator)

---

## Model Summary

| Model | Type | Parameters | Input | Output | Speed (CPU) |
|---|---|---|---|---|---|
| YOLOv8n | Object Detection | 3.2M | 640×640 image | Bounding boxes + classes + confidence | ~80ms |
| YOLOv8 (fire) | Fire Detection | ~3.2M | 640×640 image | Fire/Smoke boxes + confidence | ~80ms |
| MobileNet (DeepSort) | Feature Extraction | ~3.4M | Cropped bbox | 128-D embedding vector | ~5ms/crop |
| Kalman Filter | Motion Model | N/A (analytical) | Previous state | Predicted position | <1ms |

**Total inference per frame**: ~170ms on CPU → **~6 FPS** real-time processing

---

## References

1. **YOLOv8** — Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
2. **DeepSort** — Wojke, N., Bewley, A., Paulus, D. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. IEEE ICIP.
3. **SORT** — Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B. (2016). *Simple Online and Realtime Tracking*. IEEE ICIP.
4. **COCO Dataset** — Lin, T.Y., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV.
5. **Kalman Filter** — Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. ASME Journal of Basic Engineering.
6. **MobileNet** — Howard, A.G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861.
7. **Hungarian Algorithm** — Kuhn, H.W. (1955). *The Hungarian Method for the Assignment Problem*. Naval Research Logistics.

---

*TrackGuard AI — Built for safer railways, powered by machine learning.*
