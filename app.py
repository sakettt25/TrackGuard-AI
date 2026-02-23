"""
TrackGuard AI - Web Interface
Streamlit-based web app for CCTV Railway Management System
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import datetime
import random
import tempfile
import json
from ultralytics import YOLO
from deep_sort1.deep_sort import DeepSort

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="TrackGuard AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
    .status-box {
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .fire-safe {
        background-color: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    .fire-danger {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# ─── COCO Class Map ────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
CLASS_MAP = {i: name for i, name in enumerate(COCO_CLASSES)}

COLOR_TO_LABEL = {
    (0, 165, 255): 'Fire Personnel',
    (128, 0, 128): 'Station Personnel',
    (31, 31, 31): 'Cleaning Personnel',
    (47, 47, 47): 'Security Personnel'
}


# ─── Model Loading (cached) ────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_fire_model():
    path = "weights/best.pt"
    if os.path.exists(path):
        return YOLO(path)
    return None

@st.cache_resource
def load_tracker():
    return DeepSort(model_path="deep_sort1/deep/checkpoint/ckpt.t7", max_age=30)


# ─── Processing Function ───────────────────────────────────────
def process_video(video_path, progress_bar, status_text, frame_display,
                  metrics_placeholder, detection_threshold, fire_threshold,
                  live_stats_placeholder):
    """Process video and return results"""
    
    model = load_yolo_model()
    modelf = load_fire_model()
    tracker = load_tracker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file")
        return None, []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Output video
    out_path = os.path.join("output", "processed_output.avi")
    os.makedirs("output", exist_ok=True)
    cap_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
    frame_data_list = []
    all_class_counts = {}
    total_persons = 0
    fire_frames = 0
    frame_count = 0
    
    ret, frame = cap.read()
    
    while ret:
        frame_count += 1
        frame_data = {}
        
        # ── Fire Detection ──
        if modelf is not None:
            try:
                resultsfire = modelf.predict(source=frame, conf=fire_threshold, verbose=False)
                bboxes_fire = resultsfire[0].boxes.xyxy
                confs_fire = resultsfire[0].boxes.conf
                labels_fire = resultsfire[0].boxes.cls
                
                frame_data["fire_detected"] = len(bboxes_fire) > 0
                if frame_data["fire_detected"]:
                    fire_frames += 1
                
                for box, conf, cls in zip(bboxes_fire, confs_fire, labels_fire):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    frame_data["fire_confidence"] = conf.item()
                    color = (255, 0, 255) if int(cls.item()) == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"FIRE {conf.item():.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception:
                frame_data["fire_detected"] = False
        else:
            frame_data["fire_detected"] = False
        
        # ── Timestamp ──
        ts = datetime.datetime.now()
        frame_data["timestamp"] = ts.strftime("%A, %d %B %Y %I:%M:%S %p")
        
        # ── Object Detection ──
        filtered = {}
        try:
            results = model(frame, verbose=False)
            dict_counts = {k: 0 for k in range(80)}
            
            for result in results:
                bboxes_xywh = []
                confidence = []
                
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    bboxes_xywh.append([x1, y1, w, h])
                    confidence.append(score)
                    dict_counts[int(class_id)] += 1
                
                filtered = {CLASS_MAP[k]: v for k, v in dict_counts.items() if v != 0}
                
                # Accumulate global stats
                for cls_name, cnt in filtered.items():
                    all_class_counts[cls_name] = all_class_counts.get(cls_name, 0) + cnt
                if "person" in filtered:
                    total_persons += filtered["person"]
                
                # ── Tracking ──
                tracker.update(bboxes_xywh, confidence, frame)
                personnel_detected = []
                
                for track in tracker.tracker.tracks:
                    track_id = track.track_id
                    bbox = np.array(track.to_tlwh())
                    x, y, w, h = bbox
                    
                    shift = 0.5
                    y += int(h * shift)
                    x += int(w * shift)
                    
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
                                colors[track_id % len(colors)], 3)
                    
                    xc, yc = int(x + w / 2), int(y + h / 2)
                    if 0 <= xc < frame.shape[1] and 0 <= yc < frame.shape[0]:
                        cc = tuple(map(int, frame[yc, xc]))
                        if cc in COLOR_TO_LABEL:
                            personnel_detected.append(COLOR_TO_LABEL[cc])
                        cv2.circle(frame, (xc, yc - 5), 10, cc, -1)
                
                frame_data["personnel_detected"] = personnel_detected
                
                # ── Draw overlay ──
                cv2.rectangle(frame, (0, 0), (250, max(180, 30 + 30 * len(filtered))), (222, 49, 99), -1)
                yt = 30
                for k, v in filtered.items():
                    cv2.putText(frame, f"{k}: {v}", (10, yt),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    yt += 28
                
                frame_data["class_counts"] = filtered
                frame_data_list.append(frame_data)
        
        except Exception as e:
            frame_data["class_counts"] = {}
            frame_data_list.append(frame_data)
        
        cap_out.write(frame)
        
        # ── Update UI every 5 frames ──
        if frame_count % 5 == 0 or frame_count == 1:
            progress = frame_count / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Show current frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(display_frame, channels="RGB", use_container_width=True)
            
            # Live metrics
            with live_stats_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames Processed", f"{frame_count}/{total_frames}")
                c2.metric("Persons (this frame)", filtered.get("person", 0))
                c3.metric("Total Detections", sum(filtered.values()) if filtered else 0)
                fire_status = "🔥 YES" if frame_data.get("fire_detected") else "✅ No"
                c4.metric("Fire Alert", fire_status)
        
        ret, frame = cap.read()
    
    cap.release()
    cap_out.release()
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return out_path, frame_data_list, all_class_counts, total_persons, fire_frames, frame_count


# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    detection_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
    fire_threshold = st.slider("Fire Detection Confidence", 0.05, 0.5, 0.20, 0.05)
    
    st.markdown("---")
    st.markdown("## 📹 Video Source")
    
    video_source = st.radio("Choose input:", ["Upload Video", "Use Sample Video"])
    
    uploaded_file = None
    if video_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    st.markdown("---")
    st.markdown("## 👥 Personnel Colors")
    st.markdown("""
    | Color | Role |
    |---|---|
    | 🟠 Orange | Fire Personnel |
    | 🟣 Purple | Station Personnel |
    | ⚫ Dark Gray | Cleaning |
    | 🔘 Gray | Security |
    """)
    
    st.markdown("---")
    st.markdown("### 🚆 TrackGuard AI v1.0")


# ─── Main Content ──────────────────────────────────────────────
st.markdown('<div class="main-header">🚆 TrackGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered CCTV Railway Management System</div>', unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 🎯 Object Detection")
    st.caption("YOLOv8 — 80 COCO classes")
with col2:
    st.markdown("### 🔥 Fire Detection")
    st.caption("Real-time fire & smoke alerts")
with col3:
    st.markdown("### 👤 Person Tracking")
    st.caption("DeepSort multi-object tracking")
with col4:
    st.markdown("### 👷 Personnel ID")
    st.caption("Uniform color classification")

st.markdown("---")

# ─── Determine video path ──────────────────────────────────────
video_path = None

if video_source == "Upload Video" and uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name
    st.success(f"Uploaded: {uploaded_file.name}")

elif video_source == "Use Sample Video":
    sample_path = os.path.join("data", "test2.mp4")
    if os.path.exists(sample_path):
        video_path = sample_path
        st.info(f"Using sample video: {sample_path}")
    else:
        st.warning("No sample video found at `data/test2.mp4`. Please upload a video instead.")


# ─── Process Button ────────────────────────────────────────────
if video_path:
    # Show video info
    cap_info = cv2.VideoCapture(video_path)
    total_f = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap_info.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap_info.get(3))
    vid_h = int(cap_info.get(4))
    cap_info.release()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{vid_w}×{vid_h}")
    c2.metric("FPS", f"{vid_fps:.1f}")
    c3.metric("Total Frames", total_f)
    c4.metric("Duration", f"{total_f / vid_fps:.1f}s" if vid_fps > 0 else "N/A")
    
    st.markdown("---")
    
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        st.markdown("### 🔄 Processing...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_stats = st.empty()
        
        st.markdown("### 📺 Live Feed")
        frame_display = st.empty()
        
        start_time = time.time()
        
        result = process_video(
            video_path, progress_bar, status_text, frame_display,
            live_stats, detection_threshold, fire_threshold,
            live_stats
        )
        
        elapsed = time.time() - start_time
        
        if result and result[0]:
            out_path, frame_data_list, all_counts, total_persons, fire_frames, total_processed = result
            
            st.balloons()
            
            # ── Results Section ──
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Frames", total_processed)
            r2.metric("Processing Time", f"{elapsed:.1f}s")
            r3.metric("Avg FPS", f"{total_processed / elapsed:.1f}")
            avg_persons = total_persons / total_processed if total_processed > 0 else 0
            r4.metric("Avg Persons/Frame", f"{avg_persons:.1f}")
            
            # Fire status
            st.markdown("### 🔥 Fire Detection Summary")
            if fire_frames > 0:
                st.error(f"⚠️ Fire/Smoke detected in {fire_frames} out of {total_processed} frames ({fire_frames/total_processed*100:.1f}%)")
            else:
                fire_model_status = "active" if load_fire_model() else "not loaded (no model at weights/best.pt)"
                st.success(f"✅ No fire detected — Fire model: {fire_model_status}")
            
            # Detection breakdown
            st.markdown("### 🎯 Detection Summary (Total Across All Frames)")
            if all_counts:
                sorted_counts = dict(sorted(all_counts.items(), key=lambda x: x[1], reverse=True))
                
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(sorted_counts)
                with col_table:
                    for cls, cnt in sorted_counts.items():
                        st.write(f"**{cls}**: {cnt}")
            else:
                st.info("No objects detected in the video.")
            
            # Frame-by-frame data
            st.markdown("### 📋 Frame-by-Frame Data")
            with st.expander("View all frame data", expanded=False):
                # Build a cleaner table
                table_data = []
                for i, fd in enumerate(frame_data_list):
                    table_data.append({
                        "Frame": i + 1,
                        "Timestamp": fd.get("timestamp", ""),
                        "Fire": "🔥" if fd.get("fire_detected") else "✅",
                        "Objects": str(fd.get("class_counts", {})),
                        "Personnel": ", ".join(fd.get("personnel_detected", [])) or "—"
                    })
                st.dataframe(table_data, use_container_width=True, height=400)
            
            # Download buttons
            st.markdown("### 📥 Downloads")
            dl1, dl2 = st.columns(2)
            
            with dl1:
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Processed Video",
                            f.read(),
                            file_name="trackguard_output.avi",
                            mime="video/avi",
                            use_container_width=True
                        )
            
            with dl2:
                # Export frame data as JSON
                json_data = json.dumps(frame_data_list, indent=2, default=str)
                st.download_button(
                    "⬇️ Download Detection Log (JSON)",
                    json_data,
                    file_name="trackguard_detections.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.error("Processing failed. Please check the video file.")

else:
    # Landing state
    st.markdown("### 👈 Upload a video or select sample video from the sidebar to begin")
    st.markdown("""
    **How it works:**
    1. Upload a video or use the sample video
    2. Adjust detection settings in the sidebar
    3. Click **Start Processing**
    4. View live detection feed and results
    5. Download the annotated video and detection log
    """)
