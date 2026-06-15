# CamouflageNet — AI-Powered Camouflage Detection System
### Ground Surveillance · YOLOv8m-seg · Instance Segmentation · Flask Dashboard

---

## Overview

CamouflageNet detects camouflaged human targets in complex outdoor environments using instance segmentation. It combines three data sources, multi-scale training, and Real-ESRGAN super-resolution to detect concealed targets that conventional systems miss.

**Designed for:** perimeter surveillance, search & rescue, wildlife anti-poaching, drone-based monitoring.

---

## Architecture

```mermaid
flowchart TD
    subgraph SOURCES["Input Sources"]
        A[Image Upload]
        B[Video File]
        C[Webcam]
        D[RTSP Camera]
        E[IP Camera]
        F[RTMP Drone]
        G[Network Feed]
    end

    subgraph PRE["Pre-processing"]
        H{Width < 320px?}
        I[Real-ESRGAN 4x SR]
        J[Pass-through]
    end

    subgraph MODEL["YOLOv8m-seg"]
        K[Backbone: CSPDarknet]
        L[Neck: PANet multi-scale]
        M[Head: Detection + Segmentation]
    end

    subgraph OUT["Outputs"]
        N[Bounding Boxes]
        O[Segmentation Masks]
        P[Confidence Scores]
        Q[JSONL Event Log]
        R[SSE Real-time Stats]
    end

    A & B --> H
    C & D & E & F & G --> H
    H -- low-res --> I --> K
    H -- normal  --> J --> K
    K --> L --> M
    M --> N & O & P
    N & O & P --> Q & R
```

### Training Pipeline

```mermaid
flowchart LR
    A[Roboflow\ncamouflage-detection-girgk\nHuman camo, bbox] --> D
    B[Diplomae\n1077 camo images\nbbox] --> D
    C[COD10K-v3\n5066 images\npixel masks] --> D
    D[Albumentations\nfog blur noise\nJPEG perspective] --> E[Merged Dataset\n~4000+ train images]
    E --> F[YOLOv8m-seg\nCOCO pretrained]
    F --> G[AdamW\nlr0=0.001]
    G --> H[Cosine LR\nlrf=0.01]
    H --> I[100 epochs\nT4x2 DDP]
    I --> J[best.pt]
```

---

## Results

<div align="center">

| | | |
|:---:|:---:|:---:|
| ![Result 1](app/result1.png) | ![Result 2](app/result2.png) | ![Result 3](app/result3.png) |

| | |
|:---:|:---:|
| ![Result 4](app/result4.png) | ![Result 5](app/result5.png) |

</div>

---

## Dataset

| Source | Images | Annotation | Focus |
|--------|--------|-----------|-------|
| Roboflow `camouflage-detection-girgk` | varies | Bounding boxes | Human/military camouflage |
| Diplomae `camo-object-detection-kaggle` | 1,077 | Bounding boxes | Camo objects |
| COD10K-v3 *(optional, recommended)* | 5,066 | Pixel-level masks | Enables segmentation overlays |

> **Without COD10K:** model produces bounding boxes only.
> **With COD10K:** model produces bounding boxes + segmentation polygon overlays.

---

## Benchmark Results


| Model | Box mAP@50 | Box mAP@50:95 | Mask mAP@50 | Precision | Recall |
|-------|-----------|--------------|------------|-----------|--------|
| YOLOv8m-seg (ours) | 98.2% | 80.9% | 98.2% | 98.3% | 96.8% |

---

## Setup

### Phase 1 — Train on Kaggle

**Step 1: Get Roboflow API key**
1. Go to https://app.roboflow.com
2. Settings (bottom-left gear icon) → Roboflow API
3. Copy your **Private API Key**

**Step 2: Create Kaggle notebook**
1. Go to https://www.kaggle.com/code → New Notebook
2. Right sidebar → Session options → Accelerator: `GPU T4 x2`
3. Enable Internet (required to install packages + call Roboflow API)

**Step 3: Add COD10K dataset** *(for segmentation masks)*
1. Right sidebar → + Add Data
2. Search `cod10k-v3` → Add

**Step 4: Upload and run the notebook**
1. File → Import Notebook → select `training/camouflage_detection_kaggle.ipynb`
2. In Cell 3, replace `YOUR_ROBOFLOW_API_KEY` with your key
3. Click Run All

**Step 5: Download weights**
- After Cell 12 completes: Output tab → download `camouflage_best.pt`

---

### Phase 2 — Run the Dashboard

```bash
# 1. Extract the project
unzip camouflage-detection.zip -d ~/projects/
cd ~/projects/camouflage-detection/app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate

# 3. Install dependencies (GPU recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ../requirements.txt

# 4. Place trained weights
cp ~/Downloads/camouflage_best.pt .

# 5. Launch
python app.py
# → http://localhost:5000
```

> **⚠ CPU Warning:** Streaming inference is 1–5 seconds per frame on CPU.
> Real-time webcam/RTSP requires a CUDA GPU.

---

## Usage Guide

### Image Detection
1. Click **IMAGE** tab in sidebar
2. Drag image onto drop zone or click to browse
3. Adjust confidence slider (default 0.25)
4. Click **RUN ANALYSIS**
5. Annotated result appears with segmentation masks + bounding boxes

### Video Processing
1. Click **VIDEO** tab
2. Drop video file (MP4/AVI/MOV)
3. Click **PROCESS VIDEO**
4. Processed video with detections available for download

### Webcam Stream
1. Click **WEBCAM** tab → **▶ ENGAGE**
2. Live MJPEG feed with real-time detection overlay
3. **■ DISENGAGE** to stop

### RTSP Camera
1. Click **RTSP** tab
2. Enter URL: `rtsp://user:pass@192.168.1.100:554/stream1`
3. Click **▶ CONNECT**

### RTMP Drone Stream
1. Configure drone to push RTMP to your server
2. Click **RTMP** tab → enter pull URL: `rtmp://your-server/live/drone1`
3. Click **▶ CONNECT**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| POST | `/api/detect/image` | Single image detection |
| POST | `/api/detect/video` | Video file processing |
| GET | `/api/stream/webcam` | Webcam MJPEG stream |
| GET | `/api/stream/rtsp?url=` | RTSP camera stream |
| GET | `/api/stream/ip?url=` | IP camera stream |
| GET | `/api/stream/rtmp?url=` | RTMP drone stream |
| GET | `/api/stream/network?url=` | Network feed |
| POST | `/api/stream/stop` | Stop active stream |
| GET | `/api/events` | SSE stats + detection events |
| GET | `/api/logs?limit=50` | Recent event log entries |
| GET | `/api/stats` | Current stats snapshot |

---

## Project Structure

```
camouflage-detection/
├── training/
│   ├── generate_notebook.py
│   └── camouflage_detection_kaggle.ipynb
├── app/
│   ├── app.py                  Flask backend — all routes
│   ├── esrgan_enhance.py       Real-ESRGAN wrapper with CPU fallback
│   ├── templates/
│   │   └── index.html          Military terminal dashboard UI
│   └── static/
│       ├── css/style.css       Phosphor-green surveillance aesthetic
│       └── js/main.js          Stream control, SSE, uploads
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Path to YOLOv8 weights |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8m-seg (Ultralytics 8.1.0) |
| Super-resolution | Real-ESRGAN x4plus |
| Augmentation | Albumentations 1.3.1 |
| Backend | Flask 2.x, threaded |
| Streaming | MJPEG over HTTP, SSE |
| Video capture | OpenCV + FFMPEG |
| Training | Kaggle T4×2, DDP |
| Dataset API | Roboflow Python SDK |
