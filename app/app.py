"""
app.py — CamouflageNet Surveillance Dashboard
Flask backend supporting:
  - Image upload detection
  - Video file processing
  - Webcam MJPEG stream
  - RTSP / IP camera stream
  - RTMP drone stream
  - Generic network surveillance feed
  - Server-Sent Events for real-time stats
  - Automated JSONL event logging
"""

import os
import cv2
import json
import time
import uuid
import queue
import threading
import datetime
import numpy as np
from pathlib import Path
from flask import (
    Flask, render_template, request,
    Response, jsonify, send_from_directory,
    stream_with_context
)
from ultralytics import YOLO
from esrgan_enhance import ESRGANEnhancer

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB upload cap

BASE_DIR   = Path(__file__).parent
LOG_DIR    = BASE_DIR / 'logs'
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
for d in [LOG_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Global State
# ─────────────────────────────────────────────────────────────────────────────
model:    YOLO            = None
enhancer: ESRGANEnhancer  = None

# Live-stream control
stream_active = False
stream_lock   = threading.Lock()

# Stats broadcast to SSE clients
_stats = {
    'fps': 0.0, 'total': 0, 'current': 0,
    'avg_conf': 0.0, 'source': 'none', 'status': 'idle'
}
_stats_lock = threading.Lock()

# SSE client message queues
_sse_queues: list = []
_sse_lock   = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Startup: load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str = None) -> None:
    global model, enhancer

    target = path or os.environ.get('MODEL_PATH', 'best.pt')

    if not Path(target).exists():
        # Graceful fallback: use pretrained model if custom weights absent
        print(f"[WARN] '{target}' not found — loading pretrained yolov8m-seg.pt")
        target = 'yolov8m-seg.pt'

    print(f"[INFO] Loading YOLO model: {target}")
    model = YOLO(target)
    print("[INFO] Model loaded.")

    print("[INFO] Initialising Real-ESRGAN enhancer...")
    enhancer = ESRGANEnhancer()
    print(f"[INFO] ESRGAN available: {enhancer.available}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _infer(frame: np.ndarray,
           conf: float = 0.25,
           iou:  float = 0.45):
    """
    Run YOLO segmentation inference on one frame.

    ⚠ CPU WARNING: Without CUDA, each call may take 1-5 seconds.
    For real-time streaming, a GPU (RTX 3060 or better) is required.

    Low-resolution frames (< 320 px wide) are enhanced with Real-ESRGAN
    before inference to improve detection of distant/small targets.
    """
    if enhancer.available and frame.shape[1] < 320:
        frame = enhancer.enhance(frame)
    return model(frame, conf=conf, iou=iou, verbose=False)[0]


def _parse_detections(result) -> list:
    """Convert YOLO result to clean list of dicts."""
    out = []
    if result.boxes is None:
        return out
    for i, box in enumerate(result.boxes):
        out.append({
            'id':         i,
            'class':      result.names[int(box.cls.item())],
            'confidence': round(float(box.conf.item()), 4),
            'bbox':       [round(float(v), 2) for v in box.xyxy[0].tolist()],
            'has_mask':   result.masks is not None,
        })
    return out


def _log_event(detections: list, source: str, frame_id: str = None) -> None:
    """Append detection event to today's JSONL log and push to SSE clients."""
    if not detections:
        return
    event = {
        'timestamp':       datetime.datetime.utcnow().isoformat(),
        'frame_id':        frame_id or str(uuid.uuid4())[:8],
        'source':          source,
        'detection_count': len(detections),
        'detections':      detections,
    }
    log_file = LOG_DIR / f"detections_{datetime.date.today()}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(event) + '\n')
    _push_sse('detection', event)


def _update_stats(fps: float, detections: list, source: str) -> None:
    with _stats_lock:
        _stats['fps']     = round(fps, 1)
        _stats['current'] = len(detections)
        _stats['total']  += len(detections)
        _stats['source']  = source
        _stats['status']  = 'active'
        if detections:
            _stats['avg_conf'] = round(
                sum(d['confidence'] for d in detections) / len(detections), 4
            )
    _push_sse('stats', dict(_stats))


def _push_sse(event_type: str, data: dict) -> None:
    """Fan-out a message to all connected SSE clients."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG Frame Generator  (used by all live-stream routes)
# ─────────────────────────────────────────────────────────────────────────────
def _mjpeg_generator(cap: cv2.VideoCapture, label: str):
    """
    Generator that reads frames from any cv2.VideoCapture source,
    runs YOLO inference, annotates, and yields MJPEG bytes.

    Yields multipart/x-mixed-replace frames compatible with <img src=...>.
    """
    global stream_active
    t_prev = time.time()

    try:
        while stream_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Run segmentation inference
            result     = _infer(frame)
            detections = _parse_detections(result)

            # Compute FPS
            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # Update stats + log (async-safe — just dict writes)
            _update_stats(fps, detections, label)
            _log_event(detections, label)

            # Draw segmentation masks, bounding boxes, confidence scores
            annotated = result.plot(
                boxes=True, masks=True, conf=True, line_width=2
            )

            # Overlay HUD: FPS + detection count
            cv2.putText(
                annotated,
                f"FPS {fps:.1f}  |  Targets: {len(detections)}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 255, 100), 2, cv2.LINE_AA
            )

            # Encode as JPEG (quality 75 balances bandwidth vs clarity)
            ok, buf = cv2.imencode(
                '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )

    finally:
        cap.release()
        with _stats_lock:
            _stats['status'] = 'idle'
        _push_sse('stats', dict(_stats))


def _open_stream(url, label: str) -> Response:
    """Helper: open a VideoCapture and return a streaming Response."""
    global stream_active
    with stream_lock:
        stream_active = True

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        with stream_lock:
            stream_active = False
        return jsonify({'error': f'Cannot open stream: {url}'}), 400

    return Response(
        stream_with_context(_mjpeg_generator(cap, label)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Static Detection (image / video file)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Upload an image, run detection, return annotated result URL + detections."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    conf = float(request.form.get('conf', 0.25))
    iou  = float(request.form.get('iou',  0.45))

    # Save upload
    fname   = f"{uuid.uuid4().hex}_{file.filename}"
    up_path = UPLOAD_DIR / fname
    file.save(up_path)

    frame = cv2.imread(str(up_path))
    if frame is None:
        return jsonify({'error': 'Cannot read image file'}), 400

    result     = _infer(frame, conf=conf, iou=iou)
    detections = _parse_detections(result)

    # Save annotated output
    annotated  = result.plot(boxes=True, masks=True, conf=True, line_width=2)
    out_name   = f"result_{fname}"
    cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)

    _log_event(detections, 'image_upload', fname)

    return jsonify({
        'success':         True,
        'detection_count': len(detections),
        'detections':      detections,
        'result_url':      f'/outputs/{out_name}',
    })


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """
    Upload a video file. Process frame-by-frame and return annotated MP4.
    Long videos can take several minutes without GPU.

    ⚠ CPU WARNING: ~1-5 seconds per frame on CPU. Use GPU for any video > 30s.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file  = request.files['file']
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    file.save(UPLOAD_DIR / fname)

    cap = cv2.VideoCapture(str(UPLOAD_DIR / fname))
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 400

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = f"result_{Path(fname).stem}.mp4"
    out_path = OUTPUT_DIR / out_name
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h)
    )

    all_dets   = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result     = _infer(frame)
        detections = _parse_detections(result)
        all_dets.extend(detections)
        annotated  = result.plot(boxes=True, masks=True, conf=True)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    _log_event(all_dets, 'video_upload', fname)

    return jsonify({
        'success':          True,
        'frames_processed': frame_count,
        'total_detections': len(all_dets),
        'result_url':       f'/outputs/{out_name}',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Live Streams (MJPEG)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/stream/webcam')
def stream_webcam():
    """Local webcam (device index 0). No URL parameter needed."""
    global stream_active
    with stream_lock:
        stream_active = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with stream_lock:
            stream_active = False
        return jsonify({'error': 'Webcam not found (device 0)'}), 404

    # Request 1280×720 from hardware; driver will do its best
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return Response(
        stream_with_context(_mjpeg_generator(cap, 'webcam')),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stream/rtsp')
def stream_rtsp():
    """
    RTSP camera stream.
    Query param: ?url=rtsp://user:pass@192.168.1.1:554/stream1
    """
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'Missing ?url= parameter'}), 400
    return _open_stream(url, 'rtsp')


@app.route('/api/stream/ip')
def stream_ip():
    """
    Generic IP camera (HTTP MJPEG, RTSP, ONVIF).
    Query param: ?url=http://192.168.1.100:8080/video
    """
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'Missing ?url= parameter'}), 400
    return _open_stream(url, 'ip_camera')


@app.route('/api/stream/rtmp')
def stream_rtmp():
    """
    RTMP drone / broadcasting stream (pull mode).
    Query param: ?url=rtmp://localhost/live/drone1

    ⚠ RTMP PUSH: Drones push to an RTMP ingest server.
       Set up nginx-rtmp, then pull from it:
       rtmp://localhost/live/<stream_key>
    """
    url = request.args.get('url', 'rtmp://localhost/live/stream')
    return _open_stream(url, 'rtmp_drone')


@app.route('/api/stream/network')
def stream_network():
    """
    Generic network surveillance feed (RTSP/RTMP/HTTP).
    Query param: ?url=<any OpenCV-compatible URL>
    """
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'Missing ?url= parameter'}), 400
    return _open_stream(url, 'network_feed')


@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Signal the active stream to stop. The generator will exit on next iteration."""
    global stream_active
    with stream_lock:
        stream_active = False
    return jsonify({'success': True})


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Server-Sent Events
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/events')
def sse_events():
    """
    SSE endpoint — pushes real-time stats and detection events.
    Connect from JS:  const es = new EventSource('/api/events');
    """
    def _stream():
        q = queue.Queue(maxsize=100)
        with _sse_lock:
            _sse_queues.append(q)

        # Send current stats immediately on connect
        yield f"event: stats\ndata: {json.dumps(_stats)}\n\n"

        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ': keepalive\n\n'   # prevent proxy timeouts
        finally:
            with _sse_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    return Response(
        stream_with_context(_stream()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Data / Files
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/logs')
def get_logs():
    """Return the most recent N log entries (default 50)."""
    limit    = int(request.args.get('limit', 50))
    log_file = LOG_DIR / f"detections_{datetime.date.today()}.jsonl"

    if not log_file.exists():
        return jsonify({'logs': [], 'total': 0})

    raw   = [l for l in log_file.read_text().strip().split('\n') if l.strip()]
    total = len(raw)
    logs  = []
    for line in reversed(raw[-limit:]):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return jsonify({'logs': logs, 'total': total})


@app.route('/api/stats')
def get_stats():
    """Return current detection statistics (snapshot)."""
    with _stats_lock:
        return jsonify(dict(_stats))


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve processed images and videos from the outputs directory."""
    return send_from_directory(str(OUTPUT_DIR), filename)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    # threaded=True is required so MJPEG + SSE streams run concurrently
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
