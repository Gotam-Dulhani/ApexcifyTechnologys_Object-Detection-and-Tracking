import io
import tempfile
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import numpy as np
import cv2

app = FastAPI(title="YOLOv8 Object Detection & Tracking")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8n.onnx")

_session = None
_model_error = None


def _get_session():
    global _session, _model_error
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        return _session
    except Exception as e:
        _model_error = str(e)
        return None


COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]

CONF = 0.25
IOU = 0.45
IMG = 640


def _letterbox(img):
    h, w = img.shape[:2]
    r = min(IMG / h, IMG / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (IMG - nw) / 2, (IMG - nh) / 2
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
    lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bot, lft, rgt, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (dw, dh)


def _xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (a1 + a2 - inter + 1e-6)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


def _detect(img):
    sess = _get_session()
    if sess is None:
        raise RuntimeError(f"Model not loaded: {_model_error}")

    padded, ratio, (dw, dh) = _letterbox(img)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)

    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    pred = out[0]
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T

    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    max_s = scores_all.max(axis=1)
    cls_id = scores_all.argmax(axis=1)

    m = max_s > CONF
    boxes_xywh, max_s, cls_id = boxes_xywh[m], max_s[m], cls_id[m]

    if len(max_s) == 0:
        return img

    bxy = _xywh2xyxy(boxes_xywh)
    bxy[:, [0, 2]] = (bxy[:, [0, 2]] - dw) / ratio
    bxy[:, [1, 3]] = (bxy[:, [1, 3]] - dh) / ratio
    h_o, w_o = img.shape[:2]
    bxy[:, [0, 2]] = np.clip(bxy[:, [0, 2]], 0, w_o)
    bxy[:, [1, 3]] = np.clip(bxy[:, [1, 3]], 0, h_o)

    keep = _nms(bxy, max_s, IOU)
    bxy, max_s, cls_id = bxy[keep], max_s[keep], cls_id[keep]

    out_img = img.copy()
    for box, sc, ci in zip(bxy, max_s, cls_id):
        x1, y1, x2, y2 = map(int, box)
        label = f"{COCO[ci]} {sc:.2f}"
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out_img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out_img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out_img


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Object Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #fff; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
        h1 { margin: 2rem 0 0.5rem; font-size: 1.8rem; }
        p.subtitle { color: #aaa; margin-bottom: 2rem; }
        .upload-zone { border: 2px dashed #444; border-radius: 12px; padding: 3rem 4rem; text-align: center; cursor: pointer; transition: border-color 0.2s; width: 90%; max-width: 500px; }
        .upload-zone:hover, .upload-zone.dragover { border-color: #00aaff; }
        .upload-zone input { display: none; }
        .upload-zone label { cursor: pointer; color: #ccc; font-size: 1rem; }
        .btn { margin-top: 1.5rem; padding: 0.7rem 2.5rem; background: #00aaff; color: #fff; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
        .btn:hover { background: #0088cc; }
        .btn:disabled { background: #333; cursor: not-allowed; }
        #status { margin-top: 1rem; color: #aaa; min-height: 1.5rem; }
        .result { margin-top: 2rem; width: 90%; max-width: 700px; text-align: center; }
        .result img, .result video { max-width: 100%; border-radius: 8px; }
        .filename { color: #00aaff; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <h1>YOLOv8 Object Detection & Tracking</h1>
    <p class="subtitle">Upload an image or video to detect and track objects</p>
    <div class="upload-zone" id="dropZone">
        <label for="fileInput">
            Drag &amp; drop a file here, or click to browse<br><br>
            Supports: JPG, PNG, MP4, AVI
        </label>
        <input type="file" id="fileInput" accept="image/*,video/*">
    </div>
    <p class="filename" id="fileName"></p>
    <button class="btn" id="detectBtn" disabled>Detect Objects</button>
    <p id="status"></p>
    <div class="result" id="result"></div>
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const detectBtn = document.getElementById('detectBtn');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        const fileName = document.getElementById('fileName');
        let selectedFile = null;

        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault(); dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFile(e.dataTransfer.files[0]); }
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

        function handleFile(file) {
            selectedFile = file;
            fileName.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)';
            detectBtn.disabled = false;
            result.innerHTML = '';
            status.textContent = '';
        }

        detectBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            detectBtn.disabled = true;
            status.textContent = 'Processing... this may take a few seconds.';
            result.innerHTML = '';
            const formData = new FormData();
            formData.append('file', selectedFile);
            const isVideo = selectedFile.type.startsWith('video/');
            const endpoint = isVideo ? '/api/detect-video' : '/api/detect';
            try {
                const resp = await fetch(endpoint, { method: 'POST', body: formData });
                if (!resp.ok) { const err = await resp.text(); status.textContent = 'Error: ' + err; detectBtn.disabled = false; return; }
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                result.innerHTML = isVideo
                    ? '<video controls autoplay loop><source src="' + url + '" type="video/mp4"></video>'
                    : '<img src="' + url + '" alt="Detection result">';
                status.textContent = 'Done!';
            } catch (e) { status.textContent = 'Error: ' + e.message; }
            detectBtn.disabled = false;
        });
    </script>
</body>
</html>"""


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/health")
async def health():
    return JSONResponse({"model_loaded": _session is not None, "error": _model_error, "model_path": MODEL_PATH, "model_exists": os.path.exists(MODEL_PATH)})


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    try:
        annotated = _detect(img)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    _, buf = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        return JSONResponse({"error": "Cannot open video"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = _detect(frame)
            frames.append(annotated)
    finally:
        cap.release()

    if not frames:
        os.unlink(tmp_path)
        return JSONResponse({"error": "No frames processed"}, status_code=400)

    h, w = frames[0].shape[:2]
    tmp_out = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    with open(tmp_out, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)
    os.unlink(tmp_out)

    return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
