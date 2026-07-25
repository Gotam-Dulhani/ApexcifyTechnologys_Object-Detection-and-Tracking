import io
import os
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw
import numpy as np

app = FastAPI(title="YOLOv8 Object Detection & Tracking")

_imports_ok = False
_import_error = None
_model = None
_model_error = None

try:
    import onnxruntime as ort
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8n.onnx")
    if os.path.exists(MODEL_PATH):
        _model = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    else:
        _model_error = f"Model not found: {MODEL_PATH}"
    _imports_ok = True
except Exception:
    _import_error = traceback.format_exc()

IMG_SIZE = 640
CONF = 0.25
IOU_THR = 0.45
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


def _letterbox(arr):
    h, w = arr.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nw, nh = int(w * r), int(h * r)
    resized = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))
    dw, dh = (IMG_SIZE - nw) / 2, (IMG_SIZE - nh) / 2
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
    lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[top:top + nh, lft:lft + nw] = resized
    return padded, r, (dw, dh)


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


def _detect(img_np):
    padded, ratio, (dw, dh) = _letterbox(img_np)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)

    out = _model.run(None, {_model.get_inputs()[0].name: blob})[0]
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
        return img_np

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    bxy = np.stack([x1, y1, x2, y2], axis=1)
    bxy[:, [0, 2]] = (bxy[:, [0, 2]] - dw) / ratio
    bxy[:, [1, 3]] = (bxy[:, [1, 3]] - dh) / ratio
    h_o, w_o = img_np.shape[:2]
    bxy[:, [0, 2]] = np.clip(bxy[:, [0, 2]], 0, w_o)
    bxy[:, [1, 3]] = np.clip(bxy[:, [1, 3]], 0, h_o)

    keep = _nms(bxy, max_s, IOU_THR)
    bxy, max_s, cls_id = bxy[keep], max_s[keep], cls_id[keep]

    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    for box, sc, ci in zip(bxy, max_s, cls_id):
        xi1, yi1, xi2, yi2 = map(int, box)
        label = f"{COCO[ci]} {sc:.2f}"
        draw.rectangle([xi1, yi1, xi2, yi2], outline="green", width=2)
        tw = len(label) * 8 + 4
        draw.rectangle([xi1, yi1 - 18, xi1 + tw, yi1], fill="green")
        draw.text((xi1 + 2, yi1 - 16), label, fill="black")
    return np.array(pil_img)


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
        .upload-zone:hover { border-color: #00aaff; }
        .upload-zone input { display: none; }
        .upload-zone label { cursor: pointer; color: #ccc; font-size: 1rem; }
        .btn { margin-top: 1.5rem; padding: 0.7rem 2.5rem; background: #00aaff; color: #fff; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
        .btn:hover { background: #0088cc; }
        .btn:disabled { background: #333; cursor: not-allowed; }
        #status { margin-top: 1rem; color: #aaa; min-height: 1.5rem; }
        .result { margin-top: 2rem; width: 90%; max-width: 700px; text-align: center; }
        .result img { max-width: 100%; border-radius: 8px; }
        .filename { color: #00aaff; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <h1>YOLOv8 Object Detection</h1>
    <p class="subtitle">Upload an image to detect objects</p>
    <div class="upload-zone" id="dropZone">
        <label for="fileInput">Drag &amp; drop an image here, or click to browse</label>
        <input type="file" id="fileInput" accept="image/*">
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
            status.textContent = 'Processing...';
            result.innerHTML = '';
            const formData = new FormData();
            formData.append('file', selectedFile);
            try {
                const resp = await fetch('/api/detect', { method: 'POST', body: formData });
                if (!resp.ok) { const err = await resp.json(); status.textContent = 'Error: ' + (err.error || 'Unknown'); detectBtn.disabled = false; return; }
                const blob = await resp.blob();
                result.innerHTML = '<img src="' + URL.createObjectURL(blob) + '" alt="Result">';
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
    return JSONResponse({
        "imports_ok": _imports_ok,
        "import_error": _import_error,
        "model_loaded": _model is not None,
        "model_error": _model_error,
    })


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    if not _imports_ok:
        return JSONResponse({"error": f"Dependencies failed: {_import_error}"}, status_code=500)
    if _model is None:
        return JSONResponse({"error": f"Model not loaded: {_model_error}"}, status_code=500)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    try:
        annotated = _detect(img_np)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    out_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=90)
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/jpeg")
