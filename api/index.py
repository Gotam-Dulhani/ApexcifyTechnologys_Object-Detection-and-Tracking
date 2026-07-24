import io
import tempfile
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="YOLOv8 Object Detection & Tracking")

model = YOLO("yolov8n.pt")

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


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return StreamingResponse(io.BytesIO(b""), status_code=400, media_type="text/plain")

    results = model(img)
    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        return StreamingResponse(io.BytesIO(b""), status_code=400, media_type="text/plain")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True)
        annotated = results[0].plot()
        frames.append(annotated)
    cap.release()

    if not frames:
        os.unlink(tmp_path)
        return StreamingResponse(io.BytesIO(b""), status_code=400, media_type="text/plain")

    h, w = frames[0].shape[:2]
    tmp_out = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    with open(tmp_out, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)
    os.unlink(tmp_out)

    return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
