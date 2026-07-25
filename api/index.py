import io
import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Object Detection</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Segoe UI',sans-serif;background:#0f0f0f;color:#fff;min-height:100vh;display:flex;flex-direction:column;align-items:center}
        h1{margin:2rem 0 .5rem;font-size:1.8rem}
        .sub{color:#aaa;margin-bottom:2rem}
        .zone{border:2px dashed #444;border-radius:12px;padding:3rem;text-align:center;cursor:pointer;width:90%;max-width:500px;transition:border-color .2s}
        .zone:hover{border-color:#00aaff}
        .zone.dragover{border-color:#00aaff;background:rgba(0,170,255,.05)}
        .zone input{display:none}
        .zone label{cursor:pointer;color:#ccc;font-size:1rem}
        .btn{margin-top:1.5rem;padding:.7rem 2.5rem;background:#00aaff;color:#fff;border:none;border-radius:8px;font-size:1rem;cursor:pointer}
        .btn:hover{background:#0088cc}
        .btn:disabled{background:#333;cursor:not-allowed}
        #st{margin-top:1rem;color:#aaa;min-height:1.5rem}
        .res{margin-top:2rem;width:90%;max-width:700px;text-align:center}
        .res img{max-width:100%;border-radius:8px}
        .fn{color:#00aaff;margin-top:.5rem}
    </style>
</head>
<body>
    <h1>YOLOv8 Object Detection</h1>
    <p class="sub">Upload an image to detect objects</p>
    <div class="zone" id="dz">
        <label for="fi">Drag &amp; drop an image here, or click to browse</label>
        <input type="file" id="fi" accept="image/*">
    </div>
    <p class="fn" id="fn"></p>
    <button class="btn" id="db" disabled>Detect Objects</button>
    <p id="st"></p>
    <div class="res" id="res"></div>
    <script>
        const dz=document.getElementById('dz'),fi=document.getElementById('fi'),db=document.getElementById('db'),st=document.getElementById('st'),res=document.getElementById('res'),fn=document.getElementById('fn');let sf=null;
        dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
        dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
        dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');if(e.dataTransfer.files.length){fi.files=e.dataTransfer.files;hf(e.dataTransfer.files[0])}});
        fi.addEventListener('change',()=>{if(fi.files.length)hf(fi.files[0])});
        function hf(f){sf=f;fn.textContent=f.name+' ('+(f.size/1024/1024).toFixed(2)+' MB)';db.disabled=false;res.innerHTML='';st.textContent=''}
        db.addEventListener('click',async()=>{if(!sf)return;db.disabled=true;st.textContent='Processing...';res.innerHTML='';const fd=new FormData();fd.append('file',sf);try{const r=await fetch('/api/detect',{method:'POST',body:fd});if(!r.ok){const e=await r.json();st.textContent='Error: '+(e.error||'Unknown');db.disabled=false;return}const b=await r.blob();res.innerHTML='<img src="'+URL.createObjectURL(b)+'" alt="Result">';st.textContent='Done!'}catch(e){st.textContent='Error: '+e.message}db.disabled=false});
    </script>
</body>
</html>"""


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import onnxruntime as ort
    except ImportError as e:
        return JSONResponse({"error": f"Missing dependency: {e}"}, status_code=500)

    MODEL_URL = "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n.onnx"
    MODEL_PATH = "/tmp/yolov8n.onnx"

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            return JSONResponse({"error": f"Failed to download model: {e}"}, status_code=500)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        return JSONResponse({"error": "Model file not available"}, status_code=500)

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

    IMG_SIZE = 640
    CONF = 0.25
    IOU_THR = 0.45

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        h_orig, w_orig = arr.shape[:2]

        r = min(IMG_SIZE / h_orig, IMG_SIZE / w_orig)
        nw, nh = int(w_orig * r), int(h_orig * r)
        resized = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))
        dw, dh = (IMG_SIZE - nw) / 2, (IMG_SIZE - nh) / 2
        top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
        lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        padded[top:top + nh, lft:lft + nw] = resized

        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
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

        pil_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_img)

        if len(max_s) > 0:
            x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
            bxy = np.stack([x1, y1, x2, y2], axis=1)

            bxy[:, [0, 2]] = (bxy[:, [0, 2]] - dw) / r
            bxy[:, [1, 3]] = (bxy[:, [1, 3]] - dh) / r
            bxy[:, [0, 2]] = np.clip(bxy[:, [0, 2]], 0, w_orig)
            bxy[:, [1, 3]] = np.clip(bxy[:, [1, 3]], 0, h_orig)

            order = max_s.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break
                xx1 = np.maximum(bxy[i, 0], bxy[order[1:], 0])
                yy1 = np.maximum(bxy[i, 1], bxy[order[1:], 1])
                xx2 = np.minimum(bxy[i, 2], bxy[order[1:], 2])
                yy2 = np.minimum(bxy[i, 3], bxy[order[1:], 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                a1 = (bxy[i, 2] - bxy[i, 0]) * (bxy[i, 3] - bxy[i, 1])
                a2 = (bxy[order[1:], 2] - bxy[order[1:], 0]) * (bxy[order[1:], 3] - bxy[order[1:], 1])
                iou = inter / (a1 + a2 - inter + 1e-6)
                order = order[np.where(iou <= IOU_THR)[0] + 1]

            bxy, max_s, cls_id = bxy[keep], max_s[keep], cls_id[keep]
            for box, sc, ci in zip(bxy, max_s, cls_id):
                xi1, yi1, xi2, yi2 = map(int, box)
                label = f"{COCO[int(ci)]} {sc:.2f}"
                draw.rectangle([xi1, yi1, xi2, yi2], outline="green", width=2)
                tw = len(label) * 8 + 4
                draw.rectangle([xi1, yi1 - 18, xi1 + tw, yi1], fill="green")
                draw.text((xi1 + 2, yi1 - 16), label, fill="black")

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/jpeg")

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)
