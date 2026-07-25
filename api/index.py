from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection &amp; Tracking</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root{--bg:#0a0a0b;--card:#141416;--border:#232328;--accent:#6366f1;--accent2:#818cf8;--green:#22c55e;--text:#e4e4e7;--muted:#71717a;--radius:16px}
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
        .bg-glow{position:fixed;top:-200px;left:50%;transform:translateX(-50%);width:600px;height:600px;background:radial-gradient(circle,rgba(99,102,241,.12) 0%,transparent 70%);pointer-events:none;z-index:0}
        .container{position:relative;z-index:1;max-width:800px;margin:0 auto;padding:2rem 1.5rem}
        header{text-align:center;margin-bottom:2.5rem}
        .badge{display:inline-flex;align-items:center;gap:.4rem;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);color:var(--accent2);font-size:.75rem;font-weight:600;padding:.35rem .8rem;border-radius:99px;margin-bottom:1rem;letter-spacing:.02em}
        .badge::before{content:'';width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
        h1{font-size:2.2rem;font-weight:700;letter-spacing:-.03em;margin-bottom:.5rem;background:linear-gradient(135deg,#fff 30%,var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .sub{color:var(--muted);font-size:.95rem;line-height:1.5}
        .upload-zone{position:relative;border:2px dashed var(--border);border-radius:var(--radius);padding:3rem 2rem;text-align:center;cursor:pointer;transition:all .25s;background:var(--card)}
        .upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:rgba(99,102,241,.04);box-shadow:0 0 30px rgba(99,102,241,.06)}
        .upload-zone.has-file{border-color:var(--green);background:rgba(34,197,94,.03)}
        .upload-zone input{display:none}
        .upload-icon{width:48px;height:48px;margin:0 auto 1rem;border-radius:12px;background:rgba(99,102,241,.1);display:flex;align-items:center;justify-content:center}
        .upload-icon svg{width:24px;height:24px;stroke:var(--accent2);fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
        .upload-title{font-weight:600;font-size:1rem;margin-bottom:.35rem}
        .upload-hint{color:var(--muted);font-size:.82rem}
        .file-info{margin-top:1rem;display:none;align-items:center;gap:.6rem;justify-content:center;color:var(--green);font-size:.85rem;font-weight:500}
        .file-info.show{display:flex}
        .file-info svg{width:16px;height:16px;stroke:var(--green);fill:none;stroke-width:2}
        .actions{margin-top:1.5rem;display:flex;gap:.75rem;justify-content:center}
        .btn{padding:.7rem 2rem;border-radius:12px;font-size:.9rem;font-weight:600;cursor:pointer;border:none;transition:all .2s;font-family:inherit}
        .btn-primary{background:var(--accent);color:#fff;box-shadow:0 4px 15px rgba(99,102,241,.3)}
        .btn-primary:hover:not(:disabled){background:var(--accent2);transform:translateY(-1px);box-shadow:0 6px 20px rgba(99,102,241,.4)}
        .btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
        .btn-secondary{background:var(--card);color:var(--text);border:1px solid var(--border)}
        .btn-secondary:hover{border-color:var(--muted)}
        .status{text-align:center;margin-top:1rem;min-height:1.5rem}
        .status-text{color:var(--muted);font-size:.85rem}
        .status-text.error{color:#ef4444}
        .spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(99,102,241,.3);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;margin-right:.5rem;vertical-align:middle}
        @keyframes spin{to{transform:rotate(360deg)}}
        .result{margin-top:1.5rem;display:none;animation:fadeIn .4s ease}
        .result.show{display:block}
        @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        .result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
        .result-header{padding:1rem 1.25rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
        .result-header h3{font-size:.9rem;font-weight:600}
        .result-header a{color:var(--accent2);font-size:.8rem;text-decoration:none;font-weight:500}
        .result-header a:hover{text-decoration:underline}
        .result-img{width:100%;display:block}
        .detections{padding:1rem 1.25rem}
        .det-list{display:flex;flex-wrap:wrap;gap:.4rem}
        .det-chip{display:inline-flex;align-items:center;gap:.3rem;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.15);color:var(--accent2);font-size:.75rem;font-weight:500;padding:.3rem .65rem;border-radius:8px}
        .det-chip .score{color:var(--muted);font-weight:400}
        .det-count{font-size:.8rem;color:var(--muted)}
        footer{text-align:center;margin-top:3rem;padding:1.5rem;color:var(--muted);font-size:.78rem;border-top:1px solid var(--border)}
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="container">
        <header>
            <div class="badge">YOLOv8 Live</div>
            <h1>Object Detection</h1>
            <p class="sub">Upload any image to detect and label objects in real-time using YOLOv8 neural network</p>
        </header>

        <div class="upload-zone" id="dz">
            <input type="file" id="fi" accept="image/*">
            <div class="upload-icon">
                <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <div class="upload-title">Drop your image here</div>
            <div class="upload-hint">or click to browse &middot; JPG, PNG, WebP</div>
            <div class="file-info" id="fiInfo">
                <svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                <span id="fiName"></span>
            </div>
        </div>

        <div class="actions">
            <button class="btn btn-primary" id="db" disabled>Detect Objects</button>
            <button class="btn btn-secondary" id="rb" style="display:none">Upload New</button>
        </div>

        <div class="status" id="st"></div>

        <div class="result" id="res">
            <div class="result-card">
                <div class="result-header">
                    <h3>Detection Results</h3>
                    <a id="dl" download="detection.jpg">Download</a>
                </div>
                <img class="result-img" id="ri">
                <div class="detections">
                    <div class="det-count" id="dc"></div>
                    <div class="det-list" id="dl2"></div>
                </div>
            </div>
        </div>

        <footer>Powered by YOLOv8 + ONNX Runtime &middot; Built with FastAPI</footer>
    </div>

    <script>
        const dz=document.getElementById('dz'),fi=document.getElementById('fi'),db=document.getElementById('db'),rb=document.getElementById('rb'),st=document.getElementById('st'),res=document.getElementById('res'),ri=document.getElementById('ri'),dc=document.getElementById('dc'),dl2=document.getElementById('dl2'),fiInfo=document.getElementById('fiInfo'),fiName=document.getElementById('fiName'),dlLink=document.getElementById('dl');
        let sf=null;

        dz.addEventListener('click',()=>fi.click());
        dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
        dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
        dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');if(e.dataTransfer.files.length){fi.files=e.dataTransfer.files;handleFile(e.dataTransfer.files[0])}});
        fi.addEventListener('change',()=>{if(fi.files.length)handleFile(fi.files[0])});

        function handleFile(f){
            sf=f;
            fiName.textContent=f.name+' ('+(f.size/1024/1024).toFixed(2)+' MB)';
            fiInfo.classList.add('show');
            dz.classList.add('has-file');
            db.disabled=false;
            res.classList.remove('show');
            st.innerHTML='';
            rb.style.display='none';
        }

        rb.addEventListener('click',()=>{
            sf=null;fi.value='';
            fiInfo.classList.remove('show');dz.classList.remove('has-file');
            db.disabled=true;res.classList.remove('show');st.innerHTML='';
            rb.style.display='none';
        });

        db.addEventListener('click',async()=>{
            if(!sf)return;
            db.disabled=true;
            st.innerHTML='<span class="spinner"></span><span class="status-text">Analyzing image...</span>';
            res.classList.remove('show');
            const fd=new FormData();fd.append('file',sf);
            try{
                const r=await fetch('/api/detect',{method:'POST',body:fd});
                if(!r.ok){
                    let msg='Unknown error';
                    try{const e=await r.json();msg=e.error||msg}catch(ex){}
                    st.innerHTML='<span class="status-text error">'+msg+'</span>';
                    db.disabled=false;return;
                }
                const blob=await r.blob();
                const url=URL.createObjectURL(blob);
                ri.src=url;
                dlLink.href=url;
                st.innerHTML='';
                res.classList.add('show');
                rb.style.display='inline-block';
            }catch(e){
                st.innerHTML='<span class="status-text error">Network error: '+e.message+'</span>';
                db.disabled=false;
            }
        });
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
async def detect_image(request: Request):
    import io
    import os
    import traceback
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import onnxruntime as ort
    import urllib.request as _urlreq
    from fastapi.responses import StreamingResponse

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    MODEL_URL = "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n.onnx"
    MODEL_PATH = "/tmp/yolov8n.onnx"

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        try:
            _urlreq.urlretrieve(MODEL_URL, MODEL_PATH)
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

    COLORS = [
        "#6366f1","#f43f5e","#22c55e","#f59e0b","#3b82f6","#ec4899",
        "#14b8a6","#f97316","#8b5cf6","#06b6d4","#ef4444","#10b981",
        "#eab308","#a855f7","#0ea5e9","#d946ef","#84cc16","#fb923c",
    ]

    IMG_SIZE = 640
    CONF = 0.25
    IOU_THR = 0.45

    try:
        contents = await upload.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        h_orig, w_orig = arr.shape[:2]

        r = min(IMG_SIZE / h_orig, IMG_SIZE / w_orig)
        nw, nh = int(w_orig * r), int(h_orig * r)
        resized = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))
        dw, dh = (IMG_SIZE - nw) / 2, (IMG_SIZE - nh) / 2
        top = int(round(dh - 0.1))
        lft = int(round(dw - 0.1))
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
            detected = []
            for idx, (box, sc, ci) in enumerate(zip(bxy, max_s, cls_id)):
                xi1, yi1, xi2, yi2 = map(int, box)
                color = COLORS[int(ci) % len(COLORS)]
                label = f"{COCO[int(ci)]} {sc:.0%}"
                draw.rectangle([xi1, yi1, xi2, yi2], outline=color, width=3)
                tw = len(label) * 9 + 12
                draw.rectangle([xi1, yi1 - 24, xi1 + tw, yi1], fill=color)
                try:
                    draw.text((xi1 + 6, yi1 - 20), label, fill="#ffffff")
                except Exception:
                    draw.text((xi1 + 6, yi1 - 20), label, fill="white")
                detected.append({"label": COCO[int(ci)], "confidence": float(sc)})
        else:
            detected = []

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)
