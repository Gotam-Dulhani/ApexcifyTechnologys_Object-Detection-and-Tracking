from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

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
        .zone:hover,.zone.dragover{border-color:#00aaff;background:rgba(0,170,255,.05)}
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
async def detect():
    return JSONResponse({"message": "Detection endpoint working - ML deps not yet installed"})
