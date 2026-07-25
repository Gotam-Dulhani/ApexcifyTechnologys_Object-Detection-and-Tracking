from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html><head><title>YOLOv8 Object Detection</title></head>
<body><h1>YOLOv8 Object Detection</h1><p>Test page - FastAPI working!</p></body></html>"""


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/api/detect")
async def detect():
    return JSONResponse({"message": "Detection endpoint working - ML deps not yet installed"})
