from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
async def root():
    return JSONResponse({"status": "ok"})


@app.get("/api/health")
async def health():
    return JSONResponse({"status": "healthy"})
