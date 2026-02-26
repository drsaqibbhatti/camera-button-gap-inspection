from __future__ import annotations
import argparse, base64
from pathlib import Path
import yaml
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from src.pipeline import run_measurement
from src.io_utils import load_yaml
from src.vis import draw_overlay

app = FastAPI(title="Gap Inspection API")
CFG = None
CONFIG_PATH = None

class MeasureResponse(BaseModel):
    ok: bool
    decision: str | None = None
    gap_px: float | None = None
    gap_mm: float | None = None
    reason: str | None = None
    overlay_jpg_base64: str | None = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()

def bytes_to_bgr(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/measure", response_model=MeasureResponse)
async def measure(file: UploadFile = File(...), return_overlay: bool = False):
    img = bytes_to_bgr(await file.read())
    res, debug = run_measurement(CONFIG_PATH, img)

    resp = {**res, "overlay_jpg_base64": None}
    if return_overlay:
        vis = draw_overlay(img, res, debug, CFG)
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            resp["overlay_jpg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return resp

def main():
    global CFG, CONFIG_PATH
    args = parse_args()
    CONFIG_PATH = args.config
    CFG = load_yaml(CONFIG_PATH)
    import uvicorn
    uvicorn.run("src.server:app", host=str(CFG["server"]["host"]), port=int(CFG["server"]["port"]), reload=False)

if __name__ == "__main__":
    main()
