from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from src.io_utils import load_yaml, load_json
from src.measure import measure_gap

def load_mm_per_pixel(cfg: dict) -> float:
    mmpp = float(cfg["calibration"].get("mm_per_pixel", 0.0))
    calib_path = cfg["calibration"].get("calib_json", "")
    if calib_path:
        try:
            c = load_json(str(Path("python")/calib_path)) if not Path(calib_path).exists() else load_json(calib_path)
            mmpp = float(c.get("mm_per_pixel", mmpp))
        except Exception:
            pass
    if mmpp <= 0:
        raise ValueError("mm_per_pixel must be > 0 (set in config.yaml or calib.json)")
    return mmpp

def undistort_if_needed(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    ud = cfg["preproc"]["undistort"]
    if not ud.get("enabled", False):
        return img_bgr
    K = np.array(ud["camera_matrix"], dtype=np.float32)
    D = np.array(ud["dist_coeffs"], dtype=np.float32).reshape(-1,1)
    return cv2.undistort(img_bgr, K, D)

def run_measurement(cfg_path: str, img_bgr: np.ndarray) -> dict:
    cfg = load_yaml(cfg_path)
    rois = load_json(cfg["rois"]["path"])
    mmpp = load_mm_per_pixel(cfg)

    img_bgr = undistort_if_needed(img_bgr, cfg)
    m = measure_gap(img_bgr, rois, cfg)
    if m is None:
        return {"ok": False, "reason": "parts_not_found"}

    gap_px = float(m["gap_px"])
    gap_mm = gap_px * mmpp

    tol = cfg["tolerance"]
    ok = (gap_mm >= float(tol["gap_min_mm"])) and (gap_mm <= float(tol["gap_max_mm"]))
    decision = "PASS" if ok else "FAIL"

    return {
        "ok": True,
        "decision": decision,
        "gap_px": gap_px,
        "gap_mm": gap_mm,
        "pA": m["pA"],
        "pB": m["pB"],
    }, m.get("debug")
