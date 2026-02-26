# Camera Module & Power Button Gap Inspection

Vision-based **gap measurement and verification** for assembled components (camera module / power button).  
Implements a robust **geometric measurement pipeline** with automated **PASS/FAIL** against tolerance thresholds and
real-time feedback for assembly QA.

> Training/evaluation media are omitted due to data confidentiality.
https://drsaqibbhatti.com/projects/gap-inspection.html

---

## What this repo provides

- **Calibration** (mm/pixel) using a known reference distance or optional checkerboard.
- **ROI tool** to define where the camera module and power button appear.
- **Gap measurement** using classical CV (edges + morphology + contour extraction + closest-distance).
- **PASS/FAIL** decision using tolerances (in mm) from `config.yaml`.
- **Offline**: image folder + saved video
- **Real-time**: webcam/RTSP stream with overlay + FPS
- Optional **FastAPI** endpoint `/measure` for integration.

This template is designed so you can later swap in a PyTorch model (segmentation/keypoints)
without changing the UI/scripts — only `src/detect_parts.py`.

---

## Install

```bash
cd python
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Quick start

### 1) Set calibration (mm per pixel)
If you know a reference distance in the image (e.g., a gauge block length):
```bash
python -m src.calibrate --image sample.jpg --p1 100 200 --p2 540 200 --distance_mm 10.0 --out calib.json
```

Or edit `python/calib.json` manually (mm_per_pixel).

### 2) Define ROIs for the two parts (recommended)
```bash
python -m src.roi_tool --image sample.jpg --out rois.json
```

### 3) Run on images
```bash
python -m src.measure_images --config config.yaml --source path/to/images --out outputs
```

### 4) Run real-time
```bash
python -m src.realtime --config config.yaml --source 0
# or RTSP:
python -m src.realtime --config config.yaml --source "rtsp://user:pass@ip/stream"
```

---

## Configuration

Edit `python/config.yaml`:

- `calibration.mm_per_pixel` (or point to `calib.json`)
- `rois.path` (polygon ROIs for each part)
- `tolerance.gap_min_mm`, `tolerance.gap_max_mm`
- `preproc` parameters (Canny thresholds, morphology)

---

## Output

- Annotated frames show:
  - detected edges/contours
  - closest points
  - measured gap in **px** and **mm**
  - PASS/FAIL

---

## Notes on accuracy
Sub-millimeter accuracy requires:
- stable optics and fixed camera pose
- good lighting
- a reliable calibration (mm/px) for the working distance
- optionally lens distortion correction (supported in `config.yaml`)
