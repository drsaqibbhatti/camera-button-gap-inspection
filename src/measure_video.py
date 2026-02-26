from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from src.io_utils import load_yaml
from src.pipeline import run_measurement
from src.vis import draw_overlay

def parse_args():
    p = argparse.ArgumentParser(description="Gap measurement on a saved video.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "video_results.jsonl"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_mp4 = str(out_dir / "annotated.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0,h0))

    with jsonl.open("w", encoding="utf-8") as f:
        it = range(nframes) if nframes>0 else iter(int,1)
        for _ in tqdm(it, total=nframes if nframes>0 else None):
            ok, frame = cap.read()
            if not ok:
                break
            res, debug = run_measurement(args.config, frame)
            vis = draw_overlay(frame, res, debug, cfg)
            writer.write(vis)
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            f.write(json.dumps({"frame": fid, **res}) + "\n")

    cap.release(); writer.release()
    print(f"Saved: {out_mp4}")

if __name__ == "__main__":
    main()
