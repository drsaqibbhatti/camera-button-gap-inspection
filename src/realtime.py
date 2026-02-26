from __future__ import annotations
import argparse, time
import cv2
from src.io_utils import load_yaml
from src.pipeline import run_measurement
from src.vis import draw_overlay

def parse_args():
    p = argparse.ArgumentParser(description="Real-time gap measurement.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default="0", help="0 webcam | rtsp://... | path/to/video")
    p.add_argument("--record", default="", help="optional output mp4 path")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    if args.record:
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0,h0))

    prev = time.time(); fps_ema = 0.0
    cv2.namedWindow("Gap Inspection", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res, debug = run_measurement(args.config, frame)
        vis = draw_overlay(frame, res, debug, cfg)

        now = time.time()
        dt = now - prev
        prev = now
        fps = 1.0 / max(dt, 1e-6)
        fps_ema = fps if fps_ema == 0 else 0.9*fps_ema + 0.1*fps
        cv2.putText(vis, f"FPS: {fps_ema:.1f}", (20, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if writer is not None:
            writer.write(vis)

        cv2.imshow("Gap Inspection", vis)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
