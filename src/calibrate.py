from __future__ import annotations
import argparse
import cv2
import numpy as np
from src.io_utils import save_json

def parse_args():
    p = argparse.ArgumentParser(description="Create mm-per-pixel calibration from a known distance.")
    p.add_argument("--image", required=True)
    p.add_argument("--p1", nargs=2, type=float, required=True, help="x y")
    p.add_argument("--p2", nargs=2, type=float, required=True, help="x y")
    p.add_argument("--distance_mm", type=float, required=True)
    p.add_argument("--out", default="calib.json")
    return p.parse_args()

def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    p1 = np.array(args.p1, dtype=np.float32)
    p2 = np.array(args.p2, dtype=np.float32)
    dist_px = float(np.linalg.norm(p2 - p1))
    if dist_px <= 0:
        raise ValueError("Invalid points: zero distance")
    mm_per_pixel = float(args.distance_mm) / dist_px

    save_json(args.out, {"mm_per_pixel": mm_per_pixel, "distance_mm": float(args.distance_mm), "dist_px": dist_px, "p1": args.p1, "p2": args.p2})
    print(f"Saved {args.out} -> mm_per_pixel={mm_per_pixel:.6f}")

if __name__ == "__main__":
    main()
