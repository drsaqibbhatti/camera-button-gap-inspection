from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from src.io_utils import load_yaml
from src.pipeline import run_measurement
from src.vis import draw_overlay

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def parse_args():
    p = argparse.ArgumentParser(description="Gap measurement on images/folder.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="outputs")
    return p.parse_args()

def iter_images(src: Path):
    if src.is_file() and src.suffix.lower() in IMG_EXTS:
        yield src
    elif src.is_dir():
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        raise ValueError(f"Unsupported source: {src}")

def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "results.jsonl"
    cfg = load_yaml(args.config)

    with jsonl.open("w", encoding="utf-8") as f:
        for p in tqdm(list(iter_images(Path(args.source)))):
            img = cv2.imread(str(p))
            if img is None:
                continue
            res, debug = run_measurement(args.config, img)
            vis = draw_overlay(img, res, debug, cfg)
            cv2.imwrite(str(out_dir / f"{p.stem}_gap.jpg"), vis)
            f.write(json.dumps({"file": str(p), **res}) + "\n")

    print(f"Saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
