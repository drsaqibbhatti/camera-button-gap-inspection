from __future__ import annotations
import argparse
import cv2
import numpy as np
from src.io_utils import save_json

# Simple polygon ROI tool: click points, press Enter to save, keys:
# 1 = select camera_module, 2 = select power_button
# Right click = undo, c = clear current, s = save, esc = exit

CURRENT_NAME = "camera_module"
POLYS = {"camera_module": [], "power_button": []}

def parse_args():
    p = argparse.ArgumentParser(description="Draw polygon ROIs for the two parts.")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="rois.json")
    return p.parse_args()

def draw(img):
    vis = img.copy()
    for name, pts in POLYS.items():
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0,255,0) if name==CURRENT_NAME else (0,255,255), thickness=2)
        for (x,y) in pts:
            cv2.circle(vis, (x,y), 4, (0,255,0) if name==CURRENT_NAME else (0,255,255), -1)
        if len(pts) >= 3:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0,255,0) if name==CURRENT_NAME else (0,255,255), thickness=2)
    cv2.putText(vis, f"Editing: {CURRENT_NAME} (press 1/2 to switch, Enter to close current)", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return vis

def on_mouse(event, x, y, flags, param):
    global CURRENT_NAME
    if event == cv2.EVENT_LBUTTONDOWN:
        POLYS[CURRENT_NAME].append([int(x), int(y)])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if POLYS[CURRENT_NAME]:
            POLYS[CURRENT_NAME].pop()

def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    cv2.namedWindow("ROI Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Tool", on_mouse)

    while True:
        vis = draw(img)
        cv2.imshow("ROI Tool", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k == ord('1'):
            CURRENT_NAME = "camera_module"
        if k == ord('2'):
            CURRENT_NAME = "power_button"
        if k == ord('c'):
            POLYS[CURRENT_NAME] = []
        if k == ord('s'):
            save_json(args.out, {"polygons": POLYS, "note": "Polygon ROIs in pixel coordinates."})
            print(f"Saved {args.out}")
        if k == 13:  # Enter: close polygon by keeping as-is; nothing special
            pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
