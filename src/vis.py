from __future__ import annotations
import cv2
import numpy as np

def draw_overlay(img_bgr, result, debug=None, cfg=None):
    out = img_bgr.copy()
    if not result.get("ok", False):
        cv2.putText(out, f"FAIL: {result.get('reason','error')}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return out

    decision = result["decision"]
    gap_mm = result["gap_mm"]
    gap_px = result["gap_px"]

    color = (0,255,0) if decision=="PASS" else (0,0,255)
    cv2.putText(out, f"{decision}  gap={gap_mm:.3f}mm ({gap_px:.1f}px)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if result.get("pA") and result.get("pB"):
        pA = tuple(map(int, result["pA"]))
        pB = tuple(map(int, result["pB"]))
        cv2.circle(out, pA, 6, (255,0,0), -1)
        cv2.circle(out, pB, 6, (0,0,255), -1)
        cv2.line(out, pA, pB, (0,255,255), 2)

    if debug is not None and cfg and cfg["runtime"].get("draw_debug", True):
        # draw contours
        cntA = debug.get("cntA"); cntB = debug.get("cntB")
        if cntA is not None:
            cv2.drawContours(out, [cntA], -1, (0,255,255), 2)
        if cntB is not None:
            cv2.drawContours(out, [cntB], -1, (255,255,0), 2)

    return out
