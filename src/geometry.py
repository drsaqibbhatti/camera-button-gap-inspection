from __future__ import annotations
import numpy as np
import cv2

def mask_from_polygon(shape_hw, polygon_xy):
    h, w = shape_hw
    mask = np.zeros((h,w), dtype=np.uint8)
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1,1,2)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def bbox_from_mask(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1,y1,x2,y2]

def crop_by_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)
