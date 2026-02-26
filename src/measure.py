from __future__ import annotations
import numpy as np
import cv2
from src.geometry import mask_from_polygon, crop_by_mask

def preprocess_edges(img_bgr, cfg):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = int(cfg["preproc"]["gray_blur_ksize"])
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k|1, k|1), 0)
    edges = cv2.Canny(gray, int(cfg["preproc"]["canny_low"]), int(cfg["preproc"]["canny_high"]))
    k2 = int(cfg["preproc"]["morph_kernel"])
    it = int(cfg["preproc"]["morph_iters"])
    if k2 > 0 and it > 0:
        kernel = np.ones((k2,k2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=it)
    return edges

def largest_contour(edge_u8):
    cnts, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts[0]

def contour_points(cnt):
    return cnt.reshape(-1,2).astype(np.float32)

def closest_points_between_contours(cntA, cntB, direction="auto"):
    # brute force (fast enough for typical contour sizes); optimize if needed
    A = contour_points(cntA)
    B = contour_points(cntB)
    # If we know expected direction, reduce comparisons
    if direction in ("x","y"):
        # sample points to reduce compute
        stepA = max(1, A.shape[0]//400)
        stepB = max(1, B.shape[0]//400)
        A = A[::stepA]
        B = B[::stepB]
    # Compute pairwise squared distances
    d2 = ((A[:,None,:] - B[None,:,:])**2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    pA = A[i]
    pB = B[j]
    dist = float(np.sqrt(d2[i,j]))
    # directional gap (signed)
    if direction == "x":
        gap = float(abs(pB[0]-pA[0]))
    elif direction == "y":
        gap = float(abs(pB[1]-pA[1]))
    else:
        gap = dist
    return pA, pB, dist, gap

def bbox_gap(cntA, cntB, direction="x"):
    xA,yA,wA,hA = cv2.boundingRect(cntA)
    xB,yB,wB,hB = cv2.boundingRect(cntB)
    if direction == "x":
        # assume A left of B (swap if not)
        if xA > xB:
            xA,yA,wA,hA, xB,yB,wB,hB = xB,yB,wB,hB, xA,yA,wA,hA
        gap = max(0, xB - (xA + wA))
        return gap
    if direction == "y":
        if yA > yB:
            xA,yA,wA,hA, xB,yB,wB,hB = xB,yB,wB,hB, xA,yA,wA,hA
        gap = max(0, yB - (yA + hA))
        return gap
    # auto: return euclidean between bbox centers
    cA = np.array([xA+wA/2, yA+hA/2], dtype=np.float32)
    cB = np.array([xB+wB/2, yB+hB/2], dtype=np.float32)
    return float(np.linalg.norm(cB-cA))

def measure_gap(img_bgr, rois, cfg):
    h,w = img_bgr.shape[:2]
    polyA = rois["polygons"][cfg["rois"]["part_a_name"]]
    polyB = rois["polygons"][cfg["rois"]["part_b_name"]]
    maskA = mask_from_polygon((h,w), polyA)
    maskB = mask_from_polygon((h,w), polyB)

    edges = preprocess_edges(img_bgr, cfg)
    edgesA = cv2.bitwise_and(edges, edges, mask=maskA)
    edgesB = cv2.bitwise_and(edges, edges, mask=maskB)

    cntA = largest_contour(edgesA)
    cntB = largest_contour(edgesB)
    if cntA is None or cntB is None:
        return None

    direction = cfg["measurement"]["direction"]
    method = cfg["measurement"]["method"]

    debug = {"edges": edges, "edgesA": edgesA, "edgesB": edgesB, "cntA": cntA, "cntB": cntB}

    if method == "bbox_gap":
        gap_px = bbox_gap(cntA, cntB, direction=direction if direction in ("x","y") else "x")
        pA=pB=None; dist=gap_px
    else:
        pA, pB, dist, gap_px = closest_points_between_contours(cntA, cntB, direction=direction)

    return {"gap_px": float(gap_px), "pA": None if pA is None else pA.tolist(), "pB": None if pB is None else pB.tolist(), "debug": debug}
