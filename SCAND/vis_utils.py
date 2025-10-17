
import numpy as np
import math
from typing import Tuple, Optional
import cv2
import json

def point_to_traj(r: float,
                theta: float,
                T_horizon: float,
                num: int,
                x_base: float,
                y_base: float) -> Tuple[np.ndarray, float, float]:

    T_end = T_horizon
    w = theta / T_end if T_end > 1e-9 else 0.0
    v = r*w if theta !=0.0 else x_base / T_end

    t = np.linspace(0.0, T_end, num)
    x = r * np.sin(w * t) if theta != 0.0 else (x_base / T_end) * t
    y = r * (1.0 - np.cos(w * t)) 
    z = np.zeros_like(x)
    pts_b = np.stack([x, y, z], axis=1)
    theta_samples = w * t
    return pts_b, v, w, t, theta_samples

def make_corridor_polygon(traj_b: np.ndarray,
                          theta_samples: np.ndarray,
                          width_m: float, 
                          bridge_pts: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given centerline (N,3) and heading samples (N,), create left/right offsets and a closed polygon.
    width_m: robot width; offsets at ±width_m/2.
    Returns:
      left_b, right_b: (N,3) in base_link
      poly_b: (2N,3) polygon points (left forward then right backward)
    """
    d = width_m * 0.5
    x = traj_b[:, 0]
    y = traj_b[:, 1]
    # normal to heading (x-forward, y-left):
    n_x = -np.sin(theta_samples)
    n_y =  np.cos(theta_samples)

    xL = x + d * n_x
    yL = y + d * n_y
    xR = x - d * n_x
    yR = y - d * n_y

    z = np.zeros_like(x)
    left_b  = np.stack([xL, yL, z], axis=1)
    right_b = np.stack([xR, yR, z], axis=1)

    if bridge_pts > 0:
        bx = np.linspace(xL[-1], xR[-1], bridge_pts)
        by = np.linspace(yL[-1], yR[-1], bridge_pts)
        bridge_end = np.stack([bx, by, np.zeros_like(bx)], axis=1)
    # Build polygon: left (0→N-1) + right (N-1→0)
    poly_b = np.vstack([left_b, bridge_end,right_b[::-1]])
    return left_b, right_b, poly_b

def draw_polyline(img: np.ndarray, pts2d: np.ndarray, thickness: int, color):
    H, W = img.shape[:2]
    poly = []
    for (uu, vv) in pts2d:
        ui, vi = int(round(uu)), int(round(vv))
        if 0 <= ui < W and 0 <= vi < H:
            poly.append((ui, vi))
    for i in range(len(poly) - 1):
        cv2.line(img, poly[i], poly[i + 1], color, thickness, lineType=cv2.LINE_AA)

def draw_corridor(img: np.ndarray, poly_2d: np.ndarray, left_2d: np.ndarray, right_2d: np.ndarray,
                  fill_alpha: float = 0.35,
                  fill_color = (0,0,255),   # BGR
                  edge_color = (0,0,200),
                  edge_thickness: int = 2,):
    H, W = img.shape[:2]
    # Clip to image bounds
    def clip_pts(uv, polygon=False):
        pts = []
        for (u,v) in uv:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                pts.append([ui, vi])
        if polygon and len(pts) >= 3:
            # Ensure polygon is closed
            if pts[0] != pts[-1]:
                pts.append(pts[0])

        return np.array(pts, dtype=np.int32)

    poly = clip_pts(poly_2d, polygon=True)
    L = clip_pts(left_2d)
    R = clip_pts(right_2d)

    if len(poly) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], fill_color)
        img[:] = cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0)

    if len(L) >= 2:
        cv2.polylines(img, [L], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    if len(R) >= 2:
        cv2.polylines(img, [R], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

def project_points_cam(K: np.ndarray, dist, P_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points to pixels. No distortion if dist is None."""
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(P_cam.astype(np.float64), rvec, tvec, K, None)
    return pts2d.reshape(-1, 2)

def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points; returns Nx3."""
    assert T.shape == (4, 4)
    N = P.shape[0]
    Ph = np.hstack([P, np.ones((N, 1))])
    Qh = (T @ Ph.T).T
    return Qh[:, :3]

def load_calibration(json_path: str, fx: float, fy: float, cx: float, cy: float, mode: str = "jackal"):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    entry = data.get(mode, None)
    if entry is None or "H_cam_bl" not in entry:
        raise ValueError(f"Missing '{mode}' in {json_path}")

    h = entry["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [ 0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll),  math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3]  = np.array([xt, yt, zt], dtype=np.float64)

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam


def camray_to_ground_in_base(u: float, v: float,
                             K: np.ndarray,
                             T_b_from_c: np.ndarray) -> Optional[np.ndarray]:
    """
    Pixel (u,v) --> ray in cam --> transform to base --> intersect ground z=0 (base_link).
    """
    # print(u, v)
    # T_b_from_c = np.linalg.inv(T_cam_from_base)

    R_b_from_c = T_b_from_c[:3, :3]
    t_b_from_c = T_b_from_c[:3, 3]

    O_b = t_b_from_c  # camera origin in base
    rc = pixel_to_ray_cam(u, v, K)
    dir_b = R_b_from_c @ rc

    if abs(dir_b[2]) < 1e-8:
        return None
    s = -O_b[2] / dir_b[2]
    if s <= 0:
        return None

    P_b = O_b + s * dir_b
    return P_b  # (x,y,0)

def pixel_to_ray_cam(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Back-project pixel to a unit ray in camera frame (ignores distortion)."""
    Kinv = np.linalg.inv(K)
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    rc = Kinv @ uv1
    return rc / np.linalg.norm(rc)