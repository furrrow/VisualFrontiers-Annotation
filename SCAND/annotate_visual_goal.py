#!/usr/bin/env python3

import os
import csv
import glob
import math
import random
from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np

from dataclasses import dataclass

# --- ROS1 imports ---
import rosbag
from cv_bridge import CvBridge

# ===========================
# Configs
# ===========================

bag_dir = "SCAND/rosbags"   # Point to path with rosbags being annotated for the day
bag_idx = 4
annotate_n_bags = 1
annotations_root = "./SCAND/Annotations"
calib_path = "./SCAND/tf.json"
skip_json_path = "./SCAND/bags_to_skip.json"

fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
T_horizon = 2.0      # Path generation options
num_t_samples = 1000
robot_width_min = 0.35
robot_width_max = 0.7
undersampling_factor = 6

# Colors (BGR)
COLOR_PATH = (0, 0, 255)    # RED
COLOR_LAST = (0, 165, 255)  # ORANGE
COLOR_CLICK = (255, 0, 0)   # BLUE


# ===========================
# Camera & Geometry helpers
# ===========================
def load_calibration(json_path: str, mode: str = "jackal"):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    entry = data.get(mode, None)
    if entry is None or "H_cam_bl" not in entry:
        raise ValueError(f"Missing '{mode}.H_cam_bl' in {json_path}")

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


def pixel_to_ray_cam(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Back-project pixel to a unit ray in camera frame (ignores distortion)."""
    Kinv = np.linalg.inv(K)
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    rc = Kinv @ uv1
    return rc / np.linalg.norm(rc)

def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points; returns Nx3."""
    assert T.shape == (4, 4)
    N = P.shape[0]
    Ph = np.hstack([P, np.ones((N, 1))])
    Qh = (T @ Ph.T).T
    return Qh[:, :3]


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


def solve_arc_from_point(x: float, y: float) -> Optional[Tuple[float, float]]:
    """
    Given target (x,y) in base_link, solve for (r, theta):
      x = r sinθ,  y = r (1 - cosθ)
      => r = (x^2 + y^2)/(2y),  θ = 2 atan2(y, x)
    """
    if abs(y) < 1e-6:
        # Straight line ahead
        r = 1e9
        theta = 0.0
        return r, theta
    r = (x*x + y*y) / (2.0*y)
    theta = 2.0 * math.atan2(y, x)   # Correct half-angle form
    return r, theta


def arc_to_traj(r: float,
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

def project_points_cam(K: np.ndarray, dist, P_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points to pixels. No distortion if dist is None."""
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(P_cam.astype(np.float64), rvec, tvec, K, None)
    return pts2d.reshape(-1, 2)


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


@dataclass
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img: np.ndarray
    
# ===========================
# Main Annotator
# ===========================
class Annotator:
    def __init__(self):
        self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
        self.bridge = CvBridge()

        self.window_name = "SCAND Annotator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        self.current_img = None
        self.current_img_show = None
        self.image_topic = None

        self.latest_pts2d = None
        self.latest_left2d  = None
        self.latest_right2d = None
        self.latest_poly2d  = None
        self.robot_width = None

        self.last_carry_pts2d = None
        self.last_carry_left2d  = None
        self.last_carry_right2d = None
        self.last_carry_poly2d  = None
        self.last_carry_robot_width = None

        self.current_click_uv = None
        self.current_target_base = None   # (x,y,0) in base_link
        self.current_r_theta_vw = None    # (r, theta, v, w)

        self.writer = None

        self.last_target_base = None
        self.last_selection_record = None  # (r,θ,v,ω,thick)
        self.last_click_uv = None
        self.frame_idx = -1
        self.bag_name = ""
        self.frame_stamp = None

        self.bag_doc = None
        self.frames : None
        self.output_path = None

    def _open_bag_doc(self):
        self.bag_doc = {
            "bag": self.bag_name,
            "image_topic": self.image_topic,
            "annotations_by_stamp": {}
        }

    def _close_bag_doc(self):
        if self.bag_doc is not None:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.bag_doc, f, ensure_ascii=False, indent=2)
            self.bag_doc = None
    
    def _clear_last(self):
        self.last_click_uv = None
        self.last_target_base = None
        self.last_selection_record = None
        self.last_carry_pts2d = None
        self.last_carry_left2d = None
        self.last_carry_right2d = None
        self.last_carry_poly2d = None
        self.last_carry_robot_width = None

        self.current_click_uv = None
        self.current_target_base = None
        self.current_r_theta_vw = None
        self.latest_pts2d = None
        self.latest_poly2d = None
        self.latest_left2d = None
        self.latest_right2d = None

    def on_mouse(self, event, x, y, flags, userdata):
        if self.current_img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_click_uv = (float(x), float(y))
            self.last_click_uv = self.current_click_uv

            P_b = camray_to_ground_in_base(x, y, self.K, self.T_base_from_cam)
            if P_b is None:
                print("[WARN] Ray did not hit ground plane (z=0) in front of camera.")
                return
            xb, yb = float(P_b[0]), float(P_b[1])
            r, theta = solve_arc_from_point(xb, yb)

            traj_b, v, w, t_arr, theta_arr = arc_to_traj(r, theta, T_horizon, num_t_samples, xb, yb)
            robot_width = random.uniform(robot_width_min, robot_width_max)
            left_b, right_b, poly_b = make_corridor_polygon(traj_b, theta_arr, robot_width)

            # 4) Transform to camera and project
            traj_c = transform_points(self.T_cam_from_base, traj_b)
            left_c = transform_points(self.T_cam_from_base, left_b)
            right_c= transform_points(self.T_cam_from_base, right_b)
            poly_c = transform_points(self.T_cam_from_base, poly_b)

            ctr_2d  = project_points_cam(self.K, self.dist, traj_c)
            left_2d = project_points_cam(self.K, self.dist, left_c)
            right_2d= project_points_cam(self.K, self.dist, right_c)
            poly_2d = project_points_cam(self.K, self.dist, poly_c)


            # Keep your red centerline if desired:
            self.latest_pts2d = ctr_2d  # centerline

            # New: store corridor for redraw
            self.latest_left2d  = left_2d
            self.latest_right2d = right_2d
            self.latest_poly2d  = poly_2d
            self.robot_width = robot_width

            self.current_target_base = (xb, yb)
            self.last_target_base = self.current_target_base
            self.current_r_theta_vw = (r, theta, v, w, robot_width)

            self.last_carry_pts2d = ctr_2d.copy()
            self.last_carry_left2d  = left_2d.copy()
            self.last_carry_right2d = right_2d.copy()
            self.last_carry_poly2d  = poly_2d.copy()
            self.last_carry_robot_width = robot_width
            self.last_selection_record = (r, theta, v, w, robot_width)

            print(f"[INFO] Frame : {self.frame_idx} / {len(self.frames)}, Click {(x,y)} → base ({xb:.3f},{yb:.3f}), r={r:.3f}, θ={np.rad2deg(theta):.3f}, v={v:.3f}, ω={w:.3f}, ")

            self.redraw()

    def redraw(self):
        if self.current_img is None:
            return
        img = self.current_img.copy()

        # Draw carried-over corridor if no new latest
        if self.last_carry_pts2d is not None and self.latest_pts2d is None:
            draw_polyline(img, self.last_carry_pts2d, 2, COLOR_LAST)

        if self.last_carry_poly2d is not None and self.latest_poly2d is None:
            draw_corridor(img, self.last_carry_poly2d, self.last_carry_left2d, self.last_carry_right2d,
                        fill_alpha=0.35, fill_color=COLOR_LAST, edge_color=COLOR_LAST, edge_thickness=2)

        # Draw current corridor (translucent fill + solid edges)
        if self.latest_poly2d is not None:
            draw_corridor(img, self.latest_poly2d, self.latest_left2d, self.latest_right2d,
                        fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=(0,0,200), edge_thickness=2)

        # Draw centerline in red
        if self.latest_pts2d is not None:
            draw_polyline(img, self.latest_pts2d, 2, COLOR_PATH)

        if self.current_click_uv is not None:
            cv2.circle(img, (int(self.current_click_uv[0]), int(self.current_click_uv[1])), 5, COLOR_CLICK, -1)

        self.current_img_show = img
        cv2.imshow(self.window_name, self.current_img_show)

    def log_frame(self):
        if self.bag_doc is None:
            raise RuntimeError("bag doc not open")
        if not (self.last_click_uv and self.last_target_base and self.last_selection_record):
            return
        
        u, v = self.last_click_uv
        xb, yb = self.last_target_base
        r, theta, _, _, _ = self.last_selection_record            

        if self.frame_stamp is None:
            return  # nothing to log

        stamp_key = str(self.frame_stamp)
        # print(stamp_key)
        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "click": {"u": u, "v": v},
            "goal_base": {"x": xb, "y": yb, "z": 0.0},
            "arc": {"r": r, "theta": theta}, 
            "robot_width": self.last_carry_robot_width
        }

        # clear per-frame transient state
        self.current_click_uv = None
        self.current_target_base = None
        self.current_r_theta_vw = None
        self.latest_pts2d = None
        self.latest_poly2d = None
        self.latest_left2d = None
        self.latest_right2d = None

    def log_stop(self):
        if self.bag_doc is None or self.frame_stamp is None:
            return

        stamp_key = str(self.frame_stamp)  # keep same format you use in log_frame

        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "stop": True,
            "click": None,                     # no pixel click
            "goal_base": {"x": 0.0, "y": 0.0, "z": 0.0},
            "arc": {"r": 0.0, "theta": 0.0}
        }

    def process_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.output_path = os.path.join(annotations_root, f"{stem}.json")

        print(f"\n=== Processing {self.bag_name} ===")

        with rosbag.Bag(bag_path, "r") as bag:
            if "Jackal" in self.bag_name:
                self.image_topic = "/camera/rgb/image_raw/compressed"
            elif "Spot" in self.bag_name:
                self.image_topic = "/image_raw/compressed"
            
            print(f"[INFO] Using image topic: {self.image_topic}")
            # print(bag.get_type_and_topic_info()[1][self.image_topic])
            # total_len = bag.get_type_and_topic_info()[1][self.image_topic].message_count
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic])):
                if i % undersampling_factor != 0:
                    continue

                self.frame_idx = i
                self.frame_stamp = t

                # Convert ROS Image -> BGR
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.frames.append(FrameItem(idx=i, stamp=t, img=cv_img))

        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return

        self._open_bag_doc()
        i = 0
        try:
            while 0 <= i < len(self.frames):
                fr = self.frames[i]
                self.frame_idx = fr.idx
                self.frame_stamp = fr.stamp
                self.current_img = fr.img

                # clear “current” state (but keep last_* to allow carry-over)
                self.current_click_uv = None
                self.current_target_base = None
                self.current_r_theta_vw = None
                self.latest_pts2d = None
                self.latest_poly2d = None
                self.latest_left2d = None
                self.latest_right2d = None

                self.redraw()
                key = cv2.waitKey(0) & 0xFF

                if key in (ord('q'), 27):   # q or ESC
                    print("[INFO] Quit requested.")
                    return

                elif key == 83:  # Right Arrow → save (using last_*) then next
                    self.log_frame()
                    i += 1

                elif key == 81:  # Left Arrow → go back one (no save)
                    print("[INFO] Back one frame.")
                    i = max(0, i - 1)
                elif key == ord('0'):  # STOP action
                    self.log_stop()
                    self._clear_last()
                    self.redraw()
                    continue 
                else:
                    continue

        finally:
            self._close_bag_doc()

            #         self.current_img = cv_img
            #         self.redraw()

            #         key = cv2.waitKey(0) & 0xFF
            #         if key == ord('q') or key == 27:  # 'q' or ESC
            #             print("[INFO] Quit requested.")
            #             return
            #         elif key == 83:  # Right Arrow
            #             self.log_frame()
            #             self.redraw()

            #     # # log final frame
            #     # self.log_frame()
            # finally:
            #     self._close_bag_doc()

    def run(self):
        bag_files = sorted(glob.glob(os.path.join(bag_dir, "*.bag")))
        if bag_idx is not None:
            bag_files = bag_files[bag_idx: bag_idx+annotate_n_bags]

        with open(skip_json_path, 'r') as f:
            bags_to_skip = json.load(f)
        if not bag_files:
            print(f"[ERROR] No .bag files found in {bag_dir}")
            return
        for bp in bag_files:
            print(os.path.basename(bp))
            if bags_to_skip.get(os.path.basename(bp), False):
                print(f"[INFO] Skipping {bp}")
                continue
            self.frames : list[FrameItem] = []
            self.process_bag(bp)
        print(f"\n[DONE] Annotations written to {self.output_path}")

if __name__ == "__main__":
    Annotator().run()
