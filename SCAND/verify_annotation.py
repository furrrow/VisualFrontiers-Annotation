import os
import rosbag
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import math

bagfile_root = "SCAND/rosbags"
bag_idx = 4
annotations_root = "/home/jim/Downloads/unverified"
fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
T_horizon = 2.0      # Path generation options
num_t_samples = 1000
calib_path = "./SCAND/tf.json"

COLOR_PATH = (0, 0, 255) #red
COLOR_CLICK = (255, 0, 0) #green

@dataclass
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img: np.ndarray
    r: float
    theta: float
    sub_goal : list   # [x, y, z] in base_link
    width_m : float
    u : float
    v : float
    stop : bool

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
    theta = 2.0 * math.atan2(y, x) 
    return r, theta

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

def draw(frame_item: FrameItem, K: np.ndarray, dist, T_cam_from_base: np.ndarray, window: str = "SCAND Verification"):

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    current_img = frame_item.img
    x_base, y_base, z_base = frame_item.sub_goal
    width_m = frame_item.width_m
    r, theta = frame_item.r, frame_item.theta
    u, v = frame_item.u, frame_item.v
    stop = frame_item.stop

    if not stop: 
        path_points, v, w, t_samples, theta_samples = point_to_traj(r, theta, T_horizon, num_t_samples, x_base, y_base)
        left_b, right_b, poly_b = make_corridor_polygon(path_points, theta_samples, width_m, bridge_pts=20)

        # 4) Transform to camera and project
        traj_c = transform_points(T_cam_from_base, path_points)
        left_c = transform_points(T_cam_from_base, left_b)
        right_c= transform_points(T_cam_from_base, right_b)
        poly_c = transform_points(T_cam_from_base, poly_b)

        ctr_2d  = project_points_cam(K, dist, traj_c)
        left_2d = project_points_cam(K, dist, left_c)
        right_2d= project_points_cam(K, dist, right_c)
        poly_2d = project_points_cam(K, dist, poly_c)
        # Project to image

        if current_img is None:
            return
        img = current_img.copy()


        draw_polyline(img, ctr_2d, 2, COLOR_PATH)
        draw_corridor(img, poly_2d, left_2d, right_2d, 
                    fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=COLOR_PATH, edge_thickness=2)
        
        cv2.circle(img, (int(u), int(v)), 5, COLOR_CLICK, -1)
    else:
        ## transclucent red overlay for stop
        img = current_img.copy()
        overlay = img.copy()
        overlay[:] = (0,0,255)
        img[:] = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    cv2.imshow(window, img)

def verify_annotations():

    annotations_files = sorted([f for f in os.listdir(annotations_root) if f.endswith('.json')])
    bridge = CvBridge()

    for ann_file in annotations_files:

        if "Jackal" in ann_file:
            K, dist, T_base_from_cam = load_calibration(calib_path, mode="jackal")
            T_cam_from_base = np.linalg.inv(T_base_from_cam)
        elif "Spot" in ann_file:
            K, dist, T_base_from_cam = load_calibration(calib_path, mode="spot")
            T_cam_from_base = np.linalg.inv(T_base_from_cam)

        ann_path = os.path.join(annotations_root, ann_file)
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        bag_name = ann_data.get("bag", False)
        if not bag_name:
            print(f"[ERROR] No bag_name found in {ann_file}")
            continue

        img_topic = ann_data.get("image_topic", False)
        if not img_topic:
            print(f"[ERROR] No image_topic found in {ann_file}")
            continue

        bag_path = os.path.join(bagfile_root, bag_name)

        if not os.path.exists(bag_path):
            print(f"[ERROR] Bag file {bag_name} not found for annotation {ann_file}")
            continue

        print(f"[INFO] Verifying annotations in {ann_file} against bag {bag_name}")

        annotation_timstamps = ann_data.get("annotations_by_stamp")
        frame_items : list[FrameItem] = []
        with rosbag.Bag(bag_path, 'r') as bag:

            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=[img_topic])):
                annotation_item = annotation_timstamps.get(str(t), False)
                if not annotation_item:
                    continue

                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                sub_goal = annotation_item.get("goal_base")
                sub_goal = [sub_goal["x"], sub_goal["y"], sub_goal["z"]]
                width_m = annotation_item.get("robot_width", 0.5)

                stop = annotation_item.get("stop", False)

                if stop:
                    u, v = 0,0
                else:
                    u = annotation_item.get("click")["u"]
                    v = annotation_item.get("click")["v"]
                
                r = annotation_item.get("arc")["r"]
                theta = annotation_item.get("arc")["theta"]
                
                frame_items.append(FrameItem(idx=idx, 
                                             stamp=t, 
                                             img=cv_img, 
                                             r=r, 
                                             theta=theta, 
                                             sub_goal=sub_goal, 
                                             width_m=width_m,
                                             u=u,
                                             v=v,
                                             stop=stop))
            i = 0
            while 0 <= i < len(frame_items):
                fr = frame_items[i]
                draw(fr, K, dist, T_cam_from_base)
                key = cv2.waitKey(0) & 0xFF

                if key == 83:  # Right Arrow → save (using last_*) then next
                    i += 1

                elif key == 81:  # Left Arrow → go back one (no save)
                    print("[INFO] Back one frame.")
                    i = max(0, i - 1)
                
                elif: key in (ord('q'), 27):   # q or ESC
                    print("[INFO] Quit requested.")
                    return
                else:
                    continue
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_annotations()