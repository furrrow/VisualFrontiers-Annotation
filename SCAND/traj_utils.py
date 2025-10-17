import math
from typing import Optional, Tuple
import numpy as np

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
