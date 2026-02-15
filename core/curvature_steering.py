"""
Curvature-based steering controller (no PID)
- Uses lane curvature feedforward + small lateral error trim
- Applies rate limiting and output clamping for smoothness
"""

import math
from typing import Optional


class CurvatureSteeringController:
    def __init__(self,
                 wheelbase_m: float = 2.9,
                 k_p_lat: float = 0.12,
                 out_limit: float = 0.25,
                 rate_limit: float = 0.03,
                 ff_gain: float = 1.0):
        self.L = wheelbase_m
        self.k_p_lat = k_p_lat
        self.out_limit = out_limit
        self.rate_limit = rate_limit
        self.ff_gain = ff_gain

    def _compute_centerline_curvature(self, lane_detector) -> Optional[float]:
        """Estimate centerline curvature (1/m) from BEV polynomial fits.
        Returns None if unavailable.
        """
        # Access last fitted coefficients (Q2, Q1, Q0) for left/right lanes
        cL = getattr(lane_detector, 'last_coeff_left', None)
        cR = getattr(lane_detector, 'last_coeff_right', None)
        if cL is None and cR is None:
            return None

        # Evaluation y in BEV pixels (near bottom/look-ahead)
        bev_h = getattr(lane_detector, 'bev_h', None)
        if bev_h is None:
            return None
        y_eval_px = bev_h - 60  # mirrors LOOK_Y_OFFSET = 60

        # Pixel-to-meter scales in BEV
        px_to_m_x = getattr(lane_detector, 'px_to_m_x', None)
        px_to_m_y = getattr(lane_detector, 'px_to_m_y', None)
        if px_to_m_x is None or px_to_m_y is None or px_to_m_y == 0:
            return None

        def lane_curvature_px(coeff):
            # coeff = (Q2, Q1, Q0) for x(y)
            Q2, Q1, _ = coeff
            dx_dy = 2.0 * Q2 * y_eval_px + Q1         # px/px
            d2x_dy2 = 2.0 * Q2                        # px/px^2
            # Convert derivatives to meters
            dx_dy_m = dx_dy * (px_to_m_x / px_to_m_y)
            d2x_dy2_m = d2x_dy2 * (px_to_m_x / (px_to_m_y ** 2))
            # Curvature in 1/m for param y: (x(y), y)
            denom = (1.0 + dx_dy_m * dx_dy_m) ** 1.5
            if denom <= 1e-6:
                return 0.0
            return abs(d2x_dy2_m) / denom

        curvatures = []
        if cL is not None:
            curvatures.append(lane_curvature_px(cL))
        if cR is not None:
            curvatures.append(lane_curvature_px(cR))

        if len(curvatures) == 0:
            return None
        return sum(curvatures) / len(curvatures)

    def step(self,
             lane_detector,
             lateral_error_m: Optional[float],
             speed_kmh: float,
             last_out: Optional[float] = None) -> float:
        """Compute steering using curvature feedforward + small lateral trim.
        Returns steering in [-out_limit, out_limit].
        """
        # Feedforward from curvature
        kappa = self._compute_centerline_curvature(lane_detector)
        if kappa is None:
            # No geometry available: hold steer gently
            u = 0.0 if last_out is None else last_out * 0.9
        else:
            delta_ff = math.atan(self.L * kappa)  # radians
            # Map radians to normalized steer roughly 1:1, clamp later
            u = self.ff_gain * delta_ff

        # Add small trim from lateral error (if available)
        if lateral_error_m is not None:
            # Clamp lateral error to avoid spikes
            e_lat = max(-0.8, min(0.8, float(lateral_error_m)))
            u += self.k_p_lat * e_lat

        # Clamp output
        u = max(-self.out_limit, min(self.out_limit, u))

        # Rate limit
        if last_out is not None:
            delta = u - last_out
            if delta > self.rate_limit:
                u = last_out + self.rate_limit
            elif delta < -self.rate_limit:
                u = last_out - self.rate_limit

        return u
