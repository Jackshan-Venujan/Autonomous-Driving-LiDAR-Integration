import math
import csv
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from evaluation.ground_truth_logger import FrameGT, GroundTruthLogger


CLASSES = ['vehicle', 'pedestrian', 'cyclist']

YOLO_TO_CLASS = {
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle',
    'motorcycle': 'vehicle', 'van': 'vehicle',
    'person': 'pedestrian',
    'bicycle': 'cyclist',
}


@dataclass
class ExperimentResults:
    experiment_name: str
    # Detection accuracy
    map_iou05: float = 0.0
    map_iou07: float = 0.0
    per_class_ap_iou05: Dict[str, float] = field(default_factory=dict)
    per_class_ap_iou07: Dict[str, float] = field(default_factory=dict)
    # Depth accuracy
    depth_rmse: float = float('nan')
    depth_mae: float = float('nan')
    depth_delta_1_25: float = 0.0
    # Safety
    fnr_per_class: Dict[str, float] = field(default_factory=dict)
    fpr_per_class: Dict[str, float] = field(default_factory=dict)
    recall_per_class: Dict[str, float] = field(default_factory=dict)
    # Real-time
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_fps: float = 0.0
    # Robustness by weather tag
    map_by_weather: Dict[str, float] = field(default_factory=dict)


class MetricsEngine:
    """
    Accumulates per-frame detection data and computes all evaluation metrics.

    Usage:
        engine = MetricsEngine('lidar_fusion', focal_px, img_width)
        for frame in scenario:
            t0 = time.perf_counter()
            result = agent.process_frame(image)
            latency = (time.perf_counter() - t0) * 1000
            engine.update(frame_id, result, frame_gt, latency, weather_tag)
        results = engine.compute()
        engine.save_csv('output/exp1_metrics.csv')
    """

    def __init__(
        self,
        experiment_name: str,
        focal_length_px: float,
        img_width: int,
        azimuth_tolerance_deg: float = 15.0,
        distance_tolerance_m: float = 8.0,
    ):
        self._name = experiment_name
        self._focal = focal_length_px
        self._img_width = img_width
        self._az_tol = azimuth_tolerance_deg
        self._dist_tol = distance_tolerance_m

        # Per-frame accumulation lists
        self._latencies: List[float] = []
        self._depth_pairs: List[Tuple[float, float]] = []   # (predicted, gt)
        self._weather_tags: List[str] = []

        # Per-class detection accumulation
        # Each entry: {'conf': float, 'matched': bool, 'weather': str}
        self._detections: Dict[str, List[Dict]] = defaultdict(list)
        # Per-class GT counts per frame: {class: [count_per_frame, ...]}
        self._gt_counts: Dict[str, List[int]] = defaultdict(list)

        self._results: Optional[ExperimentResults] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        detection_result: Dict,
        frame_gt: FrameGT,
        latency_ms: float,
        weather_tag: str = 'clear',
    ):
        """Process one frame — accumulate data for metric computation."""
        self._latencies.append(latency_ms)
        self._weather_tags.append(weather_tag)

        # Extract detections from DrivingAgent result dict
        obs_data = detection_result.get('obstacle_data', {}) or {}
        raw_dets = obs_data.get('lane_detections', []) or []

        # Count GT actors per class for this frame
        gt_class_counts: Dict[str, int] = defaultdict(int)
        for gt_actor in frame_gt.actors:
            gt_class_counts[gt_actor.class_name] += 1
        for cls in CLASSES:
            self._gt_counts[cls].append(gt_class_counts.get(cls, 0))

        # Track which GT actors have been matched (greedy, one match per GT)
        matched_gt_ids = set()

        for det in raw_dets:
            raw_cls = det.get('class', '')
            cls = YOLO_TO_CLASS.get(raw_cls)
            if cls is None:
                continue
            conf = float(det.get('confidence', det.get('conf', 0.5)))
            det_dist = det.get('stereo_distance') or det.get('distance')

            # Compute detection azimuth from bbox centre
            bbox = det.get('bbox')
            det_azimuth = None
            if bbox is not None:
                cx = (bbox[0] + bbox[2]) / 2.0
                det_azimuth = math.degrees(math.atan2(cx - self._img_width / 2.0, self._focal))

            # Find best matching GT actor
            matched = False
            best_gt = None
            best_score = float('inf')

            for gt_actor in frame_gt.actors:
                if gt_actor.actor_id in matched_gt_ids:
                    continue
                if gt_actor.class_name != cls:
                    continue
                if det_dist is None or det_azimuth is None:
                    continue
                az_diff = abs(det_azimuth - gt_actor.azimuth_deg)
                dist_diff = abs(det_dist - gt_actor.distance_m)
                if az_diff <= self._az_tol and dist_diff <= self._dist_tol:
                    score = az_diff + dist_diff
                    if score < best_score:
                        best_score = score
                        best_gt = gt_actor

            if best_gt is not None:
                matched = True
                matched_gt_ids.add(best_gt.actor_id)
                # Accumulate depth error
                if det_dist is not None:
                    self._depth_pairs.append((det_dist, best_gt.distance_m))

            self._detections[cls].append({
                'conf': conf,
                'matched': matched,
                'weather': weather_tag,
            })

    def compute(self) -> ExperimentResults:
        """Compute all metrics from accumulated data."""
        mAP05, per_cls_05 = self._compute_map(0.5)
        mAP07, per_cls_07 = self._compute_map(0.7)
        rmse, mae, delta = self._compute_depth_metrics()
        fnr, fpr, recall = self._compute_safety_metrics()
        mean_lat, p95_lat, mean_fps = self._compute_performance_metrics()
        map_weather = self._compute_robustness()

        self._results = ExperimentResults(
            experiment_name=self._name,
            map_iou05=mAP05,
            map_iou07=mAP07,
            per_class_ap_iou05=per_cls_05,
            per_class_ap_iou07=per_cls_07,
            depth_rmse=rmse,
            depth_mae=mae,
            depth_delta_1_25=delta,
            fnr_per_class=fnr,
            fpr_per_class=fpr,
            recall_per_class=recall,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            mean_fps=mean_fps,
            map_by_weather=map_weather,
        )
        return self._results

    def save_csv(self, output_path: str):
        """Save ExperimentResults to a flat CSV file."""
        if self._results is None:
            self._results = self.compute()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        flat = self._flatten_results(self._results)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    def _compute_map(self, iou_threshold: float) -> Tuple[float, Dict[str, float]]:
        """VOC 11-point mAP using azimuth+distance proximity as IoU surrogate."""
        per_class_ap = {}
        for cls in CLASSES:
            dets = self._detections[cls]
            total_gt = sum(self._gt_counts[cls])
            if total_gt == 0:
                per_class_ap[cls] = 0.0
                continue
            if not dets:
                per_class_ap[cls] = 0.0
                continue
            dets_sorted = sorted(dets, key=lambda d: d['conf'], reverse=True)
            tp = np.zeros(len(dets_sorted))
            fp = np.zeros(len(dets_sorted))
            for i, d in enumerate(dets_sorted):
                if d['matched']:
                    tp[i] = 1
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / (total_gt + 1e-9)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            per_class_ap[cls] = self._voc_ap(precisions, recalls)
        mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
        return mean_ap, per_class_ap

    def _compute_depth_metrics(self) -> Tuple[float, float, float]:
        """RMSE, MAE, delta<1.25 from accumulated (pred, gt) pairs."""
        if not self._depth_pairs:
            return float('nan'), float('nan'), 0.0
        preds = np.array([p for p, _ in self._depth_pairs], dtype=np.float32)
        gts = np.array([g for _, g in self._depth_pairs], dtype=np.float32)
        valid = (gts > 0) & np.isfinite(preds) & np.isfinite(gts)
        if valid.sum() == 0:
            return float('nan'), float('nan'), 0.0
        p, g = preds[valid], gts[valid]
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        mae = float(np.mean(np.abs(p - g)))
        ratio = np.maximum(p / (g + 1e-9), g / (p + 1e-9))
        delta = float(np.mean(ratio < 1.25))
        return rmse, mae, delta

    def _compute_safety_metrics(self) -> Tuple[Dict, Dict, Dict]:
        """FNR, FPR, Recall per class."""
        fnr, fpr, recall = {}, {}, {}
        total_frames = max(len(self._latencies), 1)
        for cls in CLASSES:
            dets = self._detections[cls]
            tp = sum(1 for d in dets if d['matched'])
            fp = sum(1 for d in dets if not d['matched'])
            total_gt = sum(self._gt_counts[cls])
            fn = max(0, total_gt - tp)
            # TN: frames where no GT of this class AND no detection
            gt_presence = [c > 0 for c in self._gt_counts[cls]]
            tn = sum(1 for present in gt_presence if not present) - fp
            tn = max(0, tn)
            fnr[cls] = fn / (fn + tp + 1e-9)
            fpr[cls] = fp / (fp + tn + 1e-9)
            recall[cls] = tp / (tp + fn + 1e-9)
        return fnr, fpr, recall

    def _compute_performance_metrics(self) -> Tuple[float, float, float]:
        """mean latency, p95 latency, mean FPS."""
        if not self._latencies:
            return 0.0, 0.0, 0.0
        lats = np.array(self._latencies)
        mean_lat = float(np.mean(lats))
        p95_lat = float(np.percentile(lats, 95))
        mean_fps = float(np.mean(1000.0 / (lats + 1e-9)))
        return mean_lat, p95_lat, mean_fps

    def _compute_robustness(self) -> Dict[str, float]:
        """mAP@0.5 grouped by weather tag."""
        tags = set(self._weather_tags)
        result = {}
        for tag in tags:
            frame_indices = [i for i, t in enumerate(self._weather_tags) if t == tag]
            if not frame_indices:
                continue
            tag_ap_list = []
            for cls in CLASSES:
                dets = [d for d in self._detections[cls] if d['weather'] == tag]
                gt_total = sum(
                    self._gt_counts[cls][i]
                    for i in frame_indices
                    if i < len(self._gt_counts[cls])
                )
                if gt_total == 0 or not dets:
                    tag_ap_list.append(0.0)
                    continue
                dets_sorted = sorted(dets, key=lambda d: d['conf'], reverse=True)
                tp = np.array([1.0 if d['matched'] else 0.0 for d in dets_sorted])
                fp = 1 - tp
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recalls = tp_cum / (gt_total + 1e-9)
                precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
                tag_ap_list.append(self._voc_ap(precisions, recalls))
            result[tag] = float(np.mean(tag_ap_list)) if tag_ap_list else 0.0
        return result

    @staticmethod
    def _voc_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """VOC 2010 11-point interpolation."""
        ap = 0.0
        for threshold in np.linspace(0, 1, 11):
            mask = recalls >= threshold
            if mask.any():
                ap += precisions[mask].max()
        return ap / 11.0

    @staticmethod
    def _flatten_results(r: ExperimentResults) -> Dict:
        flat = {'experiment_name': r.experiment_name}
        flat['map_iou05'] = r.map_iou05
        flat['map_iou07'] = r.map_iou07
        for cls in CLASSES:
            flat[f'ap05_{cls}'] = r.per_class_ap_iou05.get(cls, 0.0)
            flat[f'ap07_{cls}'] = r.per_class_ap_iou07.get(cls, 0.0)
        flat['depth_rmse'] = r.depth_rmse
        flat['depth_mae'] = r.depth_mae
        flat['depth_delta_1_25'] = r.depth_delta_1_25
        for cls in CLASSES:
            flat[f'fnr_{cls}'] = r.fnr_per_class.get(cls, 0.0)
            flat[f'fpr_{cls}'] = r.fpr_per_class.get(cls, 0.0)
            flat[f'recall_{cls}'] = r.recall_per_class.get(cls, 0.0)
        flat['mean_latency_ms'] = r.mean_latency_ms
        flat['p95_latency_ms'] = r.p95_latency_ms
        flat['mean_fps'] = r.mean_fps
        for tag, val in r.map_by_weather.items():
            flat[f'map_{tag}'] = val
        return flat
