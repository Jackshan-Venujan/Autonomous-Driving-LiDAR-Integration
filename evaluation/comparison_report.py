import csv
import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

from evaluation.metrics_engine import ExperimentResults, CLASSES

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# Scoring weights — safety-first ordering
CATEGORY_WEIGHTS = {
    'safety':    0.35,
    'detection': 0.25,
    'depth':     0.15,
    'realtime':  0.15,
    'robustness': 0.10,
}

# (metric_key_in_flat_dict, category, higher_is_better)
METRIC_SPECS = [
    ('map_iou05',         'detection',  True),
    ('map_iou07',         'detection',  True),
    ('depth_rmse',        'depth',      False),
    ('depth_mae',         'depth',      False),
    ('depth_delta_1_25',  'depth',      True),
    ('fnr_vehicle',       'safety',     False),
    ('fnr_pedestrian',    'safety',     False),
    ('fnr_cyclist',       'safety',     False),
    ('fpr_vehicle',       'safety',     False),
    ('fpr_pedestrian',    'safety',     False),
    ('fpr_cyclist',       'safety',     False),
    ('recall_pedestrian', 'safety',     True),
    ('mean_latency_ms',   'realtime',   False),
    ('mean_fps',          'realtime',   True),
    ('map_rain',          'robustness', True),
    ('map_night',         'robustness', True),
    ('map_cloudy_evening','robustness', True),
]

CATEGORY_AXIS_LABELS = {
    'safety':     'Safety',
    'detection':  'Detection\nAccuracy',
    'depth':      'Depth\nAccuracy',
    'realtime':   'Real-time\nPerformance',
    'robustness': 'Robustness',
}


class ComparisonReport:
    """
    Generates all comparison artefacts from two ExperimentResults.

    Call generate_all() to produce:
      - comparison_table.csv
      - radar_chart.png
      - console summary with weighted scores and winner recommendation
    """

    def __init__(
        self,
        result_exp1: ExperimentResults,
        result_exp2: ExperimentResults,
        output_dir: str = 'output/comparison',
    ):
        self._r1 = result_exp1
        self._r2 = result_exp2
        self._out = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._flat1 = self._flatten_results(result_exp1)
        self._flat2 = self._flatten_results(result_exp2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self):
        self.save_comparison_csv()
        self.render_radar_chart()
        score1, score2, winner = self.compute_weighted_scores()
        self.print_summary(score1, score2, winner)

    def save_comparison_csv(self, filename: str = 'comparison_table.csv'):
        path = os.path.join(self._out, filename)
        rows = []
        for metric, cat, higher_better in METRIC_SPECS:
            v1 = self._flat1.get(metric)
            v2 = self._flat2.get(metric)
            if v1 is None and v2 is None:
                continue
            v1 = v1 or 0.0
            v2 = v2 or 0.0
            if higher_better:
                winner = self._r1.experiment_name if v1 > v2 else (self._r2.experiment_name if v2 > v1 else 'tie')
            else:
                winner = self._r1.experiment_name if v1 < v2 else (self._r2.experiment_name if v2 < v1 else 'tie')
            rows.append({
                'metric': metric,
                'category': cat,
                f'{self._r1.experiment_name}': round(v1, 5),
                f'{self._r2.experiment_name}': round(v2, 5),
                'winner': winner,
            })
        fieldnames = ['metric', 'category', self._r1.experiment_name, self._r2.experiment_name, 'winner']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Report] Saved comparison table to {path}")

    def render_radar_chart(self, filename: str = 'radar_chart.png'):
        if not _HAS_MATPLOTLIB:
            print("[Report] matplotlib not available — skipping radar chart")
            return

        cat_scores1, cat_scores2 = self._per_category_scores()
        categories = list(CATEGORY_WEIGHTS.keys())
        n = len(categories)
        angles = [2 * math.pi * i / n for i in range(n)]
        angles += angles[:1]  # close polygon

        vals1 = [cat_scores1.get(c, 0.0) for c in categories] + [cat_scores1.get(categories[0], 0.0)]
        vals2 = [cat_scores2.get(c, 0.0) for c in categories] + [cat_scores2.get(categories[0], 0.0)]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([CATEGORY_AXIS_LABELS[c] for c in categories], size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8)

        ax.plot(angles, vals1, 'o-', linewidth=2, color='steelblue', label=self._r1.experiment_name)
        ax.fill(angles, vals1, alpha=0.25, color='steelblue')
        ax.plot(angles, vals2, 'o-', linewidth=2, color='darkorange', label=self._r2.experiment_name)
        ax.fill(angles, vals2, alpha=0.25, color='darkorange')

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Sensor Fusion Comparison', size=14, pad=20)

        path = os.path.join(self._out, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Report] Saved radar chart to {path}")

    def compute_weighted_scores(self) -> Tuple[float, float, str]:
        """
        Compute final weighted score for each experiment.
        Returns (score_exp1, score_exp2, winner_name).
        """
        cat_scores1, cat_scores2 = self._per_category_scores()
        score1 = sum(CATEGORY_WEIGHTS[c] * cat_scores1.get(c, 0.0) for c in CATEGORY_WEIGHTS)
        score2 = sum(CATEGORY_WEIGHTS[c] * cat_scores2.get(c, 0.0) for c in CATEGORY_WEIGHTS)
        if score1 > score2:
            winner = self._r1.experiment_name
        elif score2 > score1:
            winner = self._r2.experiment_name
        else:
            winner = 'tie'
        return score1, score2, winner

    def print_summary(self, score1: float, score2: float, winner: str):
        sep = '-' * 70
        print(f"\n{sep}")
        print("  ADAS SENSOR FUSION COMPARISON REPORT")
        print(sep)

        # Per-category breakdown
        cat_scores1, cat_scores2 = self._per_category_scores()
        print(f"\n{'Category':<20} {'Weight':>6}  {self._r1.experiment_name:>20}  {self._r2.experiment_name:>20}")
        print('-' * 70)
        for cat, weight in CATEGORY_WEIGHTS.items():
            s1 = cat_scores1.get(cat, 0.0)
            s2 = cat_scores2.get(cat, 0.0)
            print(f"  {CATEGORY_AXIS_LABELS[cat]:<18} {weight:>6.2f}  {s1:>20.4f}  {s2:>20.4f}")

        print(sep)
        print(f"  WEIGHTED TOTAL             {score1:>28.4f}  {score2:>20.4f}")
        print(sep)

        print(f"\n  WINNER: {winner.upper()}")
        if winner == self._r1.experiment_name:
            print("  → LiDAR + monocular fusion is recommended for physical implementation.")
            print("    Strong safety and depth accuracy from active ranging outweigh higher cost.")
        elif winner == self._r2.experiment_name:
            print("  → Stereo camera setup is recommended for physical implementation.")
            print("    Passive depth with lower power, cost, and weight is competitive overall.")
        else:
            print("  → Both setups are equivalent. Choose based on deployment constraints.")
        print(sep + '\n')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _per_category_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute normalised [0,1] score per category for each experiment."""
        cat_scores1: Dict[str, float] = {c: 0.0 for c in CATEGORY_WEIGHTS}
        cat_scores2: Dict[str, float] = {c: 0.0 for c in CATEGORY_WEIGHTS}
        cat_counts: Dict[str, int] = {c: 0 for c in CATEGORY_WEIGHTS}

        for metric, cat, higher_better in METRIC_SPECS:
            v1 = self._flat1.get(metric)
            v2 = self._flat2.get(metric)
            if v1 is None and v2 is None:
                continue
            v1 = float(v1) if (v1 is not None and not _is_nan(v1)) else 0.0
            v2 = float(v2) if (v2 is not None and not _is_nan(v2)) else 0.0
            n1, n2 = self._normalize_pair(v1, v2, higher_better)
            cat_scores1[cat] += n1
            cat_scores2[cat] += n2
            cat_counts[cat] += 1

        # Average within each category
        for cat in CATEGORY_WEIGHTS:
            if cat_counts[cat] > 0:
                cat_scores1[cat] /= cat_counts[cat]
                cat_scores2[cat] /= cat_counts[cat]

        return cat_scores1, cat_scores2

    @staticmethod
    def _normalize_pair(v1: float, v2: float, higher_is_better: bool) -> Tuple[float, float]:
        """Normalize a pair to [0, 1]. Handles edge case where both are 0."""
        max_val = max(abs(v1), abs(v2))
        if max_val < 1e-9:
            return 0.5, 0.5
        if higher_is_better:
            return v1 / max_val, v2 / max_val
        else:
            return 1.0 - v1 / max_val, 1.0 - v2 / max_val

    @staticmethod
    def _flatten_results(r: ExperimentResults) -> Dict:
        flat = {}
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


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
