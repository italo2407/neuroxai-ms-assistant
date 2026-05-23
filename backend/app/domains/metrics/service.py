import numpy as np


class MetricsService:
    def compute(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Compute DICE, IoU, precision, recall, F1 for binary masks."""
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou = float(tp / (tp + fp + fn + 1e-8))
        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))

        total_px = pred_bin.size

        return {
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "gt_lesion_pixels": int(gt_bin.sum()),
            "pred_lesion_pixels": int(pred_bin.sum()),
            "gt_lesion_pct": round(100 * float(gt_bin.sum()) / total_px, 3),
            "pred_lesion_pct": round(100 * float(pred_bin.sum()) / total_px, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }


metrics_service = MetricsService()
