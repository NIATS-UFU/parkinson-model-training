"""Evaluation helpers for classification metrics."""


def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    if not y_true:
        msg = "ROC AUC requires at least one sample"
        raise ValueError(msg)
    paired = list(zip(y_true, y_scores, strict=True))
    paired.sort(key=lambda x: x[1], reverse=True)

    n_pos = sum(1 for y, _ in paired if y == 1)
    n_neg = len(paired) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    auc = 0.0
    fp, tp = 0, 0
    prev_fp, prev_tp = 0, 0
    prev_score = float("-inf")

    for y, score in paired:
        if score != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
            prev_score = score
        if y == 1:
            tp += 1
        else:
            fp += 1

    auc += (n_neg - prev_fp) * (n_pos + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    """Compute Accuracy, Balanced Accuracy, Sensitivity, Specificity, and ROC AUC for binary classification."""
    if not y_true:
        msg = "Metrics require at least one sample"
        raise ValueError(msg)
    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        msg = "Metric inputs must have the same length"
        raise ValueError(msg)

    tp, fp, tn, fn = 0, 0, 0, 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    roc_auc = compute_roc_auc(y_true, y_prob)
    accuracy = (tp + tn) / len(y_true)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
    }
