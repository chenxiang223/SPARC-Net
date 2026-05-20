from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch


def _to_class_count_tensor(
    class_counts: Optional[Iterable[int] | Mapping[int, int] | torch.Tensor],
    num_classes: int,
) -> torch.Tensor:
    if class_counts is None:
        return torch.ones(num_classes, dtype=torch.float32)

    if isinstance(class_counts, Mapping):
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for key, value in class_counts.items():
            key = int(key)
            if 0 <= key < num_classes:
                counts[key] = float(value)
        return counts

    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1:
        raise ValueError(f"class_counts must be 1D, got shape {tuple(counts.shape)}.")
    if counts.numel() != num_classes:
        raise ValueError(f"class_counts has {counts.numel()} classes, expected {num_classes}.")
    return counts


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp(min=1e-12)


@dataclass(frozen=True)
class GroupMetrics:
    class_ids: tuple[int, ...]
    oa: float
    aa: float
    macro_f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Full evaluation bundle for long-tail HSI classification.

    oa:
        Overall accuracy.
    aa:
        Average class accuracy, i.e. mean per-class recall.
    kappa:
        Cohen's kappa based on the confusion matrix.
    macro_f1:
        Mean class-wise F1 over valid classes.
    confusion_matrix:
        Rows are ground truth classes, columns are predicted classes.
    """

    oa: float
    aa: float
    kappa: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion_matrix: torch.Tensor
    support: torch.Tensor
    per_class_precision: torch.Tensor
    per_class_recall: torch.Tensor
    per_class_accuracy: torch.Tensor
    per_class_f1: torch.Tensor
    class_groups: Dict[str, GroupMetrics]

    def to_summary_dict(self) -> Dict[str, float]:
        summary = {
            "oa": self.oa,
            "aa": self.aa,
            "kappa": self.kappa,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }
        for group_name, metrics in self.class_groups.items():
            summary[f"{group_name}_oa"] = metrics.oa
            summary[f"{group_name}_aa"] = metrics.aa
            summary[f"{group_name}_macro_f1"] = metrics.macro_f1
        return summary


def compute_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    preds = torch.as_tensor(preds, dtype=torch.long).view(-1).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    if preds.shape[0] != labels.shape[0]:
        raise ValueError("preds and labels must contain the same number of samples.")

    if num_classes is None:
        if labels.numel() == 0:
            num_classes = 0
        else:
            num_classes = int(max(int(preds.max().item()), int(labels.max().item())) + 1)
    if num_classes == 0:
        return torch.empty(0, 0, dtype=torch.long)

    flat_index = labels * num_classes + preds
    matrix = torch.bincount(flat_index, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes).to(torch.long)


def build_class_frequency_groups(
    class_counts: Optional[Iterable[int] | Mapping[int, int] | torch.Tensor],
    num_classes: int,
    *,
    head_fraction: float = 0.34,
    tail_fraction: float = 0.34,
) -> Dict[str, tuple[int, ...]]:
    """
    Partition classes into head / medium / tail by training-set frequency rank.

    The split is rank-based rather than threshold-based, which is more stable
    across HSI datasets with very different absolute pixel counts.
    """

    if num_classes <= 0:
        return {"head": tuple(), "medium": tuple(), "tail": tuple()}
    if not 0.0 <= head_fraction <= 1.0 or not 0.0 <= tail_fraction <= 1.0:
        raise ValueError("head_fraction and tail_fraction must be in [0, 1].")

    counts = _to_class_count_tensor(class_counts, num_classes=num_classes)
    sorted_ids = torch.argsort(counts, descending=True).tolist()

    if num_classes == 1:
        return {"head": (sorted_ids[0],), "medium": tuple(), "tail": tuple()}
    if num_classes == 2:
        return {"head": (sorted_ids[0],), "medium": tuple(), "tail": (sorted_ids[1],)}

    head_n = max(1, int(round(num_classes * head_fraction)))
    tail_n = max(1, int(round(num_classes * tail_fraction)))
    if head_n + tail_n >= num_classes:
        tail_n = max(1, num_classes - head_n - 1)
    medium_n = num_classes - head_n - tail_n
    if medium_n <= 0:
        medium_n = 1
        if tail_n > 1:
            tail_n -= 1
        else:
            head_n = max(1, head_n - 1)

    head_ids = tuple(sorted_ids[:head_n])
    medium_ids = tuple(sorted_ids[head_n : head_n + medium_n])
    tail_ids = tuple(sorted_ids[head_n + medium_n :])
    return {"head": head_ids, "medium": medium_ids, "tail": tail_ids}


def _compute_group_metrics(
    class_ids: Sequence[int],
    confusion_matrix: torch.Tensor,
    per_class_recall: torch.Tensor,
    per_class_f1: torch.Tensor,
    support: torch.Tensor,
) -> GroupMetrics:
    if len(class_ids) == 0:
        return GroupMetrics(class_ids=tuple(), oa=0.0, aa=0.0, macro_f1=0.0, support=0)

    idx = torch.as_tensor(class_ids, dtype=torch.long)
    group_support_tensor = support.index_select(0, idx)
    valid_mask = group_support_tensor > 0
    group_support = int(group_support_tensor.sum().item())
    sub_matrix = confusion_matrix.index_select(0, idx).index_select(1, idx)
    oa = 0.0 if group_support == 0 else float(sub_matrix.diag().sum().item() / max(group_support, 1))
    if bool(valid_mask.any().item()):
        aa = float(per_class_recall.index_select(0, idx)[valid_mask].mean().item())
        macro_f1 = float(per_class_f1.index_select(0, idx)[valid_mask].mean().item())
    else:
        aa = 0.0
        macro_f1 = 0.0
    return GroupMetrics(
        class_ids=tuple(int(i) for i in class_ids),
        oa=oa,
        aa=aa,
        macro_f1=macro_f1,
        support=group_support,
    )


def compute_classification_metrics(
    *,
    labels: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    preds: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    class_counts: Optional[Iterable[int] | Mapping[int, int] | torch.Tensor] = None,
    head_fraction: float = 0.34,
    tail_fraction: float = 0.34,
) -> ClassificationMetrics:
    """
    Compute the full metric bundle used in imbalanced HSI evaluation.
    """

    labels = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    if preds is None:
        if logits is None:
            raise ValueError("Either logits or preds must be provided.")
        logits = torch.as_tensor(logits)
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {tuple(logits.shape)}.")
        preds = logits.argmax(dim=1)
        if num_classes is None:
            num_classes = int(logits.shape[1])
    else:
        preds = torch.as_tensor(preds, dtype=torch.long).view(-1).cpu()

    if preds.shape[0] != labels.shape[0]:
        raise ValueError("labels and predictions must contain the same number of samples.")

    if num_classes is None:
        if labels.numel() == 0:
            num_classes = 0
        else:
            num_classes = int(max(int(preds.max().item()), int(labels.max().item())) + 1)

    confusion_matrix = compute_confusion_matrix(preds, labels, num_classes=num_classes).to(torch.float32)
    if num_classes == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return ClassificationMetrics(
            oa=0.0,
            aa=0.0,
            kappa=0.0,
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            confusion_matrix=confusion_matrix.to(torch.long),
            support=empty,
            per_class_precision=empty,
            per_class_recall=empty,
            per_class_accuracy=empty,
            per_class_f1=empty,
            class_groups={"head": GroupMetrics(tuple(), 0.0, 0.0, 0.0, 0), "medium": GroupMetrics(tuple(), 0.0, 0.0, 0.0, 0), "tail": GroupMetrics(tuple(), 0.0, 0.0, 0.0, 0)},
        )

    support = confusion_matrix.sum(dim=1)
    pred_support = confusion_matrix.sum(dim=0)
    tp = confusion_matrix.diag()

    per_class_precision = _safe_divide(tp, pred_support)
    per_class_recall = _safe_divide(tp, support)
    per_class_f1 = _safe_divide(2.0 * per_class_precision * per_class_recall, per_class_precision + per_class_recall)
    per_class_accuracy = per_class_recall.clone()

    valid_mask = support > 0
    macro_precision = float(per_class_precision[valid_mask].mean().item()) if bool(valid_mask.any().item()) else 0.0
    macro_recall = float(per_class_recall[valid_mask].mean().item()) if bool(valid_mask.any().item()) else 0.0
    macro_f1 = float(per_class_f1[valid_mask].mean().item()) if bool(valid_mask.any().item()) else 0.0
    oa = float(tp.sum().item() / confusion_matrix.sum().clamp(min=1.0).item())
    aa = macro_recall

    total = confusion_matrix.sum().clamp(min=1.0)
    expected = (support * pred_support).sum() / (total * total)
    observed = tp.sum() / total
    kappa_denom = 1.0 - expected
    if abs(float(kappa_denom.item())) < 1e-12:
        kappa = 0.0
    else:
        kappa = float(((observed - expected) / kappa_denom).item())

    group_ids = build_class_frequency_groups(
        class_counts if class_counts is not None else support,
        num_classes=num_classes,
        head_fraction=head_fraction,
        tail_fraction=tail_fraction,
    )
    group_metrics = {
        name: _compute_group_metrics(ids, confusion_matrix, per_class_recall, per_class_f1, support)
        for name, ids in group_ids.items()
    }

    return ClassificationMetrics(
        oa=oa,
        aa=aa,
        kappa=kappa,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        confusion_matrix=confusion_matrix.to(torch.long),
        support=support,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_accuracy=per_class_accuracy,
        per_class_f1=per_class_f1,
        class_groups=group_metrics,
    )


def format_metric_summary(metrics: ClassificationMetrics, prefix: str = "") -> str:
    parts = []
    if prefix:
        parts.append(prefix.strip())
    parts.extend(
        [
            f"OA={metrics.oa:.4f}",
            f"AA={metrics.aa:.4f}",
            f"Kappa={metrics.kappa:.4f}",
            f"Macro-F1={metrics.macro_f1:.4f}",
        ]
    )
    for name in ("head", "medium", "tail"):
        group = metrics.class_groups.get(name)
        if group is not None and len(group.class_ids) > 0:
            parts.append(f"{name.capitalize()}-AA={group.aa:.4f}")
    return " | ".join(parts)


def format_per_class_table(
    metrics: ClassificationMetrics,
    *,
    inverse_label_mapping: Optional[Mapping[int, int]] = None,
) -> str:
    headers = ["Class", "Orig", "Support", "Acc", "Prec", "Recall", "F1", "Group"]
    rows = ["\t".join(headers)]
    group_lookup: Dict[int, str] = {}
    for group_name, group in metrics.class_groups.items():
        for class_id in group.class_ids:
            group_lookup[int(class_id)] = group_name

    num_classes = int(metrics.support.numel())
    for class_id in range(num_classes):
        orig_label = class_id if inverse_label_mapping is None else inverse_label_mapping.get(class_id, class_id)
        rows.append(
            "\t".join(
                [
                    str(class_id),
                    str(orig_label),
                    str(int(metrics.support[class_id].item())),
                    f"{float(metrics.per_class_accuracy[class_id].item()):.4f}",
                    f"{float(metrics.per_class_precision[class_id].item()):.4f}",
                    f"{float(metrics.per_class_recall[class_id].item()):.4f}",
                    f"{float(metrics.per_class_f1[class_id].item()):.4f}",
                    group_lookup.get(class_id, "-"),
                ]
            )
        )
    return "\n".join(rows)


def format_per_class_accuracy_table(
    metrics: ClassificationMetrics,
    *,
    class_names: Optional[Mapping[int, str]] = None,
    inverse_label_mapping: Optional[Mapping[int, int]] = None,
) -> str:
    headers = ["ID", "Class Name", "Total", "Correct", "Acc (%)"]
    rows = []
    num_classes = int(metrics.support.numel())

    for class_id in range(num_classes):
        display_id = class_id + 1 if inverse_label_mapping is None else int(inverse_label_mapping.get(class_id, class_id + 1))
        class_name = (
            class_names.get(class_id, f"Class_{display_id}")
            if class_names is not None
            else f"Class_{display_id}"
        )
        total = int(metrics.support[class_id].item())
        correct = int(metrics.confusion_matrix[class_id, class_id].item())
        acc_percent = float(metrics.per_class_accuracy[class_id].item()) * 100.0
        rows.append(
            [
                str(display_id),
                class_name,
                str(total),
                str(correct),
                f"{acc_percent:.2f}",
            ]
        )

    widths = [
        max(len(headers[col]), max((len(row[col]) for row in rows), default=0))
        for col in range(len(headers))
    ]
    header_line = "    ".join(f"{headers[col]:<{widths[col]}}" for col in range(len(headers)))
    separator_line = "-" * len(header_line)
    body_lines = [
        "    ".join(f"{row[col]:<{widths[col]}}" for col in range(len(headers)))
        for row in rows
    ]
    summary_lines = [
        separator_line,
        f"Overall Accuracy (OA): {metrics.oa * 100.0:.2f}%",
        f"Average Accuracy (AA): {metrics.aa * 100.0:.2f}%",
        f"Kappa Coefficient    : {metrics.kappa:.4f}",
    ]
    return "\n".join([header_line, separator_line, *body_lines, *summary_lines])
