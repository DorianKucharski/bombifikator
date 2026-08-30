from __future__ import annotations

import threading
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from reference_builder.reference_models import ClusterLabel
from sharedkernel.utils.paths import ensure_directory


def _to_document(label: ClusterLabel) -> dict[str, Any]:
    return {
        "cluster_id": label.cluster_id,
        "slug": label.slug,
        "confidence": round(label.confidence, 3),
        "reasoning": label.reasoning,
    }


def _to_label(document: dict[str, Any]) -> ClusterLabel:
    return ClusterLabel(
            cluster_id=document["cluster_id"],
            slug=document["slug"],
            confidence=document["confidence"],
            reasoning=document["reasoning"],
    )


def read_labels(labels_path: Path) -> tuple[ClusterLabel, ...]:
    if not labels_path.is_file():
        return ()
    with labels_path.open("rb") as handle:
        document = tomllib.load(handle)
    return tuple(_to_label(label) for label in document["labels"])


class LabelStore:
    def __init__(self, labels_path: Path) -> None:
        self._labels_path = labels_path
        self._lock = threading.Lock()
        self._labels: dict[int, ClusterLabel] = {label.cluster_id: label for label in read_labels(labels_path)}

    def contains(self, cluster_id: int) -> bool:
        return cluster_id in self._labels

    def record(self, label: ClusterLabel) -> None:
        with self._lock:
            self._labels[label.cluster_id] = label
            ensure_directory(self._labels_path.parent)
            with self._labels_path.open("wb") as handle:
                tomli_w.dump({
                    "label_count": len(self._labels),
                    "labels": [_to_document(self._labels[cluster_id]) for cluster_id in sorted(self._labels)],
                }, handle)

    def labels(self) -> tuple[ClusterLabel, ...]:
        return tuple(self._labels[cluster_id] for cluster_id in sorted(self._labels))
