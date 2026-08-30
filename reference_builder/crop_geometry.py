from __future__ import annotations

from reference_builder.reference_models import Detection
from sharedkernel.utils.geometry import BoundingBox


def _padded_span(low: int, high: int, padding: int, limit: int) -> tuple[int, int]:
    return max(0, low - padding), min(limit, high + padding)


def pad_box(box: BoundingBox, padding_ratio: float, frame_width: int, frame_height: int) -> BoundingBox:
    padding = int(round(box.shorter_side * padding_ratio))
    left, right = _padded_span(box.left, box.right, padding, frame_width)
    top, bottom = _padded_span(box.top, box.bottom, padding, frame_height)
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def clamp_box(left: float, top: float, right: float, bottom: float,
              frame_width: int, frame_height: int) -> BoundingBox:
    return BoundingBox(
            left=max(0, min(frame_width, int(round(left)))),
            top=max(0, min(frame_height, int(round(top)))),
            right=max(0, min(frame_width, int(round(right)))),
            bottom=max(0, min(frame_height, int(round(bottom)))),
    )


def covers_whole_frame(box: BoundingBox, frame_width: int, frame_height: int, coverage_ratio: float) -> bool:
    frame_area = frame_width * frame_height
    return frame_area > 0 and box.width * box.height >= coverage_ratio * frame_area


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    overlap_width = max(0, min(first.right, second.right) - max(first.left, second.left))
    overlap_height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = overlap_width * overlap_height
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def suppress_overlapping(detections: tuple[Detection, ...], iou_threshold: float) -> tuple[Detection, ...]:
    distinct: list[Detection] = []
    for detection in detections:
        overlaps_kept = any(intersection_over_union(detection.box, kept.box) >= iou_threshold for kept in distinct)
        if not overlaps_kept:
            distinct.append(detection)
    return tuple(distinct)
