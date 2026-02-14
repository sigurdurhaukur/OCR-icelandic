from typing import Any


def _copy_paragraph_bboxes(
    paragraph_bboxes: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Create a deep copy of paragraph bounding boxes."""
    if not paragraph_bboxes:
        return []
    return [{**bbox, "bbox": list(bbox.get("bbox", []))} for bbox in paragraph_bboxes]


def _clamp_value(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to the specified range."""
    return max(minimum, min(value, maximum))


def _round_bbox(coords: list[float]) -> list[int]:
    """Round bbox coordinates to integers."""
    return [int(round(value)) for value in coords]
