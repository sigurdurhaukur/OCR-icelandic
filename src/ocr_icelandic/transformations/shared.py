from typing import Any

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


def _copy_paragraph_bboxes(
    paragraph_bboxes: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not paragraph_bboxes:
        return []
    result = [{**bbox, "bbox": list(bbox.get("bbox", []))} for bbox in paragraph_bboxes]
    logger.debug("Copied %d paragraph bounding boxes", len(result))
    return result


def _clamp_value(value: float, minimum: float, maximum: float) -> float:
    clamped = max(minimum, min(value, maximum))
    if clamped != value:
        logger.debug("Clamped value %.2f to range [%.2f, %.2f] -> %.2f", value, minimum, maximum, clamped)
    return clamped


def _round_bbox(coords: list[float]) -> list[int]:
    result = [int(round(value)) for value in coords]
    logger.debug("Rounded bbox coordinates: %s -> %s", coords, result)
    return result
