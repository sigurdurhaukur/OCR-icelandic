def _copy_paragraph_bboxes(paragraph_bboxes: list[dict] | None) -> list[dict]:
    if not paragraph_bboxes:
        return []
    return [{**bbox, "bbox": list(bbox.get("bbox", []))} for bbox in paragraph_bboxes]


def _clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _round_bbox(coords: list[float]) -> list[int]:
    return [int(round(value)) for value in coords]
