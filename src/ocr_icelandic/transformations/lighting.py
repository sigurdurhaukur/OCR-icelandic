"""Lighting and shadow transformations for synthetic OCR images.

This module contains transformations that simulate lighting conditions
during document photography or scanning:
- light_reflection: Simulated camera flash or light spots
- shadow_overlay: Edge shadows from scanning or photography
- shadow_gradient: Gradient shadow effects for depth
"""

from typing import Any

from ocr_icelandic import randomness

from PIL import Image, ImageDraw, ImageFilter

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes

logger = get_logger(__name__)


def light_reflection(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Simulate light reflection on the image.

    Creates an elliptical bright spot to simulate camera flash
    or overhead lighting reflection.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (image with reflection, metadata dict, unchanged bboxes)
    """
    logger.debug("Applying light reflection transformation")
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # Get image dimensions
    width, height = image.size

    # Get position for the reflection center
    center_x = randomness.randint(int(width * 0.2), int(width * 0.8))
    center_y = randomness.randint(int(height * 0.2), int(height * 0.8))

    # Get ellipse size
    ellipse_width = randomness.randint(width // 8, width // 4)
    ellipse_height = randomness.randint(height // 8, height // 6)
    logger.debug(
        "Light reflection: center=(%d,%d), ellipse=(%d,%d)",
        center_x,
        center_y,
        ellipse_width,
        ellipse_height,
    )

    # Create a mask for the reflection
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)

    # Draw ellipse on mask
    left = center_x - ellipse_width // 2
    top = center_y - ellipse_height // 2
    right = center_x + ellipse_width // 2
    bottom = center_y + ellipse_height // 2

    mask_draw.ellipse([left, top, right, bottom], fill=255)

    # Apply blur for softer edges
    mask = mask.filter(
        ImageFilter.GaussianBlur(radius=(ellipse_width + ellipse_height) // 4)
    )

    # Create overlay
    if isinstance(bg_color, str):
        base_light_color = [255, 255, 255, 180]  # Semi-transparent white
    else:
        if len(bg_color) == 3:
            base_light_color = [*bg_color, 180]  # Semi-transparent version of bg_color
        else:
            base_light_color = list(bg_color)
    for i in range(3):
        base_light_color[i] = min(255, base_light_color[i] + randomness.randint(10, 30))

    light_color = tuple(base_light_color)
    reflection = Image.new("RGBA", (width, height), light_color)
    reflection.putalpha(mask)

    # Overlay over image
    result = Image.alpha_composite(image.convert("RGBA"), reflection)

    return (
        result.convert(image.mode),
        {
            "transformation": "light_reflection",
            "center_x": center_x,
            "center_y": center_y,
            "ellipse_width": ellipse_width,
            "ellipse_height": ellipse_height,
        },
        paragraph_bboxes_copy,
    )


def shadow_overlay(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Cast a random uneven shadow from one edge with fuzzy borders.

    Simulates shadows cast during document photography, such as
    from the camera or photographer's hand.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (shadowed image, metadata dict, unchanged bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Shadow layer
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    # Random edge selection (0=top, 1=right, 2=bottom, 3=left)
    edge = randomness.randint(0, 3)

    # Random shadow parameters - how far the shadow extends and its opacity
    max_depth = randomness.uniform(0.15, 0.5) * min(image.width, image.height)
    opacity = randomness.randint(20, 120)

    # Create uneven shadow polygon with integer coordinates
    points: list[tuple[float, float]] = []
    polygons_points = 3
    if edge == 0:  # Top edge
        points = [(0.0, 0.0), (float(image.width), 0.0)]
        for i in range(polygons_points):
            x = (i + 1) * image.width / 6
            y = randomness.uniform(max_depth * 0.3, max_depth)
            points.append((x, y))
        points.append((0.0, randomness.uniform(max_depth * 0.3, max_depth)))
    elif edge == 1:  # Right edge
        points = [(float(image.width), 0.0), (float(image.width), float(image.height))]
        for i in range(polygons_points):
            x = image.width - randomness.uniform(max_depth * 0.3, max_depth)
            y = (i + 1) * image.height / 6
            points.append((x, y))
        points.append(
            (image.width - randomness.uniform(max_depth * 0.3, max_depth), 0.0)
        )
    elif edge == 2:  # Bottom edge
        points = [(0.0, float(image.height)), (float(image.width), float(image.height))]
        for i in range(polygons_points):
            x = (i + 1) * image.width / 6
            y = image.height - randomness.uniform(max_depth * 0.3, max_depth)
            points.append((x, y))
        points.append(
            (0.0, image.height - randomness.uniform(max_depth * 0.3, max_depth))
        )
    else:  # Left edge
        points = [(0.0, 0.0), (0.0, float(image.height))]
        for i in range(polygons_points):
            x = randomness.uniform(max_depth * 0.3, max_depth)
            y = (i + 1) * image.height / 6
            points.append((x, y))
        points.append((randomness.uniform(max_depth * 0.3, max_depth), 0.0))

    # Draw shadow polygon
    shadow_draw.polygon(points, fill=(0, 0, 0, opacity))

    # Apply blur for fuzzy edges
    blur_radius = randomness.uniform(10, 30)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Composite with original image
    image = Image.alpha_composite(image, shadow)

    return (
        image,
        {
            "transformation": "shadow_overlay",
            "edge": edge,
            "max_depth": round(max_depth, 2),
            "opacity": opacity,
            "blur_radius": round(blur_radius, 2),
        },
        paragraph_bboxes_copy,
    )


def shadow_gradient(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Apply a shadow gradient overlay to simulate lighting effects.

    Creates a vertical gradient from transparent to semi-transparent
    black to simulate uneven lighting conditions.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (gradient-shadowed image, metadata dict, unchanged bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    width, height = image.size

    # Create gradient
    gradient = Image.new("L", (1, height), color=0xFF)
    gradient_opacity = randomness.uniform(0.3, 0.7)
    for y in range(height):
        # Gradient from transparent to semi-transparent black
        gradient.putpixel((0, y), int(255 * (y / height) * gradient_opacity))
    alpha_gradient = gradient.resize((width, height))

    # Create shadow overlay
    shadow_overlay_layer = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    shadow_overlay_layer.putalpha(alpha_gradient)

    # Composite with original image
    image = Image.alpha_composite(image, shadow_overlay_layer)

    return (
        image,
        {
            "transformation": "shadow_gradient",
            "gradient_opacity": round(gradient_opacity, 2),
        },
        paragraph_bboxes_copy,
    )
