"""Content effect transformations for synthetic OCR images.

This module contains transformations that simulate document aging,
damage, and printing artifacts:
- blur: Gaussian blur to simulate camera focus issues
- textured_stains: Coffee/tea stain textures from asset files
- dusty_paper: Grainy paper texture overlay
- reverse_bleed_through: Text bleeding through from reverse side
- paper_edge_unevenness: Light wavy/irregular paper edges
"""

from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes

logger = get_logger(__name__)


def blur(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Apply Gaussian blur to simulate camera focus issues.

    Args:
        image: Input image to blur
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged by blur)

    Returns:
        Tuple of (blurred image, metadata dict, unchanged bboxes)
    """
    logger.debug("Applying blur transformation")
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    radius = random.uniform(0.1, 0.5)
    logger.debug("Blur radius: %.2f", radius)
    return (
        image.filter(ImageFilter.GaussianBlur(radius)),
        {
            "transformation": "blur",
            "radius": round(radius, 2),
        },
        paragraph_bboxes_copy,
    )


# Load stain textures from assets directory
stain_textures = list(Path("assets/stains").rglob("*.png")) + list(Path("assets/stains").rglob("*.jpg"))


def textured_stains(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Apply coffee/tea stain textures using multiply blending.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (stained image, metadata dict, unchanged bboxes)
    """
    logger.debug("Applying textured stains transformation")
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    texture = random.choice(stain_textures)
    logger.debug("Using stain texture: %s", texture)
    stain = Image.open(texture).convert("RGBA")
    # Adjust scale factor to ensure stain fits within image
    max_scale = min(image.width / stain.width, image.height / stain.height) * 0.8
    scale_factor = random.uniform(0.5, min(1.5, max_scale))
    new_size = (int(stain.width * scale_factor), int(stain.height * scale_factor))
    stain = stain.resize(new_size, Image.Resampling.LANCZOS)

    # Reduce opacity to 80%
    alpha = stain.split()[3]
    alpha = alpha.point(lambda p: int(p * 0.8))
    stain.putalpha(alpha)

    # Apply edge gradient for smooth transition at boundaries
    # Create a distance-from-edge mask that fades to transparency
    stain_w, stain_h = stain.size
    edge_fade_size = int(min(stain_w, stain_h) * 0.15)  # 15% of smaller dimension

    if edge_fade_size > 0:
        # Create coordinate arrays for distance calculation
        y_coords, x_coords = np.ogrid[:stain_h, :stain_w]

        # Calculate distance from each edge
        dist_left = x_coords
        dist_right = stain_w - 1 - x_coords
        dist_top = y_coords
        dist_bottom = stain_h - 1 - y_coords

        # Minimum distance to any edge
        dist_to_edge = np.minimum(
            np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom)
        )

        # Create gradient: 0 at edge, 1 at edge_fade_size distance
        edge_gradient = np.clip(dist_to_edge / edge_fade_size, 0, 1).astype(np.float32)

        # Apply smooth easing (ease-in-out) for more natural transition
        edge_gradient = edge_gradient * edge_gradient * (3 - 2 * edge_gradient)

        # Apply gradient to existing alpha channel
        alpha_array = np.array(stain.split()[3], dtype=np.float32)
        alpha_array = alpha_array * edge_gradient
        stain.putalpha(Image.fromarray(alpha_array.astype(np.uint8)))

    # Allow stain to be positioned partially outside image bounds
    pos_x = random.randint(-stain.width // 2, image.width - stain.width // 2)
    pos_y = random.randint(-stain.height // 2, image.height - stain.height // 2)

    # Ensure image is RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Apply multiply blending for more realistic stain effect
    # Multiply blend: result = (paper * stain) / 255
    # This makes stains naturally darken the paper while texture shows through
    img_array = np.array(image, dtype=np.float32)

    # Create a full-size stain layer with white (neutral for multiply) background
    stain_layer = np.ones((image.height, image.width, 4), dtype=np.float32) * 255.0
    stain_alpha_layer = np.zeros((image.height, image.width), dtype=np.float32)

    # Calculate the region where stain overlaps with image
    stain_x1 = max(0, pos_x)
    stain_y1 = max(0, pos_y)
    stain_x2 = min(image.width, pos_x + stain.width)
    stain_y2 = min(image.height, pos_y + stain.height)

    # Corresponding region in stain texture
    tex_x1 = max(0, -pos_x)
    tex_y1 = max(0, -pos_y)
    tex_x2 = tex_x1 + (stain_x2 - stain_x1)
    tex_y2 = tex_y1 + (stain_y2 - stain_y1)

    # Get stain data as numpy
    stain_array = np.array(stain, dtype=np.float32)

    # Copy stain RGB and alpha to full-size layers
    if stain_x2 > stain_x1 and stain_y2 > stain_y1:
        stain_layer[stain_y1:stain_y2, stain_x1:stain_x2, :3] = stain_array[
            tex_y1:tex_y2, tex_x1:tex_x2, :3
        ]
        stain_alpha_layer[stain_y1:stain_y2, stain_x1:stain_x2] = (
            stain_array[tex_y1:tex_y2, tex_x1:tex_x2, 3] / 255.0
        )

    # Apply multiply blend for RGB channels
    # Where stain is present (alpha > 0), blend = (img * stain) / 255
    # Where stain is absent (alpha = 0), result = original
    multiplied = (img_array[:, :, :3] * stain_layer[:, :, :3]) / 255.0

    # Blend based on stain alpha: result = multiplied * alpha + original * (1 - alpha)
    result = np.zeros_like(img_array)
    for c in range(3):
        result[:, :, c] = multiplied[:, :, c] * stain_alpha_layer + img_array[
            :, :, c
        ] * (1 - stain_alpha_layer)
    # Preserve original alpha channel
    result[:, :, 3] = img_array[:, :, 3]

    result = np.clip(result, 0, 255).astype(np.uint8)
    combined = Image.fromarray(result, mode="RGBA")

    return (
        combined,
        {
            "transformation": "coffee_stains",
            "position": (pos_x, pos_y),
            "scale_factor": round(scale_factor, 2),
            "blend_mode": "multiply",
            "edge_fade_size": edge_fade_size,
        },
        paragraph_bboxes_copy,
    )


def dusty_paper(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Create grainy overlay to simulate dusty paper.

    Varies in grain size and intensity.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (dusty image, metadata dict, unchanged bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    grain_size = random.randint(1, 3)
    intensity = random.uniform(0.05, 0.15)
    noise = Image.effect_noise(image.size, grain_size * 10)
    grainy_overlay = noise.convert("RGBA" if image.mode == "RGBA" else "RGB")
    dusty_image = Image.blend(image, grainy_overlay, intensity)
    return (
        dusty_image,
        {
            "transformation": "dusty-paper",
            "grain_size": grain_size,
            "intensity": round(intensity, 3),
        },
        paragraph_bboxes_copy,
    )


def reverse_bleed_through(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Simulate bleed-through effect from text on reverse side of page.

    Args:
        image: Input image
        bg_color: Background color (unused, kept for API consistency)
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (image with bleed-through, metadata dict, unchanged bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    intensity = np.random.uniform(
        0.01, 0.04
    )  # Adjust intensity of the bleed-through effect

    # Store original alpha channel if present
    has_alpha = image.mode == "RGBA"
    if has_alpha:
        alpha_channel = image.split()[3]

    # Convert PIL image to numpy array (RGB only for processing)
    if has_alpha:
        img_rgb = image.convert("RGB")
        img_array = np.array(img_rgb)
    else:
        img_array = np.array(image)

    # Flip the image horizontally
    flipped = cv2.flip(img_array, 1)

    # Apply random shift
    # Calculate minimum shift based on image size (10% of width/height)
    min_shift_x = max(3, int(img_array.shape[1] * 0.1))
    min_shift_y = max(3, int(img_array.shape[0] * 0.1))

    # Generate random shift
    shift_x = int(
        np.random.choice([-1, 1]) * np.random.randint(min_shift_x, min_shift_x + 10)
    )
    shift_y = int(
        np.random.choice([-1, 1]) * np.random.randint(min_shift_y, min_shift_y + 10)
    )

    # Create transformation matrix to shift the flipped image (bleed-through)
    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    shifted = cv2.warpAffine(
        flipped,
        M,
        (img_array.shape[1], img_array.shape[0]),
        borderValue=(255, 255, 255),
    )

    # Create mask for dark colors (low intensity values - light colors should not bleed through)
    gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    dark_mask = gray_shifted < 128

    # Apply where original image is light and shifted image is dark
    gray_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    light_mask = gray_original > 200

    # Combine masks
    bleed_mask = dark_mask & light_mask

    # Apply the effect
    result = img_array.copy().astype(np.float32)
    for i in range(3):  # Apply to each color channel
        result[:, :, i] = np.where(
            bleed_mask,
            img_array[:, :, i] * (1 - intensity) + shifted[:, :, i] * intensity,
            img_array[:, :, i],
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(result)

    # Restore alpha channel if original had it
    if has_alpha:
        result_image = result_image.convert("RGBA")
        result_image.putalpha(alpha_channel)

    return (
        result_image,
        {
            "transformation": "reverse_bleed_through",
            "intensity": round(float(intensity), 3),
            "shift_x": shift_x,
            "shift_y": shift_y,
        },
        paragraph_bboxes_copy,
    )


def paper_edge_unevenness(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, dict[str, Any], list[dict[str, Any]]]:
    """Apply light paper edge unevenness to simulate real paper edges.

    Creates slightly wavy/irregular edges to make documents look like
    real paper that has been cut or has natural edge imperfections.

    Args:
        image: Input image
        bg_color: Background color to fill outside the uneven edges
        paragraph_bboxes: Optional bounding boxes (unchanged)

    Returns:
        Tuple of (image with uneven edges, metadata dict, unchanged bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # Parameters for light edge unevenness
    max_deviation = random.randint(1, 5)  # Small deviations for subtle effect
    num_points = random.randint(10, 20)  # Control points per edge

    width, height = image.size

    # Generate wavy edge points for each edge
    def generate_edge_points(
        start: tuple[int, int],
        end: tuple[int, int],
        num_pts: int,
        max_dev: int,
        inward_direction: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Generate points along an edge with random deviations.

        Note: First and last points have no deviation so corners meet cleanly.
        """
        points = []
        for i in range(num_pts + 1):
            t = i / num_pts
            # Base position along edge
            x = int(start[0] + t * (end[0] - start[0]))
            y = int(start[1] + t * (end[1] - start[1]))
            # Add random deviation inward, but not at endpoints (corners)
            if i == 0 or i == num_pts:
                deviation = 0
            else:
                deviation = random.randint(0, max_dev)
            x += inward_direction[0] * deviation
            y += inward_direction[1] * deviation
            points.append((x, y))
        return points

    # Generate points for all four edges (deviations go inward)
    top_points = generate_edge_points(
        (0, 0), (width, 0), num_points, max_deviation, (0, 1)
    )
    right_points = generate_edge_points(
        (width, 0), (width, height), num_points, max_deviation, (-1, 0)
    )
    bottom_points = generate_edge_points(
        (width, height), (0, height), num_points, max_deviation, (0, -1)
    )
    left_points = generate_edge_points(
        (0, height), (0, 0), num_points, max_deviation, (1, 0)
    )

    # Combine all points into a polygon (going clockwise)
    polygon_points = top_points + right_points + bottom_points + left_points

    # Draw the polygon on mask (black outside, white inside)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)

    # Apply slight blur to soften the jagged edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Ensure image is RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Multiply existing alpha with wavy edge mask
    # This cuts away the edges while preserving existing transparency
    result = image.copy()
    r, g, b, a = result.split()

    mask_array = np.array(mask, dtype=np.float32) / 255.0
    alpha_array = np.array(a, dtype=np.float32) / 255.0
    new_alpha = (mask_array * alpha_array * 255).astype(np.uint8)

    result.putalpha(Image.fromarray(new_alpha))

    return (
        result,
        {
            "transformation": "paper_edge_unevenness",
            "max_deviation": max_deviation,
            "num_points": num_points,
        },
        paragraph_bboxes_copy,
    )
