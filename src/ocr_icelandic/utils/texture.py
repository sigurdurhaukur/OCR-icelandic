"""Paper texture and background utilities."""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def discover_paper_textures(papers_dir: str = "assets/papers") -> list[str]:
    """
    Discover paper texture files in the specified directory.

    Args:
        papers_dir: Path to directory containing paper texture images

    Returns:
        List of absolute paths to paper texture files
    """
    paper_paths = []
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        return paper_paths

    # Look for common image formats
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        paper_paths.extend(str(p) for p in papers_path.glob(ext))

    return sorted(paper_paths)


def apply_paper_texture(
    image: Image.Image,
    texture_path: str,
    blend_alpha: float = 0.85,
) -> Image.Image:
    """
    Apply a paper texture to an image background.

    This function extracts the texture pattern from the source image and applies it
    to the target background color, similar to GIMP's "Color to Alpha" approach.
    It preserves the texture's shadows, highlights, and surface details while
    replacing the base color.

    Args:
        image: Base image to apply texture to (with desired background color)
        texture_path: Path to paper texture image
        blend_alpha: Alpha value for texture intensity (0.0-1.0, higher = more visible texture)

    Returns:
        Image with paper texture applied
    """
    try:
        # Load the texture in RGB mode to preserve all detail
        texture = Image.open(texture_path).convert("RGB")

        # Resize or tile texture to match image size
        img_width, img_height = image.size
        tex_width, tex_height = texture.size

        # If texture is smaller, tile it
        if tex_width < img_width or tex_height < img_height:
            # Calculate how many tiles we need
            tiles_x = (img_width // tex_width) + 2
            tiles_y = (img_height // tex_height) + 2

            # Create tiled texture
            tiled = Image.new("RGB", (tex_width * tiles_x, tex_height * tiles_y))
            for i in range(tiles_x):
                for j in range(tiles_y):
                    tiled.paste(texture, (i * tex_width, j * tex_height))

            texture = tiled

        # Crop to exact size with random offset for variety
        max_offset_x = max(0, texture.width - img_width)
        max_offset_y = max(0, texture.height - img_height)
        offset_x = random.randint(0, max_offset_x) if max_offset_x > 0 else 0
        offset_y = random.randint(0, max_offset_y) if max_offset_y > 0 else 0

        texture = texture.crop(
            (offset_x, offset_y, offset_x + img_width, offset_y + img_height)
        )

        # Convert to numpy arrays for processing
        texture_array = np.array(texture, dtype=np.float32)
        img_array = np.array(image, dtype=np.float32)

        # Extract luminance of the texture
        # Using standard luminance weights for RGB
        texture_luminance = (
            0.299 * texture_array[:, :, 0]
            + 0.587 * texture_array[:, :, 1]
            + 0.114 * texture_array[:, :, 2]
        )
        mean_luminance = np.mean(texture_luminance)

        # Calculate luminance variation factor for each pixel
        # This represents how much darker/lighter each pixel is compared to average
        luminance_factor = texture_luminance / (
            mean_luminance + 1e-6
        )  # Avoid division by zero

        # Apply the luminance variation to the target background color
        # This preserves shadows and highlights from the texture
        for channel in range(3):
            img_array[:, :, channel] = img_array[:, :, channel] * luminance_factor

        # Mix the result with the original image based on blend_alpha
        # This allows control over texture intensity
        final_array = img_array * blend_alpha + np.array(image, dtype=np.float32) * (
            1 - blend_alpha
        )

        # Clip values to valid range
        final_array = np.clip(final_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        result = Image.fromarray(final_array, mode="RGB")

        return result

    except Exception as e:
        # If texture loading fails, return original image
        print(f"Warning: Failed to apply paper texture from {texture_path}: {e}")
        return image


def discover_backgrounds(
    backgrounds_dir: str = "assets/backgrounds",
) -> tuple[list[str], list[str]]:
    """
    Discover background images separated by shadow type.

    Args:
        backgrounds_dir: Path to directory containing background images

    Returns:
        Tuple of (no_shadow_backgrounds, with_shadow_backgrounds)
    """
    no_shadow_paths = []
    with_shadow_paths = []

    backgrounds_path = Path(backgrounds_dir)
    if not backgrounds_path.exists():
        return no_shadow_paths, with_shadow_paths

    # Discover no_shadow backgrounds
    no_shadow_dir = backgrounds_path / "no_shadow"
    if no_shadow_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            no_shadow_paths.extend(str(p) for p in no_shadow_dir.rglob(ext))

    # Discover with_shadow backgrounds
    with_shadow_dir = backgrounds_path / "with_shadow"
    if with_shadow_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            with_shadow_paths.extend(str(p) for p in with_shadow_dir.rglob(ext))

    return sorted(no_shadow_paths), sorted(with_shadow_paths)


def create_paper_drop_shadow(
    foreground: Image.Image,
    max_offset: int = 3,
    max_blur: int = 5,
    shadow_opacity: int = 80,
) -> tuple[Image.Image, tuple[int, int]]:
    """
    Create a drop shadow for the paper along 2-3 edges.

    Args:
        foreground: The paper image (RGBA with alpha channel defining shape)
        max_offset: Maximum shadow offset in pixels
        max_blur: Maximum blur radius for the shadow
        shadow_opacity: Base opacity of the shadow (0-255)

    Returns:
        Tuple of (shadow image RGBA, shadow offset (x, y))
    """
    # Random shadow offset (small, 1-3 pixels)
    offset_x = random.randint(1, max_offset)
    offset_y = random.randint(1, max_offset)

    # Random blur radius (small, 2-5 pixels)
    blur_radius = random.uniform(2, max_blur)

    # Random opacity variation
    opacity = random.randint(int(shadow_opacity * 0.7), shadow_opacity)

    # Create shadow from foreground alpha channel
    if foreground.mode == "RGBA":
        alpha = foreground.split()[3]
    else:
        # If no alpha, create a solid mask
        alpha = Image.new("L", foreground.size, 255)

    # Create shadow layer (black with alpha from foreground)
    shadow = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
    shadow_alpha = alpha.point(lambda p: int(p * opacity / 255))
    shadow.putalpha(shadow_alpha)

    # Apply blur
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return shadow, (offset_x, offset_y)


def apply_background_image(
    foreground: Image.Image,
    background_path: str,
    paragraph_bboxes: list[dict] | None = None,
    position: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Composite a foreground image (paper with text) onto a background image.

    Args:
        foreground: The paper image with text
        background_path: Path to background image
        paragraph_bboxes: Optional bounding boxes to transform
        position: Optional (x, y) position to place the foreground. If None, centers it.

    Returns:
        Tuple of (composited image, metadata dict, transformed bboxes)
    """
    from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes

    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    try:
        # Load the background
        background = Image.open(background_path).convert("RGBA")
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        # Convert foreground to RGBA if needed
        if foreground.mode != "RGBA":
            foreground = foreground.convert("RGBA")

        # Scale background if it's smaller than foreground
        if bg_width < fg_width or bg_height < fg_height:
            scale_factor = max(fg_width / bg_width, fg_height / bg_height) * 1.2
            new_bg_width = int(bg_width * scale_factor)
            new_bg_height = int(bg_height * scale_factor)
            background = background.resize(
                (new_bg_width, new_bg_height), Image.Resampling.BICUBIC
            )
            bg_width, bg_height = background.size

        # Determine position (center by default, or use provided position)
        if position is None:
            # Center the paper on the background
            x = (bg_width - fg_width) // 2
            y = (bg_height - fg_height) // 2
            position = (x, y)

        # Create drop shadow for the paper
        shadow, shadow_offset = create_paper_drop_shadow(foreground)
        shadow_x = position[0] + shadow_offset[0]
        shadow_y = position[1] + shadow_offset[1]

        # Create composite: first paste shadow, then paper
        composite = background.copy()
        composite.paste(shadow, (shadow_x, shadow_y), shadow)
        composite.paste(foreground, position, foreground)

        # Convert back to RGB
        composite = composite.convert("RGB")

        # Crop back to original foreground size, centered on the pasted paper
        paste_x, paste_y = position
        crop_x = paste_x
        crop_y = paste_y

        # Ensure crop stays within composite bounds
        crop_x = max(0, min(crop_x, composite.width - fg_width))
        crop_y = max(0, min(crop_y, composite.height - fg_height))

        composite = composite.crop(
            (crop_x, crop_y, crop_x + fg_width, crop_y + fg_height)
        )

        # Bounding boxes don't need adjustment since we cropped exactly where the paper was
        # The paper is in the same relative position in the final image as it was in the original

        metadata = {
            "background_path": background_path,
            "position": position,
            "background_size": (bg_width, bg_height),
            "crop_offset": (crop_x, crop_y),
            "drop_shadow_offset": shadow_offset,
        }

        return composite, metadata, paragraph_bboxes_copy

    except Exception as e:
        print(f"Warning: Failed to apply background from {background_path}: {e}")
        # Return original foreground if background application fails
        rgb_foreground = (
            foreground.convert("RGB") if foreground.mode != "RGB" else foreground
        )
        return rgb_foreground, {}, paragraph_bboxes_copy
