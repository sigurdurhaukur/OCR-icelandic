"""Paper texture and background utilities."""

from pathlib import Path

from ocr_icelandic import randomness

import cv2
import numpy as np
from PIL import Image, ImageFilter

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


def discover_paper_textures(papers_dir: str = "assets/papers") -> list[str]:
    """
    Discover paper texture files in the specified directory.

    Args:
        papers_dir: Path to directory containing paper texture images

    Returns:
        List of absolute paths to paper texture files
    """
    logger.debug("Discovering paper textures in directory: %s", papers_dir)
    paper_paths = []
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        logger.warning("Paper textures directory does not exist: %s", papers_dir)
        return paper_paths

    # Look for common image formats
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        paper_paths.extend(str(p) for p in papers_path.glob(ext))

    logger.debug("Found %d paper texture files", len(paper_paths))
    return sorted(paper_paths)


def _tile_texture_seamlessly(
    texture: Image.Image,
    target_size: tuple[int, int],
    random_offset: bool = True,
    blend_edges: bool = True,
    edge_blend_width: int = 25,
) -> Image.Image:
    """
    Tile texture to fill target size with seamless edges.

    Uses mirror tiling (flip alternating tiles) and edge blending
    to minimize visible seams at tile boundaries.

    Args:
        texture: Source texture to tile
        target_size: (width, height) of target image
        random_offset: Apply random offset for variety
        blend_edges: Apply Gaussian blending at tile boundaries
        edge_blend_width: Width of edge blend in pixels

    Returns:
        Tiled texture matching target_size
    """
    target_width, target_height = target_size
    tex_width, tex_height = texture.size

    # If texture is larger, just crop with random offset
    if tex_width >= target_width and tex_height >= target_height:
        max_offset_x = max(0, tex_width - target_width)
        max_offset_y = max(0, tex_height - target_height)
        offset_x = (
            randomness.randint(0, max_offset_x)
            if random_offset and max_offset_x > 0
            else 0
        )
        offset_y = (
            randomness.randint(0, max_offset_y)
            if random_offset and max_offset_y > 0
            else 0
        )
        return texture.crop(
            (offset_x, offset_y, offset_x + target_width, offset_y + target_height)
        )

    # Calculate tiles needed (add extra for random offset)
    tiles_x = (target_width // tex_width) + 2
    tiles_y = (target_height // tex_height) + 2

    # Create tiled texture with mirror pattern to reduce repetition
    tiled = Image.new(texture.mode, (tex_width * tiles_x, tex_height * tiles_y))
    for i in range(tiles_x):
        for j in range(tiles_y):
            # Mirror flip alternating tiles (checkerboard pattern)
            tile = texture
            if (i + j) % 2 == 1:  # Alternate tiles
                tile = texture.transpose(Image.FLIP_LEFT_RIGHT)
            tiled.paste(tile, (i * tex_width, j * tex_height))

    # Apply edge blending if enabled (smooth transitions at tile boundaries)
    if blend_edges:
        tiled = _apply_edge_blending(tiled, tex_width, tex_height, edge_blend_width)

    # Crop to target size with random offset
    max_offset_x = max(0, tiled.width - target_width)
    max_offset_y = max(0, tiled.height - target_height)
    offset_x = (
        randomness.randint(0, max_offset_x) if random_offset and max_offset_x > 0 else 0
    )
    offset_y = (
        randomness.randint(0, max_offset_y) if random_offset and max_offset_y > 0 else 0
    )

    return tiled.crop(
        (offset_x, offset_y, offset_x + target_width, offset_y + target_height)
    )


def _apply_edge_blending(
    tiled: Image.Image,
    tile_width: int,
    tile_height: int,
    blend_width: int,
) -> Image.Image:
    """
    Apply Gaussian blending at tile boundaries to hide seams.

    Creates gradient masks at tile edges and applies localized blur
    to smooth transitions between tiles.

    Args:
        tiled: Tiled texture image
        tile_width: Width of individual tiles
        tile_height: Height of individual tiles
        blend_width: Width of edge blend in pixels

    Returns:
        Tiled texture with smoothed edges
    """
    # Create blend mask for vertical seams
    for i in range(1, tiled.width // tile_width):
        x = i * tile_width
        if x < tiled.width - blend_width:
            # Apply slight blur near seam
            region = tiled.crop(
                (x - blend_width // 2, 0, x + blend_width // 2, tiled.height)
            )
            blurred = region.filter(ImageFilter.GaussianBlur(radius=1.0))
            tiled.paste(blurred, (x - blend_width // 2, 0))

    # Create blend mask for horizontal seams
    for j in range(1, tiled.height // tile_height):
        y = j * tile_height
        if y < tiled.height - blend_width:
            # Apply slight blur near seam
            region = tiled.crop(
                (0, y - blend_width // 2, tiled.width, y + blend_width // 2)
            )
            blurred = region.filter(ImageFilter.GaussianBlur(radius=1.0))
            tiled.paste(blurred, (0, y - blend_width // 2))

    return tiled


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
    logger.debug(
        "Applying paper texture from '%s' with blend_alpha=%.2f",
        texture_path,
        blend_alpha,
    )
    try:
        # Load the texture in RGB mode to preserve all detail
        texture = Image.open(texture_path).convert("RGB")
        logger.debug("Loaded texture: %dx%d", texture.width, texture.height)

        # Use seamless tiling for paper texture
        img_width, img_height = image.size
        texture = _tile_texture_seamlessly(
            texture,
            (img_width, img_height),
            random_offset=True,
            blend_edges=True,
            edge_blend_width=25,
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
        logger.debug("Paper texture applied successfully")

        return result

    except Exception as e:
        # If texture loading fails, return original image
        logger.error("Failed to apply paper texture from '%s': %s", texture_path, e)
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
    logger.debug("Discovering background images in: %s", backgrounds_dir)
    no_shadow_paths = []
    with_shadow_paths = []

    backgrounds_path = Path(backgrounds_dir)
    if not backgrounds_path.exists():
        logger.warning("Backgrounds directory does not exist: %s", backgrounds_dir)
        return no_shadow_paths, with_shadow_paths

    # Discover no_shadow backgrounds
    no_shadow_dir = backgrounds_path / "no_shadow"
    if no_shadow_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            no_shadow_paths.extend(str(p) for p in no_shadow_dir.rglob(ext))
        logger.debug("Found %d no_shadow backgrounds", len(no_shadow_paths))

    # Discover with_shadow backgrounds
    with_shadow_dir = backgrounds_path / "with_shadow"
    if with_shadow_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            with_shadow_paths.extend(str(p) for p in with_shadow_dir.rglob(ext))
        logger.debug("Found %d with_shadow backgrounds", len(with_shadow_paths))

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
    offset_x = randomness.randint(1, max_offset)
    offset_y = randomness.randint(1, max_offset)

    # Random blur radius (small, 2-5 pixels)
    blur_radius = randomness.uniform(2, max_blur)

    # Random opacity variation
    opacity = randomness.randint(int(shadow_opacity * 0.7), shadow_opacity)

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
    background: str | Image.Image,
    paragraph_bboxes: list[dict] | None = None,
    position: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """
    Composite a foreground image (paper with text) onto a background image.

    Args:
        foreground: The paper image with text
        background: Path to background image or pre-loaded/transformed Image
        paragraph_bboxes: Optional bounding boxes to transform
        position: Optional (x, y) position to place the foreground. If None, centers it.

    Returns:
        Tuple of (composited image, metadata dict, transformed bboxes)
    """
    from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes

    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    try:
        # Load or convert the background
        was_pretransformed = False
        background_path_str = None

        if isinstance(background, str):
            logger.debug("Loading background image from path: '%s'", background)
            background_path_str = background
            background_img = Image.open(background).convert("RGBA")
        elif isinstance(background, Image.Image):
            logger.debug("Using pre-transformed background image")
            was_pretransformed = True
            background_img = background.convert("RGBA")
        else:
            raise TypeError(f"background must be str or Image, got {type(background)}")

        bg_width, bg_height = background_img.size
        fg_width, fg_height = foreground.size
        logger.debug(
            "Background (%dx%d) and foreground (%dx%d)",
            bg_width,
            bg_height,
            fg_width,
            fg_height,
        )

        # Convert foreground to RGBA if needed
        if foreground.mode != "RGBA":
            foreground = foreground.convert("RGBA")

        # Scale background if it's smaller than foreground
        if bg_width < fg_width or bg_height < fg_height:
            scale_factor = max(fg_width / bg_width, fg_height / bg_height) * 1.2
            new_bg_width = int(bg_width * scale_factor)
            new_bg_height = int(bg_height * scale_factor)
            logger.debug(
                "Scaling background from (%dx%d) to (%dx%d)",
                bg_width,
                bg_height,
                new_bg_width,
                new_bg_height,
            )
            background_img = background_img.resize(
                (new_bg_width, new_bg_height), Image.Resampling.BICUBIC
            )
            bg_width, bg_height = background_img.size

        # Determine position (center by default, or use provided position)
        if position is None:
            # Center the paper on the background
            x = (bg_width - fg_width) // 2
            y = (bg_height - fg_height) // 2
            position = (x, y)
            logger.debug("Auto-centered foreground at position (%d, %d)", x, y)
        else:
            logger.debug("Using provided position: (%d, %d)", position[0], position[1])

        # Create drop shadow for the paper
        shadow, shadow_offset = create_paper_drop_shadow(foreground)
        shadow_x = position[0] + shadow_offset[0]
        shadow_y = position[1] + shadow_offset[1]

        # Create composite: first paste shadow, then paper
        composite = background_img.copy()
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
            "background_path": background_path_str,  # None if pre-transformed
            "was_pretransformed": was_pretransformed,
            "position": position,
            "background_size": (bg_width, bg_height),
            "crop_offset": (crop_x, crop_y),
            "drop_shadow_offset": shadow_offset,
        }

        return composite, metadata, paragraph_bboxes_copy

    except Exception as e:
        bg_desc = (
            background_path_str if background_path_str else "pre-transformed image"
        )
        print(f"Warning: Failed to apply background from {bg_desc}: {e}")
        # Return original foreground if background application fails
        rgb_foreground = (
            foreground.convert("RGB") if foreground.mode != "RGB" else foreground
        )
        return rgb_foreground, {}, paragraph_bboxes_copy


def texture_to_height_map(
    texture_path: str,
    size: tuple[int, int],
    blur_radius: float = 3.0,
    contrast: float = 1.5,
) -> np.ndarray:
    """
    Convert paper texture to a normalized height map using luminance.

    The height map represents surface elevation where brighter areas
    are "higher" (closer to viewer) and darker areas are "lower" (further away).
    High contrast enhancement is applied to make subtle texture variations
    more pronounced for displacement mapping.

    Args:
        texture_path: Path to paper texture image
        size: Target (width, height) to resize/tile to
        blur_radius: Gaussian blur to smooth the height map for gradual gradients
        contrast: Contrast multiplier (1.0 = no change, 3.0 = high contrast)

    Returns:
        Height map as float32 array (0.0 to 1.0) with shape (height, width)
    """
    logger.debug(
        "Converting texture to height map: size=%dx%d, blur=%.1f, contrast=%.1f",
        size[0],
        size[1],
        blur_radius,
        contrast,
    )
    # Load texture and convert to grayscale
    texture = Image.open(texture_path).convert("L")
    img_width, img_height = size
    logger.debug("Loaded texture for height map: %dx%d", texture.width, texture.height)

    # Use seamless tiling for height map texture
    texture = _tile_texture_seamlessly(
        texture,
        size,
        random_offset=True,
        blend_edges=True,
        edge_blend_width=8,  # Slightly smaller for height maps
    )

    # Apply Gaussian blur for smooth gradients
    if blur_radius > 0:
        logger.debug("Applying Gaussian blur with radius %.1f", blur_radius)
        texture = texture.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert to float32 and normalize to 0-1 range
    height_map = np.array(texture, dtype=np.float32) / 255.0

    # Apply contrast enhancement to make texture variations more pronounced
    # First, stretch to full range (histogram stretching)
    min_val = height_map.min()
    max_val = height_map.max()
    logger.debug("Height map range before stretching: [%.4f, %.4f]", min_val, max_val)
    if max_val > min_val:
        height_map = (height_map - min_val) / (max_val - min_val)

    # Then apply contrast multiplier around the midpoint
    if contrast != 1.0:
        height_map = (height_map - 0.5) * contrast + 0.5
        height_map = np.clip(height_map, 0.0, 1.0)
        logger.debug("Applied contrast enhancement: multiplier=%.1f", contrast)

    logger.debug("Height map created successfully")
    return height_map


def apply_displacement_from_texture(
    image: Image.Image,
    texture_path: str,
    displacement_strength: float = 0.5,
) -> Image.Image:
    """
    Apply displacement warping and lighting based on paper texture.

    This makes text "hug" the paper folds/creases by:
    1. Converting paper texture to a height map (luminance)
    2. Computing gradients to determine displacement direction
    3. Warping pixels using cv2.remap
    4. Optionally applying lighting based on surface normals

    Args:
        image: Image to warp (typically with text already blended)
        texture_path: Path to paper texture image
        displacement_strength: Pixel displacement multiplier (1.0-5.0 typical)

    Returns:
        Warped (and optionally lit) image
    """
    logger.debug(
        "Applying displacement from texture: '%s' with strength=%.2f",
        texture_path,
        displacement_strength,
    )
    # Get image dimensions
    img_width, img_height = image.size
    logger.debug("Image dimensions: %dx%d", img_width, img_height)

    # Generate height map from texture
    height_map = texture_to_height_map(
        texture_path, (img_width, img_height), blur_radius=1.0
    )

    # Compute gradients for displacement
    grad_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=5)

    # Normalize gradients to prevent extreme displacement
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_grad = grad_magnitude.max()
    if max_grad > 0:
        grad_x /= max_grad
        grad_y /= max_grad
        logger.debug("Gradient normalization complete: max_gradient=%.4f", max_grad)

    # Create displacement maps
    # Pixels shift perpendicular to gradient (along contour lines)
    h, w = height_map.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Displacement is in the direction of the gradient
    map_x = (x_coords + grad_x * displacement_strength).astype(np.float32)
    map_y = (y_coords + grad_y * displacement_strength).astype(np.float32)

    # Convert image to numpy array
    img_array = np.array(image)

    # Apply remapping with bilinear interpolation
    # BORDER_REFLECT prevents edge artifacts
    logger.debug("Applying displacement mapping")
    remapped = cv2.remap(
        img_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    logger.debug("Displacement mapping complete")
    # Convert back to PIL Image
    return Image.fromarray(remapped)
