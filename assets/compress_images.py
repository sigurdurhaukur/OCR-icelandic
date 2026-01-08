import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import math
import logging
import psutil
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

INPUT_FOLDER = Path(".")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
MAX_PIXELS = 12_000_000  # 12 MP
JPEG_QUALITY = 90

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)


def is_image(file_path: Path) -> bool:
    is_img = file_path.suffix.lower() in IMAGE_EXTENSIONS
    if not is_img:
        logger.debug(f"Skipped non-image file: {file_path.name}")
    return is_img


def process_image(image_path: Path) -> None:
    try:
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif")

            width, height = img.size
            total_pixels = width * height

            # Check if resize is needed
            needs_resize = total_pixels > MAX_PIXELS

            # Check if image has meaningful transparency
            has_transparency = False
            if img.mode in ("RGBA", "LA"):
                alpha = img.getchannel("A")
                alpha_values = set(list(alpha.getdata()))  # Convert to list first
                # If alpha has more than one unique value and not all 255, keep transparency
                has_transparency = len(alpha_values) > 1 and not (alpha_values == {255})
                logger.debug(
                    f"{image_path.name}: Mode={img.mode}, Alpha values={len(alpha_values)}, Has transparency={has_transparency}"
                )
            elif img.mode == "P" and "transparency" in img.info:
                has_transparency = True
                logger.debug(f"{image_path.name}: Palette mode with transparency")
            else:
                logger.debug(
                    f"{image_path.name}: Mode={img.mode}, No transparency channel"
                )

            # Determine correct format
            current_ext = image_path.suffix.lower()
            if has_transparency:
                needs_format_change = current_ext != ".png"
            else:
                needs_format_change = current_ext not in {".jpg", ".jpeg"}

            # Check if processing is needed
            if not needs_resize and not needs_format_change:
                logger.debug(
                    f"Skipped {image_path.name} - already compliant ({width}x{height}, {total_pixels:,} pixels, format={current_ext})"
                )
                return

            # Process image if needed
            if needs_resize:
                scale = math.sqrt(MAX_PIXELS / total_pixels)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized {image_path.name} from {width}x{height} to {new_width}x{new_height}"
                )
            else:
                logger.debug(
                    f"No resize needed for {image_path.name} ({width}x{height}, {total_pixels:,} pixels)"
                )

            if has_transparency:
                # Keep original format with transparency
                if img.mode == "P":
                    img = img.convert("RGBA")
                    logger.debug(f"Converted {image_path.name} from palette to RGBA")
                output_path = image_path.with_suffix(".png")
                img.save(output_path, "PNG", optimize=True)
                logger.debug(f"Saved {image_path.name} as PNG (preserved transparency)")
            else:
                # Convert to JPEG
                original_mode = img.mode
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                    logger.debug(
                        f"Converted {image_path.name} from {original_mode} to RGB"
                    )
                output_path = image_path.with_suffix(".jpg")
                if exif_data:
                    img.save(
                        output_path,
                        "JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True,
                        exif=exif_data,
                    )
                    logger.debug(f"Saved {image_path.name} as JPEG (preserved EXIF)")
                else:
                    img.save(output_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
                    logger.debug(f"Saved {image_path.name} as JPEG")

            # Remove original if it's different from output
            if image_path != output_path:
                image_path.unlink()
                logger.debug(f"Removed original file: {image_path.name}")
            else:
                logger.debug(f"Overwrote original file: {image_path.name}")

    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")


def main():
    if not INPUT_FOLDER.exists() or not INPUT_FOLDER.is_dir():
        logger.error(f"{INPUT_FOLDER} is not a valid directory")
        sys.exit(1)

    subfolders = [
        f
        for f in sorted(INPUT_FOLDER.iterdir())
        if f.is_dir() and not f.name.startswith(".")
    ]

    for subfolder in subfolders:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            images_to_process = [
                file_path
                for file_path in subfolder.rglob("*")
                if file_path.is_file() and is_image(file_path)
            ]

            logger.debug(f"Found {len(images_to_process)} images in {subfolder.name}")

            task = progress.add_task(
                f"[cyan]Processing {subfolder.name}", total=len(images_to_process)
            )
            if images_to_process:
                with ThreadPoolExecutor(psutil.cpu_count(logical=False)) as executor:
                    executor.map(process_image, images_to_process)

            progress.update(task, advance=1)

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
