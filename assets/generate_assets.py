#!/usr/bin/env python3
"""
Asset Generation Script

Generates synthetic assets (backgrounds, papers, stains) using Replicate's
Nano Banana AI image generation model. Automatically extends existing numbered
assets and creates attribution files.

Usage:
    python generate_assets.py                    # Generate all categories
    python generate_assets.py --category papers  # Generate specific category
    python generate_assets.py --count 10         # Generate specific count per category
    python generate_assets.py --dry-run          # Show prompts without generating
    python generate_assets.py --list             # List available categories
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path
from typing import Any

import replicate
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Initialize Rich console
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("asset_generator")

# Base directory for assets (directory containing this script)
ASSETS_DIR = Path(__file__).parent

# Attribution text for generated images
AI_MODEL = "prunaai/flux-fast"  # "google/nano-banana"
ATTRIBUTION = "prunaai/flux-fast"

# Retry configuration
MAX_RETRIES = 3
RETRY_MIN_WAIT = 2  # seconds
RETRY_MAX_WAIT = 10  # seconds

# Asset configurations with prompts and placeholder variations
ASSET_CONFIGS: dict[str, dict[str, Any]] = {
    "backgrounds/no_shadow/landscapes": {
        "prompt_template": (
            "A beautiful {scene_type} landscape photograph, {time_of_day}, "
            "{weather}, professional photography, high resolution, no text, no watermarks"
        ),
        "placeholders": {
            "scene_type": [
                "mountain",
                "coastal",
                "forest",
                "desert",
                "meadow",
                "lake",
                "valley",
                "canyon",
                "hillside",
                "countryside",
            ],
            "time_of_day": [
                "golden hour",
                "midday sun",
                "blue hour",
                "soft morning light",
                "afternoon light",
            ],
            "weather": [
                "clear sky",
                "partly cloudy",
                "dramatic clouds",
                "misty",
                "sunny",
            ],
        },
        "extension": ".jpg",
        "count": 10,
    },
    "backgrounds/no_shadow/cityscapes": {
        "prompt_template": (
            "A {style} cityscape photograph, {city_feature}, {time_of_day}, "
            "{atmosphere}, professional urban photography, high resolution, no text, no watermarks"
        ),
        "placeholders": {
            "style": [
                "modern",
                "vintage",
                "aerial",
                "street level",
                "panoramic",
                "architectural",
            ],
            "city_feature": [
                "skyline",
                "downtown buildings",
                "busy street",
                "historic district",
                "waterfront",
                "bridge",
                "plaza",
                "rooftop view",
            ],
            "time_of_day": [
                "golden hour",
                "blue hour",
                "night lights",
                "sunrise",
                "midday",
                "dusk",
            ],
            "atmosphere": [
                "clear weather",
                "rainy reflections",
                "foggy morning",
                "sunset colors",
                "overcast",
            ],
        },
        "extension": ".jpg",
        "count": 10,
    },
    "backgrounds/with_shadow/desks": {
        "prompt_template": (
            "Close-up of a {desk_type} desk surface, {material} texture, "
            "{lighting} lighting, top-down view, clean workspace, professional photography, no text, "
            "{condition} condition"
        ),
        "placeholders": {
            "desk_type": [
                "wooden",
                "modern",
                "vintage",
                "minimalist",
                "rustic",
                "industrial",
                "scandinavian",
                "antique",
            ],
            "material": [
                "oak wood",
                "walnut",
                "marble",
                "concrete",
                "leather",
                "linen",
                "pine wood",
                "mahogany",
            ],
            "lighting": [
                "soft natural",
                "warm ambient",
                "bright daylight",
                "side",
                "overhead",
            ],
            "condition": [
                "well-used",
                "pristine",
                "slightly worn",
                "scratched",
                "polished",
                "textured",
            ],
        },
        "extension": ".jpg",
        "count": 10,
    },
    "papers": {
        "prompt_template": (
            "Square paper texture, {paper_type}, {condition}, {color_tone} tones, "
            "flat lay, seamless texture, high resolution, no text, no shadows"
        ),
        "placeholders": {
            "paper_type": [
                "aged parchment",
                "kraft paper",
                "cotton rag paper",
                "recycled paper",
                "watercolor paper",
                "cardstock",
                "rice paper",
                "newsprint",
                "vellum",
                "linen paper",
            ],
            "condition": [
                "slightly worn",
                "pristine",
                "weathered edges",
                "subtle creases",
                "smooth",
                "textured grain",
            ],
            "color_tone": [
                "warm cream",
                "cool white",
                "ivory",
                "antique yellow",
                "off-white",
                "light tan",
                "pale beige",
            ],
        },
        "extension": ".jpg",
        "count": 10,
    },
    "stains/coffee": {
        "prompt_template": (
            "Dried coffee stain on pure white background, {stain_shape}, {intensity} brown color, "
            "{style}, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "stain_shape": [
                "ring shaped",
                "splash pattern",
                "drip marks",
                "circular puddle",
                "scattered droplets",
                "smeared",
            ],
            "intensity": ["light", "medium", "dark", "faded"],
            "style": [
                "realistic",
                "watercolor effect",
                "dried edges",
                "fresh wet look",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
    "stains/ink": {
        "prompt_template": (
            "Dried ink stain on pure white background, {ink_color} ink, {stain_pattern}, "
            "{intensity}, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "ink_color": ["black", "dark blue", "navy", "blue-black", "sepia"],
            "stain_pattern": [
                "splatter",
                "blot",
                "smear",
                "drips",
                "fingerprint smudge",
                "brush stroke",
            ],
            "intensity": [
                "light wash",
                "medium saturation",
                "deep saturated",
                "faded",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
    "stains/wine": {
        "prompt_template": (
            "Dried wine stain on pure white background, {wine_type} wine color, {stain_shape}, "
            "{dryness}, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "wine_type": ["red", "deep burgundy", "merlot", "cabernet"],
            "stain_shape": [
                "ring mark",
                "splash",
                "spill puddle",
                "droplets",
                "smear",
            ],
            "dryness": [
                "fresh and wet",
                "dried with darker edges",
                "partially dried",
                "old faded stain",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
    "stains/tea": {
        "prompt_template": (
            "Dried tea stain on pure white background, {tea_type} color, {stain_shape}, "
            "{intensity}, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "tea_type": [
                "light amber",
                "golden brown",
                "dark tannin",
                "green tea pale",
                "chai orange-brown",
            ],
            "stain_shape": [
                "ring mark from cup",
                "splash pattern",
                "drip trail",
                "circular puddle",
                "overlapping rings",
            ],
            "intensity": [
                "faint and subtle",
                "medium wash",
                "strong saturated",
                "faded old stain",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
    "stains/watercolor": {
        "prompt_template": (
            "Dried watercolor stain on pure white background, {color} pigment, {pattern}, "
            "{wetness}, artistic texture, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "color": [
                "blue",
                "purple",
                "green",
                "orange",
                "pink",
                "yellow",
                "mixed rainbow",
                "earth tones",
            ],
            "pattern": [
                "soft bloom",
                "hard edges",
                "gradient wash",
                "splatter",
                "drip marks",
                "blended blend",
            ],
            "wetness": [
                "wet-on-wet soft edges",
                "dried with granulation",
                "partially dried blooms",
                "crisp dried edges",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
    "stains/grease": {
        "prompt_template": (
            "Dried grease stain on pure white background, {grease_type}, {stain_shape}, "
            "highly translucent, isolated, transparent edges, no other objects"
        ),
        "placeholders": {
            "grease_type": [
                "cooking oil translucent",
                "butter yellow tint",
                "machine oil dark",
                "food grease",
                "fingerprint oil marks",
            ],
            "stain_shape": [
                "circular spot",
                "smear streak",
                "splatter dots",
                "spread puddle",
                "drip pattern",
            ],
        },
        "extension": ".png",
        "count": 10,
    },
}


def find_next_number(directory: Path) -> int:
    """
    Find the next available number for asset files in a directory.

    Scans for files matching pattern NNNN.* (e.g., 0001.jpg, 0002.png)
    and returns the next available number.

    Args:
        directory: Path to the directory to scan

    Returns:
        Next available number (starting from 1 if directory is empty)
    """
    if not directory.exists():
        return 1

    # Pattern to match numbered files (0001.jpg, 0002.png, etc.)
    pattern = re.compile(r"^(\d{4})\.(jpg|jpeg|png)$", re.IGNORECASE)

    max_number = 0
    for file_path in directory.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)

    return max_number + 1


def generate_prompt(config: dict[str, Any]) -> str:
    """
    Generate a prompt by filling in template placeholders with random values.

    Args:
        config: Configuration dict containing 'prompt_template' and 'placeholders'

    Returns:
        Completed prompt string with all placeholders filled
    """
    prompt = config["prompt_template"]

    for placeholder, options in config["placeholders"].items():
        value = random.choice(options)
        prompt = prompt.replace(f"{{{placeholder}}}", value)

    return prompt


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_replicate_api(prompt: str) -> bytes:
    """
    Call Replicate API with retry logic.

    Args:
        prompt: The prompt to generate the image from

    Returns:
        Image bytes from the API response
    """
    output = replicate.run(
        AI_MODEL,
        input={"prompt": prompt},
    )
    return output.read()


def generate_image(prompt: str, output_path: Path, extension: str) -> bool:
    """
    Generate an image using an AI model and save it.

    Args:
        prompt: The prompt to generate the image from
        output_path: Path to save the image (without extension)
        extension: File extension (.jpg or .png)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Call Replicate API with retries
        image_data = _call_replicate_api(prompt)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        image_path = output_path.with_suffix(extension)
        with open(image_path, "wb") as f:
            f.write(image_data)

        # Create attribution file
        attribution_path = output_path.with_suffix(".attribution")
        with open(attribution_path, "w") as f:
            f.write(ATTRIBUTION)

        logger.debug(f"Generated: {image_path}")
        return True

    except Exception as e:
        logger.error(
            f"Failed to generate {output_path.name} after {MAX_RETRIES} attempts: {e}"
        )
        return False


def generate_assets_for_category(
    category: str,
    config: dict[str, Any],
    count: int | None = None,
    dry_run: bool = False,
    progress: Progress | None = None,
    task_id: Any | None = None,
) -> tuple[int, int]:
    """
    Generate assets for a single category.

    Args:
        category: Category path (e.g., 'backgrounds/no_shadow/landscapes')
        config: Configuration dict for this category
        count: Number of images to generate (overrides config if provided)
        dry_run: If True, show prompts without generating
        progress: Rich Progress instance for tracking
        task_id: Task ID for progress updates

    Returns:
        Tuple of (success_count, failure_count)
    """
    directory = ASSETS_DIR / category
    extension = config["extension"]
    target_count = count if count is not None else config["count"]

    start_number = find_next_number(directory)
    success_count = 0
    failure_count = 0

    for i in range(target_count):
        current_number = start_number + i
        filename = f"{current_number:04d}"
        output_path = directory / filename

        prompt = generate_prompt(config)

        if dry_run:
            console.print(f"[dim]{category}/{filename}[/dim]: {prompt}")
            success_count += 1
        else:
            if generate_image(prompt, output_path, extension):
                success_count += 1
            else:
                failure_count += 1

        if progress and task_id is not None:
            progress.update(task_id, advance=1)

    return success_count, failure_count


def list_categories() -> None:
    """Display a table of available asset categories."""
    table = Table(title="Available Asset Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Default Count", justify="right")
    table.add_column("Format", style="green")
    table.add_column("Prompt Preview", style="dim", max_width=50)

    for category, config in ASSET_CONFIGS.items():
        prompt_preview = config["prompt_template"][:47] + "..."
        table.add_row(
            category,
            str(config["count"]),
            config["extension"],
            prompt_preview,
        )

    console.print(table)


def main() -> None:
    """Main entry point for the asset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic assets using Replicate's Nano Banana model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        help="Generate only this category (e.g., 'papers', 'stains/coffee')",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        help="Number of images to generate per category (overrides default)",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show prompts without generating images",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available categories and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.list:
        list_categories()
        return

    # Determine which categories to process
    if args.category:
        if args.category not in ASSET_CONFIGS:
            console.print(f"[red]Error:[/red] Unknown category '{args.category}'")
            console.print("Use --list to see available categories")
            return
        categories = {args.category: ASSET_CONFIGS[args.category]}
    else:
        categories = ASSET_CONFIGS

    # Calculate total images to generate
    total_images = sum(
        args.count if args.count else config["count"] for config in categories.values()
    )

    # Display header
    mode = "[yellow]DRY RUN[/yellow]" if args.dry_run else "[green]GENERATING[/green]"
    console.print(
        Panel(
            f"Mode: {mode}\n"
            f"Categories: {len(categories)}\n"
            f"Total Images: {total_images}",
            title="Asset Generator",
        )
    )

    # Track overall statistics
    total_success = 0
    total_failure = 0

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:
        # Add overall progress task
        overall_task = progress.add_task(
            "[bold blue]Overall Progress", total=total_images
        )

        for category, config in categories.items():
            count = args.count if args.count else config["count"]

            # Add category-specific task
            category_task = progress.add_task(
                f"[cyan]{category}",
                total=count,
            )

            success, failure = generate_assets_for_category(
                category=category,
                config=config,
                count=args.count,
                dry_run=args.dry_run,
                progress=progress,
                task_id=category_task,
            )

            total_success += success
            total_failure += failure

            # Update overall progress
            progress.update(overall_task, advance=count)

            # Mark category as complete
            progress.update(category_task, description=f"[green]{category} [done]")

    # Display summary
    console.print()
    summary_table = Table(title="Generation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Attempted", str(total_success + total_failure))
    summary_table.add_row("Successful", f"[green]{total_success}[/green]")
    summary_table.add_row(
        "Failed", f"[red]{total_failure}[/red]" if total_failure else "0"
    )

    if total_success + total_failure > 0:
        success_rate = (total_success / (total_success + total_failure)) * 100
        summary_table.add_row("Success Rate", f"{success_rate:.1f}%")

    console.print(summary_table)

    if total_failure > 0:
        console.print(
            f"\n[yellow]Warning:[/yellow] {total_failure} images failed to generate. "
            "Check the logs for details."
        )


if __name__ == "__main__":
    main()
