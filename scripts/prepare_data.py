"""
Script to prepare a dataset with images generated from text data.
Handles text overflow by creating multiple images if necessary.
Saves the new dataset to disk and optionally pushes it to the Hugging Face Hub.

Generating english synthetic OCR dataset as an example:

python scripts/prepare_data.py \
    dataset_path="agentlans/high-quality-english-sentences" \
    text_column="text" \
    data_directory="default" \
    split="train" \
    max_entries=1 \
    column_range="[1,1]" \
    max_workers=1 \
    local_output_dir="eng_synthetic_ocr_output_v2" \
    num_examples=1000 \
    push_to_hub=True \
    hub_repo_id="Sigurdur/eng_synthetic_ocr_v2" \
    apply_random_transformations=False \
    google_fonts_directory="./english_fonts" \
    paper_texture_probability=0.3 \

Generating icelandic synthetic OCR dataset as an example:

python scripts/prepare_data.py \
    dataset_path="arnastofnun/igc-2024" \
    text_column="document" \
    data_directory="parla" \
    split="train" \
    max_entries=100 \
    column_range="[1,1]" \
    max_workers=1 \
    local_output_dir="isl_synthetic_ocr_output_v3" \
    num_examples=2000 \
    push_to_hub=True \
    hub_repo_id="Sigurdur/isl_synthetic_ocr_v3" \
    apply_random_transformations=False \
    google_fonts_directory="./icelandic_fonts" \
    paper_texture_probability=0.0 \
    background_image_probability=0.0 \
    use_random_backgrounds=False \
    use_paper_textures=False \


"""

import gc
import logging
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import cast

from datasets import (
    Dataset,
    DatasetDict,
    Image,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from omegaconf import OmegaConf
from PIL import Image as PILImage
from rich.logging import RichHandler
from tqdm import tqdm

from ocr_icelandic import randomness
from ocr_icelandic.config import DataConfig, GenerationConfig, SingleImageData
from ocr_icelandic.fonts import get_compatible_fonts, sync_google_fonts
from ocr_icelandic.image_generator import generate_single_chunk
from ocr_icelandic.text_processing import split_long_text
from ocr_icelandic.utils import discover_backgrounds, discover_paper_textures

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.ERROR)


def _save_batch_to_disk(new_data: dict, batch_dir: Path, batch_idx: int) -> str:
    """
    Save a batch of images to disk.
    """
    try:
        batch_path = batch_dir / f"batch_{batch_idx:04d}"
        batch_ds = Dataset.from_dict(dict(new_data)).cast_column("image", Image())
        batch_ds.save_to_disk(str(batch_path))
        return str(batch_path)
    except Exception as e:
        logger.error("Failed to save batch %d to disk: %s", batch_idx, e)
        raise


def _setup_fonts(cfg: DataConfig) -> list[str] | None:
    """Setup and discover available fonts."""

    google_fonts_api_key = os.environ.get("GOOGLE_FONTS_API_KEY")
    if google_fonts_api_key:
        logger.info("GOOGLE_FONTS_API_KEY found, syncing Google Fonts...")
        sync_google_fonts(google_fonts_api_key, cfg.google_fonts_directory)
    else:
        logger.warning(
            "GOOGLE_FONTS_API_KEY environment variable not set. "
            "Skipping Google Fonts sync and using only system fonts."
        )

    if not cfg.use_random_fonts:
        return None

    return get_compatible_fonts(
        language_code=cfg.language_code,
        use_cache=cfg.use_font_cache,
        cache_dir=cfg.font_cache_dir,
        google_fonts_directory=cfg.google_fonts_directory,
    )


def _setup_textures_and_backgrounds(
    cfg: DataConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Discover paper textures and backgrounds."""
    paper_textures = []
    if cfg.use_paper_textures:
        paper_textures = discover_paper_textures(cfg.paper_textures_dir)
        if paper_textures:
            logger.info(
                f"Found {len(paper_textures)} paper textures in {cfg.paper_textures_dir}"
            )
        else:
            logger.warning(
                f"No paper textures found in {cfg.paper_textures_dir}, "
                "falling back to solid colors"
            )

    no_shadow_bgs, with_shadow_bgs = [], []
    if cfg.use_background_images:
        no_shadow_bgs, with_shadow_bgs = discover_backgrounds(cfg.backgrounds_dir)
        logger.info(
            f"Found {len(no_shadow_bgs)} no-shadow backgrounds and "
            f"{len(with_shadow_bgs)} with-shadow backgrounds"
        )
        if not no_shadow_bgs and not with_shadow_bgs:
            logger.warning(
                f"No backgrounds found in {cfg.backgrounds_dir}, "
                "will not use background images"
            )

    return paper_textures, no_shadow_bgs, with_shadow_bgs


def _build_generation_config(
    cfg: DataConfig,
    fonts: list[str] | None,
    paper_textures: list[str],
    no_shadow_bgs: list[str],
    with_shadow_bgs: list[str],
) -> GenerationConfig:
    """Build the generation config with resolved settings."""
    return GenerationConfig(
        **asdict(cfg),
        available_fonts=fonts,
        available_paper_textures=paper_textures or None,
        available_no_shadow_backgrounds=no_shadow_bgs or None,
        available_with_shadow_backgrounds=with_shadow_bgs or None,
    )


def _process_image_data(
    image_data: SingleImageData, text_column: str, new_data: dict
) -> None:
    """Add single image data to the batch dictionary."""
    new_data[text_column].append(image_data.text)
    new_data["image"].append(image_data.image)
    new_data["font_path"].append(image_data.font_path)
    new_data["bg_color"].append(image_data.bg_color)
    new_data["font_color"].append(image_data.font_color)
    new_data["font_size"].append(image_data.font_size)
    new_data["image_width"].append(image_data.image_width)
    new_data["image_height"].append(image_data.image_height)
    new_data["image_dpi"].append(image_data.image_dpi)
    new_data["text_vertical_alignment"].append(image_data.text_vertical_alignment)
    new_data["text_horizontal_alignment"].append(image_data.text_horizontal_alignment)
    new_data["paragraph_bboxes"].append(image_data.paragraph_bboxes)
    new_data["transformations"].append(image_data.transformations)


def generate_image_dataset(texts: list[str], cfg: DataConfig) -> Dataset:
    """
    Generate a dataset with images from text using batch processing.

    Parallelization happens at the chunk level for better CPU utilization:
    texts are first split into chunks, then all chunks are processed in parallel.

    Args:
        texts: List of text entries to convert to images
        cfg: Configuration for image generation

    Returns:
        HuggingFace Dataset with 'text' and 'image' columns
    """
    logger.info("Generating images from text...")

    num_examples = cfg.num_examples if cfg.num_examples > 0 else len(texts)

    # Setup resources
    fonts = _setup_fonts(cfg)
    paper_textures, no_shadow_bgs, with_shadow_bgs = _setup_textures_and_backgrounds(
        cfg
    )
    generation_cfg = _build_generation_config(
        cfg, fonts, paper_textures, no_shadow_bgs, with_shadow_bgs
    )

    # Phase 1: Split all texts into chunks (fast, sequential)
    # This ensures parallelization happens at the chunk level for better CPU utilization
    all_chunks: list[str] = []
    split_texts = 0
    for text in texts[:num_examples]:
        chunks = split_long_text(text.strip(), cfg.max_text_length)
        all_chunks.extend(chunks)
        if len(chunks) > 1:
            split_texts += 1

    logger.info(
        "Split %d texts into %d chunks (%d texts required splitting).",
        num_examples,
        len(all_chunks),
        split_texts,
    )

    # Batch processing setup
    batch_dir = Path(cfg.local_output_dir) / "_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_datasets: list[str] = []
    new_data: defaultdict[str, list] = defaultdict(list)
    batch_idx = 0

    # Scale workers with chunk count, not text count
    max_workers = min(cfg.max_workers, len(all_chunks))
    logger.info(
        "Using %d parallel workers for %d chunks (batch_size=%d).",
        max_workers,
        len(all_chunks),
        cfg.batch_size,
    )

    # Phase 2: Process all chunks in parallel
    # Each worker gets a unique seed derived from base_seed + chunk_index
    # to ensure different random choices across workers
    base_seed = cfg.random_seed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_single_chunk, chunk, generation_cfg, base_seed + i
            ): i
            for i, chunk in enumerate(all_chunks)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing chunks",
            unit="chunk",
        ):
            try:
                image_data_list = future.result()

                for image_data in image_data_list:
                    _process_image_data(image_data, cfg.text_column, new_data)

                # Flush batch when threshold reached
                if len(new_data["image"]) >= cfg.batch_size:
                    batch_path = _save_batch_to_disk(new_data, batch_dir, batch_idx)
                    batch_datasets.append(batch_path)
                    logger.info(
                        "Saved batch %d with %d images to disk",
                        batch_idx,
                        len(new_data["image"]),
                    )
                    batch_idx += 1
                    new_data = defaultdict(list)
                    gc.collect()

            except (OSError, ValueError, RuntimeError) as e:
                logger.error("Error processing chunk: %s", e)
                continue

    # Save remaining data
    if new_data["image"]:
        batch_path = _save_batch_to_disk(new_data, batch_dir, batch_idx)
        batch_datasets.append(batch_path)
        logger.info(
            "Saved final batch %d with %d images to disk",
            batch_idx,
            len(new_data["image"]),
        )
        del new_data
        gc.collect()

    logger.info(
        "Generated %d batches from %d chunks.",
        len(batch_datasets),
        len(all_chunks),
    )

    # Concatenate batches
    if not batch_datasets:
        logger.warning("No images were generated.")
        return Dataset.from_dict({cfg.text_column: [], "image": []}).cast_column(
            "image", Image()
        )

    logger.info("Concatenating %d batches into final dataset...", len(batch_datasets))
    all_datasets = [load_from_disk(path) for path in batch_datasets]
    final_dataset = concatenate_datasets(all_datasets)

    # Clean up batch files
    logger.info("Cleaning up temporary batch files...")
    shutil.rmtree(batch_dir)

    return final_dataset


def create_train_test_val_split(dataset: Dataset) -> dict[str, Dataset]:
    """Create 80/10/10 train/test/validation split."""
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = split["test"].train_test_split(test_size=0.5, seed=42)
    return {
        "train": split["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }


def save_local_images(dataset_splits: dict, output_dir: str) -> None:
    """Save images to local directory."""
    local_path = Path(output_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    for split_name, items in dataset_splits.items():
        split_path = local_path / split_name
        split_path.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(items):
            image: PILImage.Image = item["image"]
            image.save(split_path / f"image_{idx:05d}.png")

    logger.info("Images saved to %s", local_path)


def display_sample(dataset: dict) -> None:
    """Display the first generated image from the dataset."""
    logger.info("\nShowing first generated image...")
    if len(dataset["train"]) > 0:
        logger.info("Text for first image:")
        logger.info("'%s'", dataset["train"][0]["text"])
        dataset["train"][0]["image"].show()


def create_image_dataset(cfg: DataConfig) -> None:
    """
    Create a dataset with images generated from text data.
    Args:
        cfg (DataConfig): Configuration for dataset creation
    """
    # Set random seed for reproducibility

    randomness.set_seed(cfg.random_seed)
    logger.info(f"Random seed set to {cfg.random_seed} for reproducibility")

    # load dataset
    dataset = cast(
        Dataset,
        load_dataset(
            cfg.dataset_path,
            cfg.data_directory,
            split=cfg.split,
            cache_dir=".cache/huggingface/datasets",
            download_mode="reuse_cache_if_exists",
        ),
    )

    if cfg.max_entries > 0:
        dataset = dataset.select(range(cfg.max_entries))

    texts = list(dataset[cfg.text_column])

    # Normalize text column name
    if cfg.text_column != "text":
        logger.info("Renaming text column '%s' to 'text'", cfg.text_column)
        dataset = dataset.rename_column(cfg.text_column, "text")
        cfg.text_column = "text"

    # Generate images
    image_dataset = generate_image_dataset(texts, cfg)

    logger.info("\nOriginal dataset size: %d", len(texts))
    logger.info("New image dataset size: %d", len(image_dataset))

    # Split and save
    final_splits = create_train_test_val_split(image_dataset)
    dataset_dict = DatasetDict(final_splits)
    dataset_dict.save_to_disk(cfg.local_output_dir)
    logger.info("Image dataset saved to %s", cfg.local_output_dir)

    # Optional: show sample
    if cfg.show_sample:
        display_sample(final_splits)

    # Optional: push to hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info("Pushing dataset to the hub at %s...", cfg.hub_repo_id)
        dataset_dict.push_to_hub(cfg.hub_repo_id)
        logger.info("Dataset pushed to the hub successfully.")

    # Optional: save local images
    if cfg.local_output_dir:
        save_local_images(final_splits, cfg.local_output_dir)


def main() -> None:
    """CLI entry point."""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))

    try:
        cfg = DataConfig(**cfg_dict)
    except TypeError as e:
        logger.error("Error: %s\n\nUsage: python prepare_data.py", e)
        sys.exit(1)

    create_image_dataset(cfg)


if __name__ == "__main__":
    main()
