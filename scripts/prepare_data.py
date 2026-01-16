"""
Script to prepare a dataset with images generated from text data.
Handles text overflow by creating multiple images if necessary.
Saves the new dataset to disk and optionally pushes it to the Hugging Face Hub.
"""

from collections import defaultdict
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
import psutil
from ocr_icelandic.fonts import (
    get_compatible_fonts,
    sync_google_fonts,
)
from ocr_icelandic.pipeline import Pipeline, PipelineState
from ocr_icelandic.pipeline.stages import (
    ApplyTransformationsStage,
    CompositeBackgroundStage,
    FinalizeImageStage,
    RenderTextStage,
    SelectBackgroundImageStage,
    SelectColorsStage,
    SelectFontStage,
    SelectLayoutStage,
    SelectPaperTextureStage,
    VisualizeBBoxesStage,
)
from ocr_icelandic.utils import (
    discover_backgrounds,
    discover_paper_textures,
)
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.logging import RichHandler
from PIL import Image as PILImage
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset creation"""

    dataset_path: str = "arnastofnun/IGC-2024"  # Hugging Face dataset path
    text_column: str = "document"  # Column in dataset containing text
    data_directory: str = "parla"  # Subdirectory or config name in the dataset
    split: str = "train"  # Which split to use from the dataset
    max_length: int = 512
    max_entries: int = 2
    show_sample: bool = False  # Whether to show a sample image after creation
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 72
    img_background_color: str = "white"
    font_path: str = "/usr/share/fonts"
    font_size: int = 12
    font_color: str = "black"
    use_random_font_colors: bool = True  # Whether to use random font colors
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 2  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    save_to_disk: bool = False  # Whether to save dataset to disk
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )
    use_random_fonts: bool = True  # Whether to use random fonts
    use_random_backgrounds: bool = True  # Whether to use random background colors
    google_fonts_directory: str = "../google_fonts"  # Directory to store Google Fonts
    language_code: str = (
        "is"  # ISO 639-1 language code (e.g., "is" for Icelandic, "de" for German)
    )
    use_font_cache: bool = True  # Whether to use SQLite caching for font compatibility
    font_cache_dir: str = ".fontcache"  # Directory to store font compatibility cache
    use_paper_textures: bool = True  # Whether to use paper textures from assets/papers
    paper_textures_dir: str = (
        "assets/papers"  # Directory containing paper texture images
    )
    backgrounds_dir: str = (
        "assets/backgrounds"  # Directory containing background images
    )
    background_image_probability: float = 0.5  # Probability of using a background image
    max_text_length: int = 2000  # Maximum characters per text before splitting
    column_gap: int = 20  # Horizontal gap in pixels between columns
    num_columns: int | None = (
        None  # Number of columns when rendering text (None => random)
    )
    min_num_columns: int = 1  # Minimum number of columns when randomizing
    max_num_columns: int = 5  # Maximum number of columns when randomizing
    column_width: int | None = None  # Fixed column width in pixels (None => random)
    min_column_width: int = 100  # Minimum column width when randomizing
    max_column_width: int = 512  # Maximum column width when randomizing
    local_output_dir: str = "./local_output"  # Local directory for temporary outputs
    # Font variation settings
    enable_font_size_variation: bool = False  # Whether to vary font size per paragraph
    font_size_min_ratio: float = (
        0.8  # Minimum font size ratio (e.g., 0.8 = 80% of base)
    )
    font_size_max_ratio: float = (
        1.2  # Maximum font size ratio (e.g., 1.2 = 120% of base)
    )
    enable_font_styles: bool = False  # Whether to apply font styles (bold/underline)
    font_bold_probability: float = 0.2  # Probability of applying bold style
    font_underline_probability: float = 0.1  # Probability of applying underline style
    random_seed: int = 42  # Random seed for reproducibility


@dataclass
class GenerationConfig(DataConfig):
    """Configuration for image generation only"""

    column_range: tuple[int, int] = (1, 1)
    column_width_range: tuple[int, int] = (100, 512)
    available_fonts: list[str] | None = None
    available_paper_textures: list[str] | None = None
    available_no_shadow_backgrounds: list[str] | None = None
    available_with_shadow_backgrounds: list[str] | None = None


@dataclass
class SingleImageData:
    """Data for a single generated image"""

    text: str
    image: PILImage.Image
    font_path: str
    bg_color: tuple[int, int, int] | str
    font_color: tuple[int, int, int] | str
    font_size: int
    image_width: int
    image_height: int
    image_dpi: int
    text_vertical_alignment: str
    text_horizontal_alignment: str
    paragraph_bboxes: list[dict]
    transformations: list[dict]
    # NEW FIELDS for font variation
    paragraph_font_sizes: list[int] | None = None  # Font size per paragraph
    paragraph_styles: list[dict] | None = None  # Style flags per paragraph


def split_long_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks at sentence boundaries to avoid mid-sentence splits.

    Args:
        text: The text to split
        max_length: Maximum length for each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    # Split on sentence boundaries
    sentences = (
        text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    )

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max_length, save current chunk and start new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_paragraph_styles(
    num_paragraphs: int,
    bold_probability: float = 0.2,
    underline_probability: float = 0.1,
) -> list[dict]:
    """
    Generate random style flags for paragraphs.

    Args:
        num_paragraphs: Number of paragraphs
        bold_probability: Probability of applying bold (0.0-1.0)
        underline_probability: Probability of applying underline (0.0-1.0)

    Returns:
        List of dicts: [{"bold": bool, "underline": bool}, ...]
    """
    from ocr_icelandic import randomness

    styles = []
    for _ in range(num_paragraphs):
        styles.append(
            {
                "bold": randomness.random() < bold_probability,
                "underline": randomness.random() < underline_probability,
            }
        )
    return styles


def _normalize_range(
    min_value: int, max_value: int, minimum: int = 1
) -> tuple[int, int]:
    """Ensure provided min/max values form a valid range."""

    min_value = max(minimum, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def generate_single_text(
    text: str, cfg: GenerationConfig
) -> tuple[list[SingleImageData], int]:
    """Generate images from text using the pipeline architecture."""
    # Split long texts first
    text_chunks = split_long_text(text.strip(), cfg.max_text_length)

    # Build the pipeline stages
    stages = [
        SelectFontStage(
            fonts=cfg.available_fonts or [],
            fixed_font=cfg.font_path if not cfg.use_random_fonts else None,
            random_selection=cfg.use_random_fonts,
        ),
        SelectColorsStage(
            random_background=cfg.use_random_backgrounds,
            random_font_color=cfg.use_random_font_colors,
            fixed_bg_color=cfg.img_background_color
            if not cfg.use_random_backgrounds
            else None,
            fixed_font_color=cfg.font_color if not cfg.use_random_font_colors else None,
        ),
        SelectLayoutStage(
            num_columns=cfg.num_columns,
            column_range=cfg.column_range,
            column_width=cfg.column_width,
            column_width_range=cfg.column_width_range,
            column_gap=cfg.column_gap,
            alignment=cfg.text_horizontal_alignment,
            vertical_alignment=cfg.text_vertical_alignment,
        ),
        SelectPaperTextureStage(
            textures=cfg.available_paper_textures or [],
            probability=1.0 if cfg.use_paper_textures else 0.0,
        ),
        SelectBackgroundImageStage(
            no_shadow_backgrounds=cfg.available_no_shadow_backgrounds or [],
            with_shadow_backgrounds=cfg.available_with_shadow_backgrounds or [],
            probability=cfg.background_image_probability,
        ),
        RenderTextStage(apply_displacement=True),
        ApplyTransformationsStage(pipeline_type="auto"),
        CompositeBackgroundStage(),
        FinalizeImageStage(use_random_composite=cfg.use_random_backgrounds),
        VisualizeBBoxesStage(enabled=True, show_labels=False),
    ]

    images: list[SingleImageData] = []
    for chunk in text_chunks:
        remaining_text = chunk
        while remaining_text:
            # Build paragraph font configs if needed
            paragraph_font_configs = None
            paragraph_font_sizes = None
            paragraph_styles = None

            if cfg.enable_font_size_variation or cfg.enable_font_styles:
                from ocr_icelandic.utils.text_layout import (
                    calculate_paragraph_font_sizes,
                )

                paragraphs = remaining_text.split("\n\n")
                num_paragraphs = len(paragraphs)

                if cfg.enable_font_size_variation:
                    paragraph_font_sizes = calculate_paragraph_font_sizes(
                        paragraphs,
                        cfg.font_size,
                        cfg.font_size_min_ratio,
                        cfg.font_size_max_ratio,
                    )
                else:
                    paragraph_font_sizes = [cfg.font_size] * num_paragraphs

                if cfg.enable_font_styles:
                    paragraph_styles = generate_paragraph_styles(
                        num_paragraphs,
                        cfg.font_bold_probability,
                        cfg.font_underline_probability,
                    )
                else:
                    paragraph_styles = [
                        {"bold": False, "underline": False}
                    ] * num_paragraphs

                # Note: ParagraphFontConfig will be built in RenderTextStage
                # when paragraph_font_configs is set on state

            # Create initial state
            initial_state = PipelineState(
                text=remaining_text,
                image_size=(cfg.image_width, cfg.image_height),
                dpi=cfg.image_dpi,
                render_scale=2,  # Render at 2x resolution, scale down for quality
                font_size=cfg.font_size,
                paragraph_font_configs=paragraph_font_configs,
            )

            # Run the pipeline
            pipeline = Pipeline(stages=stages, initial_state=initial_state)
            result = pipeline.run()

            if not result.fitted_text:
                break

            images.append(
                SingleImageData(
                    text=result.fitted_text,
                    image=result.image,
                    font_path=result.font_path or cfg.font_path,
                    bg_color=result.bg_color,
                    font_color=result.font_color,
                    font_size=cfg.font_size,
                    image_width=cfg.image_width,
                    image_height=cfg.image_height,
                    image_dpi=cfg.image_dpi,
                    text_vertical_alignment=cfg.text_vertical_alignment,
                    text_horizontal_alignment=cfg.text_horizontal_alignment,
                    paragraph_bboxes=result.paragraph_bboxes,
                    transformations=result.transformation_metadata,
                    paragraph_font_sizes=paragraph_font_sizes,
                    paragraph_styles=paragraph_styles,
                )
            )

            remaining_text = remaining_text[len(result.fitted_text) :].lstrip()

    return images, len(text_chunks)


def generate_image_dataset(texts: list[str], cfg: DataConfig) -> Dataset:
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    Args:
        texts (list of str): List of text entries to convert to images
        cfg (DataConfig): Configuration for image generation
    Returns:
        Dataset: A Hugging Face Dataset with 'text' and 'image' columns
    """

    logger.info("Generating images from text...")

    # fix number of examples to generate if specified
    num_examples = cfg.num_examples if cfg.num_examples > 0 else len(texts)

    # Check for Google Fonts API key and sync fonts if available
    google_fonts_api_key = os.environ.get("GOOGLE_FONTS_API_KEY")
    if google_fonts_api_key:
        logger.info("GOOGLE_FONTS_API_KEY found, syncing Google Fonts...")
        sync_google_fonts(google_fonts_api_key, cfg.google_fonts_directory)
    else:
        logger.warning(
            "GOOGLE_FONTS_API_KEY environment variable not set. "
            "Skipping Google Fonts sync and using only system fonts."
        )

    available_fonts = None
    if cfg.use_random_fonts:
        available_fonts = get_compatible_fonts(
            language_code=cfg.language_code,
            use_cache=cfg.use_font_cache,
            cache_dir=cfg.font_cache_dir,
            google_fonts_directory=cfg.google_fonts_directory,
        )

    available_paper_textures = None
    if cfg.use_paper_textures:
        available_paper_textures = discover_paper_textures(cfg.paper_textures_dir)
        if available_paper_textures:
            logger.info(
                f"Found {len(available_paper_textures)} paper textures in {cfg.paper_textures_dir}"
            )
        else:
            logger.warning(
                f"No paper textures found in {cfg.paper_textures_dir}, falling back to solid colors"
            )

    available_no_shadow_backgrounds = []
    available_with_shadow_backgrounds = []
    if cfg.background_image_probability > 0:
        available_no_shadow_backgrounds, available_with_shadow_backgrounds = (
            discover_backgrounds(cfg.backgrounds_dir)
        )
        logger.info(
            f"Found {len(available_no_shadow_backgrounds)} no-shadow backgrounds and {len(available_with_shadow_backgrounds)} with-shadow backgrounds"
        )
        if (
            not available_no_shadow_backgrounds
            and not available_with_shadow_backgrounds
        ):
            logger.warning(
                f"No backgrounds found in {cfg.backgrounds_dir}, will not use background images"
            )

    column_range = _normalize_range(cfg.min_num_columns, cfg.max_num_columns, minimum=1)
    column_width_range = _normalize_range(
        cfg.min_column_width, cfg.max_column_width, minimum=1
    )

    generation_cfg = GenerationConfig(
        **asdict(cfg),
        available_fonts=available_fonts,
        available_paper_textures=available_paper_textures,
        available_no_shadow_backgrounds=available_no_shadow_backgrounds,
        available_with_shadow_backgrounds=available_with_shadow_backgrounds,
        column_range=column_range,
        column_width_range=column_width_range,
    )

    new_data: defaultdict[str, list] = defaultdict(list)
    total_splits = 0
    split_texts = 0

    # Use ProcessPoolExecutor for true parallel processing (bypass GIL)
    # Use physical cores for CPU-bound tasks
    max_workers = min(psutil.cpu_count(logical=False) or 4, len(texts[:num_examples]))
    logger.info(
        f"Using {max_workers} parallel workers (physical cores) for image generation."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(generate_single_text, text, generation_cfg): text
            for text in texts[:num_examples]
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_text),
            total=len(future_to_text),
            desc="Processing texts",
            unit="text",
        ):
            try:
                image_data_list, num_splits = future.result()
                total_splits += num_splits
                split_texts += 1 if num_splits > 1 else 0

                for image_data in image_data_list:
                    new_data[cfg.text_column].append(image_data.text)
                    new_data["image"].append(image_data.image)
                    new_data["font_path"].append(image_data.font_path)
                    new_data["bg_color"].append(image_data.bg_color)
                    new_data["font_color"].append(image_data.font_color)
                    new_data["font_size"].append(image_data.font_size)
                    new_data["image_width"].append(image_data.image_width)
                    new_data["image_height"].append(image_data.image_height)
                    new_data["image_dpi"].append(image_data.image_dpi)
                    new_data["text_vertical_alignment"].append(
                        image_data.text_vertical_alignment
                    )
                    new_data["text_horizontal_alignment"].append(
                        image_data.text_horizontal_alignment
                    )
                    new_data["paragraph_bboxes"].append(image_data.paragraph_bboxes)
                    new_data["transformations"].append(image_data.transformations)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                raise e
                continue

    logger.info(
        f"Split {split_texts} long texts into multiple chunks, in total generating {total_splits} images from {len(texts)}."
    )

    # Create a new Hugging Face Dataset
    image_dataset = Dataset.from_dict(new_data).cast_column("image", DatasetImage())
    return image_dataset


def display_sample(dataset: dict) -> None:
    logger.info("\nShowing first generated image...")
    if len(dataset["train"]) > 0:
        logger.info("Text for first image:")
        logger.info(f"'{dataset['train'][0]['text']}'")
        dataset["train"][0]["image"].show()


def create_image_dataset(cfg: DataConfig) -> None:
    """
    Create a dataset with images generated from text data.
    Args:
        cfg (DataConfig): Configuration for dataset creation
    """
    # Set random seed for reproducibility
    from ocr_icelandic import randomness

    randomness.set_seed(cfg.random_seed)
    logger.info(f"Random seed set to {cfg.random_seed} for reproducibility")

    # load dataset
    dataset = cast(
        Dataset,
        load_dataset(
            cfg.dataset_path,
            cfg.data_directory if hasattr(cfg, "data_directory") else None,
            split=cfg.split,
            cache_dir=".cache/huggingface/datasets",
            download_mode="reuse_cache_if_exists",
        ),
    )

    # select number of entries if specified
    if cfg.max_entries > 0:
        dataset = dataset.select(range(cfg.max_entries))

    texts = list(dataset[cfg.text_column])

    # rename text column to 'text' if necessary
    if cfg.text_column != "text":
        logger.info(f"Renaming text column '{cfg.text_column}' to 'text'")
        dataset = dataset.rename_column(cfg.text_column, "text")
        cfg.text_column = "text"

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    logger.info(f"\nOriginal dataset size: {len(texts)}")
    logger.info(f"New image dataset size: {len(image_dataset)}")

    # Create a train/test/validation split (80/10/10)
    split_dataset = image_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    final_dataset = {
        "train": split_dataset["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }

    # Save the new dataset
    output_path = cfg.output_path
    # Use DatasetDict for saving splits

    dataset_dict = DatasetDict(list(final_dataset.items()))
    if cfg.save_to_disk:
        dataset_dict.save_to_disk(output_path)
    logger.info(f"Image dataset saved to {output_path}")

    # Display the first image as an example
    if cfg.show_sample:
        display_sample(final_dataset)

    # upload to huggingface dataset hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing dataset to the hub at {cfg.hub_repo_id}...")
        dataset_dict.push_to_hub(cfg.hub_repo_id)
        logger.info("Dataset pushed to the hub successfully.")

    if cfg.local_output_dir:
        local_output_path = Path(cfg.local_output_dir)
        local_output_path.mkdir(parents=True, exist_ok=True)

        for split, items in final_dataset.items():
            split_path = local_output_path / split

            split_path.mkdir(parents=True, exist_ok=True)
            for idx, item in enumerate(items):
                image: PILImage.Image = item["image"]
                image_save_path = split_path / f"image_{idx:05d}.png"
                image_metadata_path = split_path / f"image_{idx:05d}.json"
                image.save(image_save_path)

                image_metadata = {
                    "text": item["text"],
                    "font_path": item.get("font_path"),
                    "bg_color": item.get("bg_color"),
                    "font_color": item.get("font_color"),
                    "font_size": item.get("font_size"),
                    "image_width": item.get("image_width"),
                    "image_height": item.get("image_height"),
                    "image_dpi": item.get("image_dpi"),
                    "text_vertical_alignment": item.get("text_vertical_alignment"),
                    "text_horizontal_alignment": item.get("text_horizontal_alignment"),
                    "paragraph_bboxes": item.get("paragraph_bboxes"),
                    "transformations": item.get("transformations"),
                }
                image_metadata_path.write_text(
                    json.dumps(image_metadata, ensure_ascii=False, indent=4),
                    encoding="utf-8",
                )

        logger.info(f"Image dataset also saved to local directory {local_output_path}")


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = DataConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    create_image_dataset(cfg)


if __name__ == "__main__":
    main()
