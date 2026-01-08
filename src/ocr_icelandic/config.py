"""Configuration dataclasses for synthetic OCR dataset generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage


@dataclass
class DataConfig:
    """Configuration for synthetic OCR dataset creation."""

    # Dataset source
    dataset_path: str = "arnastofnun/IGC-2024"
    text_column: str = "document"
    data_directory: str = "parla"
    split: str = "train"
    max_length: int = 512
    max_entries: int = 2
    num_examples: int = 2

    # Image settings
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 144  # Standard set by SmolDocling paper
    img_background_color: str = "white"

    # Font settings
    font_size: int | None = None  # Fixed size, or None to use range
    font_size_range: tuple[int, int] = (11, 24)
    font_color: str = "black"
    use_random_fonts: bool = True
    use_random_font_sizes: bool = True
    use_random_font_colors: bool = True

    # Text layout
    max_text_length: int = 2000
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    num_columns: int | None = None  # Fixed count, or None to use range
    column_range: tuple[int, int] = (1, 5)
    column_width: int | None = None  # Fixed width, or None to use range
    column_width_range: tuple[int, int] = (100, 512)
    column_gap: int = 20

    # Background/texture settings
    use_random_backgrounds: bool = True
    use_paper_textures: bool = True
    paper_textures_dir: str = "assets/papers"
    use_background_images: bool = True
    backgrounds_dir: str = "assets/backgrounds"
    background_image_probability: float = 1.0

    # Transformations
    apply_random_transformations: bool = True

    # Output settings
    output_path: str = "isl_synthetic_ocr_output"
    local_output_dir: str = "./local_output"
    save_to_disk: bool = False
    push_to_hub: bool = False
    hub_repo_id: str = "Sigurdur/isl_synthetic_ocr"
    show_sample: bool = False

    # Font discovery
    google_fonts_directory: str = "../google_fonts"
    language_code: str = "is"  # ISO 639-1 language code
    use_font_cache: bool = True
    font_cache_dir: str = ".fontcache"

    # Processing
    max_workers: int = 1
    batch_size: int = 50


@dataclass
class GenerationConfig(DataConfig):
    """Runtime configuration with cached/resolved resources."""

    available_fonts: list[str] | None = None
    available_paper_textures: list[str] | None = None
    available_no_shadow_backgrounds: list[str] | None = None
    available_with_shadow_backgrounds: list[str] | None = None


@dataclass
class SingleImageData:
    """Data for a single generated image."""

    text: str
    image: "PILImage.Image"
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
