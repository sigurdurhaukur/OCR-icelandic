# OCR-icelandic: Language-Agnostic OCR Model Training Pipeline

## Overview

Training recipe for OCR Vision Transformer models for languages with limited image-text paired datasets. Generates realistic synthetic document images from plain text, then fine-tunes vision-language models (SmolVLM, IDEFICS3) using LoRA adapters.

**Key Innovation**: Creates unlimited synthetic OCR training data from text-only corpora with realistic backgrounds, textures, and transformations.

## Repository Structure

```
OCR-icelandic/
├── scripts/
│   ├── prepare_data.py          # Synthetic OCR dataset generator
│   ├── smol_vlm_ft.py           # SmolVLM fine-tuning with LoRA
│   ├── train_llm.py             # Text-to-text model fine-tuning
│   ├── build_gold_data.py       # Evaluation dataset builder
│   └── webui.py                 # Gradio inference interface
│
├── src/ocr_icelandic/
│   ├── config.py                # DataConfig, GenerationConfig
│   ├── randomness.py            # Centralized RNG for reproducibility
│   ├── image_generator.py       # Pipeline-based image generation
│   │
│   ├── pipeline/
│   │   ├── core.py              # Pipeline, PipelineState, Stage protocol
│   │   └── stages/
│   │       ├── selection.py     # Font/color/layout/texture/background
│   │       ├── rendering.py     # Text rendering to image
│   │       ├── transformations.py
│   │       └── postprocessing.py
│   │
│   ├── transformations/
│   │   ├── pipeline.py          # Transformation orchestration
│   │   ├── effects.py           # Blur, dusty, stains, bleed-through
│   │   ├── perspective.py       # 3D perspective distortions
│   │   ├── rotate.py            # Rotation transformations
│   │   ├── lighting.py          # Light reflection, shadows
│   │   ├── tight_crop.py        # Content-aware cropping
│   │   └── shared.py            # Utility functions
│   │
│   ├── utils/
│   │   ├── image_creation.py    # Core text rendering
│   │   ├── text_layout.py       # Text wrapping and layout
│   │   ├── texture.py           # Paper textures and noise
│   │   ├── visualization.py     # Bounding box visualization
│   │   ├── font.py              # Font loading utilities
│   │   └── color.py             # Color generation
│   │
│   ├── font_cache.py            # Font caching system
│   ├── language_support.py      # Language character sets
│   ├── fonts.py                 # Font discovery
│   └── colors.py                # Color utilities
│
├── assets/
│   ├── papers/                  # 19 paper textures
│   ├── stains/                  # 77 stain textures (coffee, tea, ink, etc.)
│   ├── backgrounds/
│   │   ├── no_shadow/           # Distant backgrounds (landscapes, cityscapes)
│   │   └── with_shadow/         # Close backgrounds (desks)
│   └── generate_assets.py       # Replicate API asset generator
│
├── tests/
│   ├── test_transformations.py
│   ├── test_transformation_snapshots.py
│   └── __snapshots__/
│
├── slurm/                       # SLURM job scripts
└── notebooks/                   # Jupyter notebooks
```

## Pipeline Architecture

The core is a modular pipeline that decomposes image generation into composable stages.

**Location**: `src/ocr_icelandic/pipeline/core.py`

### PipelineState

Immutable data container passing through all stages:

```python
@dataclass
class PipelineState:
    text: str
    image_size: tuple[int, int]
    dpi: int
    image: Image.Image | None
    fitted_text: str
    paragraph_bboxes: list[dict]
    font_path: str | None
    font_size: int
    font_color: tuple[int, int, int] | str
    bg_color: tuple[int, int, int] | str
    paper_texture_path: str | None
    background_image: Image.Image | None
    num_columns: int
    column_gap: int
    alignment: str
    vertical_alignment: str
    transformation_metadata: list[dict]
    stage_metadata: dict
```

### Stage Protocol

```python
class Stage(Protocol):
    @property
    def name(self) -> str: ...
    def __call__(self, state: PipelineState) -> PipelineState: ...
```

### Standard Stages

**Selection** (`pipeline/stages/selection.py`):
- `SelectFontStage` - Choose font from compatible fonts
- `SelectColorsStage` - Generate contrasting colors
- `SelectLayoutStage` - Column count and alignment
- `SelectPaperTextureStage` - Paper texture selection
- `SelectBackgroundImageStage` - Background image selection

**Rendering** (`pipeline/stages/rendering.py`):
- `RenderTextStage` - Render text to RGBA image with bboxes

**Transformation** (`pipeline/stages/transformations.py`):
- `ApplyTransformationsStage` - Apply random document transformations

**Postprocessing** (`pipeline/stages/postprocessing.py`):
- `CompositeBackgroundStage` - Blend onto background
- `FinalizeImageStage` - Convert RGBA to RGB
- `VisualizeBBoxesStage` - Debug visualization

### Custom Pipeline Example

```python
from ocr_icelandic.pipeline.core import Pipeline
from ocr_icelandic.pipeline.stages import (
    SelectFontStage, SelectColorsStage, SelectLayoutStage,
    SelectPaperTextureStage, SelectBackgroundImageStage,
    RenderTextStage, ApplyTransformationsStage,
    CompositeBackgroundStage, FinalizeImageStage
)

pipeline = Pipeline([
    SelectFontStage(),
    SelectColorsStage(),
    SelectLayoutStage(),
    SelectPaperTextureStage(),
    SelectBackgroundImageStage(),
    RenderTextStage(),
    ApplyTransformationsStage(),
    CompositeBackgroundStage(),
    FinalizeImageStage(),
])

state = PipelineState(text="Your text", image_size=(512, 512))
state = pipeline(state)
image = state.image
bboxes = state.paragraph_bboxes
```

## Reproducibility

**Location**: `src/ocr_icelandic/randomness.py`

```python
from ocr_icelandic.randomness import set_seed, get_seed, reset, rng, np_rng

set_seed(42)           # Set both Python random and NumPy
seed = get_seed()      # Get current seed
reset()                # Reset to non-reproducible

value = rng.random()                    # Random float [0, 1)
choice = rng.choice(['a', 'b', 'c'])   # Random choice
array = np_rng.uniform(0, 1, (10,))    # NumPy random
```

Generate reproducible dataset:
```bash
python scripts/prepare_data.py max_entries=1000 random_seed=42
```

## Background Images

**Location**: `src/ocr_icelandic/pipeline/stages/postprocessing.py`

| Category | Location | Use Case |
|----------|----------|----------|
| Landscapes | `assets/backgrounds/no_shadow/landscapes/` | Distant backgrounds |
| Cityscapes | `assets/backgrounds/no_shadow/cityscapes/` | Distant backgrounds |
| Desks | `assets/backgrounds/with_shadow/desks/` | Close backgrounds |

Pipeline automatically adapts transformations based on background type (shadow vs no-shadow).

## Asset Generation

**Location**: `assets/generate_assets.py`

| Category | Count |
|----------|-------|
| Papers | 19 |
| Coffee/Tea/Ink/Wine stains | 77 |
| Landscapes | 18 |
| Cityscapes | 15 |
| Desks | 18 |

```bash
python assets/generate_assets.py                    # Generate all
python assets/generate_assets.py --category papers  # Specific category
python assets/generate_assets.py --dry-run          # Preview prompts
```

## Data Generation

### Workflow

```
Text Corpus (HuggingFace)
    → Text Selection & Splitting
    → Run Image Generation Pipeline
    → Apply Random Transformations
    → Create Train/Val/Test Splits
    → Save to Disk / Push to Hub
```

### Entry Point

**Location**: `src/ocr_icelandic/image_generator.py`

```python
def generate_single_text(text: str, cfg: GenerationConfig) -> list[SingleImageData]:
    """Generate images for a single text entry using pipeline."""
```

Returns `SingleImageData` objects with: `image`, `text`, `font_path`, `font_size`, `font_color`, `bg_color`, `paragraph_bboxes`, `transformations`.

### Configuration

See `src/ocr_icelandic/config.py` for full `DataConfig` options. Key settings:

- **Dataset**: `dataset_path`, `text_column`, `max_entries`
- **Image**: `image_width`, `image_height`, `image_dpi`
- **Font**: `font_size_range`, `use_random_fonts`, `language_code`
- **Layout**: `column_range`, `column_gap`, `text_horizontal_alignment`
- **Bounding Boxes**: `bbox_per_column`, `bbox_max_chars`
- **Textures**: `use_paper_textures`, `use_background_images`
- **Output**: `local_output_dir`, `save_to_disk`, `push_to_hub`

### Usage Examples

```bash
# Basic
python scripts/prepare_data.py

# With backgrounds and reproducibility
python scripts/prepare_data.py \
  max_entries=5000 \
  use_background_images=True \
  random_seed=42

# Custom dataset
python scripts/prepare_data.py \
  dataset_path=my_org/my_corpus \
  text_column=text_field \
  language_code=en \
  max_entries=10000

# Multi-column with textures
python scripts/prepare_data.py \
  num_columns=2 \
  column_gap=30 \
  use_paper_textures=True
```

## Bounding Box Configuration

Control how bounding boxes are generated for training data:

### bbox_per_column

**Type**: `bool` (default: `False`)

When `True`, creates separate bounding boxes when a paragraph spans multiple columns. When `False`, creates a single union bbox for the entire paragraph across all columns.

**Use case**: Column-aware document analysis where each column section needs independent bbox annotations.

**Example**:
```bash
# Split bboxes at column boundaries
python scripts/prepare_data.py \
  bbox_per_column=True \
  num_columns=3
```

**Output format**:
- Paragraph spanning columns 0→1→2 creates 3 separate bboxes
- Each bbox has `sequence_number`: 0, 1, 2...
- Each bbox has `columns`: [0], [1], [2] respectively

### bbox_max_chars

**Type**: `int | None` (default: `None`)

Maximum characters per bounding box. When set, splits bboxes at rendered line boundaries when the character limit is exceeded. `None` means no character limit.

**Use case**: Limiting bbox size in training data to prevent extremely large bounding boxes that are difficult for models to process.

**Example**:
```bash
# Limit bbox size to 100 characters
python scripts/prepare_data.py bbox_max_chars=100
```

**Behavior**:
- Character count uses visible text only (strips whitespace)
- Splits occur at line boundaries (never mid-line)
- If a single line exceeds the limit, it still gets a bbox (no mid-line splits)
- Empty lines contribute 0 characters

### Combined Usage

Both settings work together and are independent:

```bash
# Split by both column boundaries and character limit
python scripts/prepare_data.py \
  bbox_per_column=True \
  bbox_max_chars=150 \
  num_columns=2
```

**Priority**: Column splits are checked first, then character splits within each column section.

### Bbox Output Format

When splitting is enabled, bbox dictionaries include additional metadata:

```python
{
    "paragraph_index": 0,          # Original paragraph index
    "sequence_number": 0,          # 0, 1, 2, ... for split bboxes
    "paragraph_text": "Full...",   # Complete paragraph text (same for all splits)
    "columns": [2],                # List of columns this bbox spans
    "char_count": 85,              # Character count in this bbox
    "bbox": [100, 200, 300, 250]   # Bounding box coordinates [x0, y0, x1, y1]
}
```

**Backward compatibility**: All bboxes include these fields. When no splitting occurs:
- `sequence_number` = 0
- `columns` = list with single column
- `char_count` = total characters in paragraph

## Core Modules

### config.py
- `DataConfig` - Complete configuration for dataset generation
- `GenerationConfig` - Extended config with resolved resources
- `SingleImageData` - Generated image with metadata

### image_generator.py
- `generate_single_text()` - Main entry point for pipeline-based generation

### pipeline/core.py
- `Pipeline` - Orchestrates stages
- `PipelineState` - State container
- `Stage` - Protocol for stages

### utils/image_creation.py
- `create_image_with_text()` - Core text rendering with multi-column layout

### utils/texture.py
- `load_paper_textures()`, `generate_paper_texture()`, `apply_texture()`

### font_cache.py

```python
from ocr_icelandic.font_cache import FontCache

cache = FontCache(".fontcache", language_code="is")
compatible_fonts = cache.get_fonts(font_dirs=["/usr/share/fonts"], language_code="is")
```

### language_support.py

```python
from ocr_icelandic.language_support import LanguageRegistry

lang = LanguageRegistry.get("is")  # Icelandic
required_chars = lang.required_characters
```

Supported: Icelandic, English, Norwegian, Danish, Swedish, German, Spanish, French, Portuguese

## Transformations

**Location**: `src/ocr_icelandic/transformations/`

### pipeline.py

Three configs based on background type:
- `PIPELINE_NO_BACKGROUND_PROBABILITIES`
- `PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES`
- `PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES`

```python
def apply_random_transformation(
    image: Image.Image,
    paragraph_bboxes: list[dict],
    stain_textures: list[Image.Image] | None = None,
    background_receives_shadow: bool = False,
) -> tuple[Image.Image, list[dict], list[str]]:
```

### effects.py
- `blur()` - Gaussian blur
- `dusty_paper()` - Grainy texture
- `reverse_bleed_through()` - Text showing through
- `textured_stains()` - Coffee/tea stains
- `ink_splashes()` - Ink splatter
- `paper_edge_unevenness()` - Torn edges

### perspective.py
- `_apply_perspective_distortion()` - Main perspective transformation
- Supports: book curve, camera angle, combined effects

### rotate.py
- `_rotate_within_bounds()` - Rotation with padding
- `_transform_paragraph_bboxes_for_rotation()`

### lighting.py
- `light_reflection()` - Camera flash spots
- `shadow_overlay()` - Edge shadows
- `shadow_gradient()` - Directional gradient

### tight_crop.py
- `tight_crop()` - Remove excess whitespace

## Testing

```bash
uv sync --group dev                           # Install test deps
pytest tests/ -v                              # Run all tests
pytest tests/test_transformations.py -v       # Specific file
pytest tests/ --cov=src/ocr_icelandic         # With coverage
```

### Snapshot Tests

Visual regression tests for transformations.

```bash
# Generate initial snapshots
pytest tests/test_transformation_snapshots.py --snapshot-update

# Run tests (detect changes)
pytest tests/test_transformation_snapshots.py -v

# Update after intentional changes
pytest tests/test_transformation_snapshots.py --snapshot-update
```

Snapshots in `tests/__snapshots__/`: `.png` files for images, `.amber` for bboxes.

All snapshot tests use fixed seeds (`random.seed(42)`) for determinism.

## Training & Inference

### smol_vlm_ft.py

Fine-tunes SmolVLM/IDEFICS3 with LoRA adapters.

```bash
python scripts/smol_vlm_ft.py \
  model_id=HuggingFaceTB/SmolVLM-Base \
  hf_dataset_id=Sigurdur/isl_synthetic_ocr \
  output_dir=./lora_results \
  num_train_epochs=3 \
  learning_rate=1e-4
```

Features: LoRA/QLoRA, custom OCR metrics (WER, CER), W&B integration, 8-bit optimizers.

### webui.py

Gradio interface for OCR inference.

```bash
python scripts/webui.py  # Opens http://localhost:7860
```

## Installation

```bash
pip install -e .                    # Base
pip install -e ".[training]"        # With training deps
uv sync                             # Or use UV (recommended)
```

### Core Dependencies
- `datasets`, `pillow`, `opencv-python` - Data processing
- `omegaconf`, `rich`, `tqdm` - Configuration & utilities

### Training Dependencies (optional)
- `torch`, `transformers`, `accelerate`, `peft`, `bitsandbytes`
- `wandb`, `gradio`

## Development Workflow

```bash
# Generate synthetic data
python scripts/prepare_data.py \
  dataset_path=your/corpus \
  max_entries=10000 \
  random_seed=42 \
  save_to_disk=True

# Fine-tune model
python scripts/smol_vlm_ft.py \
  hf_dataset_id=your/dataset \
  num_train_epochs=3

# Evaluate
python scripts/webui.py
```

### SLURM

```bash
sbatch slurm/generate_synthetic_data.slurm
sbatch slurm/train_smolVLM.slurm
```

### Code Quality

```bash
pre-commit install
pre-commit run --all-files
```

## Language Adaptation

1. **Prepare corpus**: HuggingFace dataset or local files
2. **Select fonts**: Use `language_code` parameter for automatic font filtering
3. **Generate data**: `python scripts/prepare_data.py dataset_path=... language_code=de`
4. **Fine-tune**: `python scripts/smol_vlm_ft.py hf_dataset_id=...`
5. **Evaluate**: `python scripts/webui.py`

For non-Latin scripts: ensure Unicode font support, consider larger image sizes.

## Troubleshooting

**No compatible fonts found**
- Install fonts for your language
- Use `font_path=/path/to/font.ttf` directly
- Use `language_code=your_code`

**Out of memory during training**
- Reduce `per_device_train_batch_size`
- Use QLoRA: `load_in_4bit=True`
- Reduce `lora_r`

**Images look unrealistic**
- Enable `use_background_images=True`
- Enable `use_paper_textures=True`

**Text doesn't fit**
- Increase `image_width`/`image_height`
- Reduce `font_size` or `max_text_length`
- Adjust `column_range`

**Snapshot tests failing**
- Review differences carefully
- If intentional: `pytest ... --snapshot-update`
- If unintended: fix bug and re-run

**Datasets not reproducible**
- Always use `random_seed=42` parameter

**Font cache outdated**
- Delete `.fontcache/` and regenerate

## Conventions

- Use `uv` for dependency management
- Use `pytest` for testing
- Follow PEP 8 (enforced by pre-commit)
- All random operations use centralized `randomness` module
- Document line numbers as `file.py:123` format
