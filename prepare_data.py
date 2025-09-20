import logging
import sys
from dataclasses import dataclass

from datasets import Dataset, Image, load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import create_image_with_text

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    dataset_path: str = "mideind/is_prototyping_corpus"
    max_length: int = 512
    max_entries: int = 10
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 72
    img_background_color: str = "white"
    font_path: str = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_size: int = 12
    font_color: str = "black"
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 1000  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )


def generate_image_dataset(texts, cfg):
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    """
    new_data = {"text": [], "image": []}

    # Get settings from config
    width = cfg.image_width
    height = cfg.image_height
    dpi = cfg.image_dpi
    font_size = cfg.font_size
    alignment = cfg.text_horizontal_alignment
    font_path = cfg.font_path
    bg_color = cfg.img_background_color
    font_color = cfg.font_color
    vertical_alignment = cfg.text_vertical_alignment

    # fix number of examples to generate if specified
    if cfg.num_examples:
        texts = texts[: cfg.num_examples]

    print("Generating images from text...")
    for text in tqdm(texts):
        remaining_text = text.strip()
        while remaining_text:
            image, fitted_text = create_image_with_text(
                remaining_text,
                image_size=(width, height),
                alignment=alignment,
                font_size=font_size,
                font_path=font_path,
                bg_color=bg_color,
                font_color=font_color,
                vertical_alignment=vertical_alignment,
                dpi=dpi,
            )

            if not fitted_text:
                # No text could be fitted, break to avoid infinite loop
                break

            new_data["text"].append(fitted_text)
            new_data["image"].append(image)

            # Update remaining text
            # This assumes create_image_with_text preserves original whitespace
            # and returns a prefix of the input text.
            remaining_text = remaining_text[len(fitted_text) :].lstrip()

    # Create a new Hugging Face Dataset
    image_dataset = Dataset.from_dict(new_data).cast_column("image", Image())
    return image_dataset


def create_image_dataset(cfg: DataConfig) -> None:
    # load dataset
    dataset = load_dataset(
        cfg.dataset_path,
        "igc",
        split=f"train",
    )

    # select number of entries if specified
    if cfg.max_entries:
        dataset = dataset.select(range(cfg.max_entries))

    texts = dataset["text"]

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    print(f"\nOriginal dataset size: {len(texts)}")
    print(f"New image dataset size: {len(image_dataset)}")

    # Save the new dataset
    output_path = cfg.output_path
    image_dataset.save_to_disk(output_path)
    print(f"Image dataset saved to {output_path}")

    # You can also push it to the hub
    # image_dataset.push_to_hub("your-username/your-dataset-name")

    # Display the first image as an example
    print("\nShowing first generated image...")
    if len(image_dataset) > 0:
        print("Text for first image:")
        print(f"'{image_dataset[0]['text']}'")
        image_dataset[0]["image"].show()

    # upload to huggingface dataset hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        print(f"Pushing dataset to the hub at {cfg.hub_repo_id}...")
        image_dataset.push_to_hub(cfg.hub_repo_id)
        print("Dataset pushed to the hub successfully.")


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
