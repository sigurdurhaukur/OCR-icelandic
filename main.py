from datasets import Dataset, Image, load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import create_image_with_text


def generate_image_dataset(texts, cfg):
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    """
    new_data = {"text": [], "image": []}

    # Get settings from config
    width = cfg.image.width
    height = cfg.image.height
    dpi = cfg.image.dpi
    font_size = cfg.font.size
    alignment = cfg.text.horizontal_alignment
    font_path = cfg.font.path
    bg_color = cfg.image.background_color
    font_color = cfg.font.color
    vertical_alignment = cfg.text.vertical_alignment

    # fix number of examples to generate if specified
    if cfg.output.get("num_examples"):
        texts = texts[: cfg.output.num_examples]

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


if __name__ == "__main__":
    # Load configuration
    cfg = OmegaConf.load("config.yaml")

    # load dataset
    text_dataset_cfg = cfg.text_dataset
    dataset = load_dataset(
        text_dataset_cfg.dataset_path,
        "igc",
        split=f"train",
    )

    # select number of entries if specified
    if text_dataset_cfg.get("max_entries"):
        dataset = dataset.select(range(text_dataset_cfg.max_entries))

    texts = dataset["text"]

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    print(f"\nOriginal dataset size: {len(texts)}")
    print(f"New image dataset size: {len(image_dataset)}")

    # Save the new dataset
    output_path = cfg.output.path
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
    if cfg.output.get("push_to_hub") and cfg.output.get("hub_repo_id"):
        print(f"Pushing dataset to the hub at {cfg.output.hub_repo_id}...")
        image_dataset.push_to_hub(cfg.output.hub_repo_id)
        print("Dataset pushed to the hub successfully.")
