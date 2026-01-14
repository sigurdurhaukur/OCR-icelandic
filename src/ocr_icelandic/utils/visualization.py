"""Visualization and debugging utilities."""

import random

from PIL import Image, ImageDraw, ImageFont

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


def dummy_text_with_line_breaks(num_sentences: int = 5) -> str:
    """Generate dummy text for testing.

    Args:
        num_sentences: Number of sentences to generate

    Returns:
        String with randomly selected test sentences
    """
    logger.debug("Generating dummy text with %d sentences", num_sentences)
    sentences = [
        "Icelandic characters: ð, þ, æ, ö, á, é, í, ó, ú.",
        # "This is a sample sentence for OCR training.",
        # "Pillow makes it easy to create images with text.",
        # "Line breaks should be handled properly.",
        # "Tabs and spaces can affect text alignment.",
        # "This is the last sentence in this example.",
        # "Additional text to test overflow handling.",
        # "More text that might get cut off.",
        # "Even more text for testing purposes.",
        # "This line might not fit in smaller images.",
        # "Final line that definitely won't fit in tiny images.",
        "„Megi hann fara og vera en ég vona svo sannarlega að hann komi aldrei aftur til Íslands,“ segir Helgi Magnús Gunnarsson fyrrverandi vararíkssaksóknari um nýjustu vendingar í máli Mohamads Kourani. Helgi, sem sætti líflátshótunum",
    ]
    selected_sentences = random.choices(sentences, k=num_sentences)
    result = "\n".join(selected_sentences)
    logger.debug("Generated dummy text of length %d", len(result))
    return result


def visualise_bboxes(
    image: Image.Image,
    paragraph_bboxes: list[dict],
    line_width: int = 2,
    show_labels: bool = True,
    max_label_chars: int = 20,
) -> Image.Image:
    """
    Draw bounding boxes on an image to visualize paragraph locations.

    Args:
        image: PIL Image object to draw on
        paragraph_bboxes: List of bbox dictionaries with format:
            [{"paragraph_index": int, "paragraph_text": str, "column": int, "bbox": [x1, y1, x2, y2]}]
        line_width: Width of the rectangle border in pixels
        show_labels: Whether to show paragraph text preview labels
        max_label_chars: Maximum number of characters to show in label preview

    Returns:
        PIL Image object with bounding boxes drawn
    """
    logger.debug("Visualizing %d bounding boxes on image (size=%dx%d)", len(paragraph_bboxes), image.width, image.height)
    # Create a copy to avoid modifying the original
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image)

    # Define color palette for sequential cycling
    color_palette = [
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    # Load a small font for labels
    try:
        label_font = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        label_font = ImageFont.load_default()

    # Draw each bbox
    drawn_count = 0
    for idx, bbox_data in enumerate(paragraph_bboxes):
        # Get bbox coordinates
        bbox = bbox_data.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            logger.debug("Skipping bbox %d: invalid format", idx)
            continue

        x1, y1, x2, y2 = bbox
        drawn_count += 1

        # Select color from palette (cycle sequentially)
        color = color_palette[idx % len(color_palette)]

        logger.debug("Drawing bbox %d: coordinates (%d, %d, %d, %d)", idx, x1, y1, x2, y2)
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label if enabled
        if show_labels:
            paragraph_text = bbox_data.get("paragraph_text", "")
            if paragraph_text:
                # Truncate text to max_label_chars
                label_text = paragraph_text[:max_label_chars]
                if len(paragraph_text) > max_label_chars:
                    label_text += "..."

                # Calculate label background size
                label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]

                # Position label at top-left of bbox with padding
                label_x = x1
                label_y = y1 - label_height - 4  # 4px padding

                # If label would go above image, place it inside the bbox
                if label_y < 0:
                    label_y = y1 + 2

                # Draw semi-transparent background for label
                background_padding = 2
                draw.rectangle(
                    [
                        label_x - background_padding,
                        label_y - background_padding,
                        label_x + label_width + background_padding,
                        label_y + label_height + background_padding,
                    ],
                    fill=(0, 0, 0, 200),  # Black with some transparency
                )

                # Draw label text
                draw.text(
                    (label_x, label_y),
                    label_text,
                    fill=(255, 255, 255),
                    font=label_font,
                )

    logger.debug("Visualization complete: drew %d bounding boxes", drawn_count)
    return visualized_image
