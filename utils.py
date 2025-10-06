import random

from PIL import Image, ImageDraw, ImageFont


def load_font(font: str = "Arial.ttf", font_size: int = 20) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font or default if not found.
    Args:
        font: Path to the .ttf font file
        font_size: Size of the font
    Returns:
        ImageFont.FreeTypeFont object
    """
    # Load a font
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    return font


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    tab_width: int = 4,
    dpi: int = 72,
) -> list[str]:
    """
    Wrap text to fit within a maximum width.

    Args:
        draw: ImageDraw object.
        text: The text to wrap.
        font: The font to use.
        max_width: The maximum width for a line of text.
        tab_width: The number of spaces to replace tabs with.
        dpi: Dots per inch for the image.

    Returns:
        A list of strings, where each string is a wrapped line.
    """
    paragraphs = text.split("\n")
    lines = []

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            lines.append("")
            continue

        leading_whitespace = ""
        stripped_paragraph = paragraph.lstrip()
        if len(paragraph) > len(stripped_paragraph):
            leading_whitespace = paragraph[: len(paragraph) - len(stripped_paragraph)]
            leading_whitespace = leading_whitespace.replace("\t", " " * tab_width)

        stripped_paragraph = stripped_paragraph.replace("\t", " " * tab_width)
        words = stripped_paragraph.split()
        current_line = []
        is_first_line = True

        for word in words:
            test_line_base = " ".join(current_line + [word])
            test_line = (
                leading_whitespace + test_line_base if is_first_line else test_line_base
            )
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    line_to_add = (
                        leading_whitespace if is_first_line else ""
                    ) + " ".join(current_line)
                    lines.append(line_to_add)
                    is_first_line = False
                current_line = [word]
                # Handle case where a single word is too long
                if not lines or lines[-1] != (
                    (leading_whitespace if is_first_line else "") + word
                ):
                    test_line_base = " ".join(current_line)
                    test_line = (
                        leading_whitespace + test_line_base
                        if is_first_line
                        else test_line_base
                    )
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if bbox[2] - bbox[0] > max_width:
                        lines.append(
                            (leading_whitespace if is_first_line else "") + word
                        )
                        is_first_line = False
                        current_line = []

        if current_line:
            lines.append(
                (leading_whitespace if is_first_line else "") + " ".join(current_line)
            )

        if para_idx < len(paragraphs) - 1:
            lines.append("")
    return lines


def get_text_drawing_details(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    image_size: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    vertical_alignment: str = "center",
) -> tuple[int, int, list[str]]:
    """
    Calculate vertical positioning and determine which lines fit.

    Args:
        draw: ImageDraw object.
        lines: A list of text lines.
        image_size: The (width, height) of the image.
        font: The font being used.
        vertical_alignment: Vertical text alignment ('top', 'center', 'bottom').

    Returns:
        A tuple containing the starting Y position, effective line height, and the list of fitted lines.
    """
    line_height = (
        draw.textbbox((0, 0), "Ag", font=font)[3]
        - draw.textbbox((0, 0), "Ag", font=font)[1]
    )
    line_spacing = int(line_height * 0.2)
    effective_line_height = line_height + line_spacing
    total_text_height = len(lines) * effective_line_height - line_spacing

    if vertical_alignment == "top":
        start_y = 0
    elif vertical_alignment == "bottom":
        start_y = max(0, image_size[1] - total_text_height)
    else:  # center
        start_y = max(0, (image_size[1] - total_text_height) // 2)

    fitted_lines = []
    for i, line in enumerate(lines):
        y_position = start_y + i * effective_line_height
        if y_position + line_height <= image_size[1]:
            fitted_lines.append(line)
        else:
            break
    return int(start_y), int(effective_line_height), fitted_lines


def draw_text_lines(
    draw: ImageDraw.ImageDraw,
    fitted_lines: list[str],
    start_y: int,
    effective_line_height: int,
    font: ImageFont.FreeTypeFont,
    image_size: tuple[int, int],
    alignment: str,
    margin_x: int,
    font_color: str = "black",
):
    """
    Draw the text lines onto the image.

    Args:
        draw: ImageDraw object.
        fitted_lines: The lines of text that fit on the image.
        start_y: The starting Y position for drawing.
        effective_line_height: The height of each line including spacing.
        font: The font to use.
        image_size: The (width, height) of the image.
        alignment: The text alignment ('left', 'center', 'right').
        margin_x: The horizontal margin.
        font_color: The color of the font.
    """
    for i, line in enumerate(fitted_lines):
        y_position = start_y + i * effective_line_height
        if line:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            if alignment == "left":
                x_position = margin_x
            elif alignment == "right":
                x_position = image_size[0] - margin_x - line_width
            else:  # center
                x_position = (image_size[0] - line_width) // 2
            draw.text(
                (int(x_position), int(y_position)), line, fill=font_color, font=font
            )


def create_image_with_text(
    text: str,
    image_size: tuple[int, int] = (400, 100),
    font_path: str = "Arial.ttf",
    font_size: int = 20,
    font_color: str = "black",
    bg_color: str = "white",
    max_width_ratio: float = 0.9,
    tab_width: int = 4,
    alignment: str = "center",
    vertical_alignment: str = "center",
    dpi: int = 72,
) -> tuple[Image.Image, str]:
    """
    Create an image with text for OCR training.

    Args:
        text: Text to render
        image_size: Tuple of (width, height) in pixels at default DPI (72)
        font_path: Path to the .ttf font file
        font_size: Size of the font in points at default DPI (72)
        font_color: Color of the font
        bg_color: Background color of the image
        max_width_ratio: Ratio of image width to use for text (0.0-1.0)
        tab_width: Number of spaces to replace tabs with
        alignment: Text alignment - 'center', 'left', or 'right'
        vertical_alignment: Vertical text alignment - 'top', 'center', or 'bottom'
        dpi: Dots per inch for the image

    Returns:
        tuple: (PIL Image object, string of text that actually fits in the image)
    """
    # Scale image size and font size based on DPI
    scale_factor = dpi / 72.0
    scaled_image_size = (
        int(image_size[0] * scale_factor),
        int(image_size[1] * scale_factor),
    )
    scaled_font_size = int(font_size * scale_factor)

    image = Image.new("RGB", scaled_image_size, color=bg_color)
    image.info["dpi"] = (dpi, dpi)
    draw = ImageDraw.Draw(image)
    font_path = load_font(font=font_path, font_size=scaled_font_size)

    max_text_width = int(scaled_image_size[0] * max_width_ratio)
    margin_x = (scaled_image_size[0] - max_text_width) // 2

    lines = wrap_text(draw, text, font_path, max_text_width, tab_width)

    start_y, effective_line_height, fitted_lines = get_text_drawing_details(
        draw, lines, scaled_image_size, font_path, vertical_alignment=vertical_alignment
    )

    draw_text_lines(
        draw,
        fitted_lines,
        start_y,
        effective_line_height,
        font_path,
        scaled_image_size,
        alignment,
        margin_x,
        font_color=font_color,
    )

    actual_text_lines = []
    for line in fitted_lines:
        actual_text_lines.append(line)

    while actual_text_lines and not actual_text_lines[-1].strip():
        actual_text_lines.pop()

    actual_text = "\n".join(actual_text_lines)

    return image, actual_text


def dummy_text_with_line_breaks(num_sentences=5):
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
    return "\n".join(selected_sentences)
