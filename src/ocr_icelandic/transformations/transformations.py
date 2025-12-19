import random

from PIL import Image, ImageDraw, ImageFilter

from ocr_icelandic.transformations.perspective import perspective
from ocr_icelandic.transformations.rotate import rotate
from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes
from ocr_icelandic.transformations.skew import skew


def blur(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    radius = random.uniform(0.1, 0.5)
    return (
        image.filter(ImageFilter.GaussianBlur(radius)),
        {
            "transformation": "blur",
            "radius": round(radius, 2),
        },
        paragraph_bboxes_copy,
    )


def ink_splashes(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    splashes = random.randint(3, 6)
    for _ in range(splashes):
        radius = random.randint(10, 30)
        cx = random.randint(0, image.width)
        cy = random.randint(0, image.height)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        color = (0, 0, 0, random.randint(80, 150))

        # Create temporary image for single splash with blur
        splash = Image.new("RGBA", image.size, (0, 0, 0, 0))
        splash_draw = ImageDraw.Draw(splash)
        splash_draw.ellipse(bbox, fill=color)
        splash = splash.filter(ImageFilter.GaussianBlur(radius=2))

        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, splash)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return (
        combined.convert("RGB"),
        {
            "transformation": "ink_splashes",
            "splashes": splashes,
        },
        paragraph_bboxes_copy,
    )


def dusty_paper(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, dict, list[dict]]:
    """create grainy overlay to simulate dusty paper
    varies in grain size and intensity"""
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    grain_size = random.randint(1, 3)
    intensity = random.uniform(0.05, 0.15)
    noise = Image.effect_noise(image.size, grain_size * 10)
    grainy_overlay = noise.convert("RGB")
    dusty_image = Image.blend(image, grainy_overlay, intensity)
    return (
        dusty_image,
        {
            "transformation": "dusty-paper",
            "grain_size": grain_size,
            "intensity": round(intensity, 3),
        },
        paragraph_bboxes_copy,
    )


CONTENT_TRANSFORMATIONS = [
    blur,
    ink_splashes,
    dusty_paper,
]
PERSPECTIVE_TRANSFORMATIONS = [
    rotate,
    skew,
    perspective,
]


def _get_random_subset(transformations: list) -> list:
    k = random.randint(0, len(transformations))
    return random.sample(transformations, k)


def apply_random_transformation(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
) -> tuple[Image.Image, list[dict], list[dict]]:
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    transformations_to_apply = [
        *_get_random_subset(CONTENT_TRANSFORMATIONS),
        *_get_random_subset(PERSPECTIVE_TRANSFORMATIONS),
    ]

    transformation_meta: list[dict] = []
    for transform in transformations_to_apply:
        image, meta, paragraph_bboxes_copy = transform(
            image, bg_color, paragraph_bboxes_copy
        )

        transformation_meta.append(meta)

    return image, transformation_meta, paragraph_bboxes_copy
