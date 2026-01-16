"""Post-processing stages for finalizing images."""

from PIL import Image

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.pipeline.core import BaseStage, PipelineState
from ocr_icelandic.pipeline.stages.selection import get_random_background_color

logger = get_logger(__name__)


class CompositeBackgroundStage(BaseStage):
    """
    Composite the document onto a background image.

    Only applies if state.background_image is set.
    """

    def __init__(self, position: tuple[int, int] | None = None):
        super().__init__("CompositeBackground")
        self.position = position

    def __call__(self, state: PipelineState) -> PipelineState:
        if state.background_image is None or state.image is None:
            return state

        from ocr_icelandic.utils.texture import apply_background_image

        state.image, bg_meta, state.paragraph_bboxes = apply_background_image(
            state.image,
            state.background_image,
            paragraph_bboxes=state.paragraph_bboxes,
            position=self.position,
        )

        state.transformation_metadata.append({"transformation": "background", **bg_meta})
        logger.debug("Composited image onto background")

        return state


class FinalizeImageStage(BaseStage):
    """
    Convert image to final RGB format and scale down if render_scale > 1.

    Handles RGBA to RGB conversion with proper background color compositing,
    then scales down to target size using high-quality resampling.
    """

    def __init__(
        self,
        composite_color: tuple[int, int, int] | str | None = None,
        use_random_composite: bool = True,
    ):
        super().__init__("FinalizeImage")
        self.composite_color = composite_color
        self.use_random_composite = use_random_composite

    def __call__(self, state: PipelineState) -> PipelineState:
        if state.image is None:
            return state

        if state.image.mode == "RGBA":
            # Determine composite color
            if self.composite_color:
                bg = self.composite_color
            elif self.use_random_composite:
                bg = get_random_background_color()
            else:
                bg = state.bg_color

            # Ensure bg is a tuple for Image.new
            if isinstance(bg, str):
                from PIL import ImageColor

                bg = ImageColor.getrgb(bg)

            # Create RGB background and paste RGBA image
            rgb_background = Image.new("RGB", state.image.size, bg)
            rgb_background.paste(state.image, (0, 0), state.image)
            state.image = rgb_background
            logger.debug("Converted RGBA to RGB with background color")

        elif state.image.mode != "RGB":
            state.image = state.image.convert("RGB")
            logger.debug("Converted image to RGB from mode: %s", state.image.mode)

        # Scale down if render_scale > 1
        scale = state.render_scale
        if scale > 1:
            state.image = state.image.resize(state.image_size, Image.Resampling.LANCZOS)

            # Scale down bounding boxes
            for bbox in state.paragraph_bboxes:
                bbox["bbox"][0] = int(bbox["bbox"][0] / scale)
                bbox["bbox"][1] = int(bbox["bbox"][1] / scale)
                bbox["bbox"][2] = int(bbox["bbox"][2] / scale)
                bbox["bbox"][3] = int(bbox["bbox"][3] / scale)

            logger.debug("Scaled down from %dx to target size", scale)

        return state


class VisualizeBBoxesStage(BaseStage):
    """Debug stage to visualize bounding boxes on the image."""

    def __init__(self, show_labels: bool = True, enabled: bool = True):
        super().__init__("VisualizeBBoxes")
        self.show_labels = show_labels
        self.enabled = enabled

    def __call__(self, state: PipelineState) -> PipelineState:
        if not self.enabled or state.image is None:
            return state

        from ocr_icelandic.utils.visualization import visualise_bboxes

        state.image = visualise_bboxes(
            state.image,
            state.paragraph_bboxes,
            show_labels=self.show_labels,
        )
        logger.debug("Visualized %d bounding boxes", len(state.paragraph_bboxes))

        return state


class CropToContentStage(BaseStage):
    """Optionally crop the image to content bounds with padding."""

    def __init__(
        self,
        enabled: bool = True,
        padding: int = 10,
        min_coverage: float = 0.5,
    ):
        super().__init__("CropToContent")
        self.enabled = enabled
        self.padding = padding
        self.min_coverage = min_coverage

    def __call__(self, state: PipelineState) -> PipelineState:
        if not self.enabled or state.image is None:
            return state

        # Only crop if there are bboxes to calculate content area
        if not state.paragraph_bboxes:
            return state

        # Calculate content bounds from bboxes
        min_x = min(bbox["bbox"][0] for bbox in state.paragraph_bboxes)
        min_y = min(bbox["bbox"][1] for bbox in state.paragraph_bboxes)
        max_x = max(bbox["bbox"][2] for bbox in state.paragraph_bboxes)
        max_y = max(bbox["bbox"][3] for bbox in state.paragraph_bboxes)

        content_area = (max_x - min_x) * (max_y - min_y)
        image_area = state.image.width * state.image.height

        # Only crop if content is less than min_coverage of image
        if content_area / image_area >= self.min_coverage:
            return state

        # Apply padding
        crop_x1 = max(0, min_x - self.padding)
        crop_y1 = max(0, min_y - self.padding)
        crop_x2 = min(state.image.width, max_x + self.padding)
        crop_y2 = min(state.image.height, max_y + self.padding)

        state.image = state.image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Update bboxes to new coordinates
        for bbox in state.paragraph_bboxes:
            bbox["bbox"][0] -= crop_x1
            bbox["bbox"][1] -= crop_y1
            bbox["bbox"][2] -= crop_x1
            bbox["bbox"][3] -= crop_y1

        logger.debug("Cropped image to content bounds with %dpx padding", self.padding)

        return state
