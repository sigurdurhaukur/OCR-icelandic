"""Core pipeline infrastructure for OCR image generation."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineState:
    """Mutable state passed through all pipeline stages."""

    # Core inputs
    text: str = ""
    image_size: tuple[int, int] = (512, 512)
    dpi: int = 72
    render_scale: int = 1  # Render at Nx resolution, scale down at end

    # Core outputs
    image: Image.Image | None = None
    fitted_text: str = ""
    paragraph_bboxes: list[dict[str, Any]] = field(default_factory=list)

    # Selected resources
    font_path: str | None = None
    font_size: int = 12
    font_color: tuple[int, int, int] | str = "black"
    bg_color: tuple[int, int, int] | str = "white"
    paper_texture_path: str | None = None
    background_image: Image.Image | None = None
    background_receives_shadow: bool = False

    # Layout settings
    num_columns: int = 1
    column_gap: int = 20
    column_width: int | None = None
    alignment: str = "left"
    vertical_alignment: str = "center"
    hyphenation_lang: str = "is"  # ISO 639-1 language code for word hyphenation

    # Bounding box settings
    bbox_per_column: bool = False
    bbox_max_chars: int | None = None

    # Font variation settings
    paragraph_font_configs: list | None = None

    # Accumulated metadata
    transformation_metadata: list[dict[str, Any]] = field(default_factory=list)
    stage_metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PipelineState":
        """Create a shallow copy of the state."""
        import copy as copy_module

        return copy_module.copy(self)


@runtime_checkable
class Stage(Protocol):
    """Protocol defining the interface for pipeline stages."""

    @property
    def name(self) -> str:
        """Human-readable name for logging and metadata."""
        ...

    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute the stage, modifying and returning the state."""
        ...


class BaseStage(ABC):
    """
    Abstract base class for pipeline stages.

    Provides common functionality while enforcing the Stage protocol.
    """

    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute the stage logic."""
        pass

    def _add_metadata(self, state: PipelineState, key: str, value: Any) -> None:
        """Helper to add stage-specific metadata."""
        state.stage_metadata[f"{self.name}.{key}"] = value


class LambdaStage:
    """
    Stage wrapper for simple functions.

    Allows using plain functions as stages:
        LambdaStage("my_stage", lambda state: ...)
    """

    def __init__(self, name: str, func: Callable[[PipelineState], PipelineState]):
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, state: PipelineState) -> PipelineState:
        return self._func(state)


class Pipeline:
    """
    Configurable image generation pipeline.

    Executes a sequence of stages, each operating on shared PipelineState.
    Stages can read and modify any part of the state, enabling flexible
    composition of operations.

    Example:
        pipeline = Pipeline(
            stages=[
                SelectFontStage(fonts=[...], random_selection=True),
                SelectColorsStage(random_background=True),
                RenderTextStage(),
                ApplyTransformationsStage(),
                FinalizeImageStage(),
            ],
            initial_state=PipelineState(
                text="Hello world",
                image_size=(512, 512),
            ),
        )
        result = pipeline.run()
    """

    def __init__(
        self,
        stages: list[Stage],
        initial_state: PipelineState | None = None,
    ):
        self.stages = stages
        self.initial_state = initial_state or PipelineState()

    def run(self, text: str | None = None) -> PipelineState:
        """
        Execute all stages in order.

        Args:
            text: Optional text to render (overrides initial_state.text)

        Returns:
            Final PipelineState with generated image and metadata
        """
        state = self.initial_state.copy()
        if text is not None:
            state.text = text

        logger.debug("Starting pipeline with %d stages", len(self.stages))

        for i, stage in enumerate(self.stages):
            logger.debug(
                "Executing stage %d/%d: %s", i + 1, len(self.stages), stage.name
            )
            state = stage(state)

        logger.debug("Pipeline completed successfully")
        return state

    def with_stages(self, stages: list[Stage]) -> "Pipeline":
        """Return a new Pipeline with different stages but same initial state."""
        return Pipeline(stages=stages, initial_state=self.initial_state)

    def with_initial_state(self, state: PipelineState) -> "Pipeline":
        """Return a new Pipeline with different initial state but same stages."""
        return Pipeline(stages=self.stages, initial_state=state)
