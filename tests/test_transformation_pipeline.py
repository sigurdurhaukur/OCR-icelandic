#!/usr/bin/env python3
"""Test suite for transformation pipeline configurations."""

import pytest
from PIL import Image

from ocr_icelandic.transformations.transformations import (
    apply_random_transformation,
    PIPELINE_NO_BACKGROUND_PROBABILITIES,
    PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES,
    PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES,
)


@pytest.fixture
def test_image():
    """Create a test RGBA image."""
    return Image.new("RGBA", (512, 512), color=(255, 255, 255, 255))


@pytest.fixture
def test_bboxes():
    """Create test bounding boxes."""
    return [
        {"bbox": [10, 10, 100, 50]},
        {"bbox": [10, 60, 100, 100]},
    ]


class TestPipelineConfigurations:
    """Test suite for different pipeline configurations."""

    def test_pipeline_no_background(self, test_image, test_bboxes):
        """Test Pipeline 1: No photo background."""
        transformed_img, metadata, updated_bboxes, transformed_background = (
            apply_random_transformation(
                image=test_image,
                bg_color=(255, 255, 255),
                paragraph_bboxes=test_bboxes,
                use_background=False,
                background_has_shadow=False,
            )
        )

        assert isinstance(transformed_img, Image.Image)
        assert isinstance(metadata, list)
        assert len(metadata) > 0
        assert isinstance(updated_bboxes, list)
        assert len(updated_bboxes) == len(test_bboxes)

        # Verify each metadata entry has transformation name
        for meta in metadata:
            assert "transformation" in meta

    def test_pipeline_background_with_shadow(self, test_image, test_bboxes):
        """Test Pipeline 2: Photo background with shadow."""
        transformed_img, metadata, updated_bboxes, transformed_background = (
            apply_random_transformation(
                image=test_image,
                bg_color=(255, 255, 255),
                paragraph_bboxes=test_bboxes,
                use_background=True,
                background_has_shadow=True,
            )
        )

        assert isinstance(transformed_img, Image.Image)
        assert isinstance(metadata, list)
        assert len(metadata) > 0
        assert isinstance(updated_bboxes, list)

        for meta in metadata:
            assert "transformation" in meta

    def test_pipeline_background_no_shadow(self, test_image, test_bboxes):
        """Test Pipeline 3: Photo background without shadow."""
        transformed_img, metadata, updated_bboxes, transformed_background = (
            apply_random_transformation(
                image=test_image,
                bg_color=(255, 255, 255),
                paragraph_bboxes=test_bboxes,
                use_background=True,
                background_has_shadow=False,
            )
        )

        assert isinstance(transformed_img, Image.Image)
        assert isinstance(metadata, list)
        assert len(metadata) > 0
        assert isinstance(updated_bboxes, list)

        for meta in metadata:
            assert "transformation" in meta

    def test_probability_overrides(self, test_image, test_bboxes):
        """Test custom probability overrides functionality."""
        custom_probabilities = {"blur": 1.0, "rotate": 0.0}

        transformed_img, metadata, updated_bboxes, transformed_background = (
            apply_random_transformation(
                image=test_image,
                bg_color=(255, 255, 255),
                paragraph_bboxes=test_bboxes,
                use_background=False,
                background_has_shadow=False,
                probability_overrides=custom_probabilities,
            )
        )

        assert isinstance(transformed_img, Image.Image)
        assert isinstance(metadata, list)

        # Verify transformation names are recorded
        for meta in metadata:
            assert "transformation" in meta


class TestProbabilityConfigurations:
    """Test suite for probability configuration values."""

    def test_pipeline_no_background_probabilities(self):
        """Verify Pipeline 1 probability configuration exists and has expected keys."""
        assert isinstance(PIPELINE_NO_BACKGROUND_PROBABILITIES, dict)
        assert "light_reflection" in PIPELINE_NO_BACKGROUND_PROBABILITIES

    def test_pipeline_background_with_shadow_probabilities(self):
        """Verify Pipeline 2 probability configuration exists and has expected keys."""
        assert isinstance(PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES, dict)
        assert "light_reflection" in PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES

    def test_pipeline_background_no_shadow_probabilities(self):
        """Verify Pipeline 3 probability configuration exists and has expected keys."""
        assert isinstance(PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES, dict)
        assert "light_reflection" in PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES
        assert "shadow_overlay" in PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES

    def test_probability_values_are_valid(self):
        """Verify all probability values are between 0 and 1."""
        all_configs = [
            PIPELINE_NO_BACKGROUND_PROBABILITIES,
            PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES,
            PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES,
        ]

        for config in all_configs:
            for key, value in config.items():
                assert 0.0 <= value <= 1.0, (
                    f"Probability {key}={value} is out of range [0, 1]"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
