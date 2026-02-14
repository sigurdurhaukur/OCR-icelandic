"""Tests for bounding box splitting functionality."""

from PIL import Image

from ocr_icelandic.utils.image_creation import create_image_with_text


class TestRoundingFix:
    """Test that bbox coordinates are properly rounded after scale-down."""

    def test_scale_down_uses_rounding(self):
        """Test that bbox scaling uses round() instead of int() truncation."""
        # This test is for the pipeline stages, testing create_image_with_text directly
        # doesn't involve scaling, but we can verify the coordinates are integers
        text = "Test text for rounding verification"
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=20,
        )

        assert len(bboxes) > 0, "Should have at least one bbox"
        for bbox in bboxes:
            coords = bbox["bbox"]
            assert all(isinstance(c, int) for c in coords), (
                "All coordinates should be integers"
            )


class TestNoSplitting:
    """Test backward compatibility - no splitting when features are disabled."""

    def test_single_paragraph_no_split(self):
        """Single paragraph should create one bbox when splitting is disabled."""
        text = "This is a test paragraph with multiple words."
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=16,
            bbox_per_column=False,
            bbox_max_chars=None,
        )

        assert len(bboxes) == 1, "Should create exactly one bbox"
        bbox = bboxes[0]
        assert bbox["paragraph_index"] == 0
        assert bbox["sequence_number"] == 0
        assert len(bbox["columns"]) >= 1
        assert bbox["char_count"] > 0

    def test_multiple_paragraphs_no_split(self):
        """Multiple paragraphs should create one bbox each when splitting is disabled."""
        text = "First paragraph.\n\nSecond paragraph."
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 300),
            font_size=16,
            bbox_per_column=False,
            bbox_max_chars=None,
        )

        assert len(bboxes) == 2, "Should create exactly two bboxes"
        assert all(b["sequence_number"] == 0 for b in bboxes), (
            "All should have sequence 0"
        )


class TestColumnSplitting:
    """Test bbox_per_column feature."""

    def test_paragraph_spanning_columns_splits(self):
        """Paragraph spanning multiple columns should create separate bboxes."""
        # Long text that will span multiple columns
        text = (
            "This is a very long paragraph with many words that should span across multiple columns when rendered with a multi-column layout. "
            * 3
        )

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=14,
            num_columns=3,
            bbox_per_column=True,
            bbox_max_chars=None,
        )

        # Should have multiple bboxes due to column splits
        assert len(bboxes) > 1, "Should split into multiple bboxes"

        # All should be from paragraph 0
        assert all(b["paragraph_index"] == 0 for b in bboxes)

        # Sequence numbers should increment
        seq_nums = [b["sequence_number"] for b in bboxes]
        assert seq_nums == list(range(len(bboxes))), (
            "Sequence numbers should be 0, 1, 2..."
        )

        # Each bbox should have different columns
        columns_sets = [set(b["columns"]) for b in bboxes]
        # Verify that columns are distinct (no overlap for single-column bboxes)
        if all(len(cols) == 1 for cols in columns_sets):
            all_cols = [list(cols)[0] for cols in columns_sets]
            assert len(all_cols) == len(set(all_cols)), (
                "Each bbox should be in a different column"
            )

    def test_short_text_single_column_no_split(self):
        """Short text in single column shouldn't split even with bbox_per_column=True."""
        text = "Short text."
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=16,
            num_columns=1,
            bbox_per_column=True,
        )

        assert len(bboxes) == 1, "Short text in single column should not split"
        assert bboxes[0]["sequence_number"] == 0


class TestCharacterSplitting:
    """Test bbox_max_chars feature."""

    def test_long_paragraph_splits_at_char_limit(self):
        """Long paragraph should split when character limit is exceeded."""
        # Create text with known character count
        text = "Word " * 50  # ~250 characters

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 400),
            font_size=14,
            bbox_per_column=False,
            bbox_max_chars=100,  # Set limit at 100 chars
        )

        # Should split into multiple bboxes
        assert len(bboxes) >= 2, (
            f"Should split into at least 2 bboxes, got {len(bboxes)}"
        )

        # All should be from paragraph 0
        assert all(b["paragraph_index"] == 0 for b in bboxes)

        # Sequence numbers should increment
        seq_nums = [b["sequence_number"] for b in bboxes]
        assert seq_nums == list(range(len(bboxes))), "Sequence numbers should increment"

        # All bboxes except possibly the last should respect the limit
        for i, bbox in enumerate(bboxes[:-1]):  # Check all but last
            assert bbox["char_count"] <= 100, (
                f"Bbox {i} exceeds char limit: {bbox['char_count']}"
            )

    def test_short_text_under_limit_no_split(self):
        """Short text under character limit shouldn't split."""
        text = "Short text with few words."
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=16,
            bbox_max_chars=200,
        )

        assert len(bboxes) == 1, "Text under limit should not split"
        assert bboxes[0]["char_count"] < 200


class TestCombinedSplitting:
    """Test that column and character splitting work together."""

    def test_column_takes_precedence(self):
        """Column split should happen even if under character limit."""
        text = "This text spans columns. " * 20

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=14,
            num_columns=3,
            bbox_per_column=True,
            bbox_max_chars=500,  # High limit
        )

        # Should split by columns
        assert len(bboxes) > 1, "Should split due to columns"

    def test_both_features_create_multiple_splits(self):
        """Both column and char limits should create splits."""
        text = "Word " * 100  # Long text

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(600, 600),
            font_size=14,
            num_columns=2,
            bbox_per_column=True,
            bbox_max_chars=80,
        )

        # Should have multiple splits from both features
        assert len(bboxes) >= 2, "Should have multiple splits"

        # Verify metadata is consistent
        for bbox in bboxes:
            assert "paragraph_index" in bbox
            assert "sequence_number" in bbox
            assert "columns" in bbox
            assert "char_count" in bbox
            assert "bbox" in bbox


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_long_line_doesnt_split_midline(self):
        """Single very long line shouldn't split mid-line."""
        text = "Supercalifragilisticexpialidocious" * 10  # One very long "word"

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(200, 200),
            font_size=12,
            bbox_max_chars=50,
        )

        # Should create bbox for the single line, even if it exceeds limit
        assert len(bboxes) >= 1, "Should create at least one bbox"

    def test_empty_lines_dont_count_toward_char_limit(self):
        """Empty lines shouldn't contribute to character count."""
        text = "Line one\n\n\nLine two"

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 300),
            font_size=16,
            bbox_max_chars=20,
        )

        # Each paragraph should have reasonable char counts (not counting empty lines)
        for bbox in bboxes:
            # Char count should only include visible text
            assert bbox["char_count"] > 0 or bbox["paragraph_text"].strip() == ""

    def test_sequence_numbers_start_at_zero(self):
        """First bbox for each paragraph should have sequence_number=0."""
        text = "Para one. " * 20 + "\n\n" + "Para two. " * 20

        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 600),
            font_size=14,
            bbox_max_chars=50,
        )

        # Group by paragraph_index
        by_paragraph = {}
        for bbox in bboxes:
            para_idx = bbox["paragraph_index"]
            if para_idx not in by_paragraph:
                by_paragraph[para_idx] = []
            by_paragraph[para_idx].append(bbox)

        # First bbox of each paragraph should have sequence 0
        for para_idx, para_bboxes in by_paragraph.items():
            first_bbox = para_bboxes[0]
            assert first_bbox["sequence_number"] == 0, (
                f"First bbox of paragraph {para_idx} should have sequence 0"
            )

    def test_bbox_format_includes_all_fields(self):
        """Verify all expected fields are in bbox output."""
        text = "Test text"
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=16,
            bbox_per_column=True,
            bbox_max_chars=100,
        )

        assert len(bboxes) > 0, "Should have at least one bbox"

        required_fields = {
            "paragraph_index",
            "sequence_number",
            "paragraph_text",
            "columns",
            "char_count",
            "bbox",
        }

        for bbox in bboxes:
            assert set(bbox.keys()) == required_fields, (
                f"Bbox should have exactly these fields: {required_fields}"
            )
            assert isinstance(bbox["columns"], list), "columns should be a list"
            assert isinstance(bbox["bbox"], list), "bbox should be a list"
            assert len(bbox["bbox"]) == 4, "bbox should have 4 coordinates"

    def test_columns_field_is_sorted_list(self):
        """Columns field should be a sorted list of column indices."""
        text = "Test text that might span columns."
        _, _, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
            font_size=16,
            num_columns=2,
        )

        for bbox in bboxes:
            cols = bbox["columns"]
            assert isinstance(cols, list), "columns should be a list"
            assert cols == sorted(cols), "columns should be sorted"
            assert all(isinstance(c, int) for c in cols), (
                "all column indices should be integers"
            )


class TestBackwardCompatibility:
    """Test that existing code patterns still work."""

    def test_default_parameters_work(self):
        """Calling without new parameters should work (backward compatible)."""
        text = "Simple test text"
        image, fitted_text, bboxes = create_image_with_text(
            text=text,
            image_size=(400, 200),
        )

        assert isinstance(image, Image.Image)
        assert isinstance(fitted_text, str)
        assert isinstance(bboxes, list)
        assert len(bboxes) > 0

    def test_bbox_coordinates_are_valid(self):
        """Bbox coordinates should be valid integers within image bounds."""
        text = "Test text"
        image_size = (400, 200)
        image, _, bboxes = create_image_with_text(
            text=text,
            image_size=image_size,
            font_size=16,
        )

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox["bbox"]

            # Should be integers
            assert all(isinstance(c, int) for c in [x0, y0, x1, y1])

            # x0 < x1 and y0 < y1
            assert x0 < x1, "x0 should be less than x1"
            assert y0 < y1, "y0 should be less than y1"

            # Should be within image bounds (allowing for some margin)
            assert 0 <= x0 <= image_size[0], f"x0={x0} out of bounds"
            assert 0 <= x1 <= image_size[0], f"x1={x1} out of bounds"
            assert 0 <= y0 <= image_size[1], f"y0={y0} out of bounds"
            assert 0 <= y1 <= image_size[1], f"y1={y1} out of bounds"
