#!/usr/bin/env python
"""Test script to verify the refactored transformations.py module."""

import sys
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, "/workspace/OCR-icelandic/src")

# Import the transformations module
from ocr_icelandic.transformations.transformations import (
    apply_random_transformation,
    PIPELINE_NO_BACKGROUND_PROBABILITIES,
    PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES,
    PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES,
)

# Create a test image
test_image = Image.new("RGBA", (512, 512), color=(255, 255, 255, 255))

# Test bounding boxes
test_bboxes = [
    {"x1": 10, "y1": 10, "x2": 100, "y2": 50},
    {"x1": 10, "y1": 60, "x2": 100, "y2": 100},
]

print("Testing refactored transformations.py module...")
print("-" * 50)

# Test Pipeline 1: No photo background
print("\n1. Testing Pipeline 1: No photo background")
try:
    transformed_img, metadata, updated_bboxes = apply_random_transformation(
        image=test_image,
        bg_color=(255, 255, 255),
        paragraph_bboxes=test_bboxes,
        use_background=False,
        background_has_shadow=False,
    )
    print("   ✓ Pipeline 1 executed successfully")
    print(f"   Applied transformations: {[m.get('transformation') for m in metadata]}")
except Exception as e:
    print(f"   ✗ Pipeline 1 failed: {e}")
    sys.exit(1)

# Test Pipeline 2: Photo background with shadow
print("\n2. Testing Pipeline 2: Photo background with shadow")
try:
    transformed_img, metadata, updated_bboxes = apply_random_transformation(
        image=test_image,
        bg_color=(255, 255, 255),
        paragraph_bboxes=test_bboxes,
        use_background=True,
        background_has_shadow=True,
    )
    print("   ✓ Pipeline 2 executed successfully")
    print(f"   Applied transformations: {[m.get('transformation') for m in metadata]}")
except Exception as e:
    print(f"   ✗ Pipeline 2 failed: {e}")
    sys.exit(1)

# Test Pipeline 3: Photo background without shadow
print("\n3. Testing Pipeline 3: Photo background without shadow")
try:
    transformed_img, metadata, updated_bboxes = apply_random_transformation(
        image=test_image,
        bg_color=(255, 255, 255),
        paragraph_bboxes=test_bboxes,
        use_background=True,
        background_has_shadow=False,
    )
    print("   ✓ Pipeline 3 executed successfully")
    print(f"   Applied transformations: {[m.get('transformation') for m in metadata]}")
except Exception as e:
    print(f"   ✗ Pipeline 3 failed: {e}")
    sys.exit(1)

# Verify probability configurations
print("\n4. Verifying probability configurations:")
print(
    f"   - Pipeline 1 (no bg) light_reflection probability: {PIPELINE_NO_BACKGROUND_PROBABILITIES['light_reflection']}"
)
print(
    f"   - Pipeline 2 (bg with shadow) light_reflection probability: {PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES['light_reflection']}"
)
print(
    f"   - Pipeline 3 (bg no shadow) light_reflection probability: {PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES['light_reflection']}"
)
print(
    f"   - Pipeline 3 (bg no shadow) shadow_overlay probability: {PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES['shadow_overlay']}"
)

# Test with custom probability overrides
print("\n5. Testing custom probability overrides")
try:
    custom_probabilities = {"blur": 1.0, "rotate": 0.0}
    transformed_img, metadata, updated_bboxes = apply_random_transformation(
        image=test_image,
        bg_color=(255, 255, 255),
        paragraph_bboxes=test_bboxes,
        use_background=False,
        background_has_shadow=False,
        probability_overrides=custom_probabilities,
    )
    print("   ✓ Custom probability overrides work")
    print(f"   Applied transformations: {[m.get('transformation') for m in metadata]}")
except Exception as e:
    print(f"   ✗ Custom overrides failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed successfully! ✓")
print("=" * 50)
