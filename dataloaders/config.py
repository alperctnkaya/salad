"""
Centralised dataset paths for the hotel benchmark.

Every module that needs a dataset path should import from here
so there is a single place to update when paths change.
"""
import os

# ── Repository root (dynamically finding the parent of the salad submodule) ──
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Dataset paths ────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(_REPO_ROOT, "data/full")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "images")

GALLERY_METADATA = os.path.join(DATASET_ROOT, "metadata_gallery.json")
TEST_OBJECT_METADATA = os.path.join(DATASET_ROOT, "metadata_test_object.json")
TEST_NON_OBJECT_METADATA = os.path.join(DATASET_ROOT, "metadata_test_non_object.json")
