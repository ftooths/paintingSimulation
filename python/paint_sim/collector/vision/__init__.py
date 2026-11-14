"""Manual click-seeded and Gemini Vision auto-detection segmentation & lossless mask refinement."""
from __future__ import annotations

from paint_sim.collector.vision.gemini_detect import (
    DetectedSampleItem,
    SampleSheetDetectionResponse,
    build_genai_client,
    detect_and_refine_samples_auto,
    get_public_secrets_status,
    get_secrets_path,
    list_available_vision_models,
    load_secrets,
    norm1000_box_to_px,
    norm1000_point_to_px,
    redact_secrets,
    save_secrets,
    test_genai_connection,
)
from paint_sim.collector.vision.manual_segment import (
    SAM2PointBoxSegmenter,
    segment_from_clicks_oklab_grabcut,
    segment_multiple_samples,
    segment_single_sample,
)
from paint_sim.collector.vision.refine import (
    compute_mask_iou,
    cross_check_mask_against_box,
    find_overlapping_mask_pairs,
    refine_mask_and_export_lossless_rgba,
    validate_layout_hint,
    validate_mask_geometry,
)

__all__ = [
    "DetectedSampleItem",
    "SampleSheetDetectionResponse",
    "build_genai_client",
    "detect_and_refine_samples_auto",
    "get_public_secrets_status",
    "get_secrets_path",
    "list_available_vision_models",
    "load_secrets",
    "norm1000_box_to_px",
    "norm1000_point_to_px",
    "redact_secrets",
    "save_secrets",
    "test_genai_connection",
    "SAM2PointBoxSegmenter",
    "segment_from_clicks_oklab_grabcut",
    "segment_multiple_samples",
    "segment_single_sample",
    "compute_mask_iou",
    "cross_check_mask_against_box",
    "find_overlapping_mask_pairs",
    "refine_mask_and_export_lossless_rgba",
    "validate_layout_hint",
    "validate_mask_geometry",
]
