"""Shared mask refinement, validation, cross-check, and lossless 32-bit RGBA PNG export (Sections 3.2, 3.3, 4.3)."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def refine_mask_and_export_lossless_rgba(
    rgb_u8: np.ndarray,
    raw_mask_bool: np.ndarray,
    *,
    seed_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
    raw_alpha_u8: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Shared post-segmentation mask refinement and lossless RGBA crop exporter (Section 3.2):
      1. Keeps connected component(s) containing `seed_points_xy` (or largest component if no seeds).
      2. Fills interior specular highlight holes (`binary_fill_holes`), applies 5x5 morphological
         closing (`binary_closing`), and re-fills holes.
      3. Erodes by 2px (`structure=np.ones((5, 5))`) to define `core_opaque`.
      4. Preserves bit-exact original `[R, G, B]` inside `core_opaque` (`max_rgb_diff == 0`),
         while decontaminating the 2px boundary rim via `distance_transform_edt` nearest-interior RGB
         (same lossless rim logic as `isolate_spoon_rgba_lossless`).
      5. Crops tightly on integer pixel boundaries with padding (no resampling) and exports
         32-bit lossless RGBA PNG bytes plus full-image binary mask PNG bytes.
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 RGB array, got shape {rgb_u8.shape}")
    orig_rgb = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    h, w, _ = orig_rgb.shape
    fg = np.asarray(raw_mask_bool, dtype=bool)
    if fg.shape != (h, w):
        raise ValueError(f"Mask shape {fg.shape} does not match image shape {(h, w)}")

    if not np.any(fg):
        raise ValueError("Segmentation produced an empty foreground mask.")

    # Slice to the non-zero bounding box of `fg` first so `ndimage.label` and all morphological
    # operations run on the local sample crop rather than the full multi-sample sheet.
    f_ys, f_xs = np.where(fg)
    pad_x = max(6, w // 50)
    pad_y = max(6, h // 50)
    morph_pad = max(pad_x, pad_y) + 8
    by0 = max(0, int(np.min(f_ys)) - morph_pad)
    by1 = min(h, int(np.max(f_ys)) + morph_pad + 1)
    bx0 = max(0, int(np.min(f_xs)) - morph_pad)
    bx1 = min(w, int(np.max(f_xs)) + morph_pad + 1)

    sub_fg = fg[by0:by1, bx0:bx1]
    sub_h, sub_w = sub_fg.shape
    labeled, num_features = ndimage.label(sub_fg)
    if num_features > 1:
        sub_keep = np.zeros((sub_h, sub_w), dtype=bool)
        if seed_points_xy:
            for sx, sy in seed_points_xy:
                cx = max(0, min(sub_w - 1, int(sx) - bx0))
                cy = max(0, min(sub_h - 1, int(sy) - by0))
                cid = int(labeled[cy, cx])
                if cid > 0:
                    sub_keep |= labeled == cid
        if not np.any(sub_keep):
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            sub_keep = labeled == int(np.argmax(counts))
    else:
        sub_keep = sub_fg

    sub_rgb = orig_rgb[by0:by1, bx0:bx1]

    sub_solid = ndimage.binary_fill_holes(sub_keep)
    sub_solid = ndimage.binary_closing(sub_solid, structure=np.ones((5, 5), dtype=bool))
    sub_solid = ndimage.binary_fill_holes(sub_solid)

    # 2px erosion defines 100% uncontaminated interior (`core_opaque`)
    sub_core = ndimage.binary_erosion(sub_solid, structure=np.ones((5, 5), dtype=bool))
    if not np.any(sub_core):
        sub_core = sub_solid

    # Rim decontamination via Euclidean Distance Transform (Ground Rule 3: max_rgb_diff == 0 inside core_opaque)
    _, nearest_idx = ndimage.distance_transform_edt(~sub_core, return_indices=True)
    sub_rim_rgb = sub_rgb[nearest_idx[0], nearest_idx[1]]
    sub_final_rgb = np.where(sub_core[:, :, None], sub_rgb, sub_rim_rgb)

    # Verify lossless interior invariant
    max_rgb_diff = int(
        np.max(np.abs(sub_final_rgb[sub_core].astype(np.int32) - sub_rgb[sub_core].astype(np.int32)))
    )
    if max_rgb_diff != 0:
        raise RuntimeError(f"Lossless interior invariant violated: max_rgb_diff={max_rgb_diff}")

    if raw_alpha_u8 is not None and raw_alpha_u8.shape == (h, w):
        sub_rim_alpha = np.clip(raw_alpha_u8[by0:by1, bx0:bx1], 128, 255).astype(np.uint8)
    else:
        sub_rim_alpha = np.full(sub_solid.shape, 220, dtype=np.uint8)

    sub_clean_alpha = np.where(sub_core, 255, np.where(sub_solid, sub_rim_alpha, 0)).astype(np.uint8)
    sub_rgba = np.dstack([sub_final_rgb, sub_clean_alpha])

    solid_mask = np.zeros((h, w), dtype=bool)
    solid_mask[by0:by1, bx0:bx1] = sub_solid
    core_opaque = np.zeros((h, w), dtype=bool)
    core_opaque[by0:by1, bx0:bx1] = sub_core

    ys_local, xs_local = np.where(sub_clean_alpha > 16)
    ymin = max(0, by0 + int(np.min(ys_local)) - pad_y)
    ymax = min(h, by0 + int(np.max(ys_local)) + pad_y + 1)
    xmin = max(0, bx0 + int(np.min(xs_local)) - pad_x)
    xmax = min(w, bx0 + int(np.max(xs_local)) + pad_x + 1)

    rgba_crop = sub_rgba[ymin - by0 : ymax - by0, xmin - bx0 : xmax - bx0]

    crop_buf = io.BytesIO()
    Image.fromarray(rgba_crop, mode="RGBA").save(crop_buf, format="PNG", compress_level=1)

    mask_u8 = (solid_mask.astype(np.uint8)) * 255
    mask_buf = io.BytesIO()
    Image.fromarray(mask_u8, mode="L").save(mask_buf, format="PNG", compress_level=1)

    return {
        "refined_mask_bool": solid_mask,
        "core_opaque_bool": core_opaque,
        "rgba_crop_u8": rgba_crop,
        "crop_png_bytes": crop_buf.getvalue(),
        "mask_png_bytes": mask_buf.getvalue(),
        "crop_bbox_xyxy": (xmin, ymin, xmax, ymax),
        "max_rgb_diff": max_rgb_diff,
    }


def validate_mask_geometry(
    mask_bool: np.ndarray,
    *,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.60,
    max_borders: int = 2,
) -> Dict[str, Any]:
    """
    Validate a sample mask against Section 3.3 geometry rules:
      - Reject masks covering < 0.2% or > 60% of the image.
      - Reject masks touching more than 2 image borders.
    Returns `{"valid": bool, "reasons": [...], "area_ratio": float, "borders_touched": int}`.
    """
    fg = np.asarray(mask_bool, dtype=bool)
    h, w = fg.shape[:2]
    total_px = max(1, h * w)
    fg_px = int(np.count_nonzero(fg))
    area_ratio = fg_px / float(total_px)

    borders_touched = int(
        bool(np.any(fg[0, :]))
        + bool(np.any(fg[-1, :]))
        + bool(np.any(fg[:, 0]))
        + bool(np.any(fg[:, -1]))
    )

    reasons: List[str] = []
    if area_ratio < min_area_ratio:
        reasons.append(f"Mask area {area_ratio * 100:.2f}% is below minimum {min_area_ratio * 100:.2f}%.")
    if area_ratio > max_area_ratio:
        reasons.append(f"Mask area {area_ratio * 100:.1f}% exceeds maximum {max_area_ratio * 100:.1f}%.")
    if borders_touched > max_borders:
        reasons.append(f"Mask touches {borders_touched} image borders (max allowed: {max_borders}).")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "area_ratio": round(area_ratio, 6),
        "borders_touched": borders_touched,
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two boolean masks."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter / max(1, union))


def find_overlapping_mask_pairs(
    masks: Sequence[np.ndarray],
    iou_threshold: float = 0.30,
    bboxes_xyxy: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of mask index pairs whose IoU exceeds `iou_threshold` (Section 3.3)."""
    warnings_list: List[Dict[str, Any]] = []
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_xyxy is not None and i < len(bboxes_xyxy) and j < len(bboxes_xyxy):
                ax0, ay0, ax1, ay1 = bboxes_xyxy[i]
                bx0, by0, bx1, by1 = bboxes_xyxy[j]
                if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
                    continue
                ux0, uy0 = min(ax0, bx0), min(ay0, by0)
                ux1, uy1 = max(ax1, bx1), max(ay1, by1)
                iou = compute_mask_iou(masks[i][uy0:uy1, ux0:ux1], masks[j][uy0:uy1, ux0:ux1])
            else:
                iou = compute_mask_iou(masks[i], masks[j])
            if iou > iou_threshold:
                warnings_list.append(
                    {
                        "index_a": i + 1,
                        "index_b": j + 1,
                        "iou": round(iou, 4),
                        "message": f"Sample #{i + 1} and Sample #{j + 1} overlap with IoU {iou:.2f} (> {iou_threshold:.2f}).",
                    }
                )
    return warnings_list


def cross_check_mask_against_box(
    mask_bool: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    *,
    min_box_coverage: float = 0.40,
    max_outside_spill: float = 0.15,
) -> Dict[str, Any]:
    """
    Cross-check refined local mask against LLM-proposed bounding box (Section 4.3):
      - If refined mask covers < 40% of its box area, or
      - If > 15% of the mask pixels spill outside the box,
      mark `needs_review=True` with detailed reasons.
    """
    fg = np.asarray(mask_bool, dtype=bool)
    h, w = fg.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    x0_c = max(0, min(w - 1, int(x0)))
    y0_c = max(0, min(h - 1, int(y0)))
    x1_c = max(x0_c + 1, min(w, int(x1)))
    y1_c = max(y0_c + 1, min(h, int(y1)))

    box_area = max(1, (x1_c - x0_c) * (y1_c - y0_c))
    inside_mask = np.zeros((h, w), dtype=bool)
    inside_mask[y0_c:y1_c, x0_c:x1_c] = True

    fg_total = int(np.count_nonzero(fg))
    fg_inside = int(np.count_nonzero(fg & inside_mask))
    fg_outside = fg_total - fg_inside

    box_coverage = fg_inside / float(box_area)
    spill_ratio = (fg_outside / float(fg_total)) if fg_total > 0 else 1.0

    reasons: List[str] = []
    if box_coverage < min_box_coverage:
        reasons.append(
            f"Refined mask covers only {box_coverage * 100:.1f}% of LLM box (< {min_box_coverage * 100:.0f}%)."
        )
    if spill_ratio > max_outside_spill:
        reasons.append(
            f"Refined mask spills {spill_ratio * 100:.1f}% outside LLM box (> {max_outside_spill * 100:.0f}%)."
        )

    return {
        "needs_review": len(reasons) > 0,
        "reasons": reasons,
        "box_coverage": round(box_coverage, 4),
        "spill_ratio": round(spill_ratio, 4),
    }


def validate_layout_hint(
    sample_boxes_xyxy: Sequence[Tuple[int, int, int, int]],
    layout_hint: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate detected sample count and left-to-right ordering against an optional site adapter
    `layout_hint` without blind cropping (Section 7.2).
    """
    if not layout_hint:
        return {"valid": True, "needs_review": False, "reasons": []}

    reasons: List[str] = []
    expected_count = layout_hint.get("expected_count")
    if expected_count is not None and len(sample_boxes_xyxy) != int(expected_count):
        reasons.append(
            f"Layout hint expected {expected_count} samples, but vision found {len(sample_boxes_xyxy)}."
        )

    grid = str(layout_hint.get("grid", "")).lower()
    if grid.startswith("1x") and len(sample_boxes_xyxy) >= 2:
        centers_x = [0.5 * (b[0] + b[2]) for b in sample_boxes_xyxy]
        if any(centers_x[i] >= centers_x[i + 1] for i in range(len(centers_x) - 1)):
            reasons.append("Samples are not strictly ordered left-to-right as expected by 1xN layout hint.")

    return {
        "valid": len(reasons) == 0,
        "needs_review": len(reasons) > 0,
        "reasons": reasons,
    }
