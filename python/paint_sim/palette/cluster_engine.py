"""Palette-First Clustering, Gemini Sample-Point Grounding & Precedence Resolver (Modules 3 & 5)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from paint_sim.physics.engine_bridge import NativePaintCoreBridge

PROTECTED_CATEGORIES: Set[str] = {
    "clear_part",
    "decal_sticker",
    "background",
    "stand_base",
}


@dataclass
class AtomicRegion:
    region_id: int
    part_id: str
    part_category: str
    color_group_id: str
    median_linear_rgb: List[float]
    is_protected: bool = False
    unlocked_by_user: bool = False


class PaletteAndPrecedenceEngine:
    """
    Implements:
    1. Default `COLOR_GROUP` (Palette-First) mode with seamless switching to `PART` and `REGION`.
    2. Strict precedence resolution: `Region Override > Part Assignment > Color-Group Assignment > Identity`.
    3. Protected region immunity (`clear_part`, `decal_sticker`, `background`, `stand_base`).
    4. Gemini Call 1 Sample-Point Validity Gate (rejecting clipped highlights & dark panel lines).
    5. Eyedropper & Delta E_00 tolerance grouping.
    """

    def __init__(self, bridge: Optional[NativePaintCoreBridge] = None) -> None:
        self.bridge = bridge or NativePaintCoreBridge()
        self.active_selection_mode: str = "COLOR_GROUP"  # Default Palette-First per PRD D5 / FR-4.1
        self.color_group_stacks: Dict[str, Dict[str, Any]] = {}
        self.part_stacks: Dict[str, Dict[str, Any]] = {}
        self.region_overrides: Dict[int, Dict[str, Any]] = {}

    def resolve_region_stack(self, region: AtomicRegion) -> Dict[str, Any]:
        """
        Resolve effective layer stack for an atomic region r = (part_id, color_group_id)
        according to SRS Section 3.3.3:
          1. Protected regions -> Identity (unmodified) unless explicitly unlocked by user.
          2. Region Override (if present)
          3. Part Assignment (if present)
          4. Color-Group Assignment (if present)
          5. Identity (unmodified)
        """
        if (region.is_protected or region.part_category in PROTECTED_CATEGORIES) and not region.unlocked_by_user:
            return {"is_modified": False, "is_protected": True, "source": "PROTECTED_LOCKED", "layers": []}

        if region.region_id in self.region_overrides:
            stack = copy.deepcopy(self.region_overrides[region.region_id])
            stack["is_modified"] = True
            stack["source"] = "REGION_OVERRIDE"
            return stack

        if region.part_id in self.part_stacks:
            stack = copy.deepcopy(self.part_stacks[region.part_id])
            stack["is_modified"] = True
            stack["source"] = "PART_ASSIGNMENT"
            return stack

        if region.color_group_id in self.color_group_stacks:
            stack = copy.deepcopy(self.color_group_stacks[region.color_group_id])
            stack["is_modified"] = True
            stack["source"] = "COLOR_GROUP_ASSIGNMENT"
            return stack

        return {"is_modified": False, "is_protected": False, "source": "IDENTITY", "layers": []}

    def fuse_with_gemini_samples(
        self,
        albedo_map_hwc: np.ndarray,
        clip_mask_hw: np.ndarray,
        panel_mask_hw: np.ndarray,
        sample_points_yx_norm: Sequence[Sequence[int]],
        panel_threshold: int = 45,
    ) -> Tuple[List[float], int]:
        """
        Anti-Hallucination Sample-Point Validity Gate (TC-VIS-002):
        Filters out any Gemini-proposed [y, x] sample points (0..1000 normalized)
        that land on clipped specular highlights (M_clip == 1) or dark panel lines
        (P_panel >= panel_threshold), and returns the robust median linear RGB albedo
        of the valid points.
        """
        h, w, _ = albedo_map_hwc.shape
        valid_rgbs: List[np.ndarray] = []
        fallback_rgbs: List[np.ndarray] = []

        for pt in sample_points_yx_norm:
            py = int(np.clip(round((pt[0] / 1000.0) * (h - 1)), 0, h - 1))
            px = int(np.clip(round((pt[1] / 1000.0) * (w - 1)), 0, w - 1))
            rgb = albedo_map_hwc[py, px]
            fallback_rgbs.append(rgb)

            is_clipped = bool(clip_mask_hw[py, px] > 0) or float(np.max(rgb)) >= 0.96
            is_panel = int(panel_mask_hw[py, px]) >= panel_threshold or float(np.max(rgb)) <= 0.04
            if not is_clipped and not is_panel:
                valid_rgbs.append(rgb)

        chosen = valid_rgbs if valid_rgbs else fallback_rgbs
        arr = np.asarray(chosen, dtype=np.float64)
        lab = self.bridge.linear_srgb_to_oklab(arr)
        med_lab = np.median(lab, axis=0)
        med_rgb = self.bridge.oklab_to_linear_srgb(med_lab)
        return [float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2])], len(valid_rgbs)

    def select_regions_within_delta_e00(
        self,
        seed_region: AtomicRegion,
        all_regions: Sequence[AtomicRegion],
        tolerance_delta_e00: float = 4.0,
    ) -> List[int]:
        """
        Eyedropper & Delta E_00 Tolerance Selector (TC-PAL-004):
        Uses the seed region's regularized median albedo (ignoring pixel-level highlight/shadow noise)
        and returns all non-protected region IDs within `tolerance_delta_e00`.
        """
        matched_ids: List[int] = []
        for reg in all_regions:
            if (reg.is_protected or reg.part_category in PROTECTED_CATEGORIES) and not reg.unlocked_by_user:
                continue
            de = self.bridge.delta_e00_linear_rgb(seed_region.median_linear_rgb, reg.median_linear_rgb)
            if de <= tolerance_delta_e00:
                matched_ids.append(reg.region_id)
        return matched_ids
