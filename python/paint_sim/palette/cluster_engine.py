"""Palette-First OKLab k-Means++ Clustering, Gemini Sample-Point Grounding & Precedence Resolver (Modules 3 & 5)."""
from __future__ import annotations

import copy
import math
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
    6. Data-driven Adaptive OKLab k-Means++ clustering for arbitrary plastic model kits (Gundam, Zaku, military, automotive, etc.).
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

    @staticmethod
    def describe_oklab_color(lab: Sequence[float]) -> Tuple[str, str, bool]:
        """
        Generate a kit-agnostic descriptive name and slug from OKLab (L, a, b).
        Works for any plastic model color (Green Zaku, Purple Eva, Tan Military, Silver Car, Gundam Tricolor, etc.).
        """
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        C = math.hypot(a, b)
        h_deg = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0

        if C < 0.032:
            if L >= 0.72:
                return "cg_white", "Molded White / Light Gray", True
            if L >= 0.45:
                return "cg_neutral_gray", "Molded Neutral Gray", True
            return "cg_dark_frame", "Molded Dark Gray / Inner Frame", True

        if h_deg < 42.0 or h_deg >= 342.0:
            return "cg_red", "Molded Red / Crimson", False
        if h_deg < 72.0:
            return ("cg_orange", "Molded Orange / Amber", False) if C >= 0.08 else ("cg_tan", "Molded Tan / Desert Brown", False)
        if h_deg < 112.0:
            return ("cg_yellow", "Molded Yellow / Gold", False) if L >= 0.62 else ("cg_olive", "Molded Olive / Khaki", False)
        if h_deg < 175.0:
            return "cg_green", "Molded Green / Zaku Drab", False
        if h_deg < 220.0:
            return "cg_cyan", "Molded Cyan / Teal", False
        if h_deg < 280.0:
            return "cg_blue", "Molded Blue / Cobalt", False
        if h_deg < 320.0:
            return "cg_purple", "Molded Purple / Violet", False
        return "cg_magenta", "Molded Pink / Magenta", False

    def extract_oklab_kmeans_palette(
        self,
        linear_rgb_hwc: np.ndarray,
        k_init: int = 6,
        merge_delta_e00: float = 3.5,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic OKLab k-Means++ clustering with automatic Delta E_00 cluster merging.
        Replaces hardcoded 6-cluster Gundam heuristics so any kit's molded colors are discovered.
        """
        h, w, _ = linear_rgb_hwc.shape
        flat_rgb = linear_rgb_hwc.reshape(-1, 3)
        n_pix = flat_rgb.shape[0]

        # Separate dark background pixels (L < 0.14 and low chroma)
        lab_all = self.bridge.linear_srgb_to_oklab(flat_rgb)
        chroma_all = np.hypot(lab_all[:, 1], lab_all[:, 2])
        fg_mask = ~((lab_all[:, 0] < 0.14) & (chroma_all < 0.025))
        fg_indices = np.flatnonzero(fg_mask)
        if fg_indices.size < 16:
            fg_indices = np.arange(n_pix)

        # Subsample foreground deterministically for fast k-means++
        step = max(1, fg_indices.size // 4096)
        sample_idx = fg_indices[::step]
        sample_lab = lab_all[sample_idx]
        # Weight chroma (a, b) higher than lightness L for shadow invariance on 3D plastic parts
        feats = np.column_stack([sample_lab[:, 0] * 0.45, sample_lab[:, 1] * 2.2, sample_lab[:, 2] * 2.2])

        k = min(max(2, int(k_init)), sample_idx.size)
        centers = np.zeros((k, 3), dtype=np.float64)
        # Deterministic k-means++ seeding: start with brightest pixel, then farthest-point sampling
        first_i = int(np.argmax(sample_lab[:, 0]))
        centers[0] = feats[first_i]
        min_sq_dists = np.sum((feats - centers[0]) ** 2, axis=1)
        for c_idx in range(1, k):
            next_i = int(np.argmax(min_sq_dists))
            centers[c_idx] = feats[next_i]
            sq_d = np.sum((feats - centers[c_idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, sq_d)

        # Lloyd iterations
        labels = np.zeros(sample_idx.size, dtype=np.int32)
        for _ in range(12):
            dists = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_idx in range(k):
                members = feats[labels == c_idx]
                if members.shape[0] > 0:
                    centers[c_idx] = np.mean(members, axis=0)

        # Collect cluster medians in linear RGB & merge perceptually duplicate clusters
        raw_clusters: List[Dict[str, Any]] = []
        for c_idx in range(k):
            mask_c = labels == c_idx
            count_c = int(np.sum(mask_c))
            if count_c == 0 or (count_c / float(sample_idx.size)) < 0.01:
                continue
            med_lab = np.median(sample_lab[mask_c], axis=0)
            med_rgb = self.bridge.oklab_to_linear_srgb(med_lab)
            orig_pix_indices = sample_idx[mask_c]
            # Pick 2 representative interior sample points in normalized 0..1000 coordinates
            p1 = int(orig_pix_indices[len(orig_pix_indices) // 3])
            p2 = int(orig_pix_indices[(2 * len(orig_pix_indices)) // 3])
            pts_norm = [
                [int(round(((p1 // w) / max(1, h - 1)) * 1000)), int(round(((p1 % w) / max(1, w - 1)) * 1000))],
                [int(round(((p2 // w) / max(1, h - 1)) * 1000)), int(round(((p2 % w) / max(1, w - 1)) * 1000))],
            ]
            raw_clusters.append({
                "count": count_c,
                "med_lab": med_lab,
                "med_rgb": [float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2])],
                "sample_points": pts_norm,
            })

        raw_clusters.sort(key=lambda item: item["count"], reverse=True)
        merged: List[Dict[str, Any]] = []
        for cand in raw_clusters:
            is_dup = False
            for existing in merged:
                if self.bridge.delta_e00_linear_rgb(cand["med_rgb"], existing["med_rgb"]) < merge_delta_e00:
                    existing["count"] += cand["count"]
                    is_dup = True
                    break
            if not is_dup:
                merged.append(cand)

        palette_out: List[Dict[str, Any]] = []
        used_ids: Dict[str, int] = {}
        for item in merged:
            base_id, desc_name, is_achrom = self.describe_oklab_color(item["med_lab"])
            occ = used_ids.get(base_id, 0) + 1
            used_ids[base_id] = occ
            cg_id = base_id if occ == 1 else f"{base_id}_{occ}"
            full_name = desc_name if occ == 1 else f"{desc_name} #{occ}"
            palette_out.append({
                "color_group_id": cg_id,
                "descriptive_name": full_name,
                "is_achromatic": is_achrom,
                "median_linear_rgb": item["med_rgb"],
                "sample_points": item["sample_points"],
            })

        return palette_out
