"""Click-seeded manual segmentation engines: SAM 2 (VRAM-managed) and OKLab Flood-Fill + GrabCut (Sections 3.2, 11.2)."""
from __future__ import annotations

from collections import OrderedDict
import contextlib
import hashlib
import inspect
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Official OpenCV GrabCut docs: https://docs.opencv.org/4.12.0/d8/d83/tutorial_py_grabcut.html
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_fill_holes, label as nd_label

from paint_sim.collector.sources import load_collector_settings
from paint_sim.collector.vision.refine import (
    find_overlapping_mask_pairs,
    refine_mask_and_export_lossless_rgba,
    validate_mask_geometry,
)
from paint_sim.db.sample_ingestor import isolate_spoon_rgba_lossless, measure_swatch_optics
from paint_sim.physics.engine_bridge import NativePaintCoreBridge
from paint_sim.vision.vram_manager import VRAMManager

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]


def segment_from_clicks_oklab_grabcut(
    rgb_u8: np.ndarray,
    pos_points_xy: list[tuple[int, int]],
    neg_points_xy: list[tuple[int, int]] | None = None,
    box_xyxy: tuple[int, int, int, int] | None = None,
    oklab_tol: float = 0.06,
    grabcut_iters: int = 4,
    precomputed_oklab: np.ndarray | None = None,
) -> np.ndarray:
    """
    Produces a boolean foreground mask (H, W) around user-clicked or LLM-seeded points (Section 11.2).
    1. Computes OKLab distance from each positive seed point and merges connected regions.
    2. Fills enclosed specular highlight islands via binary_fill_holes.
    3. Refines boundary via cv2.grabCut(GC_INIT_WITH_MASK).
       Source: https://docs.opencv.org/4.12.0/d8/d83/tutorial_py_grabcut.html
    """
    h, w = rgb_u8.shape[:2]
    if not pos_points_xy:
        raise ValueError("At least one positive seed point (x, y) is required.")

    # When `box_xyxy` is provided, operate on the padded ROI sub-array (5x fewer pixels on multi-spoon sheets)
    if box_xyxy is not None:
        bx0, by0, bx1, by1 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
        bx0, bx1 = min(bx0, bx1), max(bx0, bx1)
        by0, by1 = min(by0, by1), max(by0, by1)
        pad = 12
        rx0 = max(0, bx0 - pad)
        ry0 = max(0, by0 - pad)
        rx1 = min(w, bx1 + pad)
        ry1 = min(h, by1 + pad)
        local_pos = [(int(px) - rx0, int(py) - ry0) for px, py in pos_points_xy]
        local_neg = [(int(nx) - rx0, int(ny) - ry0) for nx, ny in (neg_points_xy or [])]
        local_box = (
            max(0, bx0 - rx0),
            max(0, by0 - ry0),
            min(rx1 - rx0, bx1 - rx0),
            min(ry1 - ry0, by1 - ry0),
        )
        sub_rgb = np.ascontiguousarray(rgb_u8[ry0:ry1, rx0:rx1])
        sub_lab = precomputed_oklab[ry0:ry1, rx0:rx1] if precomputed_oklab is not None else None
        sub_mask = _segment_oklab_grabcut_roi(
            sub_rgb,
            local_pos,
            local_neg,
            local_box,
            oklab_tol=oklab_tol,
            grabcut_iters=grabcut_iters,
            precomputed_oklab=sub_lab,
        )
        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[ry0:ry1, rx0:rx1] = sub_mask
        return full_mask

    return _segment_oklab_grabcut_roi(
        rgb_u8,
        pos_points_xy,
        neg_points_xy,
        None,
        oklab_tol=oklab_tol,
        grabcut_iters=grabcut_iters,
        precomputed_oklab=precomputed_oklab,
    )


def _segment_oklab_grabcut_roi(
    rgb_u8: np.ndarray,
    pos_points_xy: list[tuple[int, int]],
    neg_points_xy: list[tuple[int, int]] | None = None,
    box_xyxy: tuple[int, int, int, int] | None = None,
    oklab_tol: float = 0.06,
    grabcut_iters: int = 4,
    precomputed_oklab: np.ndarray | None = None,
) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    if precomputed_oklab is not None and precomputed_oklab.shape[:2] == (h, w):
        lab = precomputed_oklab
    else:
        lin_rgb = NativePaintCoreBridge.srgb_u8_to_linear(rgb_u8)
        lab = NativePaintCoreBridge.linear_srgb_to_oklab(lin_rgb)

    # 8-connectivity structuring element for OKLab flood fill (Section 3.2)
    conn8 = np.ones((3, 3), dtype=int)
    seed_mask = np.zeros((h, w), dtype=bool)
    for sx, sy in pos_points_xy:
        sx_c = max(0, min(w - 1, int(sx)))
        sy_c = max(0, min(h - 1, int(sy)))
        seed_lab = lab[sy_c, sx_c]
        # Perceptual OKLab Euclidean distance (lightness weighted 0.65 to tolerate curved shading)
        d_l = (lab[..., 0] - seed_lab[0]) * 0.65
        d_a = lab[..., 1] - seed_lab[1]
        d_b = lab[..., 2] - seed_lab[2]
        dist = np.sqrt(d_l * d_l + d_a * d_a + d_b * d_b)
        within = dist <= oklab_tol
        if neg_points_xy:
            for nx, ny in neg_points_xy:
                nx_c = max(0, min(w - 1, int(nx)))
                ny_c = max(0, min(h - 1, int(ny)))
                neg_lab = lab[ny_c, nx_c]
                nd_l = (lab[..., 0] - neg_lab[0]) * 0.65
                nd_a = lab[..., 1] - neg_lab[1]
                nd_b = lab[..., 2] - neg_lab[2]
                neg_dist = np.sqrt(nd_l * nd_l + nd_a * nd_a + nd_b * nd_b)
                within &= dist <= neg_dist
        labeled, _ = nd_label(within, structure=conn8)
        comp_id = labeled[sy_c, sx_c]
        if comp_id > 0:
            seed_mask |= labeled == comp_id

    norm_box: tuple[int, int, int, int] | None = None
    if box_xyxy is not None:
        x0, y0, x1, y1 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        norm_box = (x0, y0, x1, y1)
        bx0, by0 = max(0, x0), max(0, y0)
        bx1, by1 = min(w, x1), min(h, y1)
        box_gate = np.zeros((h, w), dtype=bool)
        box_gate[by0:by1, bx0:bx1] = True
        bg_samples = []
        if neg_points_xy:
            for nx, ny in neg_points_xy:
                bg_samples.append(lab[max(0, min(h - 1, int(ny))), max(0, min(w - 1, int(nx)))])
        if not bg_samples and (bx1 - bx0) > 6 and (by1 - by0) > 6:
            bg_samples.extend(
                [
                    lab[by0, bx0],
                    lab[by0, bx1 - 1],
                    lab[by1 - 1, bx0],
                    lab[by1 - 1, bx1 - 1],
                ]
            )
        if bg_samples:
            bg_med = np.median(np.stack(bg_samples, axis=0), axis=0)
            bd_l = (lab[..., 0] - bg_med[0]) * 0.65
            bd_a = lab[..., 1] - bg_med[1]
            bd_b = lab[..., 2] - bg_med[2]
            bg_dist = np.sqrt(bd_l * bd_l + bd_a * bd_a + bd_b * bd_b)
            non_bg = (bg_dist > oklab_tol) & box_gate
            labeled_nb, _ = nd_label(non_bg, structure=conn8)
            for sx, sy in pos_points_xy:
                cid = labeled_nb[max(0, min(h - 1, int(sy))), max(0, min(w - 1, int(sx)))]
                if cid > 0:
                    seed_mask |= labeled_nb == cid
        seed_mask &= box_gate

    seed_mask = binary_fill_holes(seed_mask)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    if norm_box is not None:
        x0, y0, x1, y1 = norm_box
        gc_mask[: max(0, y0), :] = cv2.GC_BGD
        gc_mask[min(h, y1) :, :] = cv2.GC_BGD
        gc_mask[:, : max(0, x0)] = cv2.GC_BGD
        gc_mask[:, min(w, x1) :] = cv2.GC_BGD
    else:
        gc_mask[0, :] = gc_mask[-1, :] = gc_mask[:, 0] = gc_mask[:, -1] = cv2.GC_BGD

    gc_mask[seed_mask] = cv2.GC_PR_FGD
    for sx, sy in pos_points_xy:
        cv2.circle(gc_mask, (int(sx), int(sy)), radius=3, color=int(cv2.GC_FGD), thickness=-1)
    for nx, ny in neg_points_xy or []:
        cv2.circle(gc_mask, (int(nx), int(ny)), radius=3, color=int(cv2.GC_BGD), thickness=-1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    if grabcut_iters > 0 and np.any(gc_mask == cv2.GC_BGD) and np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
        try:
            # Source: https://docs.opencv.org/ref/master/d3/d47/group__imgproc__segmentation.html
            cv2.grabCut(rgb_u8, gc_mask, None, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
            fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            if box_xyxy is not None:
                fg |= seed_mask
            if not np.any(fg):
                fg = seed_mask
        except cv2.error:
            fg = seed_mask
    else:
        fg = seed_mask

    fg = binary_fill_holes(fg)
    # Keep connected components containing at least one positive point
    labeled_fg, _ = nd_label(fg, structure=conn8)
    keep = np.zeros((h, w), dtype=bool)
    for sx, sy in pos_points_xy:
        cid = labeled_fg[max(0, min(h - 1, int(sy))), max(0, min(w - 1, int(sx)))]
        if cid > 0:
            keep |= labeled_fg == cid
    return keep if np.any(keep) else seed_mask


class SAM2PointBoxSegmenter:
    """
    Lazy-loaded SAM 2 / HQ-SAM 2 interactive segmenter coordinated with `VRAMManager` (Section 3.2):
      - Caches image embeddings per image SHA-256 so repeated clicks only run the lightweight mask decoder.
      - Maintains LRU cache of part logits (up to 64 entries) for iterative mask refinement.
      - Supports model variants: hq_sam2_large, sam2.1_base_plus, sam2.1_small, sam2.1_tiny.
      - Falls back deterministically to OKLab GrabCut when checkpoint is missing.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "hq_sam2_large": {
            "checkpoint": "checkpoints/sam2.1_hq_hiera_large.pt",
            "model_cfg": "configs/sam2.1/sam2.1_hq_hiera_l.yaml",
            "vram_gb": 2.4,
            "requires_hq_package": True,
        },
        "sam2.1_base_plus": {
            "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
            "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "vram_gb": 2.0,
            "requires_hq_package": False,
        },
        "sam2.1_small": {
            "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
            "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "vram_gb": 1.8,
            "requires_hq_package": False,
        },
        "sam2.1_tiny": {
            "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
            "model_cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "vram_gb": 1.5,
            "requires_hq_package": False,
        },
    }

    @staticmethod
    def _check_hq_support() -> bool:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            return "hq_token_only" in inspect.signature(SAM2ImagePredictor.predict).parameters
        except Exception:
            return False

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        model_cfg: Optional[str] = None,
        vram_manager: Optional[VRAMManager] = None,
        model_variant: Optional[str] = None,
        bridge: Optional[NativePaintCoreBridge] = None,
    ) -> None:
        cfg = load_collector_settings()

        # Variant precedence: explicit model_variant > collector settings > default "sam2.1_base_plus"
        if model_variant is not None:
            variant_name = str(model_variant)
        elif cfg.get("sam2_model_variant"):
            variant_name = str(cfg["sam2_model_variant"])
        else:
            variant_name = "sam2.1_base_plus"

        if variant_name not in self.MODEL_VARIANTS:
            variant_name = "sam2.1_base_plus"

        vspec = self.MODEL_VARIANTS[variant_name]
        self.model_variant = variant_name
        self.vram_footprint_gb = float(vspec["vram_gb"])

        # HQ detection & downgrade
        if self.model_variant == "hq_sam2_large" and not self._check_hq_support():
            logger.warning("HQ-SAM 2 not supported in current environment; downgrading to sam2.1_base_plus.")
            self.model_variant = "sam2.1_base_plus"
            vspec = self.MODEL_VARIANTS["sam2.1_base_plus"]
            self.vram_footprint_gb = float(vspec["vram_gb"])

        # Checkpoint path precedence: explicit checkpoint_path > model_variant > collector settings > default
        if checkpoint_path is not None:
            raw_chk = checkpoint_path
        elif model_variant is not None:
            raw_chk = vspec["checkpoint"]
        elif cfg.get("sam2_checkpoint"):
            raw_chk = cfg["sam2_checkpoint"]
        else:
            raw_chk = vspec["checkpoint"]

        chk_p = Path(raw_chk)
        self.checkpoint_path = chk_p if chk_p.is_absolute() else (REPO_ROOT / chk_p)

        # Model config precedence: explicit model_cfg > model_variant > collector settings > default
        if model_cfg is not None:
            self.model_cfg = str(model_cfg)
        elif model_variant is not None:
            self.model_cfg = str(vspec["model_cfg"])
        elif cfg.get("sam2_model_cfg"):
            self.model_cfg = str(cfg["sam2_model_cfg"])
        else:
            self.model_cfg = str(vspec["model_cfg"])

        self.vram_manager = vram_manager or VRAMManager()
        self.bridge = bridge or NativePaintCoreBridge()
        self._lock = threading.RLock()
        self.max_part_cache: int = 64
        self._part_logits_cache: OrderedDict[Tuple[str, str], Any] = OrderedDict()
        self._current_panel_mask: Optional[np.ndarray] = None
        self._current_lin_rgb: Optional[np.ndarray] = None
        self._current_image_sha256: Optional[str] = None
        self._current_object_mask: Optional[np.ndarray] = None
        self._current_model_bbox: Optional[Tuple[int, int, int, int]] = None
        self._current_roi_bbox: Optional[Tuple[int, int, int, int]] = None
        self._predictor_roi_key: Optional[Tuple[str, Tuple[int, int, int, int]]] = None
        self._predictor: Any = None
        self._predictor_image_sha: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import sam2  # noqa: F401

            return self.checkpoint_path.is_file()
        except Exception:
            return False

    def _ensure_predictor(self) -> Any:
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"SAM 2 checkpoint not found at {self.checkpoint_path}. "
                    "Switch segmentation_engine to 'grabcut' or provide a valid SAM 2.1 checkpoint."
                )
            # Source: https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            cuda_avail = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_avail else "cpu"
            logger.info(
                "SAM 2 Predictor Diagnostics: cuda_available=%s, device_name=%s, model_variant=%s, checkpoint_path=%s",
                cuda_avail,
                device_name,
                self.model_variant,
                self.checkpoint_path,
            )

            device = "cuda" if cuda_avail else "cpu"
            sam_model = build_sam2(self.model_cfg, str(self.checkpoint_path), device=device)
            self.vram_manager.register_model("sam_segmentation", sam_model)
            orig_jit_script = torch.jit.script

            def _safe_jit_script(obj: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    return orig_jit_script(obj, *args, **kwargs)
                except OSError:
                    return obj

            try:
                torch.jit.script = _safe_jit_script  # type: ignore[assignment]
                self._predictor = SAM2ImagePredictor(sam_model)
            finally:
                torch.jit.script = orig_jit_script  # type: ignore[assignment]
            return self._predictor

    def detect_foreground_object(self, rgb_u8: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = rgb_u8.shape[:2]
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[: min(h, 2), :] = True
        border_mask[max(0, h - 2) :, :] = True
        border_mask[:, : min(w, 2)] = True
        border_mask[:, max(0, w - 2) :] = True

        lin_rgb = NativePaintCoreBridge.srgb_u8_to_linear(rgb_u8)
        lab = NativePaintCoreBridge.linear_srgb_to_oklab(lin_rgb)

        border_lab = lab[border_mask]
        median_bg_lab = np.median(border_lab, axis=0)

        d_l = (lab[..., 0] - median_bg_lab[0]) * 0.65
        d_a = lab[..., 1] - median_bg_lab[1]
        d_b = lab[..., 2] - median_bg_lab[2]
        dist = np.sqrt(d_l * d_l + d_a * d_a + d_b * d_b)

        cand_fg = dist > 0.035
        fg_mask = binary_fill_holes(cand_fg)

        fg_count = int(np.count_nonzero(fg_mask))
        total_px = h * w
        if 0.02 * total_px <= fg_count <= 0.95 * total_px:
            ys, xs = np.where(fg_mask)
            model_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            fg_mask = np.ones((h, w), dtype=bool)
            model_bbox = (0, 0, w - 1, h - 1)

        return fg_mask, model_bbox

    def _ensure_image_state(self, rgb_u8: np.ndarray, img_sha: str) -> None:
        if self._current_image_sha256 != img_sha:
            h, w = rgb_u8.shape[:2]
            self._current_lin_rgb = NativePaintCoreBridge.srgb_u8_to_linear(rgb_u8)
            self._current_panel_mask = self.bridge.compute_panel_lines(self._current_lin_rgb)
            fg_mask, model_bbox = self.detect_foreground_object(rgb_u8)
            self._current_object_mask = fg_mask
            self._current_model_bbox = model_bbox

            rx0 = max(0, model_bbox[0] - 32)
            ry0 = max(0, model_bbox[1] - 32)
            rx1 = min(w - 1, model_bbox[2] + 32)
            ry1 = min(h - 1, model_bbox[3] + 32)
            if (rx1 - rx0 + 1) * (ry1 - ry0 + 1) >= 0.85 * (w * h) or w <= 128 or h <= 128:
                self._current_roi_bbox = (0, 0, w - 1, h - 1)
            else:
                self._current_roi_bbox = (rx0, ry0, rx1, ry1)
            self._current_image_sha256 = img_sha

    def warmup_session_image(self, rgb_u8: np.ndarray) -> Dict[str, Any]:
        h, w = rgb_u8.shape[:2]
        with self._lock:
            img_sha = hashlib.sha256(np.ascontiguousarray(rgb_u8).tobytes()).hexdigest()
            self._ensure_image_state(rgb_u8, img_sha)
            engine_warmed = False

            if self.is_available():
                try:
                    rx0, ry0, rx1, ry1 = self._current_roi_bbox or (0, 0, w - 1, h - 1)
                    roi_rgb = np.array(rgb_u8[ry0 : ry1 + 1, rx0 : rx1 + 1], dtype=np.uint8, copy=True, order="C")
                    with self.vram_manager.acquire_stage("sam_segmentation", self.vram_footprint_gb, keep_resident=True):
                        predictor = self._ensure_predictor()
                        import torch

                        device_type = "cuda" if torch.cuda.is_available() else "cpu"
                        autocast_ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if device_type == "cuda"
                            else contextlib.nullcontext()
                        )
                        with torch.inference_mode(), autocast_ctx:
                            predictor.set_image(roi_rgb)
                            rh_w, rw_w = roi_rgb.shape[:2]
                            predictor.predict(
                                point_coords=np.array([[float(rw_w // 2), float(rh_w // 2)]], dtype=np.float32),
                                point_labels=np.array([1], dtype=np.int32),
                                multimask_output=True,
                            )
                        self._predictor_roi_key = (img_sha, (rx0, ry0, rx1, ry1))
                        self._predictor_image_sha = img_sha
                        engine_warmed = True
                except Exception as exc:
                    logger.warning("SAM 2 warmup failed (%s).", exc)
                    engine_warmed = False

            obj_mask = (
                self._current_object_mask.copy()
                if self._current_object_mask is not None
                else np.ones((h, w), dtype=bool)
            )
            model_bbox = (
                list(self._current_model_bbox)
                if self._current_model_bbox is not None
                else [0, 0, w - 1, h - 1]
            )
            roi_bbox = (
                list(self._current_roi_bbox)
                if self._current_roi_bbox is not None
                else [0, 0, w - 1, h - 1]
            )

            return {
                "image_sha256": img_sha,
                "object_mask": obj_mask,
                "model_bbox_xyxy": model_bbox,
                "roi_bbox_xyxy": roi_bbox,
                "engine_warmed": engine_warmed,
            }

    @staticmethod
    def _select_best_mask_index(
        masks: np.ndarray,
        scores: np.ndarray,
        local_pos: Sequence[Tuple[int, int]],
        multimask_output: bool,
    ) -> int:
        if not multimask_output or len(scores) <= 1:
            return int(np.argmax(scores)) if len(scores) > 0 else 0

        max_score = float(np.max(scores))
        total_px = float(masks.shape[1] * masks.shape[2])

        candidates: List[Tuple[float, int]] = []
        for i in range(len(scores)):
            m = np.asarray(masks[i] > 0.0, dtype=bool)
            area_ratio = float(np.count_nonzero(m)) / max(1.0, total_px)
            covers_seed = all(
                0 <= py < m.shape[0] and 0 <= px < m.shape[1] and bool(m[py, px])
                for px, py in local_pos
            )
            if (
                covers_seed
                and float(scores[i]) >= max(0.50, max_score - 0.22)
                and 0.0002 <= area_ratio <= 0.45
            ):
                candidates.append((area_ratio, i))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]

        return int(np.argmax(scores))

    def _cache_part_logits(
        self,
        img_sha: str,
        part_id: Optional[str],
        low_res_logits: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if part_id is None:
            return
        key = (img_sha, part_id)
        self._part_logits_cache[key] = (np.asarray(low_res_logits, dtype=np.float32), roi_bbox)
        self._part_logits_cache.move_to_end(key)
        while len(self._part_logits_cache) > self.max_part_cache:
            self._part_logits_cache.popitem(last=False)

    def clear_part_logits(self, part_id: str, image_sha256: Optional[str] = None) -> None:
        with self._lock:
            if image_sha256 is not None:
                self._part_logits_cache.pop((image_sha256, part_id), None)
            else:
                for k in list(self._part_logits_cache.keys()):
                    if k[1] == part_id:
                        self._part_logits_cache.pop(k, None)

    def predict_interactive(
        self,
        rgb_u8: np.ndarray,
        pos_points_xy: Sequence[Tuple[int, int]],
        neg_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
        box_xyxy: Optional[Tuple[int, int, int, int]] = None,
        part_id: Optional[str] = None,
        use_prev_mask: bool = True,
        snap_to_panel_lines: bool = True,
        hq_token_only: bool = False,
        tolerance: float = 5.0,
    ) -> Dict[str, Any]:
        if not pos_points_xy and box_xyxy is None and not (use_prev_mask and neg_points_xy and part_id):
            raise ValueError("At least one positive point or bounding box is required.")

        h, w = rgb_u8.shape[:2]
        clamped_pos = [(max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))) for x, y in (pos_points_xy or [])]
        clamped_neg = [(max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))) for x, y in (neg_points_xy or [])]
        if box_xyxy is not None:
            bx0 = max(0, min(w - 1, int(box_xyxy[0])))
            by0 = max(0, min(h - 1, int(box_xyxy[1])))
            bx1 = max(0, min(w - 1, int(box_xyxy[2])))
            by1 = max(0, min(h - 1, int(box_xyxy[3])))
            bx0, bx1 = min(bx0, bx1), max(bx0, bx1)
            by0, by1 = min(by0, by1), max(by0, by1)
            clamped_box: Optional[Tuple[int, int, int, int]] = (bx0, by0, bx1, by1)
        else:
            clamped_box = None

        with self._lock:
            img_sha = hashlib.sha256(np.ascontiguousarray(rgb_u8).tobytes()).hexdigest()
            self._ensure_image_state(rgb_u8, img_sha)

            active_roi = self._current_roi_bbox or (0, 0, w - 1, h - 1)
            rx0, ry0, rx1, ry1 = active_roi

            points_inside = True
            for px, py in clamped_pos:
                if not (rx0 <= px <= rx1 and ry0 <= py <= ry1):
                    points_inside = False
                    break
            if points_inside:
                for nx, ny in clamped_neg:
                    if not (rx0 <= nx <= rx1 and ry0 <= ny <= ry1):
                        points_inside = False
                        break
            if points_inside and clamped_box is not None:
                bx0, by0, bx1, by1 = clamped_box
                if not (rx0 <= bx0 <= rx1 and rx0 <= bx1 <= rx1 and ry0 <= by0 <= ry1 and ry0 <= by1 <= ry1):
                    points_inside = False

            if not points_inside:
                active_roi = (0, 0, w - 1, h - 1)
                rx0, ry0, rx1, ry1 = active_roi

            roi_rgb = np.array(rgb_u8[ry0 : ry1 + 1, rx0 : rx1 + 1], dtype=np.uint8, copy=True, order="C")
            rh, rw = roi_rgb.shape[:2]

            local_pos = [(x - rx0, y - ry0) for x, y in clamped_pos]
            local_neg = [(x - rx0, y - ry0) for x, y in clamped_neg]
            local_box = (
                clamped_box[0] - rx0,
                clamped_box[1] - ry0,
                clamped_box[2] - rx0,
                clamped_box[3] - ry0,
            ) if clamped_box is not None else None

            tol_clamped = float(np.clip(tolerance, 1.0, 20.0))
            if tol_clamped < 5.0:
                logit_thresh = float((5.0 - tol_clamped) * 0.35)
            else:
                logit_thresh = 0.0

            raw_mask: Optional[np.ndarray] = None
            iou_score = 0.85
            engine_used = self.model_variant

            if self.is_available():
                try:
                    with self.vram_manager.acquire_stage("sam_segmentation", self.vram_footprint_gb, keep_resident=True):
                        predictor = self._ensure_predictor()
                        if (
                            self._predictor_roi_key != (img_sha, active_roi)
                            or getattr(self, "_predictor_image_sha", None) != img_sha
                        ):
                            predictor.set_image(roi_rgb)
                            self._predictor_roi_key = (img_sha, active_roi)
                            self._predictor_image_sha = img_sha

                        pts_list: List[List[float]] = []
                        lbl_list: List[int] = []
                        for x, y in local_pos:
                            pts_list.append([float(x), float(y)])
                            lbl_list.append(1)
                        for x, y in local_neg:
                            pts_list.append([float(x), float(y)])
                            lbl_list.append(0)

                        point_coords = np.array(pts_list, dtype=np.float32) if pts_list else None
                        point_labels = np.array(lbl_list, dtype=np.int32) if lbl_list else None
                        box_arr = np.array(local_box, dtype=np.float32) if local_box is not None else None

                        mask_input = None
                        if use_prev_mask and part_id and (img_sha, part_id) in self._part_logits_cache:
                            cached_entry = self._part_logits_cache[(img_sha, part_id)]
                            if isinstance(cached_entry, tuple) and len(cached_entry) == 2:
                                cached_logits, cached_roi = cached_entry
                            else:
                                cached_logits, cached_roi = cached_entry, None

                            if (cached_roi is None or cached_roi == active_roi) and (len(local_pos) + len(local_neg) > 0):
                                mask_input = cached_logits

                        if mask_input is not None:
                            multimask_output = False
                        elif len(local_pos) == 1 and not local_neg and local_box is None:
                            multimask_output = True
                        else:
                            multimask_output = False

                        predict_kwargs: Dict[str, Any] = {
                            "point_coords": point_coords,
                            "point_labels": point_labels,
                            "box": box_arr,
                            "mask_input": mask_input,
                            "multimask_output": multimask_output,
                        }
                        if hq_token_only and self._check_hq_support():
                            predict_kwargs["hq_token_only"] = True

                        import torch

                        device_type = "cuda" if torch.cuda.is_available() else "cpu"
                        autocast_ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if device_type == "cuda"
                            else contextlib.nullcontext()
                        )
                        old_mask_thresh = getattr(predictor, "mask_threshold", 0.0)
                        predictor.mask_threshold = logit_thresh
                        try:
                            with torch.inference_mode(), autocast_ctx:
                                masks, scores, low_res_masks = predictor.predict(**predict_kwargs)
                        finally:
                            predictor.mask_threshold = old_mask_thresh

                        best_idx = self._select_best_mask_index(masks, scores, local_pos, multimask_output)

                        if tol_clamped < 5.0 and low_res_masks is not None and len(low_res_masks) > best_idx:
                            cur_low = low_res_masks[best_idx]
                            if cur_low.ndim == 3 and cur_low.shape[0] == 1:
                                cur_low = cur_low[0]
                            up_logits = cv2.resize(
                                np.asarray(cur_low, dtype=np.float32),
                                (rw, rh),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            m_base = masks[best_idx]
                            is_binary = (m_base.dtype == bool) or np.all(np.isin(m_base, [0, 1]))
                            if is_binary:
                                m_bool = m_base > 0
                                roi_raw_mask = m_bool & (up_logits > logit_thresh)
                            else:
                                roi_raw_mask = np.asarray(m_base > logit_thresh, dtype=bool)
                        elif tol_clamped < 5.0:
                            roi_raw_mask = np.asarray(masks[best_idx] > logit_thresh, dtype=bool)
                        else:
                            roi_raw_mask = np.asarray(masks[best_idx] > 0.0, dtype=bool)

                        raw_mask = np.zeros((h, w), dtype=bool)
                        raw_mask[ry0 : ry1 + 1, rx0 : rx1 + 1] = roi_raw_mask
                        iou_score = float(scores[best_idx]) if len(scores) > 0 else 1.0

                        if part_id is not None and low_res_masks is not None and len(low_res_masks) > best_idx:
                            self._cache_part_logits(
                                img_sha,
                                part_id,
                                low_res_masks[best_idx : best_idx + 1],
                                roi_bbox=active_roi,
                            )
                        engine_used = self.model_variant
                except Exception as exc:
                    logger.warning("SAM 2 predictor failed (%s); falling back to OKLab GrabCut.", exc)
                    raw_mask = None

            if raw_mask is None:
                fallback_pos = clamped_pos
                if not fallback_pos and clamped_box is not None:
                    bx0, by0, bx1, by1 = clamped_box
                    fallback_pos = [(int((bx0 + bx1) // 2), int((by0 + by1) // 2))]
                elif not fallback_pos and use_prev_mask and part_id and (img_sha, part_id) in self._part_logits_cache:
                    cached_entry = self._part_logits_cache[(img_sha, part_id)]
                    if isinstance(cached_entry, tuple) and len(cached_entry) == 2:
                        cached_logits, cached_roi = cached_entry
                    else:
                        cached_logits, cached_roi = cached_entry, None
                    if cached_logits is not None:
                        c_arr = cached_logits[0] if cached_logits.ndim == 3 else cached_logits
                        if cached_roi is not None:
                            crx0, cry0, crx1, cry1 = cached_roi
                            crw = max(1, crx1 - crx0 + 1)
                            crh = max(1, cry1 - cry0 + 1)
                        else:
                            crx0, cry0, crw, crh = 0, 0, w, h
                        up_prev = cv2.resize(
                            np.asarray(c_arr, dtype=np.float32),
                            (crw, crh),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        ay, ax = np.unravel_index(int(np.argmax(up_prev)), up_prev.shape)
                        if float(up_prev[ay, ax]) > 0.0:
                            prev_ys, prev_xs = np.where(up_prev > 0.0)
                            if len(prev_ys) > 0 and float(np.max(up_prev)) == float(np.min(up_prev[prev_ys, prev_xs])):
                                mid = len(prev_ys) // 2
                                ay, ax = int(prev_ys[mid]), int(prev_xs[mid])
                            fallback_pos = [(
                                max(0, min(w - 1, crx0 + int(ax))),
                                max(0, min(h - 1, cry0 + int(ay))),
                            )]

                if not fallback_pos:
                    raw_mask = np.zeros((h, w), dtype=bool)
                else:
                    oklab_tol = float(np.clip(0.06 * (tol_clamped / 5.0), 0.015, 0.22))
                    raw_mask = segment_from_clicks_oklab_grabcut(
                        rgb_u8, fallback_pos, clamped_neg, clamped_box, oklab_tol=oklab_tol
                    )
                engine_used = "oklab_grabcut_fallback"
                iou_score = 0.85
                if part_id is not None:
                    roi_slice = raw_mask[ry0 : ry1 + 1, rx0 : rx1 + 1].astype(np.float32)
                    if roi_slice.size > 0 and roi_slice.shape[0] > 0 and roi_slice.shape[1] > 0:
                        synth = cv2.resize(roi_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
                    else:
                        synth = cv2.resize(raw_mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
                    synth_logits = np.where(synth > 0.5, 5.0, -5.0).astype(np.float32)[np.newaxis, ...]
                    self._cache_part_logits(img_sha, part_id, synth_logits, roi_bbox=active_roi)

            if snap_to_panel_lines and np.any(raw_mask):
                ys, xs = np.where(raw_mask)
                margin = 16
                ry0 = max(0, int(np.min(ys)) - margin)
                ry1 = min(h, int(np.max(ys)) + margin + 1)
                rx0 = max(0, int(np.min(xs)) - margin)
                rx1 = min(w, int(np.max(xs)) + margin + 1)

                sub_lin = self._current_lin_rgb[ry0:ry1, rx0:rx1]  # type: ignore[index]
                sub_panel = self._current_panel_mask[ry0:ry1, rx0:rx1]  # type: ignore[index]
                sub_raw = raw_mask[ry0:ry1, rx0:rx1].copy()

                neg_guard = np.zeros_like(sub_raw, dtype=bool)
                neg_r = max(2, int(round(tol_clamped * 0.45)) + 1)
                neg_r2 = neg_r * neg_r
                for nx, ny in clamped_neg:
                    if ry0 <= ny < ry1 and rx0 <= nx < rx1:
                        lx, ly = nx - rx0, ny - ry0
                        y_min = max(0, ly - neg_r)
                        y_max = min(sub_raw.shape[0], ly + neg_r + 1)
                        x_min = max(0, lx - neg_r)
                        x_max = min(sub_raw.shape[1], lx + neg_r + 1)
                        for gy in range(y_min, y_max):
                            for gx in range(x_min, x_max):
                                if (gx - lx) ** 2 + (gy - ly) ** 2 <= neg_r2:
                                    neg_guard[gy, gx] = True
                for px, py in clamped_pos:
                    if ry0 <= py < ry1 and rx0 <= px < rx1:
                        neg_guard[py - ry0, px - rx0] = False

                ext_allow_feather: Optional[np.ndarray] = None
                lum_feather_mask: Optional[np.ndarray] = None
                lum_feather_alpha: Optional[np.ndarray] = None
                if (clamped_pos or np.count_nonzero(sub_raw) > 16) and tol_clamped > 1.5:
                    if sub_lin.shape[0] >= 3 and sub_lin.shape[1] >= 3:
                        sub_lin_smooth = cv2.GaussianBlur(sub_lin, (3, 3), 0.8)
                    else:
                        sub_lin_smooth = sub_lin
                    sub_lab = NativePaintCoreBridge.linear_srgb_to_oklab(sub_lin_smooth)
                    pos_labs_list = []
                    if clamped_pos:
                        pos_lin_pts = np.array(
                            [
                                [
                                    np.mean(
                                        self._current_lin_rgb[
                                            max(0, py - 1) : min(h, py + 2), max(0, px - 1) : min(w, px + 2)
                                        ],
                                        axis=(0, 1),
                                    )
                                    for px, py in clamped_pos
                                ]
                            ],  # type: ignore[index]
                            dtype=np.float32,
                        )
                        pos_labs = NativePaintCoreBridge.linear_srgb_to_oklab(pos_lin_pts)[0]
                        pos_labs_list = [pos_labs[i] for i in range(len(pos_labs))]
                    if np.count_nonzero(sub_raw) > 16:
                        med_lab = np.median(sub_lab[sub_raw], axis=0)
                        pos_labs_list.append(med_lab)

                    if np.any(sub_raw):
                        core_for_ref = binary_erosion(sub_raw, structure=np.ones((3, 3), dtype=bool))
                        if not np.any(core_for_ref):
                            core_for_ref = sub_raw
                        ref_lab = np.median(sub_lab[core_for_ref], axis=0)
                    elif pos_labs_list:
                        ref_lab = pos_labs_list[0]
                    else:
                        ref_lab = np.array([0.5, 0.0, 0.0], dtype=np.float32)

                    ref_a = float(ref_lab[1])
                    ref_b = float(ref_lab[2])
                    ref_chroma = float(np.hypot(ref_a, ref_b))

                    pos_de_maps = []
                    for seed_l, seed_a, seed_b in pos_labs_list:
                        dl = (sub_lab[..., 0] - seed_l) * 0.65
                        da = sub_lab[..., 1] - seed_a
                        db = sub_lab[..., 2] - seed_b
                        de = np.sqrt(dl * dl + da * da + db * db) * 100.0
                        pos_de_maps.append(de)
                    min_pos_de = np.minimum.reduce(pos_de_maps)

                    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    ring_inner = cv2.dilate(sub_raw.astype(np.uint8), k_inner) > 0
                    ring_outer = cv2.dilate(sub_raw.astype(np.uint8), k_outer) > 0
                    ext_ring = ring_outer & (~ring_inner)

                    ext_de = np.full(sub_lab.shape[:2], 999.0, dtype=np.float32)
                    if np.any(ext_ring):
                        ext_l, ext_a, ext_b = np.median(sub_lab[ext_ring], axis=0)
                        edl = (sub_lab[..., 0] - ext_l) * 0.65
                        eda = sub_lab[..., 1] - ext_a
                        edb = sub_lab[..., 2] - ext_b
                        ext_de = np.sqrt(edl * edl + eda * eda + edb * edb) * 100.0

                    corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
                    for cy, cx in corners:
                        if not sub_raw[cy, cx]:
                            cl, ca, cb = sub_lab[cy, cx]
                            cdl = (sub_lab[..., 0] - cl) * 0.65
                            cda = sub_lab[..., 1] - ca
                            cdb = sub_lab[..., 2] - cb
                            cde = np.sqrt(cdl * cdl + cda * cda + cdb * cdb) * 100.0
                            ext_de = np.minimum(ext_de, cde)

                    band_px = int(np.clip(round(tol_clamped * 0.55), 1, 7))
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
                    dilated = cv2.dilate(sub_raw.astype(np.uint8), kernel) > 0
                    panel_thresh = 160 if tol_clamped >= 10.0 else 95
                    ext_band = dilated & (~sub_raw) & (sub_panel < panel_thresh)

                    px_a = sub_lab[..., 1]
                    px_b = sub_lab[..., 2]
                    px_c = np.hypot(px_a, px_b)
                    hue_cos = (px_a * ref_a + px_b * ref_b) / np.maximum(px_c * ref_chroma, 1e-6)

                    closer_to_part = (min_pos_de < ext_de * 0.92) | (ext_de > 6.0)

                    if clamped_neg:
                        neg_lin_pts = np.array(
                            [
                                [
                                    np.mean(
                                        self._current_lin_rgb[
                                            max(0, ny - 1) : min(h, ny + 2), max(0, nx - 1) : min(w, nx + 2)
                                        ],
                                        axis=(0, 1),
                                    )
                                    for nx, ny in clamped_neg
                                ]
                            ],  # type: ignore[index]
                            dtype=np.float32,
                        )
                        neg_labs = NativePaintCoreBridge.linear_srgb_to_oklab(neg_lin_pts)[0]
                        neg_de_maps = []
                        for seed_l, seed_a, seed_b in neg_labs:
                            dl = (sub_lab[..., 0] - seed_l) * 0.65
                            da = sub_lab[..., 1] - seed_a
                            db = sub_lab[..., 2] - seed_b
                            de = np.sqrt(dl * dl + da * da + db * db) * 100.0
                            neg_de_maps.append(de)
                        min_neg_de = np.minimum.reduce(neg_de_maps)
                        neg_ok = min_pos_de < min_neg_de * 0.85
                    else:
                        neg_ok = True

                    if ref_chroma > 0.04:
                        # Unblurred linear luminance and local references
                        sub_y = (0.2126 * sub_lin[..., 0] + 0.7152 * sub_lin[..., 1] + 0.0722 * sub_lin[..., 2]).astype(np.float32)

                        k_lum = (15, 15)
                        core_erode = cv2.erode(sub_raw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
                        part_ref_mask = core_erode if np.any(core_erode) else sub_raw
                        y_part_med = float(np.median(sub_y[part_ref_mask]))
                        p_num = cv2.boxFilter(sub_y * part_ref_mask.astype(np.float32), ddepth=-1, ksize=k_lum, borderType=cv2.BORDER_REPLICATE)
                        p_den = cv2.boxFilter(part_ref_mask.astype(np.float32), ddepth=-1, ksize=k_lum, borderType=cv2.BORDER_REPLICATE)
                        y_part_local = np.where(p_den >= 1e-4, p_num / np.maximum(p_den, 1e-4), y_part_med)

                        k_bg_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        k_bg_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
                        bg_ring = (cv2.dilate(sub_raw.astype(np.uint8), k_bg_out) > 0) & (~(cv2.dilate(sub_raw.astype(np.uint8), k_bg_in) > 0)) & (~neg_guard)

                        if np.any(bg_ring):
                            y_bg_med = float(np.median(sub_y[bg_ring]))
                        elif np.any(~sub_raw):
                            y_bg_med = float(np.median(sub_y[~sub_raw]))
                        else:
                            y_bg_med = y_part_med

                        bg_num = cv2.boxFilter(sub_y * bg_ring.astype(np.float32), ddepth=-1, ksize=k_lum, borderType=cv2.BORDER_REPLICATE)
                        bg_den = cv2.boxFilter(bg_ring.astype(np.float32), ddepth=-1, ksize=k_lum, borderType=cv2.BORDER_REPLICATE)
                        y_bg_local = np.where(bg_den >= 1e-4, bg_num / np.maximum(bg_den, 1e-4), y_bg_med)

                        unblur_lab = NativePaintCoreBridge.linear_srgb_to_oklab(sub_lin)
                        u_a = unblur_lab[..., 1]
                        u_b = unblur_lab[..., 2]
                        u_c = np.hypot(u_a, u_b)
                        u_hue_cos = (u_a * ref_a + u_b * ref_b) / np.maximum(u_c * ref_chroma, 1e-6)
                        u_chromatic_ok = (u_hue_cos > 0.55) & (u_c >= max(0.022, ref_chroma * 0.15))

                        denom = y_bg_local - y_part_local
                        has_lum_contrast = np.abs(denom) >= 0.03
                        safe_denom = np.where(denom >= 0.0, np.maximum(denom, 1e-4), np.minimum(denom, -1e-4))
                        alpha_y = np.clip((y_bg_local - sub_y) / safe_denom, 0.0, 1.0)

                        cand_mask = ext_band & u_chromatic_ok & neg_ok & (~neg_guard)
                        allow_band_lum = cand_mask & has_lum_contrast & (alpha_y >= 0.50)
                        feather_band_lum = cand_mask & has_lum_contrast & (alpha_y >= 0.10) & (alpha_y < 0.50)

                        lum_feather_mask = feather_band_lum
                        lum_feather_alpha = np.where(feather_band_lum, alpha_y, 0.0).astype(np.float32)

                        chromatic_ok = (hue_cos > 0.55) & (px_c >= max(0.022, ref_chroma * 0.16))
                        allow_band_no_lum = cand_mask & (~has_lum_contrast) & closer_to_part & chromatic_ok & (min_pos_de <= tol_clamped * 1.85)

                        allow_band = allow_band_lum | allow_band_no_lum
                        ext_allow_feather = closer_to_part & chromatic_ok
                        if clamped_neg:
                            ext_allow_feather = ext_allow_feather & neg_ok
                    else:
                        achromatic_ok = min_pos_de <= tol_clamped * 1.35
                        if np.any(ext_ring):
                            achromatic_ok = achromatic_ok & (min_pos_de < ext_de * 0.90)
                        allow_band = ext_band & closer_to_part & achromatic_ok
                        if clamped_neg:
                            allow_band = allow_band & neg_ok
                            ext_allow_feather = closer_to_part & achromatic_ok & neg_ok
                        else:
                            ext_allow_feather = closer_to_part & achromatic_ok

                    allow_band = allow_band & (~neg_guard)

                    sub_raw = sub_raw | allow_band

                    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    closed = (
                        (cv2.morphologyEx(sub_raw.astype(np.uint8), cv2.MORPH_CLOSE, smooth_k) > 0)
                        & (sub_panel < panel_thresh)
                        & (~neg_guard)
                    )
                    if ref_chroma > 0.04:
                        closed = closed & (px_c >= max(0.015, ref_chroma * 0.12))
                    sub_raw = (closed | sub_raw) & (~neg_guard)
                    raw_mask[ry0:ry1, rx0:rx1] = sub_raw

                sub_alpha = self.bridge.snap_and_matte_part_mask(
                    sub_lin,
                    sub_panel,
                    sub_raw,
                    snap_radius=2,
                    matte_radius=1,
                    matte_eps=1e-4,
                )
                sub_alpha = np.where(sub_raw, 1.0, sub_alpha).astype(np.float32)
                sub_alpha = np.where(sub_alpha >= 0.05, sub_alpha, 0.0).astype(np.float32)
                sub_alpha[neg_guard] = 0.0

                if ext_allow_feather is not None:
                    sub_alpha[~sub_raw & (~ext_allow_feather)] = 0.0

                alpha_cutoff = float(np.clip(0.5 - (tol_clamped - 5.0) * 0.03, 0.20, 0.65))
                sub_alpha[~sub_raw] = np.where(
                    sub_alpha[~sub_raw] >= alpha_cutoff,
                    np.minimum(sub_alpha[~sub_raw] * 0.45, alpha_cutoff * 0.85),
                    0.0,
                )

                feather_only = None
                if lum_feather_mask is not None and lum_feather_alpha is not None:
                    feather_only = lum_feather_mask & (~sub_raw) & (~neg_guard)
                    sub_alpha[feather_only] = lum_feather_alpha[feather_only]

                alpha_01 = np.zeros((h, w), dtype=np.float32)
                alpha_01[ry0:ry1, rx0:rx1] = sub_alpha
                mask_bool = alpha_01 >= alpha_cutoff
                if feather_only is not None:
                    full_feather = np.zeros((h, w), dtype=bool)
                    full_feather[ry0:ry1, rx0:rx1] = feather_only
                    mask_bool[full_feather] = False

                if (not np.any(mask_bool)) or (np.count_nonzero(raw_mask) <= 4 and not np.any(sub_alpha >= alpha_cutoff)):
                    mask_bool = raw_mask.copy()
                    alpha_01 = mask_bool.astype(np.float32)

                if engine_used == "oklab_grabcut_fallback":
                    mask_bool = mask_bool & raw_mask
                    if feather_only is not None:
                        alpha_01[(~raw_mask) & (~full_feather)] = 0.0
                    else:
                        alpha_01[~raw_mask] = 0.0

                for nx, ny in clamped_neg:
                    mask_bool[ny, nx] = False
                    alpha_01[ny, nx] = 0.0
            else:
                mask_bool = raw_mask.copy()
                for nx, ny in clamped_neg:
                    mask_bool[ny, nx] = False
                alpha_01 = mask_bool.astype(np.float32)

            return {
                "mask_bool": mask_bool,
                "alpha_01": alpha_01.astype(np.float32),
                "iou_score": float(iou_score),
                "engine_used": engine_used,
                "has_cached_logits": bool(part_id and (img_sha, part_id) in self._part_logits_cache),
            }

    def predict_mask(
        self,
        rgb_u8: np.ndarray,
        pos_points_xy: Sequence[Tuple[int, int]],
        neg_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
        box_xyxy: Optional[Tuple[int, int, int, int]] = None,
        tolerance: float = 5.0,
    ) -> np.ndarray:
        """Run SAM 2 point/box prompt prediction under VRAMManager budget control."""
        res = self.predict_interactive(
            rgb_u8=rgb_u8,
            pos_points_xy=pos_points_xy,
            neg_points_xy=neg_points_xy,
            box_xyxy=box_xyxy,
            snap_to_panel_lines=False,
            tolerance=tolerance,
        )
        return res["mask_bool"]


_GLOBAL_SAM2_SEGMENTER_LOCK = threading.Lock()
_GLOBAL_SAM2_SEGMENTER: Optional[SAM2PointBoxSegmenter] = None


def _get_sam2_segmenter() -> SAM2PointBoxSegmenter:
    global _GLOBAL_SAM2_SEGMENTER
    with _GLOBAL_SAM2_SEGMENTER_LOCK:
        if _GLOBAL_SAM2_SEGMENTER is None:
            _GLOBAL_SAM2_SEGMENTER = SAM2PointBoxSegmenter()
        return _GLOBAL_SAM2_SEGMENTER


def segment_single_sample(
    rgb_u8: np.ndarray,
    pos_points_xy: Sequence[Tuple[int, int]],
    *,
    neg_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
    box_xyxy: Optional[Tuple[int, int, int, int]] = None,
    clean_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
    engine: Optional[str] = None,
    oklab_tol: Optional[float] = None,
    grabcut_iters: Optional[int] = None,
    tolerance: Optional[float] = None,
    override_validation: bool = False,
    precomputed_oklab: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Segment one sample from `rgb_u8` using the chosen engine (`grabcut`, `sam2`, or `rembg`),
    apply lossless rim refinement (`refine.py`), validate mask geometry, and measure swatch optics.
    """
    cfg = load_collector_settings()
    chosen_engine = (engine or cfg.get("segmentation_engine", "grabcut")).lower().strip()
    tol = float(oklab_tol if oklab_tol is not None else cfg.get("oklab_flood_tolerance", 0.06))
    iters = int(grabcut_iters if grabcut_iters is not None else cfg.get("grabcut_iterations", 4))

    pos_list = [(int(p[0]), int(p[1])) for p in pos_points_xy]
    neg_list = [(int(p[0]), int(p[1])) for p in (neg_points_xy or [])]
    if box_xyxy is not None:
        bx0, by0, bx1, by1 = (int(v) for v in box_xyxy)
        box_tuple: Optional[Tuple[int, int, int, int]] = (
            min(bx0, bx1),
            min(by0, by1),
            max(bx0, bx1),
            max(by0, by1),
        )
    else:
        box_tuple = None

    engine_used = chosen_engine
    if chosen_engine == "sam2":
        sam2_seg = _get_sam2_segmenter()
        if sam2_seg.is_available():
            eff_tol = (
                float(tolerance)
                if tolerance is not None
                else float(np.clip(5.0 + (float(tol) - 0.08) * 75.0, 1.0, 20.0))
            )
            raw_mask = sam2_seg.predict_mask(rgb_u8, pos_list, neg_list, box_tuple, tolerance=eff_tol)
        else:
            logger.warning("SAM 2 checkpoint unavailable; falling back to OKLab flood-fill + GrabCut.")
            engine_used = "grabcut"
            raw_mask = segment_from_clicks_oklab_grabcut(
                rgb_u8,
                pos_list,
                neg_list,
                box_tuple,
                oklab_tol=tol,
                grabcut_iters=iters,
                precomputed_oklab=precomputed_oklab,
            )
    elif chosen_engine == "rembg":
        try:
            import rembg
        except ImportError as exc:
            raise ValueError("rembg package is not installed") from exc

        h, w = rgb_u8.shape[:2]
        if box_tuple is not None:
            x0, y0, x1, y1 = box_tuple
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid empty rembg crop box after clamping: {(x0, y0, x1, y1)}")
        sub_img = Image.fromarray(rgb_u8[y0:y1, x0:x1], mode="RGB")
        rembg_model = str(cfg.get("rembg_model", "u2net"))
        session = rembg.new_session(rembg_model)
        mask_l = rembg.remove(sub_img, session=session, only_mask=True)
        sub_mask = np.asarray(mask_l, dtype=np.uint8) > 130
        raw_mask = np.zeros((h, w), dtype=bool)
        raw_mask[y0:y1, x0:x1] = sub_mask
        if not np.any(raw_mask) and pos_list:
            raw_mask = segment_from_clicks_oklab_grabcut(
                rgb_u8,
                pos_list,
                neg_list,
                box_tuple,
                oklab_tol=tol,
                grabcut_iters=iters,
                precomputed_oklab=precomputed_oklab,
            )
    else:
        engine_used = "grabcut"
        raw_mask = segment_from_clicks_oklab_grabcut(
            rgb_u8,
            pos_list,
            neg_list,
            box_tuple,
            oklab_tol=tol,
            grabcut_iters=iters,
            precomputed_oklab=precomputed_oklab,
        )

    refined = refine_mask_and_export_lossless_rgba(
        rgb_u8,
        raw_mask,
        seed_points_xy=pos_list,
    )
    geom_val = validate_mask_geometry(
        refined["refined_mask_bool"],
        min_area_ratio=float(cfg.get("min_mask_area_ratio", 0.002)),
        max_area_ratio=float(cfg.get("max_mask_area_ratio", 0.60)),
        max_borders=int(cfg.get("max_touched_borders", 2)),
    )

    # Translate `clean_points_xy` into crop-local coordinates if present
    c_xmin, c_ymin, c_xmax, c_ymax = refined["crop_bbox_xyxy"]
    local_clean_pts: Optional[List[List[int]]] = None
    if clean_points_xy:
        local_clean_pts = [
            [max(0, int(px) - c_xmin), max(0, int(py) - c_ymin)]
            for px, py in clean_points_xy
        ]

    measured_hex, optical_params = measure_swatch_optics(
        refined["rgba_crop_u8"],
        clean_points_2d=local_clean_pts,
        points_are_norm1000=False,
    )

    needs_review = (not geom_val["valid"] and not override_validation) or (
        float(optical_params.get("gloss_margin_to_next_class", 1.0)) < 0.01
    )
    review_reasons = list(geom_val["reasons"])
    if float(optical_params.get("gloss_margin_to_next_class", 1.0)) < 0.01:
        review_reasons.append(
            f"gloss_margin_to_next_class ({optical_params['gloss_margin_to_next_class']}) < 0.01"
        )

    return {
        "engine": engine_used,
        "refined_mask_bool": refined["refined_mask_bool"],
        "crop_png_bytes": refined["crop_png_bytes"],
        "mask_png_bytes": refined["mask_png_bytes"],
        "crop_bbox_xyxy": refined["crop_bbox_xyxy"],
        "max_rgb_diff": refined["max_rgb_diff"],
        "validation": geom_val,
        "measured_srgb_hex": measured_hex,
        "calibrated_optical_params": optical_params,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
    }


def segment_multiple_samples(
    rgb_u8: np.ndarray,
    sample_prompts: Sequence[Dict[str, Any]],
    *,
    engine: Optional[str] = None,
    oklab_tol: Optional[float] = None,
    grabcut_iters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Segment multiple click-seeded samples on the same image and check pairwise mask overlap (IoU > 0.3).
    Each entry in `sample_prompts` has `{"pos": [[x, y], ...], "neg": [[x, y], ...], "box": [x0, y0, x1, y1] | None}`.
    """
    cfg = load_collector_settings()
    results: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []

    # Precompute full-image OKLab once if multiple unboxed clicks share the image
    unboxed_count = sum(1 for p in sample_prompts if not p.get("box"))
    shared_oklab: Optional[np.ndarray] = None
    if unboxed_count >= 2:
        lin_rgb = NativePaintCoreBridge.srgb_u8_to_linear(rgb_u8)
        shared_oklab = NativePaintCoreBridge.linear_srgb_to_oklab(lin_rgb)

    for idx, prompt in enumerate(sample_prompts, start=1):
        pos_pts = [tuple(pt) for pt in prompt.get("pos", [])]
        neg_pts = [tuple(pt) for pt in prompt.get("neg", [])]
        box = tuple(prompt["box"]) if prompt.get("box") else None
        clean_pts = [tuple(pt) for pt in prompt.get("clean_points", [])] if prompt.get("clean_points") else None
        res = segment_single_sample(
            rgb_u8,
            pos_pts,
            neg_points_xy=neg_pts,
            box_xyxy=box,  # type: ignore[arg-type]
            clean_points_xy=clean_pts,  # type: ignore[arg-type]
            engine=engine,
            oklab_tol=oklab_tol,
            grabcut_iters=grabcut_iters,
            override_validation=bool(prompt.get("override_validation", False)),
            precomputed_oklab=shared_oklab,
        )
        res["index"] = idx
        results.append(res)
        masks.append(res["refined_mask_bool"])

    overlaps = find_overlapping_mask_pairs(
        masks,
        iou_threshold=float(cfg.get("mask_overlap_iou_warn", 0.30)),
        bboxes_xyxy=[r["crop_bbox_xyxy"] for r in results],
    )
    if overlaps:
        for ov in overlaps:
            for target_idx in (ov["index_a"] - 1, ov["index_b"] - 1):
                if 0 <= target_idx < len(results):
                    results[target_idx]["needs_review"] = True
                    results[target_idx]["review_reasons"].append(ov["message"])

    return {
        "samples": results,
        "overlap_warnings": overlaps,
    }
