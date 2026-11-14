"""Google Cloud Vertex AI 'Nano Banana' Dual-Image Post-Refiner (Module 7)."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from paint_sim.config.prompt_loader import PromptConfigLoader

logger = logging.getLogger(__name__)


class NanoBananaRefiner:
    """
    Implements ADR-008 / SDD Section 3.2:
    Sends BOTH:
      - Image 1: Original white-balanced kit photo (`01_preprocessed_wb.png`)
      - Image 2: Physically simulated photo (`06_rendered_output.png`)
    along with the dynamically assembled prompt from `config/prompts.json`
    to the latest Nano Banana model on Google Cloud Vertex AI
    (`gemini-nano-banana-2.1` / fallback `gemini-3.1-flash-image`).
    Note: Sampling parameters (`temperature`, `top_p`, `top_k`, `seed`) are strictly
    omitted because Vertex AI `gemini-nano-banana-2.1` rejects them.
    """

    UNSUPPORTED_SAMPLING_KEYS = ("temperature", "top_p", "topP", "top_k", "topK", "seed")

    def __init__(
        self,
        project_id: Optional[str] = None,
        prompt_loader: Optional[PromptConfigLoader] = None,
        mock_vertex_transport: Optional[Callable[[Dict[str, Any]], bytes]] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")
        self.prompt_loader = prompt_loader or PromptConfigLoader()
        self.mock_vertex_transport = mock_vertex_transport
        self.last_request_payload: Optional[Dict[str, Any]] = None

    @staticmethod
    def _postprocess_resolution_and_color_lock(
        refined_png: bytes,
        physics_sim_png: bytes,
        gen_cfg: Dict[str, Any],
    ) -> bytes:
        """
        Ensures the Vertex AI Nano Banana output strictly matches the native pixel resolution
        of `physics_sim_png` (no downscaling) and optionally applies luminance-preserving
        chrominance lock (`color_lock_blend_strength`) to prevent generative hue drift.
        """
        # Only run image decoding if both byte buffers start with the standard PNG signature
        png_sig = b"\x89PNG\r\n\x1a\n"
        if not (refined_png.startswith(png_sig) and physics_sim_png.startswith(png_sig)):
            return refined_png

        try:
            import io
            from PIL import Image
            import numpy as np

            with Image.open(io.BytesIO(physics_sim_png)) as phys_img, Image.open(io.BytesIO(refined_png)) as ref_img:
                phys_rgb = phys_img.convert("RGB")
                ref_rgb = ref_img.convert("RGB")

                # 1. Enforce exact 1:1 native pixel resolution match with Image 2 (physics_sim_png)
                if gen_cfg.get("preserve_native_resolution", True) and ref_rgb.size != phys_rgb.size:
                    ref_rgb = ref_rgb.resize(phys_rgb.size, Image.Resampling.LANCZOS)

                # 2. Perceptually uniform OKLab chrominance lock toward Kubelka-Munk physics output
                if gen_cfg.get("color_lock_postprocess", True):
                    strength = float(gen_cfg.get("color_lock_blend_strength", 0.35))
                    if 0.0 < strength <= 1.0:
                        from paint_sim.physics.engine_bridge import NativePaintCoreBridge

                        ref_srgb = np.asarray(ref_rgb, dtype=np.float32) / 255.0
                        phys_srgb = np.asarray(phys_rgb, dtype=np.float32) / 255.0
                        ref_lin = NativePaintCoreBridge.srgb_to_linear(ref_srgb)
                        phys_lin = NativePaintCoreBridge.srgb_to_linear(phys_srgb)

                        ref_lab = NativePaintCoreBridge.linear_srgb_to_oklab(ref_lin)
                        phys_lab = NativePaintCoreBridge.linear_srgb_to_oklab(phys_lin)

                        # Preserve high-frequency lightness L from Nano Banana while locking chrominance (a, b) in OKLab
                        out_lab = np.empty_like(ref_lab)
                        out_lab[..., 0] = (1.0 - 0.25 * strength) * ref_lab[..., 0] + (0.25 * strength) * phys_lab[..., 0]
                        out_lab[..., 1] = (1.0 - strength) * ref_lab[..., 1] + strength * phys_lab[..., 1]
                        out_lab[..., 2] = (1.0 - strength) * ref_lab[..., 2] + strength * phys_lab[..., 2]

                        out_lin = NativePaintCoreBridge.oklab_to_linear_srgb(out_lab)
                        out_srgb = NativePaintCoreBridge.linear_to_srgb(out_lin)
                        blended = np.clip(np.round(out_srgb * 255.0), 0.0, 255.0).astype(np.uint8)
                        ref_rgb = Image.fromarray(blended, mode="RGB")

                out_buf = io.BytesIO()
                ref_rgb.save(out_buf, format="PNG", compress_level=1)
                return out_buf.getvalue()
        except Exception as exc:
            logger.debug("Skipping optional PIL/numpy post-process (%s); returning raw refined bytes.", exc)
            return refined_png

    @staticmethod
    def apply_background_mode(
        png_bytes: bytes,
        background_mode: str,
        object_mask: Optional[np.ndarray] = None,
        model_bbox_xyxy: Optional[Sequence[int]] = None,
    ) -> bytes:
        if background_mode not in ("object_transparent", "object_crop") or not png_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            return png_bytes

        try:
            import io
            from PIL import Image
            import numpy as np

            with Image.open(io.BytesIO(png_bytes)) as img:
                w, h = img.size
                if background_mode == "object_transparent":
                    rgba_img = img.convert("RGBA")
                    rgba_arr = np.asarray(rgba_img, dtype=np.uint8).copy()
                    if object_mask is not None and object_mask.shape[:2] == (h, w):
                        rgba_arr[..., 3] = np.where(object_mask, 255, 0).astype(np.uint8)
                    else:
                        rgb = rgba_arr[..., :3]
                        border_mask = np.zeros((h, w), dtype=bool)
                        border_mask[: min(h, 2), :] = True
                        border_mask[max(0, h - 2) :, :] = True
                        border_mask[:, : min(w, 2)] = True
                        border_mask[:, max(0, w - 2) :] = True
                        median_bg = np.median(rgb[border_mask], axis=0)
                        diff = np.sqrt(np.sum((rgb.astype(np.float32) - median_bg) ** 2, axis=-1))
                        fg = diff > 15.0
                        rgba_arr[..., 3] = np.where(fg, 255, 0).astype(np.uint8)

                    out_img = Image.fromarray(rgba_arr, mode="RGBA")
                    out_buf = io.BytesIO()
                    out_img.save(out_buf, format="PNG", compress_level=1)
                    return out_buf.getvalue()

                elif background_mode == "object_crop":
                    if model_bbox_xyxy is not None and len(model_bbox_xyxy) == 4:
                        x0 = max(0, int(model_bbox_xyxy[0]))
                        y0 = max(0, int(model_bbox_xyxy[1]))
                        x1 = min(w - 1, int(model_bbox_xyxy[2]))
                        y1 = min(h - 1, int(model_bbox_xyxy[3]))
                        if x1 >= x0 and y1 >= y0:
                            cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
                            out_buf = io.BytesIO()
                            cropped.save(out_buf, format="PNG", compress_level=1)
                            return out_buf.getvalue()
                    return png_bytes
        except Exception as exc:
            logger.debug("apply_background_mode failed (%s); returning original bytes.", exc)
            return png_bytes

        return png_bytes

    def refine_dual_image(
        self,
        orig_wb_png: bytes,
        physics_sim_png: bytes,
        recipe_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute dual-image post-refinement via Vertex AI Nano Banana.
        Always hot-reloads `config/prompts.json` prior to assembling the prompt.
        """
        if not orig_wb_png or not physics_sim_png:
            raise ValueError("Both orig_wb_png (Image 1) and physics_sim_png (Image 2) must be non-empty.")

        bg_mode = str(recipe_context.get("background_mode", "full_bg"))
        if bg_mode in ("object_transparent", "object_crop"):
            obj_mask = recipe_context.get("object_mask")
            m_bbox = recipe_context.get("model_bbox_xyxy")
            orig_wb_png = self.apply_background_mode(orig_wb_png, bg_mode, obj_mask, m_bbox)
            physics_sim_png = self.apply_background_mode(physics_sim_png, bg_mode, obj_mask, m_bbox)

        sys_inst, user_prompt, nb_cfg = self.prompt_loader.build_nano_banana_prompt(recipe_context)
        primary_model = nb_cfg.get("model_id", "gemini-nano-banana-2.1")
        fallback_model = nb_cfg.get("fallback_model_id", "gemini-3.1-flash-image")
        location = os.environ.get("VERTEX_LOCATION", nb_cfg.get("location", "us-central1"))
        raw_gen_cfg = nb_cfg.get("generation_config", {})
        # Strip unsupported sampling parameters (`temperature`, `top_p`, `top_k`, `seed`) for gemini-nano-banana-2.1 compatibility
        gen_cfg = {k: v for k, v in raw_gen_cfg.items() if k not in self.UNSUPPORTED_SAMPLING_KEYS}
        modalities = gen_cfg.get("response_modalities", ["IMAGE", "TEXT"])

        payload = {
            "provider": "vertex_ai",
            "project_id": self.project_id,
            "location": location,
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "system_instruction": sys_inst,
            "user_prompt": user_prompt,
            "image_1_orig_bytes_len": len(orig_wb_png),
            "image_2_physics_bytes_len": len(physics_sim_png),
            "image_1_orig_png": orig_wb_png,
            "image_2_physics_png": physics_sim_png,
            "response_modalities": modalities,
        }
        self.last_request_payload = payload

        if self.mock_vertex_transport is not None:
            raw_refined = self.mock_vertex_transport(payload)
            refined_bytes = self._postprocess_resolution_and_color_lock(raw_refined, physics_sim_png, gen_cfg)
            return {
                "status": "refined_vertex_ai",
                "model_used": primary_model,
                "prompt_used": user_prompt,
                "system_instruction": sys_inst,
                "refined_png_bytes": refined_bytes,
            }

        if not self.project_id:
            logger.info("GCP_PROJECT_ID not set; returning deterministic physics output with prompt preview.")
            return {
                "status": "fallback_physics_only (no_gcp_project_configured)",
                "model_used": "none (local_offline_mode)",
                "prompt_used": user_prompt,
                "system_instruction": sys_inst,
                "refined_png_bytes": physics_sim_png,
            }

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(vertexai=True, project=self.project_id, location=location)
            for model_name in (primary_model, fallback_model):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=orig_wb_png, mime_type="image/png"),
                            types.Part.from_bytes(data=physics_sim_png, mime_type="image/png"),
                            user_prompt,
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=sys_inst,
                            response_modalities=modalities,
                        ),
                    )
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                            refined_bytes = self._postprocess_resolution_and_color_lock(
                                part.inline_data.data, physics_sim_png, gen_cfg
                            )
                            return {
                                "status": "refined_vertex_ai",
                                "model_used": model_name,
                                "prompt_used": user_prompt,
                                "system_instruction": sys_inst,
                                "refined_png_bytes": refined_bytes,
                            }
                except Exception as inner_exc:
                    logger.warning("Vertex AI model %s failed (%s); trying fallback...", model_name, inner_exc)
        except Exception as exc:
            logger.warning("Vertex AI SDK invocation failed (%s); returning deterministic physics output.", exc)

        return {
            "status": "fallback_physics_only",
            "model_used": fallback_model,
            "prompt_used": user_prompt,
            "system_instruction": sys_inst,
            "refined_png_bytes": physics_sim_png,
        }
