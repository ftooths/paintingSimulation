"""Resumable job state persistence (`sample/collector_jobs/{job_id}/job.json`) and event engine (Section 8)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import numpy as np

from paint_sim.collector.extractors.adapters import load_site_adapter_for_url
from paint_sim.collector.extractors.generic import (
    IMAGE_EXTENSIONS,
    extract_image_candidates_from_html,
    extract_post_links_from_listing,
    filter_downloaded_image_candidate,
)
from paint_sim.collector.fetcher import FetchError, PoliteFetcher
from paint_sim.collector.sources import load_collector_settings, normalize_url, sanitize_defaults
from paint_sim.collector.vision.gemini_detect import (
    GeminiSampleDetector,
    detect_and_refine_samples_auto,
    redact_secrets,
)
from paint_sim.collector.vision.manual_segment import segment_multiple_samples, segment_single_sample
from paint_sim.db.sample_ingestor import (
    classify_paint_sample,
    ingest_samples_batch,
    load_image_normalized_srgb,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOBS_DIR = _REPO_ROOT / "sample" / "collector_jobs"

TOPCOAT_MAP_BY_INFERRED = {
    "none": ("none", "auto"),
    "gloss": ("gsi_gx100_gloss", "gloss"),
    "semi_gloss": ("gsi_b516_semigloss", "semi_gloss"),
    "matte": ("gsi_gx114_matte", "matte"),
}

SURFACER_MAP_BY_INFERRED = {
    "gray": "gsi_sf1500_gray",
    "white": "gsi_sf1500_white",
    "black": "gsi_sf1500_black",
    "pink": "gsi_sf1000_pink",
}


class CollectorJobManager:
    """
    Manages resumable collector jobs stored in `sample/collector_jobs/{job_id}/job.json` (Section 8).
    Emits structured progress events (`fetch_started`, `fetch_done`, `image_found`, `vision_started`,
    `vision_done`, `vision_failed`, `sample_ready`, `ingested`, `error`) to WebSocket and CLI listeners.
    """

    def __init__(
        self,
        jobs_dir: Optional[Path] = None,
        sample_root_dir: Optional[Path] = None,
        fetcher: Optional[PoliteFetcher] = None,
        detector: Optional[GeminiSampleDetector] = None,
        secrets_path: Optional[Path] = None,
    ) -> None:
        self.jobs_dir = Path(jobs_dir or DEFAULT_JOBS_DIR)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.sample_root_dir = sample_root_dir
        self.fetcher = fetcher or PoliteFetcher()
        self.secrets_path = secrets_path
        self.detector = detector or GeminiSampleDetector(secrets_path=secrets_path)
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._image_index: Dict[str, str] = {}
        self._sample_index: Dict[str, str] = {}
        self._jobs_cache: Dict[str, Dict[str, Any]] = {}
        self._rebuild_indices()

    def add_event_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_event_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _emit(self, job_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        safe_payload = json.loads(redact_secrets(json.dumps(payload, ensure_ascii=False), self.secrets_path))
        event = {
            "job_id": job_id,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **safe_payload,
        }
        try:
            job = self.get_job(job_id)
            logs = job.setdefault("logs", [])
            logs.append(event)
            if len(logs) > 500:
                job["logs"] = logs[-500:]
            self._save_job(job)
        except Exception:
            pass
        for cb in list(self._listeners):
            try:
                cb(job_id, event)
            except Exception:
                pass

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _job_json_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _rebuild_indices(self) -> None:
        if not self.jobs_dir.exists():
            return
        for jfile in self.jobs_dir.glob("*/job.json"):
            try:
                data = json.loads(jfile.read_text(encoding="utf-8"))
                jid = data.get("job_id")
                if not jid:
                    continue
                for img in data.get("images", []):
                    if img.get("image_id"):
                        self._image_index[img["image_id"]] = jid
                for smp in data.get("samples", []):
                    if smp.get("sample_tmp_id"):
                        self._sample_index[smp["sample_tmp_id"]] = jid
            except Exception:
                continue

    def _save_job(self, job: Dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        jdir = self._job_dir(job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        jpath = self._job_json_path(job_id)
        # Ensure no secret string ever enters job.json (Ground Rule 5 & Section 6)
        raw_json = json.dumps(job, indent=2, ensure_ascii=False)
        safe_json = redact_secrets(raw_json, self.secrets_path)
        self._jobs_cache[job_id] = json.loads(safe_json)
        fd, tmp_name = tempfile.mkstemp(prefix=".job_", suffix=".json.tmp", dir=str(jdir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(safe_json)
            os.replace(tmp_name, jpath)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id in self._jobs_cache:
            return self._jobs_cache[job_id]
        jpath = self._job_json_path(job_id)
        if not jpath.is_file():
            raise KeyError(f"Unknown job_id: {job_id!r}")
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        self._jobs_cache[job_id] = loaded
        return loaded

    def create_job(
        self,
        sources: Sequence[Dict[str, Any] | str],
        *,
        vision_mode: str = "manual",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new collector job from user-supplied URLs and optional per-URL defaults."""
        cfg = load_collector_settings()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:4]
        job_id = f"{ts}_{suffix}"

        norm_sources: List[Dict[str, Any]] = []
        for s in sources:
            if isinstance(s, str):
                url = normalize_url(s)
                defs: Dict[str, Any] = {}
            elif isinstance(s, dict):
                url = normalize_url(str(s.get("url", "")))
                defs = sanitize_defaults(s.get("defaults"))
            else:
                continue
            norm_sources.append({"url": url, "defaults": defs})

        if not norm_sources:
            raise ValueError("At least one valid HTTP/HTTPS source URL is required to create a job.")

        jdir = self._job_dir(job_id)
        (jdir / "images").mkdir(parents=True, exist_ok=True)
        (jdir / "masks").mkdir(parents=True, exist_ok=True)
        (jdir / "crops").mkdir(parents=True, exist_ok=True)

        job = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vision_mode": vision_mode if vision_mode in ("manual", "auto") else "manual",
            "model": model or cfg.get("default_vision_model", "gemini-3.8-flash"),
            "sources": norm_sources,
            "images": [],
            "samples": [],
            "logs": [],
        }
        self._save_job(job)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for jfile in sorted(self.jobs_dir.glob("*/job.json"), reverse=True):
            try:
                out.append(json.loads(jfile.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def find_job_by_image_id(self, image_id: str) -> Dict[str, Any]:
        jid = self._image_index.get(image_id)
        if jid:
            return self.get_job(jid)
        self._rebuild_indices()
        jid = self._image_index.get(image_id)
        if not jid:
            raise KeyError(f"Unknown image_id: {image_id!r}")
        return self.get_job(jid)

    def find_job_by_sample_id(self, sample_tmp_id: str) -> Dict[str, Any]:
        jid = self._sample_index.get(sample_tmp_id)
        if jid:
            return self.get_job(jid)
        self._rebuild_indices()
        jid = self._sample_index.get(sample_tmp_id)
        if not jid:
            raise KeyError(f"Unknown sample_tmp_id: {sample_tmp_id!r}")
        return self.get_job(jid)

    def get_image_bytes(self, image_id: str) -> bytes:
        job = self.find_job_by_image_id(image_id)
        img_rec = next((im for im in job["images"] if im["image_id"] == image_id), None)
        if not img_rec:
            raise KeyError(f"Image {image_id!r} not found in job {job['job_id']}.")
        img_path = self._job_dir(job["job_id"]) / img_rec["local_png"]
        return img_path.read_bytes()

    def get_sample_crop_bytes(self, sample_tmp_id: str) -> bytes:
        job = self.find_job_by_sample_id(sample_tmp_id)
        smp = next((s for s in job["samples"] if s["sample_tmp_id"] == sample_tmp_id), None)
        if not smp:
            raise KeyError(f"Sample {sample_tmp_id!r} not found.")
        return (self._job_dir(job["job_id"]) / smp["crop_png"]).read_bytes()

    def get_sample_mask_bytes(self, sample_tmp_id: str) -> bytes:
        job = self.find_job_by_sample_id(sample_tmp_id)
        smp = next((s for s in job["samples"] if s["sample_tmp_id"] == sample_tmp_id), None)
        if not smp:
            raise KeyError(f"Sample {sample_tmp_id!r} not found.")
        return (self._job_dir(job["job_id"]) / smp["mask_png"]).read_bytes()

    def register_local_image(
        self,
        job_id: str,
        image_bytes: bytes,
        *,
        image_url: str = "https://local.upload/sample.png",
        page_url: str = "https://local.upload/page",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register an image buffer directly into a job (used for local file uploads and tests)."""
        job = self.get_job(job_id)
        cfg = load_collector_settings()
        ok, info = filter_downloaded_image_candidate(image_bytes, min_image_px=int(cfg.get("min_image_px", 64)))
        if not ok:
            raise ValueError(info.get("reason", "Invalid image"))

        image_id = f"img_{len(job['images']) + 1:03d}_{info['sha256'][:8]}"
        rel_png = f"images/{image_id}.png"
        (self._job_dir(job_id) / rel_png).write_bytes(info["srgb_png_bytes"])

        rec = {
            "image_id": image_id,
            "image_url": image_url,
            "page_url": page_url,
            "local_png": rel_png,
            "sha256": info["sha256"],
            "w": info["width"],
            "h": info["height"],
            "vision_mode": job.get("vision_mode", "manual"),
            "model": job.get("model", cfg.get("default_vision_model", "gemini-3.8-flash")),
            "prompt_version": "v1",
            "defaults": sanitize_defaults(defaults),
            "status": "pending",
        }
        job["images"].append(rec)
        self._image_index[image_id] = job_id
        self._save_job(job)
        self._emit(job_id, "image_found", {"image_id": image_id, "image_url": image_url, "w": rec["w"], "h": rec["h"]})
        return rec

    def fetch_job_sources(self, job_id: str) -> Dict[str, Any]:
        """
        Fetch all source URLs in `job`, extract image candidates (or direct images),
        deduplicate by URL and content SHA-256, filter by `min_image_px`, and store sRGB-normalized PNGs.
        """
        job = self.get_job(job_id)
        cfg = load_collector_settings()
        min_px = int(cfg.get("min_image_px", 300))
        seen_sha256: set[str] = {im["sha256"] for im in job.get("images", []) if im.get("sha256")}
        seen_img_urls: set[str] = {im["image_url"] for im in job.get("images", []) if im.get("image_url")}

        self._emit(job_id, "fetch_started", {"source_count": len(job["sources"])})

        def process_downloaded_image(
            img_bytes: bytes,
            img_url: str,
            page_url: str,
            merged_defaults: Dict[str, Any],
            alt_text: str = "",
            caption_hint: str = "",
        ) -> None:
            nonlocal job
            ok, info = filter_downloaded_image_candidate(img_bytes, min_image_px=min_px)
            if not ok:
                return
            sha = info["sha256"]
            if sha in seen_sha256:
                return
            seen_sha256.add(sha)
            seen_img_urls.add(img_url)

            job = self.get_job(job_id)
            image_id = f"img_{len(job['images']) + 1:03d}_{sha[:8]}"
            rel_png = f"images/{image_id}.png"
            (self._job_dir(job_id) / rel_png).write_bytes(info["srgb_png_bytes"])

            rec = {
                "image_id": image_id,
                "image_url": img_url,
                "page_url": page_url,
                "local_png": rel_png,
                "sha256": sha,
                "w": info["width"],
                "h": info["height"],
                "alt_text": alt_text,
                "caption_hint": caption_hint,
                "vision_mode": job.get("vision_mode", "manual"),
                "model": job.get("model", cfg.get("default_vision_model", "gemini-3.8-flash")),
                "prompt_version": "v1",
                "defaults": merged_defaults,
                "status": "pending",
            }
            job["images"].append(rec)
            self._image_index[image_id] = job_id
            self._save_job(job)
            self._emit(
                job_id,
                "image_found",
                {"image_id": image_id, "image_url": img_url, "page_url": page_url, "w": rec["w"], "h": rec["h"]},
            )

        for src in list(job["sources"]):
            src_url = src["url"]
            user_defaults = dict(src.get("defaults", {}))
            adapter = load_site_adapter_for_url(src_url)
            merged_defaults = {**(adapter.default_metadata if adapter else {}), **user_defaults}

            try:
                res = self.fetcher.fetch(src_url)
            except FetchError as exc:
                self._emit(job_id, "error", {"url": src_url, "message": str(exc)})
                continue

            # Direct image URL vs HTML page URL (Section 2.1)
            parsed_path = urlparse(res.final_url.lower()).path
            if res.content_type.startswith("image/") or any(parsed_path.endswith(ext) for ext in IMAGE_EXTENSIONS):
                process_downloaded_image(res.content, res.final_url, src_url, merged_defaults)
                continue

            html_text = res.content.decode("utf-8", errors="replace")

            # Optional listing page expansion ("Latest N posts", Section 7.3)
            page_urls_to_extract = [(res.final_url, html_text)]
            if adapter and adapter.listing_url and normalize_url(src_url) == normalize_url(adapter.listing_url):
                post_links = extract_post_links_from_listing(
                    html_text,
                    res.final_url,
                    post_link_selector=adapter.post_link_selector or "article h2 a, .entry-title a",
                    max_posts=adapter.max_posts,
                )
                if post_links:
                    page_urls_to_extract = []
                    for plink in post_links:
                        try:
                            pres = self.fetcher.fetch(plink, allowed_content_types=("text/html", "application/xhtml+xml"))
                            page_urls_to_extract.append((pres.final_url, pres.content.decode("utf-8", errors="replace")))
                        except FetchError as exc:
                            self._emit(job_id, "error", {"url": plink, "message": str(exc)})

            for page_url, page_html in page_urls_to_extract:
                candidates = extract_image_candidates_from_html(
                    page_html,
                    page_url,
                    adapter=adapter,
                    min_image_px=adapter.min_image_px if adapter else min_px,
                )
                for cand in candidates:
                    if cand.image_url in seen_img_urls:
                        continue
                    try:
                        img_res = self.fetcher.fetch(cand.image_url, allowed_content_types=("image/",))
                        process_downloaded_image(
                            img_res.content,
                            img_res.final_url,
                            page_url,
                            merged_defaults,
                            alt_text=cand.alt_text,
                            caption_hint=cand.caption_hint,
                        )
                    except FetchError as exc:
                        self._emit(job_id, "error", {"url": cand.image_url, "message": str(exc)})

        final_job = self.get_job(job_id)
        self._emit(job_id, "fetch_done", {"images_count": len(final_job["images"])})
        return final_job

    def _build_initial_metadata(
        self,
        img_rec: Dict[str, Any],
        *,
        caption_text: str = "",
        inferred_topcoat: str = "unknown",
        inferred_undercoat: str = "unknown",
        sheet_text: str = "",
        slot_index_0based: Optional[int] = None,
        layout_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-fill sample metadata from per-URL defaults, `classify_paint_sample()`, and LLM captions (Sections 5.1, 7.4)."""
        defs = dict(img_rec.get("defaults", {}))
        paint_code = str(defs.get("paint_code", "UNKNOWN"))
        paint_name_en = str(defs.get("paint_name_en", "Sample Paint"))
        paint_name_ko = str(defs.get("paint_name_ko", "샘플 도료"))

        # Check caption / layout_hint for topcoat and surfacer
        topcoat_code = str(defs.get("topcoat_code", "none"))
        final_finish = str(defs.get("final_finish", "auto"))
        surfacer_code = str(defs.get("surfacer_code", "none"))

        cap_low = caption_text.lower()
        if topcoat_code == "none":
            if inferred_topcoat in TOPCOAT_MAP_BY_INFERRED and inferred_topcoat != "unknown":
                topcoat_code, finish_hint = TOPCOAT_MAP_BY_INFERRED[inferred_topcoat]
                if final_finish == "auto" and finish_hint != "auto":
                    final_finish = finish_hint
            elif "semi" in cap_low or "반광" in cap_low:
                topcoat_code, final_finish = "gsi_b516_semigloss", "semi_gloss"
            elif "flat" in cap_low or "matte" in cap_low or "무광" in cap_low:
                topcoat_code, final_finish = "gsi_gx114_matte", "matte"
            elif "gloss" in cap_low or "유광" in cap_low:
                topcoat_code, final_finish = "gsi_gx100_gloss", "gloss"
            elif layout_hint and slot_index_0based is not None:
                tc_codes = layout_hint.get("topcoat_codes") or []
                labels = layout_hint.get("labels") or []
                if 0 <= slot_index_0based < len(tc_codes):
                    topcoat_code = str(tc_codes[slot_index_0based])
                if 0 <= slot_index_0based < len(labels):
                    lbl = str(labels[slot_index_0based])
                    if lbl in ("gloss", "semi_gloss", "matte"):
                        final_finish = lbl

        if surfacer_code == "none" and inferred_undercoat in SURFACER_MAP_BY_INFERRED:
            surfacer_code = SURFACER_MAP_BY_INFERRED[inferred_undercoat]

        cls_res = classify_paint_sample(
            paint_code=paint_code,
            paint_name_en=paint_name_en,
            paint_name_ko=paint_name_ko,
            brand_hint=str(defs.get("brand_id", "")),
            solvent_hint=str(defs.get("solvent_type", "")),
            role_hint=str(defs.get("paint_role", "")),
            product_line_hint=str(defs.get("product_line", "")),
            sheet_text=f"{sheet_text} {caption_text}".strip(),
            defaults=defs,
        )

        if final_finish == "auto":
            final_finish = cls_res["default_finish"]

        return {
            "brand_id": cls_res["brand_id"],
            "solvent_type": cls_res["solvent_type"],
            "paint_role": cls_res["paint_role"],
            "product_line": cls_res["product_line"],
            "paint_code": paint_code,
            "paint_name_en": paint_name_en,
            "paint_name_ko": paint_name_ko,
            "surfacer_code": surfacer_code,
            "surfacer_grit": str(defs.get("surfacer_grit") or cls_res["surfacer_grit"]),
            "undercoat_code": str(defs.get("undercoat_code", "none")),
            "topcoat_code": topcoat_code,
            "final_finish": final_finish,
            "coats": float(defs.get("coats", 2.0)),
            "thinner_and_ratio": str(defs.get("thinner_and_ratio", "standard")),
            "substrate_color": str(defs.get("substrate_color", "white")),
            "lighting_env": str(defs.get("lighting_env", "std")),
            "source_type": str(defs.get("source_type", "web_community_spoon")),
            "source_url": img_rec["page_url"],
            "image_url": img_rec["image_url"],
            "notes": caption_text,
            "candidates": cls_res["candidates"],
            "ambiguous_fields": cls_res["ambiguous_fields"],
            "classifier_needs_review": cls_res["needs_review"],
            "classifier_review_reasons": cls_res["review_reasons"],
        }

    def segment_image_manual(
        self,
        image_id: str,
        *,
        sample_prompts: Optional[Sequence[Dict[str, Any]]] = None,
        pos_points: Optional[Sequence[Sequence[int]]] = None,
        neg_points: Optional[Sequence[Sequence[int]]] = None,
        box: Optional[Sequence[int]] = None,
        engine: Optional[str] = None,
        oklab_tol: Optional[float] = None,
        grabcut_iters: Optional[int] = None,
        replace_existing_for_image: bool = False,
        override_validation: bool = False,
    ) -> Dict[str, Any]:
        """Run manual click-seeded segmentation (single or multi-sample) on `image_id`."""
        job = self.find_job_by_image_id(image_id)
        job_id = job["job_id"]
        img_rec = next(im for im in job["images"] if im["image_id"] == image_id)

        self._emit(job_id, "vision_started", {"image_id": image_id, "mode": "manual"})

        img_bytes = self.get_image_bytes(image_id)
        rgb_u8 = np.asarray(load_image_normalized_srgb(img_bytes).convert("RGB"), dtype=np.uint8)

        if sample_prompts is None:
            sample_prompts = [
                {
                    "pos": [list(p) for p in (pos_points or [])],
                    "neg": [list(p) for p in (neg_points or [])],
                    "box": list(box) if box else None,
                    "override_validation": override_validation,
                }
            ]

        multi_res = segment_multiple_samples(
            rgb_u8,
            sample_prompts,
            engine=engine,
            oklab_tol=oklab_tol,
            grabcut_iters=grabcut_iters,
        )

        job = self.get_job(job_id)
        if replace_existing_for_image:
            job["samples"] = [s for s in job["samples"] if s["image_id"] != image_id]

        adapter = load_site_adapter_for_url(img_rec["page_url"])
        layout_hint = adapter.layout_hint if adapter else None

        created_samples: List[Dict[str, Any]] = []
        for idx, seg_item in enumerate(multi_res["samples"]):
            prompt_in = sample_prompts[idx]
            smp_id = f"smp_{image_id}_{uuid.uuid4().hex[:6]}"
            rel_mask = f"masks/{smp_id}.png"
            rel_crop = f"crops/{smp_id}.png"
            (self._job_dir(job_id) / rel_mask).write_bytes(seg_item["mask_png_bytes"])
            (self._job_dir(job_id) / rel_crop).write_bytes(seg_item["crop_png_bytes"])

            meta = self._build_initial_metadata(
                img_rec,
                slot_index_0based=idx,
                layout_hint=layout_hint,
            )
            reasons = (
                list(seg_item["review_reasons"])
                + list(meta["classifier_review_reasons"])
                + [f"Ambiguous taxonomy field: {f}" for f in meta["ambiguous_fields"]]
            )
            needs_review = bool(seg_item["needs_review"] or meta["classifier_needs_review"] or meta["ambiguous_fields"])

            smp_rec = {
                "sample_tmp_id": smp_id,
                "image_id": image_id,
                "index": len(job["samples"]) + 1,
                "prompts": {
                    "pos": [list(p) for p in prompt_in.get("pos", [])],
                    "neg": [list(p) for p in prompt_in.get("neg", [])],
                    "box": list(prompt_in["box"]) if prompt_in.get("box") else None,
                    "oklab_tol": oklab_tol,
                },
                "engine": seg_item["engine"],
                "mask_png": rel_mask,
                "crop_png": rel_crop,
                "crop_bbox_xyxy": list(seg_item["crop_bbox_xyxy"]),
                "validation": seg_item["validation"],
                "measured_srgb_hex": seg_item["measured_srgb_hex"],
                "calibrated_optical_params": seg_item["calibrated_optical_params"],
                "llm": {"caption_text": "", "confidence": 1.0},
                "metadata": meta,
                "needs_review": needs_review,
                "review_reasons": reasons,
                "status": "proposed",
                "sample_id": None,
            }
            job["samples"].append(smp_rec)
            self._sample_index[smp_id] = job_id
            created_samples.append(smp_rec)
            self._emit(job_id, "sample_ready", {"sample_tmp_id": smp_id, "image_id": image_id})

        for im in job["images"]:
            if im["image_id"] == image_id:
                im["vision_mode"] = "manual"
                im["status"] = "segmented"
        self._save_job(job)
        self._emit(
            job_id,
            "vision_done",
            {"image_id": image_id, "mode": "manual", "samples_count": len(created_samples)},
        )
        return {
            "job_id": job_id,
            "image_id": image_id,
            "samples": created_samples,
            "overlap_warnings": multi_res["overlap_warnings"],
        }

    def segment_image_auto(
        self,
        image_id: str,
        *,
        model_id: Optional[str] = None,
        segmentation_engine: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Run Gemini Vision auto-detection + local refinement on `image_id`."""
        job = self.find_job_by_image_id(image_id)
        job_id = job["job_id"]
        img_rec = next(im for im in job["images"] if im["image_id"] == image_id)

        self._emit(job_id, "vision_started", {"image_id": image_id, "mode": "auto", "model": model_id or job.get("model")})

        adapter = load_site_adapter_for_url(img_rec["page_url"])
        layout_hint = adapter.layout_hint if adapter else None
        img_bytes = self.get_image_bytes(image_id)

        auto_res = detect_and_refine_samples_auto(
            img_bytes,
            model_id=model_id or job.get("model"),
            segmentation_engine=segmentation_engine,
            layout_hint=layout_hint,
            detector=self.detector,
            use_cache=use_cache,
        )

        job = self.get_job(job_id)
        if auto_res["status"] != "ok":
            for im in job["images"]:
                if im["image_id"] == image_id:
                    im["vision_mode"] = "auto"
                    im["model"] = auto_res.get("model", model_id or "")
                    im["prompt_version"] = auto_res.get("prompt_version", "v1")
                    im["status"] = "vision_failed"
            self._save_job(job)
            self._emit(
                job_id,
                "vision_failed",
                {
                    "image_id": image_id,
                    "error": auto_res.get("error", "Gemini Vision detection failed"),
                    "offer_manual_switch": True,
                },
            )
            return auto_res

        # Replace prior samples for this image
        job["samples"] = [s for s in job["samples"] if s["image_id"] != image_id]
        sheet_text = str(auto_res.get("sheet_text", ""))

        created_samples: List[Dict[str, Any]] = []
        for idx, item in enumerate(auto_res["samples"]):
            smp_id = f"smp_{image_id}_{idx + 1:02d}_{uuid.uuid4().hex[:4]}"
            rel_mask = f"masks/{smp_id}.png"
            rel_crop = f"crops/{smp_id}.png"
            (self._job_dir(job_id) / rel_mask).write_bytes(item["mask_png_bytes"])
            (self._job_dir(job_id) / rel_crop).write_bytes(item["crop_png_bytes"])

            meta = self._build_initial_metadata(
                img_rec,
                caption_text=item.get("caption_text", ""),
                inferred_topcoat=item.get("inferred_topcoat", "unknown"),
                inferred_undercoat=item.get("inferred_undercoat", "unknown"),
                sheet_text=sheet_text,
                slot_index_0based=idx,
                layout_hint=layout_hint,
            )

            reasons = (
                list(item["review_reasons"])
                + list(meta["classifier_review_reasons"])
                + [f"Ambiguous taxonomy field: {f}" for f in meta["ambiguous_fields"]]
            )
            needs_review = bool(item["needs_review"] or meta["classifier_needs_review"] or meta["ambiguous_fields"])
            status = "accepted" if (item["status"] == "accepted" and not needs_review) else "proposed"

            smp_rec = {
                "sample_tmp_id": smp_id,
                "image_id": image_id,
                "index": idx + 1,
                "prompts": {
                    "pos": [item["point_xy"]] + item.get("clean_points_xy", []),
                    "neg": [],
                    "box": item["box_xyxy"],
                },
                "engine": item["engine"],
                "mask_png": rel_mask,
                "crop_png": rel_crop,
                "crop_bbox_xyxy": list(item["crop_bbox_xyxy"]),
                "measured_srgb_hex": item["measured_srgb_hex"],
                "calibrated_optical_params": item["calibrated_optical_params"],
                "cross_check": item["cross_check"],
                "llm": {
                    "caption_text": item.get("caption_text", ""),
                    "shape": item.get("shape", "spoon"),
                    "inferred_topcoat": item.get("inferred_topcoat", "unknown"),
                    "inferred_undercoat": item.get("inferred_undercoat", "unknown"),
                    "confidence": item.get("confidence", 0.0),
                },
                "metadata": meta,
                "needs_review": needs_review,
                "review_reasons": reasons,
                "status": status,
                "sample_id": None,
            }
            job["samples"].append(smp_rec)
            self._sample_index[smp_id] = job_id
            created_samples.append(smp_rec)
            self._emit(job_id, "sample_ready", {"sample_tmp_id": smp_id, "image_id": image_id})

        for im in job["images"]:
            if im["image_id"] == image_id:
                im["vision_mode"] = "auto"
                im["model"] = auto_res["model"]
                im["prompt_version"] = auto_res["prompt_version"]
                im["status"] = "segmented"
        self._save_job(job)
        self._emit(
            job_id,
            "vision_done",
            {
                "image_id": image_id,
                "mode": "auto",
                "from_cache": auto_res["from_cache"],
                "samples_count": len(created_samples),
            },
        )
        return {
            "status": "ok",
            "job_id": job_id,
            "image_id": image_id,
            "from_cache": auto_res["from_cache"],
            "sheet_text": sheet_text,
            "layout_check": auto_res["layout_check"],
            "overlap_warnings": auto_res["overlap_warnings"],
            "samples": created_samples,
        }

    def update_sample(
        self,
        sample_tmp_id: str,
        patch_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update a sample's prompts (re-running local segmentation if requested), metadata,
        or review status (`accepted` / `rejected` / `proposed`).
        """
        job = self.find_job_by_sample_id(sample_tmp_id)
        job_id = job["job_id"]
        smp = next(s for s in job["samples"] if s["sample_tmp_id"] == sample_tmp_id)

        # Optional re-segmentation if prompts or rerun=True provided
        if patch_data.get("rerun") or "prompts" in patch_data:
            new_prompts = patch_data.get("prompts", smp["prompts"])
            pos_pts = [tuple(p) for p in new_prompts.get("pos", smp["prompts"]["pos"])]
            neg_pts = [tuple(p) for p in new_prompts.get("neg", smp["prompts"].get("neg", []))]
            box_val = tuple(new_prompts["box"]) if new_prompts.get("box") else None
            tol_val = new_prompts.get("oklab_tol", smp["prompts"].get("oklab_tol"))

            img_bytes = self.get_image_bytes(smp["image_id"])
            rgb_u8 = np.asarray(load_image_normalized_srgb(img_bytes).convert("RGB"), dtype=np.uint8)
            seg_res = segment_single_sample(
                rgb_u8,
                pos_pts,
                neg_points_xy=neg_pts,
                box_xyxy=box_val,  # type: ignore[arg-type]
                engine=patch_data.get("engine", smp.get("engine")),
                oklab_tol=tol_val,
                override_validation=bool(patch_data.get("override_validation", False)),
            )
            (self._job_dir(job_id) / smp["mask_png"]).write_bytes(seg_res["mask_png_bytes"])
            (self._job_dir(job_id) / smp["crop_png"]).write_bytes(seg_res["crop_png_bytes"])
            smp["prompts"] = {
                "pos": [list(p) for p in pos_pts],
                "neg": [list(p) for p in neg_pts],
                "box": list(box_val) if box_val else None,
                "oklab_tol": tol_val,
            }
            smp["engine"] = seg_res["engine"]
            smp["crop_bbox_xyxy"] = list(seg_res["crop_bbox_xyxy"])
            smp["measured_srgb_hex"] = seg_res["measured_srgb_hex"]
            smp["calibrated_optical_params"] = seg_res["calibrated_optical_params"]
            smp["validation"] = seg_res["validation"]

        if "metadata" in patch_data and isinstance(patch_data["metadata"], dict):
            meta = smp.setdefault("metadata", {})
            for k, v in patch_data["metadata"].items():
                meta[k] = v
            # If user explicitly chose values for previously ambiguous fields, clear them
            if "ambiguous_fields" in meta:
                remaining_amb = [f for f in meta["ambiguous_fields"] if f not in patch_data["metadata"]]
                meta["ambiguous_fields"] = remaining_amb

        if "needs_review" in patch_data:
            smp["needs_review"] = bool(patch_data["needs_review"])
        elif not smp.get("metadata", {}).get("ambiguous_fields") and patch_data.get("status") == "accepted":
            smp["needs_review"] = False

        if "status" in patch_data and patch_data["status"] in ("proposed", "accepted", "rejected"):
            smp["status"] = patch_data["status"]

        self._save_job(job)
        return smp

    def delete_sample(self, sample_tmp_id: str) -> bool:
        job = self.find_job_by_sample_id(sample_tmp_id)
        before = len(job["samples"])
        job["samples"] = [s for s in job["samples"] if s["sample_tmp_id"] != sample_tmp_id]
        self._sample_index.pop(sample_tmp_id, None)
        self._save_job(job)
        return len(job["samples"]) < before

    def ingest_job_samples(
        self,
        job_id: str,
        *,
        sample_tmp_ids: Optional[Sequence[str]] = None,
        overwrite: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Batch-ingest all `accepted` samples (or specified `sample_tmp_ids`) in `job_id`
        via `ingest_samples_batch()` (Section 7.5).
        Rejects samples that still have unresolved `ambiguous_fields` (Section 7.4).
        """
        job = self.get_job(job_id)
        target_ids = set(sample_tmp_ids) if sample_tmp_ids else None

        to_ingest: List[Dict[str, Any]] = []
        sample_refs: List[Dict[str, Any]] = []

        for smp in job["samples"]:
            if target_ids is not None:
                if smp["sample_tmp_id"] not in target_ids:
                    continue
            else:
                if smp.get("status") != "accepted":
                    continue

            meta = smp.get("metadata", {})
            amb = meta.get("ambiguous_fields") or []
            if amb:
                raise ValueError(
                    f"Sample {smp['sample_tmp_id']} has unresolved ambiguous taxonomy fields {amb}; "
                    "choose a value in the Review & Metadata form before ingesting."
                )

            crop_bytes = (self._job_dir(job_id) / smp["crop_png"]).read_bytes()
            to_ingest.append(
                {
                    "image_bytes": crop_bytes,
                    "brand_id": meta.get("brand_id", "auto"),
                    "solvent_type": meta.get("solvent_type", "auto"),
                    "paint_role": meta.get("paint_role", "auto"),
                    "product_line": meta.get("product_line", "auto"),
                    "paint_code": meta.get("paint_code", "UNKNOWN"),
                    "paint_name_en": meta.get("paint_name_en", "Sample Paint"),
                    "paint_name_ko": meta.get("paint_name_ko", "샘플 도료"),
                    "surfacer_code": meta.get("surfacer_code", "none"),
                    "surfacer_grit": meta.get("surfacer_grit", "none"),
                    "undercoat_code": meta.get("undercoat_code", "none"),
                    "topcoat_code": meta.get("topcoat_code", "none"),
                    "final_finish": meta.get("final_finish", "auto"),
                    "coats": float(meta.get("coats", 2.0)),
                    "thinner_and_ratio": meta.get("thinner_and_ratio", "standard"),
                    "substrate_color": meta.get("substrate_color", "white"),
                    "lighting_env": meta.get("lighting_env", "std"),
                    "source_type": meta.get("source_type", "web_community_spoon"),
                    "source_url": meta.get("source_url", "local_upload"),
                    "image_url": meta.get("image_url", ""),
                    "notes": meta.get("notes", ""),
                    "overwrite": overwrite,
                }
            )
            sample_refs.append(smp)

        if not to_ingest:
            return []

        ingested_rows = ingest_samples_batch(to_ingest, root_dir=self.sample_root_dir)
        for smp_ref, row in zip(sample_refs, ingested_rows):
            smp_ref["status"] = "ingested"
            smp_ref["sample_id"] = row["sample_id"]
            smp_ref["file_rel_path"] = row["file_rel_path"]

        for im in job["images"]:
            img_smps = [s for s in job["samples"] if s["image_id"] == im["image_id"]]
            if img_smps and all(s["status"] in ("ingested", "rejected") for s in img_smps):
                im["status"] = "done"

        self._save_job(job)
        self._emit(
            job_id,
            "ingested",
            {
                "count": len(ingested_rows),
                "sample_ids": [r["sample_id"] for r in ingested_rows],
            },
        )
        return ingested_rows
