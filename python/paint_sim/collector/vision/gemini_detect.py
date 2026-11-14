"""Gemini Vision structured sample detection client, dual-auth builder, disk cache, and secret manager (Sections 4, 6, 11.1)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from paint_sim.collector.sources import load_collector_settings
from paint_sim.collector.vision.manual_segment import segment_single_sample
from paint_sim.collector.vision.refine import cross_check_mask_against_box, find_overlapping_mask_pairs, validate_layout_hint
from paint_sim.config.prompt_loader import PromptConfigLoader
from paint_sim.db.sample_ingestor import load_image_normalized_srgb

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LLM_CACHE_DIR = _REPO_ROOT / "sample" / "collector_jobs" / "_llm_cache"

# Sampling keys prohibited on models that reject temperature/top_p/top_k/seed (Section 4.1)
UNSUPPORTED_SAMPLING_KEYS = ("temperature", "top_p", "topP", "top_k", "topK", "seed")


# ============================================================================
# Section 11.1 Verified Pydantic Structured Detection Schema
# ============================================================================


class DetectedSampleItem(BaseModel):
    index: int = Field(..., ge=1)
    box_2d: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Tight 2D bounding box [ymin, xmin, ymax, xmax] normalized to 0..1000",
    )
    point_2d: List[int] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Interior foreground seed point [y, x] normalized to 0..1000",
    )
    clean_points_2d: List[List[int]] = Field(
        default_factory=list,
        description="1-3 diffuse paint points [[y, x], ...] avoiding specular glare, shadows, and text (0..1000)",
    )
    shape: Literal["spoon", "plaque", "chip", "bottle_cap", "swatch", "other"] = "spoon"
    caption_text: str = ""
    inferred_topcoat: Literal["none", "gloss", "semi_gloss", "matte", "unknown"] = "unknown"
    inferred_undercoat: Literal["gray", "white", "black", "pink", "unknown"] = "unknown"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class SampleSheetDetectionResponse(BaseModel):
    image_width_hint: int = 0
    samples: List[DetectedSampleItem]
    sheet_text: str = ""
    notes: str = ""


# ============================================================================
# Section 6: Out-of-Repo Secrets Manager & Log Redaction
# ============================================================================


def get_secrets_path() -> Path:
    """
    Resolve out-of-repo secrets file path (Section 6):
    Uses `PAINTSIM_SECRETS` env var when set, otherwise `%APPDATA%\\paintingSimulation\\secrets.json`.
    """
    env_override = os.environ.get("PAINTSIM_SECRETS", "").strip()
    if env_override:
        return Path(env_override).resolve()
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "paintingSimulation" / "secrets.json"
    return Path.home() / ".config" / "paintingSimulation" / "secrets.json"


def _restrict_file_acl(path: Path) -> None:
    """Restrict file permissions to the current user where supported."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    if os.name == "nt":
        username = os.environ.get("USERNAME", "")
        if username:
            try:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(R,W)"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                pass


_SECRETS_CACHE: Dict[str, Any] = {"path": None, "mtime": -1.0, "data": {}, "sa_tokens": []}


def load_secrets(secrets_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load credentials from out-of-repo `secrets.json` with mtime caching (never exposed directly over HTTP)."""
    path = Path(secrets_path or get_secrets_path())
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
        if _SECRETS_CACHE["path"] == path and _SECRETS_CACHE["mtime"] == mtime:
            return dict(_SECRETS_CACHE["data"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        sa_tokens: List[Tuple[str, str]] = []
        sa_path = data.get("service_account_json_path")
        if sa_path and Path(sa_path).is_file():
            try:
                sa_data = json.loads(Path(sa_path).read_text(encoding="utf-8"))
                pk = str(sa_data.get("private_key", "")).strip()
                if pk:
                    sa_tokens.append((pk, "[REDACTED_PRIVATE_KEY]"))
                pk_id = str(sa_data.get("private_key_id", "")).strip()
                if pk_id and len(pk_id) >= 6:
                    sa_tokens.append((pk_id, "[REDACTED_KEY_ID]"))
            except Exception:
                pass
        _SECRETS_CACHE["path"] = path
        _SECRETS_CACHE["mtime"] = mtime
        _SECRETS_CACHE["data"] = data
        _SECRETS_CACHE["sa_tokens"] = sa_tokens
        return dict(data)
    except Exception:
        return {}


def save_secrets(
    payload: Dict[str, Any],
    secrets_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Save Gemini API key or Vertex AI service-account JSON outside the repository (Section 6).
    Returns ONLY public status (`auth_mode`, `has_api_key`, `client_email`, `project_id`, `location`).
    """
    path = Path(secrets_path or get_secrets_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_secrets(path)

    if "auth_mode" in payload and payload["auth_mode"] in ("api_key", "vertex_service_account"):
        current["auth_mode"] = payload["auth_mode"]
    if "api_key" in payload and payload["api_key"] is not None:
        raw_key = str(payload["api_key"]).strip()
        if raw_key:
            current["api_key"] = raw_key
        elif payload.get("clear_api_key"):
            current.pop("api_key", None)

    if "location" in payload and payload["location"]:
        current["location"] = str(payload["location"]).strip()
    if "project_id" in payload and payload["project_id"] is not None:
        current["project_id"] = str(payload["project_id"]).strip()

    # Service account JSON can be passed as raw JSON dict/string or file path
    sa_json_content = payload.get("service_account_json")
    sa_src_path = payload.get("service_account_json_path")
    if sa_json_content:
        if isinstance(sa_json_content, str):
            sa_dict = json.loads(sa_json_content)
        elif isinstance(sa_json_content, dict):
            sa_dict = sa_json_content
        else:
            raise ValueError("Invalid service_account_json format.")
        dest_sa = path.parent / "vertex.serviceaccount.json"
        dest_sa.write_text(json.dumps(sa_dict, indent=2), encoding="utf-8")
        _restrict_file_acl(dest_sa)
        current["service_account_json_path"] = str(dest_sa)
        current["client_email"] = str(sa_dict.get("client_email", ""))
        if not current.get("project_id") and sa_dict.get("project_id"):
            current["project_id"] = str(sa_dict["project_id"])
    elif sa_src_path:
        src = Path(str(sa_src_path))
        if not src.is_file():
            raise ValueError("Provided service_account_json_path does not exist.")
        sa_dict = json.loads(src.read_text(encoding="utf-8"))
        dest_sa = path.parent / "vertex.serviceaccount.json"
        if src.resolve() != dest_sa.resolve():
            shutil.copyfile(src, dest_sa)
        _restrict_file_acl(dest_sa)
        current["service_account_json_path"] = str(dest_sa)
        current["client_email"] = str(sa_dict.get("client_email", ""))
        if not current.get("project_id") and sa_dict.get("project_id"):
            current["project_id"] = str(sa_dict["project_id"])

    fd, tmp_name = tempfile.mkstemp(prefix=".secrets_", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
        _restrict_file_acl(path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    return get_public_secrets_status(path)


def get_public_secrets_status(secrets_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return safe non-secret metadata for UI display (never echoes API key or private key)."""
    sec = load_secrets(secrets_path)
    return {
        "auth_mode": sec.get("auth_mode", "api_key"),
        "has_api_key": bool(sec.get("api_key")),
        "has_service_account": bool(
            sec.get("service_account_json_path") and Path(sec["service_account_json_path"]).is_file()
        ),
        "client_email": sec.get("client_email", ""),
        "project_id": sec.get("project_id", ""),
        "location": sec.get("location", "us-central1"),
    }


def redact_secrets(text: str, secrets_path: Optional[Path] = None) -> str:
    """Redact any stored API key or private key material from logs or exception messages (Section 6)."""
    if not text:
        return ""
    out = str(text)
    sec = load_secrets(secrets_path)
    api_key = str(sec.get("api_key", "")).strip()
    if api_key and len(api_key) >= 4:
        out = out.replace(api_key, "[REDACTED_API_KEY]")
    for raw_tok, replacement in _SECRETS_CACHE.get("sa_tokens", []):
        out = out.replace(raw_tok, replacement)
    return out


# ============================================================================
# Section 11.1 Dual-Auth Client & Coordinate Helpers
# ============================================================================


def build_genai_client(
    auth_mode: Literal["api_key", "vertex_service_account"],
    *,
    api_key: Optional[str] = None,
    service_account_json_path: Optional[Path] = None,
    project_id: Optional[str] = None,
    location: str = "us-central1",
):
    """
    Builds a google-genai Client for either Gemini Developer API Key or Vertex AI JSON key.
    Official docs:
      - https://googleapis.github.io/python-genai/genai.html
      - https://www.mintlify.com/googleapis/python-genai/concepts/authentication
    """
    from google import genai

    if auth_mode == "api_key":
        if not api_key:
            raise ValueError("Gemini API key is required when auth_mode='api_key'.")
        return genai.Client(api_key=api_key)

    if auth_mode == "vertex_service_account":
        if not service_account_json_path or not Path(service_account_json_path).is_file():
            raise ValueError("Valid Vertex AI service-account JSON file path is required.")
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            str(service_account_json_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        resolved_project = project_id or creds.project_id
        if not resolved_project:
            raise ValueError("project_id missing from both arguments and service-account JSON.")
        return genai.Client(
            vertexai=True,
            project=resolved_project,
            location=location,
            credentials=creds,
        )

    raise ValueError(f"Unsupported auth_mode: {auth_mode!r}")


def build_genai_client_from_secrets(secrets_path: Optional[Path] = None):
    """Instantiate `genai.Client` from stored secrets or environment variables."""
    sec = load_secrets(secrets_path)
    auth_mode = sec.get("auth_mode", "api_key")
    if auth_mode == "vertex_service_account":
        sa_path = sec.get("service_account_json_path")
        return build_genai_client(
            "vertex_service_account",
            service_account_json_path=Path(sa_path) if sa_path else None,
            project_id=sec.get("project_id"),
            location=sec.get("location", "us-central1"),
        )
    api_key = sec.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    return build_genai_client("api_key", api_key=api_key)


def norm1000_box_to_px(box_2d: List[int], width: int, height: int) -> tuple[int, int, int, int]:
    """
    Converts [ymin, xmin, ymax, xmax] in 0..1000 to clamped pixel (x0, y0, x1, y1).
    Source: https://ai.google.dev/gemini-api/docs/generate-content/image-understanding#object-detection
    """
    ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box_2d]
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    x0 = max(0, min(width - 1, int(round(xmin * width / 1000.0))))
    y0 = max(0, min(height - 1, int(round(ymin * height / 1000.0))))
    x1 = max(x0 + 1, min(width, int(round(xmax * width / 1000.0))))
    y1 = max(y0 + 1, min(height, int(round(ymax * height / 1000.0))))
    return x0, y0, x1, y1


def norm1000_point_to_px(pt_2d: List[int], width: int, height: int) -> tuple[int, int]:
    """Converts [y, x] in 0..1000 to clamped pixel (x, y)."""
    y_norm = max(0, min(1000, int(pt_2d[0])))
    x_norm = max(0, min(1000, int(pt_2d[1])))
    x = max(0, min(width - 1, int(round(x_norm * width / 1000.0))))
    y = max(0, min(height - 1, int(round(y_norm * height / 1000.0))))
    return x, y


def compute_box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute 2D bounding-box IoU for `(x0, y0, x1, y1)` pixel boxes."""
    ix0 = max(box_a[0], box_b[0])
    iy0 = max(box_a[1], box_b[1])
    ix1 = min(box_a[2], box_b[2])
    iy1 = min(box_a[3], box_b[3])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter / float(area_a + area_b - inter))


def list_available_vision_models(
    secrets_path: Optional[Path] = None,
    mock_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Populate model dropdown from `client.models.list()`, filtered to vision-capable Gemini models,
    merged with defaults from `config/collector_settings.json` (Section 4.1).
    Source: https://github.com/googleapis/python-genai#list-base-models
    """
    cfg = load_collector_settings()
    defaults = list(cfg.get("available_vision_models", ["gemini-3.8-flash", "gemini-4-argon"]))
    discovered: List[str] = []

    if mock_models is not None:
        discovered.extend(mock_models)
    else:
        try:
            client = build_genai_client_from_secrets(secrets_path)
            for m in client.models.list():
                name = str(getattr(m, "name", "") or "")
                short_name = name.split("/")[-1] if "/" in name else name
                low = short_name.lower()
                if "gemini" in low and not any(skip in low for skip in ("embedding", "text-embedding", "aqa")):
                    discovered.append(short_name)
        except Exception as exc:
            logger.debug("Model listing fell back to config defaults: %s", redact_secrets(str(exc), secrets_path))

    merged: List[str] = []
    for m_id in defaults + discovered:
        if m_id and m_id not in merged:
            merged.append(m_id)
    return merged


def test_genai_connection(
    model_id: Optional[str] = None,
    secrets_path: Optional[Path] = None,
    mock_transport: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Send a minimal ping request and return `{ok, model, latency_ms}` (Section 4.1)."""
    cfg = load_collector_settings()
    chosen_model = (model_id or cfg.get("default_vision_model", "gemini-3.8-flash")).strip()
    t0 = time.perf_counter()
    try:
        if mock_transport is not None:
            reply = mock_transport(chosen_model)
        else:
            client = build_genai_client_from_secrets(secrets_path)
            resp = client.models.generate_content(
                model=chosen_model,
                contents="Reply with OK.",
            )
            reply = getattr(resp, "text", "OK") or "OK"
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "ok": True,
            "model": chosen_model,
            "latency_ms": latency_ms,
            "reply": str(reply).strip()[:64],
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        safe_err = redact_secrets(str(exc), secrets_path)
        return {
            "ok": False,
            "model": chosen_model,
            "latency_ms": latency_ms,
            "error": safe_err,
        }


class GeminiSampleDetector:
    """
    Gemini Vision sample sheet detector with disk caching, coordinate conversion,
    IoU deduplication, and local segmentation refinement (Sections 4.2, 4.3).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        prompt_loader: Optional[PromptConfigLoader] = None,
        secrets_path: Optional[Path] = None,
        mock_transport: Optional[Callable[[str, str, str, bytes], Dict[str, Any] | str]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_LLM_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_loader = prompt_loader or PromptConfigLoader()
        self.secrets_path = secrets_path
        self.mock_transport = mock_transport
        self.network_call_count: int = 0

    @staticmethod
    def _cache_file_name(image_sha256: str, model_id: str, prompt_version: str) -> str:
        safe_model = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
        safe_ver = "".join(c if c.isalnum() or c in "-_." else "_" for c in prompt_version)
        return f"{image_sha256}_{safe_model}_{safe_ver}.json"

    def detect_samples_raw(
        self,
        image_bytes: bytes,
        model_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Gemini Vision detection (or load from `(image_sha256, model_id, prompt_version)` cache).
        On API error or timeout, logs the redacted error and returns `status='vision_failed'` with zero fake boxes.
        """
        cfg = load_collector_settings()
        prompts_cfg = self.prompt_loader.get_config()
        det_cfg = prompts_cfg.get("collector_sample_detection", {})
        prompt_version = str(det_cfg.get("prompt_version", "v1"))
        chosen_model = (
            model_id
            or det_cfg.get("model_id")
            or cfg.get("default_vision_model", "gemini-3.8-flash")
        ).strip()
        sys_inst = str(
            det_cfg.get(
                "system_instruction",
                "Detect all painted samples in the image and return JSON matching SampleSheetDetectionResponse.",
            )
        )
        user_prompt = str(
            det_cfg.get(
                "user_prompt",
                "Detect all painted samples with box_2d [ymin, xmin, ymax, xmax] and point_2d [y, x] in 0..1000.",
            )
        )

        img_sha = hashlib.sha256(image_bytes).hexdigest()
        cache_path = self.cache_dir / self._cache_file_name(img_sha, chosen_model, prompt_version)

        if use_cache and cache_path.is_file():
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = SampleSheetDetectionResponse.model_validate(cached_data["response"])
                return {
                    "status": "ok",
                    "from_cache": True,
                    "model": chosen_model,
                    "prompt_version": prompt_version,
                    "parsed": parsed,
                }
            except Exception as exc:
                logger.warning("Corrupt LLM cache entry %s (%s); re-running.", cache_path.name, exc)

        try:
            if self.mock_transport is not None:
                self.network_call_count += 1
                raw_out = self.mock_transport(chosen_model, sys_inst, user_prompt, image_bytes)
                if isinstance(raw_out, str):
                    raw_dict = json.loads(raw_out)
                else:
                    raw_dict = raw_out
            else:
                # Official google-genai structured output docs:
                # https://github.com/googleapis/python-genai#json-response-schema
                from google.genai import types

                client = build_genai_client_from_secrets(self.secrets_path)
                self.network_call_count += 1
                # Ensure no unsupported sampling keys are sent (Section 4.1)
                gen_kwargs: Dict[str, Any] = {
                    "system_instruction": sys_inst,
                    "response_mime_type": "application/json",
                    "response_json_schema": SampleSheetDetectionResponse.model_json_schema(),
                }
                for bad_k in UNSUPPORTED_SAMPLING_KEYS:
                    gen_kwargs.pop(bad_k, None)

                resp = client.models.generate_content(
                    model=chosen_model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        user_prompt,
                    ],
                    config=types.GenerateContentConfig(**gen_kwargs),
                )
                raw_dict = json.loads(resp.text)

            parsed = SampleSheetDetectionResponse.model_validate(raw_dict)
            if use_cache:
                cache_path.write_text(
                    json.dumps(
                        {
                            "image_sha256": img_sha,
                            "model": chosen_model,
                            "prompt_version": prompt_version,
                            "response": parsed.model_dump(),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            return {
                "status": "ok",
                "from_cache": False,
                "model": chosen_model,
                "prompt_version": prompt_version,
                "parsed": parsed,
            }
        except Exception as exc:
            safe_err = redact_secrets(str(exc), self.secrets_path)
            logger.error("Gemini vision detection failed (%s); marking vision_failed.", safe_err)
            return {
                "status": "vision_failed",
                "from_cache": False,
                "model": chosen_model,
                "prompt_version": prompt_version,
                "error": safe_err,
                "samples": [],
            }


def detect_and_refine_samples_auto(
    image_bytes: bytes,
    *,
    model_id: Optional[str] = None,
    segmentation_engine: Optional[str] = None,
    layout_hint: Optional[Dict[str, Any]] = None,
    detector: Optional[GeminiSampleDetector] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Full Section 4.2–4.3 Auto Vision pipeline per image:
      1. Calls Gemini Vision (`SampleSheetDetectionResponse`) with `(sha256, model, prompt_version)` disk cache.
      2. Converts `0..1000` `[ymin, xmin, ymax, xmax]` and `[y, x]` coordinates into clamped image pixels.
      3. Drops boxes smaller than `0.2%` of image area and deduplicates boxes with `IoU > 0.50`.
      4. Runs the local segmentation & lossless refinement engine for each sample using `box_2d` + `point_2d` + `clean_points_2d`.
      5. Cross-checks each refined mask against its LLM box (`< 40%` box coverage or `> 15%` outside spill -> `needs_review`).
      6. Validates sample count and order against optional `layout_hint`.
    """
    cfg = load_collector_settings()
    det = detector or GeminiSampleDetector()
    raw_res = det.detect_samples_raw(image_bytes, model_id=model_id, use_cache=use_cache)
    if raw_res["status"] != "ok":
        return raw_res

    img = load_image_normalized_srgb(image_bytes)
    rgb_u8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb_u8.shape[:2]
    img_area = max(1, h * w)

    parsed: SampleSheetDetectionResponse = raw_res["parsed"]
    min_box_area_ratio = float(cfg.get("min_mask_area_ratio", 0.002))
    dedup_iou_thresh = float(cfg.get("box_dedup_iou_threshold", 0.50))
    min_cov = float(cfg.get("box_min_coverage_ratio", 0.40))
    max_spill = float(cfg.get("box_max_spill_ratio", 0.15))
    auto_accept_conf = cfg.get("auto_accept_confidence")

    # Convert coordinates, filter small boxes (< 0.2%), and deduplicate by IoU > 0.50
    kept_items: List[Tuple[DetectedSampleItem, Tuple[int, int, int, int], Tuple[int, int], List[Tuple[int, int]]]] = []
    sorted_candidates = sorted(parsed.samples, key=lambda s: (-float(s.confidence), int(s.index)))

    for item in sorted_candidates:
        box_px = norm1000_box_to_px(item.box_2d, w, h)
        box_area = (box_px[2] - box_px[0]) * (box_px[3] - box_px[1])
        if (box_area / float(img_area)) < min_box_area_ratio:
            continue
        if any(compute_box_iou(box_px, existing_box) > dedup_iou_thresh for _, existing_box, _, _ in kept_items):
            continue
        pt_px = norm1000_point_to_px(item.point_2d, w, h)
        clean_px = [norm1000_point_to_px(cp, w, h) for cp in item.clean_points_2d if len(cp) >= 2]
        kept_items.append((item, box_px, pt_px, clean_px))

    # Re-order kept items by original index / left-to-right position
    kept_items.sort(key=lambda t: (t[0].index, t[1][0]))

    layout_check = validate_layout_hint([t[1] for t in kept_items], layout_hint)

    refined_samples: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []

    for new_idx, (item, box_px, pt_px, clean_px) in enumerate(kept_items, start=1):
        pos_seeds = [pt_px] + [cp for cp in clean_px if cp != pt_px]
        seg_res = segment_single_sample(
            rgb_u8,
            pos_seeds,
            neg_points_xy=[],
            box_xyxy=box_px,
            clean_points_xy=clean_px or [pt_px],
            engine=segmentation_engine,
        )
        cross = cross_check_mask_against_box(
            seg_res["refined_mask_bool"],
            box_px,
            min_box_coverage=min_cov,
            max_outside_spill=max_spill,
        )

        reasons = list(seg_res["review_reasons"]) + list(cross["reasons"]) + list(layout_check["reasons"])
        needs_review = bool(seg_res["needs_review"] or cross["needs_review"] or layout_check["needs_review"])
        if auto_accept_conf is None or float(item.confidence) < float(auto_accept_conf):
            needs_review = True

        status = "accepted" if (not needs_review and auto_accept_conf is not None and float(item.confidence) >= float(auto_accept_conf)) else "proposed"

        refined_samples.append(
            {
                "index": new_idx,
                "box_xyxy": list(box_px),
                "point_xy": list(pt_px),
                "clean_points_xy": [list(cp) for cp in clean_px],
                "shape": item.shape,
                "caption_text": item.caption_text,
                "inferred_topcoat": item.inferred_topcoat,
                "inferred_undercoat": item.inferred_undercoat,
                "confidence": round(float(item.confidence), 4),
                "engine": seg_res["engine"],
                "refined_mask_bool": seg_res["refined_mask_bool"],
                "crop_png_bytes": seg_res["crop_png_bytes"],
                "mask_png_bytes": seg_res["mask_png_bytes"],
                "crop_bbox_xyxy": seg_res["crop_bbox_xyxy"],
                "max_rgb_diff": seg_res["max_rgb_diff"],
                "measured_srgb_hex": seg_res["measured_srgb_hex"],
                "calibrated_optical_params": seg_res["calibrated_optical_params"],
                "cross_check": cross,
                "needs_review": needs_review,
                "review_reasons": reasons,
                "status": status,
            }
        )
        masks.append(seg_res["refined_mask_bool"])

    overlaps = find_overlapping_mask_pairs(
        masks,
        iou_threshold=float(cfg.get("mask_overlap_iou_warn", 0.30)),
        bboxes_xyxy=[tuple(s["crop_bbox_xyxy"]) for s in refined_samples],  # type: ignore[misc]
    )
    for ov in overlaps:
        for t_idx in (ov["index_a"] - 1, ov["index_b"] - 1):
            if 0 <= t_idx < len(refined_samples):
                refined_samples[t_idx]["needs_review"] = True
                refined_samples[t_idx]["status"] = "proposed"
                refined_samples[t_idx]["review_reasons"].append(ov["message"])

    return {
        "status": "ok",
        "from_cache": raw_res["from_cache"],
        "model": raw_res["model"],
        "prompt_version": raw_res["prompt_version"],
        "sheet_text": parsed.sheet_text,
        "notes": parsed.notes,
        "layout_check": layout_check,
        "overlap_warnings": overlaps,
        "samples": refined_samples,
    }
