"""Source registry for the multi-site paint sample collector (persisted in config/collector_sources.json)."""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES_PATH = _REPO_ROOT / "config" / "collector_sources.json"
DEFAULT_SETTINGS_PATH = _REPO_ROOT / "config" / "collector_settings.json"

ALLOWED_DEFAULT_KEYS = (
    "brand_id",
    "solvent_type",
    "paint_role",
    "product_line",
    "paint_code",
    "paint_name_en",
    "paint_name_ko",
    "source_type",
    "surfacer_code",
    "surfacer_grit",
    "undercoat_code",
    "topcoat_code",
    "substrate_color",
    "lighting_env",
    "coats",
    "thinner_and_ratio",
)


def _is_blocked_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if an IP address is non-public (loopback, private, link-local, multicast, reserved, unspecified)."""
    if (
        ip_obj.is_loopback
        or ip_obj.is_private
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        return True
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return _is_blocked_ip(ip_obj.ipv4_mapped)
    return False


def _validate_literal_host(hostname: Optional[str]) -> str:
    """Reject empty, localhost/.local, and non-public literal IPv4/IPv6 hosts."""
    if not hostname:
        raise ValueError("Blocked non-public URL host")
    host = hostname.strip().rstrip(".").lower()
    if not host or host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("Blocked non-public URL host")
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        return host
    if _is_blocked_ip(ip_obj):
        raise ValueError("Blocked non-public URL host")
    return host


def normalize_url(url: str) -> str:
    """Normalize a HTTP/HTTPS URL by stripping fragments, validating literal host/IP safety, and standardizing case."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("URL must not be empty.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r} in {raw!r}; only http/https allowed.")
    if not parsed.netloc:
        raise ValueError(f"URL missing host/netloc: {raw!r}")
    _validate_literal_host(parsed.hostname)
    clean = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    )
    return urlunparse(clean)


def validate_safe_public_url(url: str, *, resolve_dns: bool = False) -> str:
    """
    Validate that `url` uses http/https and points to a public host (SSRF protection).
    When `resolve_dns=True`, resolves DNS via `socket.getaddrinfo` and rejects any non-public resolved IP.
    """
    norm = normalize_url(url)
    parsed = urlparse(norm)
    host = _validate_literal_host(parsed.hostname)
    if resolve_dns:
        # Skip DNS lookup if host is already a validated literal IP
        is_literal_ip = False
        try:
            ipaddress.ip_address(host)
            is_literal_ip = True
        except ValueError:
            is_literal_ip = False
        if not is_literal_ip:
            infos = socket.getaddrinfo(host, None)
            for _fam, _stype, _proto, _canon, sockaddr in infos:
                ip_str = str(sockaddr[0]).split("%")[0]
                ip_obj = ipaddress.ip_address(ip_str)
                if _is_blocked_ip(ip_obj):
                    raise ValueError("Blocked non-public URL host")
    return norm


def sanitize_defaults(defaults: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter per-URL default metadata to allowed keys with non-empty values."""
    if not defaults:
        return {}
    cleaned: Dict[str, Any] = {}
    for k, v in defaults.items():
        if k in ALLOWED_DEFAULT_KEYS and v is not None and str(v).strip() != "":
            cleaned[k] = v
    return cleaned


_SETTINGS_CACHE: Dict[str, Dict[str, Any]] = {}


def load_collector_settings(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load non-secret collector settings from `config/collector_settings.json` with mtime caching."""
    path = Path(settings_path or DEFAULT_SETTINGS_PATH)
    if not path.exists():
        return {
            "default_vision_model": "gemini-3.8-flash",
            "fallback_vision_model": "gemini-4-argon",
            "segmentation_engine": "grabcut",
            "oklab_flood_tolerance": 0.06,
            "grabcut_iterations": 4,
            "min_image_px": 300,
            "min_mask_area_ratio": 0.002,
            "max_mask_area_ratio": 0.60,
            "max_touched_borders": 2,
            "mask_overlap_iou_warn": 0.30,
            "box_dedup_iou_threshold": 0.50,
            "box_min_coverage_ratio": 0.40,
            "box_max_spill_ratio": 0.15,
            "auto_accept_confidence": None,
            "rate_limit_requests_per_sec": 1.0,
            "max_payload_bytes": 26214400,
            "request_timeout_sec": 30.0,
            "max_retries": 2,
            "respect_robots_txt": True,
            "user_agent": "PaintSimSampleCollector/1.0 (+http://127.0.0.1:8765; local-research)",
        }
    cache_key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
        cached = _SETTINGS_CACHE.get(cache_key)
        if cached and cached.get("mtime") == mtime and cached.get("data"):
            return dict(cached["data"])
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _SETTINGS_CACHE[cache_key] = {"mtime": mtime, "data": loaded}
        return dict(loaded)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def save_collector_settings(new_settings: Dict[str, Any], settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Atomically save non-secret collector settings to `config/collector_settings.json`."""
    path = Path(settings_path or DEFAULT_SETTINGS_PATH)
    current = load_collector_settings(path)
    # Never allow secret keys to be written into collector_settings.json
    forbidden = {"api_key", "gemini_api_key", "service_account_json", "private_key", "credentials"}
    for k, v in new_settings.items():
        if k.lower() in forbidden:
            continue
        current[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".settings_", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    _SETTINGS_CACHE.pop(str(path.resolve()), None)
    return current


class SourceRegistry:
    """Manages user-supplied source URLs and their default metadata in `config/collector_sources.json`."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = Path(storage_path or DEFAULT_SOURCES_PATH)

    def list_sources(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("sources", [])) if isinstance(data, dict) else []
        except Exception:
            return []

    def _save(self, sources: List[Dict[str, Any]]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".sources_", suffix=".json.tmp", dir=str(self.storage_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"sources": sources}, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.storage_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def add_source(self, url: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        norm = normalize_url(url)
        clean_defs = sanitize_defaults(defaults)
        sources = self.list_sources()
        for existing in sources:
            if existing.get("url") == norm:
                existing["defaults"] = {**existing.get("defaults", {}), **clean_defs}
                existing["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save(sources)
                return existing
        entry = {
            "url": norm,
            "defaults": clean_defs,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        sources.append(entry)
        self._save(sources)
        return entry

    def remove_source(self, url: str) -> bool:
        norm = normalize_url(url)
        sources = self.list_sources()
        filtered = [s for s in sources if s.get("url") != norm]
        if len(filtered) == len(sources):
            return False
        self._save(filtered)
        return True
