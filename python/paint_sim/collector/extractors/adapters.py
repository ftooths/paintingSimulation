"""Data-only per-site adapter loader reading `config/site_adapters/*.json` (Section 2.3)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ADAPTERS_DIR = _REPO_ROOT / "config" / "site_adapters"


@dataclass
class SiteAdapter:
    host: str
    content_selector: str = "article, main, .entry-content"
    exclude_selectors: List[str] = field(default_factory=list)
    image_attr_priority: List[str] = field(
        default_factory=lambda: [
            "data-orig-file",
            "data-large-file",
            "data-srcset",
            "srcset",
            "data-src",
            "data-lazy-src",
            "data-original",
            "src",
        ]
    )
    min_image_px: int = 300
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    layout_hint: Optional[Dict[str, Any]] = None
    listing_url: Optional[str] = None
    post_link_selector: Optional[str] = None
    max_posts: int = 5


_ADAPTERS_CACHE: Dict[str, Any] = {"dir": None, "mtime": -1.0, "adapters": {}}


def load_all_site_adapters(adapters_dir: Optional[Path] = None) -> Dict[str, SiteAdapter]:
    """Load all JSON site adapter definitions from `config/site_adapters/*.json` with mtime caching."""
    folder = Path(adapters_dir or DEFAULT_ADAPTERS_DIR)
    if not folder.exists():
        return {}
    json_files = sorted(folder.glob("*.json"))
    max_mtime = max((f.stat().st_mtime for f in json_files), default=0.0)
    if _ADAPTERS_CACHE["dir"] == folder and _ADAPTERS_CACHE["mtime"] == max_mtime:
        return dict(_ADAPTERS_CACHE["adapters"])

    adapters: Dict[str, SiteAdapter] = {}
    for json_file in json_files:
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            host = str(raw.get("host", "")).strip().lower()
            if not host:
                continue
            adapters[host] = SiteAdapter(
                host=host,
                content_selector=str(raw.get("content_selector", "article, main, .entry-content")),
                exclude_selectors=list(raw.get("exclude_selectors", [])),
                image_attr_priority=list(
                    raw.get(
                        "image_attr_priority",
                        [
                            "data-orig-file",
                            "data-large-file",
                            "data-srcset",
                            "srcset",
                            "data-src",
                            "data-lazy-src",
                            "data-original",
                            "src",
                        ],
                    )
                ),
                min_image_px=int(raw.get("min_image_px", 300)),
                default_metadata=dict(raw.get("default_metadata", {})),
                layout_hint=raw.get("layout_hint"),
                listing_url=raw.get("listing_url"),
                post_link_selector=raw.get("post_link_selector"),
                max_posts=int(raw.get("max_posts", 5)),
            )
        except Exception:
            continue
    _ADAPTERS_CACHE["dir"] = folder
    _ADAPTERS_CACHE["mtime"] = max_mtime
    _ADAPTERS_CACHE["adapters"] = adapters
    return dict(adapters)


def load_site_adapter_for_url(url: str, adapters_dir: Optional[Path] = None) -> Optional[SiteAdapter]:
    """Return a matching `SiteAdapter` for `url` host (with or without leading `www.`), or None."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    if not host:
        return None
    adapters = load_all_site_adapters(adapters_dir)
    if host in adapters:
        return adapters[host]
    bare_host = host[4:] if host.startswith("www.") else f"www.{host}"
    return adapters.get(bare_host)
