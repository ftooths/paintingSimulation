"""HTML image candidate extractors and data-only site adapters."""
from __future__ import annotations

from paint_sim.collector.extractors.adapters import SiteAdapter, load_site_adapter_for_url
from paint_sim.collector.extractors.generic import (
    ExtractedImageCandidate,
    extract_image_candidates_from_html,
    extract_post_links_from_listing,
    filter_downloaded_image_candidate,
)

__all__ = [
    "SiteAdapter",
    "load_site_adapter_for_url",
    "ExtractedImageCandidate",
    "extract_image_candidates_from_html",
    "extract_post_links_from_listing",
    "filter_downloaded_image_candidate",
]
