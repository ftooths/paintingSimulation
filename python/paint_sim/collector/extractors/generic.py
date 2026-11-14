"""Generic HTML image candidate extractor without site-specific filename hashes (Section 2.2)."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

# Official BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup, Tag

from paint_sim.collector.extractors.adapters import SiteAdapter, load_site_adapter_for_url
from paint_sim.collector.sources import load_collector_settings, normalize_url
from paint_sim.db.sample_ingestor import load_image_normalized_srgb, resolve_highest_resolution_image_url

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
BG_URL_REGEX = re.compile(r"background(?:-image)?\s*:\s*[^;]*url\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)", re.IGNORECASE)


@dataclass
class ExtractedImageCandidate:
    image_url: str
    page_url: str
    source_kind: str
    alt_text: str = ""
    caption_hint: str = ""
    width_hint: Optional[int] = None
    height_hint: Optional[int] = None


def _is_junk_image_url(url: str, junk_patterns: Sequence[str]) -> bool:
    """Return True if the candidate URL matches known non-sample junk patterns (logos, avatars, SVG, etc.)."""
    low = url.lower()
    if low.startswith("data:"):
        return True
    parsed_path = urlparse(low).path
    if parsed_path.endswith((".svg", ".gif", ".ico")):
        return True
    for pat in junk_patterns:
        p = pat.lower()
        if p in low:
            return True
    return False


def _parse_int_attr(tag: Tag, attr: str) -> Optional[int]:
    val = tag.get(attr)
    if isinstance(val, str):
        digits = re.sub(r"[^\d]", "", val)
        if digits.isdigit():
            return int(digits)
    return None


def _extract_caption_near_tag(tag: Tag) -> str:
    """Extract nearby `<figcaption>` or parent caption text if present."""
    fig = tag.find_parent("figure")
    if fig is not None:
        cap = fig.find("figcaption")
        if cap is not None:
            return cap.get_text(" ", strip=True)
    return ""


def extract_image_candidates_from_html(
    html_text: str,
    page_url: str,
    *,
    adapter: Optional[SiteAdapter] = None,
    min_image_px: Optional[int] = None,
    junk_patterns: Optional[Sequence[str]] = None,
) -> List[ExtractedImageCandidate]:
    """
    Extract and normalize image candidates from an HTML document (Section 2.2):
      1. `<img src>`, `srcset`, `data-src`, `data-srcset`, `data-lazy-src`, `data-original`, `data-orig-file`, `data-large-file`.
      2. `<picture><source srcset>`.
      3. `og:image` and `twitter:image` `<meta>` tags.
      4. `<a href>` links pointing directly to image files (`.jpg/.jpeg/.png/.webp`).
      5. Inline `style="background-image:url(...)"`.
    Restricts to `<article>`, `<main>`, `.entry-content` (or adapter `content_selector`) when present,
    falling back to the full page if no main container exists or yields candidates.
    """
    if adapter is None:
        adapter = load_site_adapter_for_url(page_url)
    cfg = load_collector_settings()
    min_px = int(
        min_image_px
        if min_image_px is not None
        else (adapter.min_image_px if adapter else cfg.get("min_image_px", 300))
    )
    junk_list = list(junk_patterns if junk_patterns is not None else cfg.get("junk_url_patterns", []))

    # Source: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
    soup = BeautifulSoup(html_text or "", "html.parser")

    # Remove excluded selectors inside soup copy
    if adapter and adapter.exclude_selectors:
        for sel in adapter.exclude_selectors:
            for bad in soup.select(sel):
                bad.decompose()

    selector = adapter.content_selector if adapter else "article, main, .entry-content"
    roots = soup.select(selector)
    if not roots:
        roots = [soup]

    candidates: List[ExtractedImageCandidate] = []
    seen_urls: set[str] = set()

    def add_candidate(
        raw_url: str,
        raw_srcset: str,
        kind: str,
        alt_text: str = "",
        caption_hint: str = "",
        w_hint: Optional[int] = None,
        h_hint: Optional[int] = None,
    ) -> None:
        resolved_srcset_items: List[str] = []
        if raw_srcset:
            for item in raw_srcset.split(","):
                parts = item.strip().split()
                if parts:
                    abs_part = urljoin(page_url, parts[0])
                    desc = f" {parts[1]}" if len(parts) > 1 else ""
                    resolved_srcset_items.append(f"{abs_part}{desc}")
        abs_url = urljoin(page_url, raw_url.strip()) if raw_url.strip() else ""
        hi_url = resolve_highest_resolution_image_url(abs_url, ", ".join(resolved_srcset_items))
        if not hi_url:
            return
        try:
            norm = normalize_url(hi_url)
        except ValueError:
            return
        if _is_junk_image_url(norm, junk_list):
            return
        # If explicit width/height HTML attributes indicate a tiny icon (< min_px on both sides), drop early
        if w_hint is not None and h_hint is not None and min(w_hint, h_hint) < min_px:
            # Only drop if we did not upgrade from a thumbnail URL to a full-size URL
            if hi_url == abs_url:
                return
        if norm in seen_urls:
            return
        seen_urls.add(norm)
        candidates.append(
            ExtractedImageCandidate(
                image_url=norm,
                page_url=page_url,
                source_kind=kind,
                alt_text=alt_text,
                caption_hint=caption_hint,
                width_hint=w_hint,
                height_hint=h_hint,
            )
        )

    def collect_from_roots(target_roots: Sequence[Tag]) -> None:
        attr_priority = (
            adapter.image_attr_priority
            if adapter
            else [
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
        for root in target_roots:
            # 1. <picture><source srcset>
            for pic in root.find_all("picture"):
                img_in_pic = pic.find("img")
                alt = str(img_in_pic.get("alt", "")).strip() if img_in_pic else ""
                cap = _extract_caption_near_tag(pic)
                for source_tag in pic.find_all("source"):
                    srcset_val = str(source_tag.get("srcset") or source_tag.get("data-srcset") or "").strip()
                    if srcset_val:
                        first_url = srcset_val.split(",")[0].strip().split()[0]
                        add_candidate(first_url, srcset_val, "picture_source", alt, cap)

            # 2. <img> tags (standard + lazy-load attributes)
            for img in root.find_all("img"):
                alt = str(img.get("alt", "")).strip()
                cap = _extract_caption_near_tag(img)
                w_hint = _parse_int_attr(img, "width")
                h_hint = _parse_int_attr(img, "height")
                srcset_val = str(img.get("data-srcset") or img.get("srcset") or "").strip()
                chosen_src = ""
                for attr in attr_priority:
                    if attr in ("srcset", "data-srcset"):
                        continue
                    val = str(img.get(attr) or "").strip()
                    if val and not val.startswith("data:"):
                        chosen_src = val
                        break
                if not chosen_src and srcset_val:
                    chosen_src = srcset_val.split(",")[0].strip().split()[0]
                if chosen_src or srcset_val:
                    add_candidate(chosen_src, srcset_val, "img", alt, cap, w_hint, h_hint)

            # 3. <a href> lightbox full-size image links
            for a_tag in root.find_all("a", href=True):
                href = str(a_tag.get("href", "")).strip()
                parsed_href = urlparse(href.lower()).path
                if any(parsed_href.endswith(ext) for ext in IMAGE_EXTENSIONS):
                    text = a_tag.get_text(" ", strip=True)
                    cap = _extract_caption_near_tag(a_tag)
                    add_candidate(href, "", "a_href", text, cap)

            # 4. Inline style="background-image:url(...)"
            for styled in root.find_all(style=True):
                style_str = str(styled.get("style", ""))
                for match in BG_URL_REGEX.finditer(style_str):
                    bg_url = match.group(1).strip()
                    if bg_url:
                        add_candidate(bg_url, "", "background_style")

    collect_from_roots(roots)
    if not candidates and roots != [soup]:
        collect_from_roots([soup])

    # 5. Meta og:image and twitter:image (from document head)
    for meta in soup.find_all("meta"):
        prop = str(meta.get("property") or meta.get("name") or "").lower().strip()
        if prop in ("og:image", "og:image:url", "twitter:image", "twitter:image:src"):
            content = str(meta.get("content") or "").strip()
            if content:
                add_candidate(content, "", "meta_image")

    return candidates


def extract_post_links_from_listing(
    html_text: str,
    listing_url: str,
    post_link_selector: str = "article h2 a, .entry-title a",
    max_posts: int = 5,
) -> List[str]:
    """Extract up to `max_posts` post URLs from a blog/category listing page (Section 7.3)."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    links: List[str] = []
    seen: set[str] = set()
    for a_tag in soup.select(post_link_selector):
        href = str(a_tag.get("href") or "").strip()
        if not href:
            continue
        try:
            norm = normalize_url(urljoin(listing_url, href))
        except ValueError:
            continue
        if norm not in seen:
            seen.add(norm)
            links.append(norm)
            if len(links) >= max_posts:
                break
    return links


def filter_downloaded_image_candidate(
    image_bytes: bytes,
    min_image_px: int = 300,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decode downloaded image bytes (sRGB-normalized), compute content SHA-256 and dimensions,
    and verify `min(width, height) >= min_image_px` (Section 2.2).
    """
    try:
        img = load_image_normalized_srgb(image_bytes)
        w, h = img.size
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        norm_rgb = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        buf = io.BytesIO()
        norm_rgb.save(buf, format="PNG")
        srgb_png = buf.getvalue()
        if min(w, h) < min_image_px:
            return False, {
                "reason": f"Image shorter side {min(w, h)}px < min_image_px ({min_image_px}px)",
                "width": w,
                "height": h,
                "sha256": sha256,
            }
        return True, {
            "width": w,
            "height": h,
            "sha256": sha256,
            "srgb_png_bytes": srgb_png,
        }
    except Exception as exc:
        return False, {"reason": f"Invalid image payload: {exc}"}
