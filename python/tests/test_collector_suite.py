"""Comprehensive test suite for the Multi-Site Paint Sample Collector (Section 9 & 10 Acceptance Verification)."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from paint_sim.collector.cli import run_cli
from paint_sim.collector.extractors.adapters import load_site_adapter_for_url
from paint_sim.collector.extractors.generic import (
    extract_image_candidates_from_html,
    extract_post_links_from_listing,
    filter_downloaded_image_candidate,
)
from paint_sim.collector.fetcher import (
    FetchError,
    PayloadTooLargeError,
    PoliteFetcher,
    RobotsDisallowedError,
)
from paint_sim.collector.jobs import CollectorJobManager
from paint_sim.collector.server import create_collector_app
from paint_sim.collector.sources import SourceRegistry
from paint_sim.collector.vision.gemini_detect import (
    GeminiSampleDetector,
    detect_and_refine_samples_auto,
    get_public_secrets_status,
    norm1000_box_to_px,
    norm1000_point_to_px,
    redact_secrets,
    save_secrets,
)
from paint_sim.collector.vision.manual_segment import (
    SAM2PointBoxSegmenter,
    segment_from_clicks_oklab_grabcut,
    segment_multiple_samples,
    segment_single_sample,
)
from paint_sim.collector.vision.refine import (
    cross_check_mask_against_box,
    validate_layout_hint,
    validate_mask_geometry,
)
from paint_sim.db.sample_ingestor import (
    classify_paint_sample,
    evaluate_gloss_calibration_lopo_cv,
    ingest_samples_batch,
    load_image_normalized_srgb,
)
from paint_sim.physics.engine_bridge import NativePaintCoreBridge


def _make_synthetic_disc_sheet_png(width: int = 480, height: int = 360) -> tuple[bytes, np.ndarray]:
    """
    Create a synthetic RGB image containing two painted discs:
      - Disc 1 centered at (140, 180), radius 65: vivid crimson paint [205, 38, 48] with an enclosed
        white specular highlight blob at (130, 160) [252, 252, 254].
      - Disc 2 centered at (340, 180), radius 65: cobalt blue paint [32, 85, 200] with an enclosed
        specular highlight blob at (330, 160) [250, 252, 255].
      - Background: dark neutral cutting mat [42, 46, 52] + top caption text bar.
    """
    rgb = np.full((height, width, 3), [42, 46, 52], dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]

    # Disc 1 (Red) + enclosed specular highlight island
    disc1 = ((xx - 140) ** 2 + (yy - 180) ** 2) <= (65 ** 2)
    spec1 = ((xx - 130) ** 2 + (yy - 160) ** 2) <= (12 ** 2)
    rgb[disc1] = [205, 38, 48]
    rgb[spec1] = [252, 252, 254]

    # Disc 2 (Blue) + enclosed specular highlight island
    disc2 = ((xx - 340) ** 2 + (yy - 180) ** 2) <= (65 ** 2)
    spec2 = ((xx - 330) ** 2 + (yy - 160) ** 2) <= (12 ** 2)
    rgb[disc2] = [32, 85, 200]
    rgb[spec2] = [250, 252, 255]

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.text((95, 265), "No Top Coat", fill=(230, 230, 230))
    draw.text((310, 265), "Gloss", fill=(230, 230, 230))

    arr = np.asarray(img, dtype=np.uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr


class TestCollectorSuite(unittest.TestCase):
    """Verifies all Section 9 & Section 10 requirements for the Multi-Site Paint Sample Collector."""

    def setUp(self) -> None:
        self.bridge = NativePaintCoreBridge()

    # ========================================================================
    # 1. Generic Extractor & Data-Only Site Adapters (Section 2.2, 2.3)
    # ========================================================================

    def test_01_generic_extractor_html_fixtures_and_site_adapter(self) -> None:
        # Fixture A: WordPress / pm-paint style post with srcset, lazy-load, lightbox <a>, inline style, og:image, and junk icons
        pm_html = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta property="og:image" content="https://pm-paint.com/wp-content/uploads/2026/10/og_cover.jpg" />
          <meta name="twitter:image" content="https://pm-paint.com/wp-content/uploads/2026/10/tw_cover.jpg" />
        </head>
        <body>
          <header><img class="custom-logo" src="https://pm-paint.com/logo.png" /></header>
          <article class="post">
            <div class="entry-content">
              <img src="https://pm-paint.com/wp-content/uploads/2026/10/sheet_gray-768x432.jpg"
                   srcset="https://pm-paint.com/wp-content/uploads/2026/10/sheet_gray-300x169.jpg 300w,
                           https://pm-paint.com/wp-content/uploads/2026/10/sheet_gray-1024x576.jpg 1024w,
                           https://pm-paint.com/wp-content/uploads/2026/10/sheet_gray.jpg 2200w"
                   alt="Gray Surfacer Sheet" />
              <img data-lazy-src="/wp-content/uploads/2026/10/sheet_white-768x432.jpg"
                   data-srcset="/wp-content/uploads/2026/10/sheet_white.jpg 2200w" />
              <a href="https://pm-paint.com/wp-content/uploads/2026/10/sheet_black.png">Full Lightbox Black Sheet</a>
              <div style="background-image: url('/wp-content/uploads/2026/10/bg_swatch.webp')"></div>
              <img src="https://pm-paint.com/wp-includes/images/smilies/icon_smile.svg" />
              <img src="https://secure.gravatar.com/avatar/xyz.jpg" />
              <img src="https://pm-paint.com/tiny_thumb.jpg" width="48" height="48" />
            </div>
          </article>
        </body>
        </html>
        """
        adapter = load_site_adapter_for_url("https://pm-paint.com/some-post/")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.host, "pm-paint.com")
        self.assertEqual(adapter.layout_hint["expected_count"], 5)

        cands = extract_image_candidates_from_html(
            pm_html,
            "https://pm-paint.com/some-post/",
            adapter=adapter,
        )
        urls = [c.image_url for c in cands]
        # Highest resolution picked, relative URLs resolved, thumbnails stripped
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/sheet_gray.jpg", urls)
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/sheet_white.jpg", urls)
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/sheet_black.png", urls)
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/bg_swatch.webp", urls)
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/og_cover.jpg", urls)
        self.assertIn("https://pm-paint.com/wp-content/uploads/2026/10/tw_cover.jpg", urls)
        # Junk filtered out
        for u in urls:
            self.assertNotIn("logo", u)
            self.assertNotIn("gravatar", u)
            self.assertNotIn(".svg", u)
            self.assertNotIn("tiny_thumb", u)

        # Fixture B: <picture><source srcset> + data-original lazy-load page
        pic_html = """
        <main>
          <picture>
            <source srcset="https://hobby.example.org/img/swatch_1x.webp 600w, https://hobby.example.org/img/swatch_2x.webp 1600w" />
            <img src="https://hobby.example.org/img/swatch_fallback.jpg" alt="Picture Swatch" />
          </picture>
          <img data-original="https://hobby.example.org/img/lazy_plaque.png" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" />
        </main>
        """
        pic_cands = extract_image_candidates_from_html(pic_html, "https://hobby.example.org/gallery")
        pic_urls = [c.image_url for c in pic_cands]
        self.assertIn("https://hobby.example.org/img/swatch_2x.webp", pic_urls)
        self.assertIn("https://hobby.example.org/img/lazy_plaque.png", pic_urls)

        # Listing page post link extraction
        listing_html = """
        <article><h2><a href="/post-1">Post 1</a></h2></article>
        <article><h2><a href="/post-2">Post 2</a></h2></article>
        <article><h2><a href="/post-3">Post 3</a></h2></article>
        """
        links = extract_post_links_from_listing(listing_html, "https://pm-paint.com/", max_posts=2)
        self.assertEqual(links, ["https://pm-paint.com/post-1", "https://pm-paint.com/post-2"])

    # ========================================================================
    # 2. Polite Fetcher with Mocked httpx (Section 2.4)
    # ========================================================================

    def test_02_fetcher_robots_ratelimit_cache_and_size_limit(self) -> None:
        clock_now = [100.0]
        slept_durations: List[float] = []

        def fake_clock() -> float:
            return clock_now[0]

        def fake_sleep(dt: float) -> None:
            slept_durations.append(dt)
            clock_now[0] += dt

        png_bytes, _ = _make_synthetic_disc_sheet_png(320, 320)

        def mock_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(
                    200,
                    text="User-agent: *\nDisallow: /private/\nAllow: /public/\n",
                    headers={"content-type": "text/plain"},
                )
            if path.startswith("/private/"):
                return httpx.Response(200, text="secret", headers={"content-type": "text/html"})
            if path == "/public/swatch.png":
                return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
            if path == "/public/oversize.png":
                return httpx.Response(
                    200,
                    content=b"X" * 2048,
                    headers={"content-type": "image/png", "content-length": "99999999"},
                )
            if path == "/public/bad_type":
                return httpx.Response(200, content=b"binary", headers={"content-type": "application/zip"})
            return httpx.Response(404)

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PoliteFetcher(
                cache_dir=Path(tmpdir) / "http_cache",
                rate_limit_per_sec=2.0,  # 0.5s interval
                max_payload_bytes=1024 * 1024,
                transport=httpx.MockTransport(mock_handler),
                clock_fn=fake_clock,
                sleep_fn=fake_sleep,
            )

            # 1. robots.txt blocks /private/
            with self.assertRaises(RobotsDisallowedError):
                fetcher.fetch("https://mock.example.com/private/secret.html")

            # 2. Allowed image fetch succeeds and rate-limits after robots.txt request
            res1 = fetcher.fetch("https://mock.example.com/public/swatch.png")
            self.assertFalse(res1.from_cache)
            self.assertEqual(res1.content, png_bytes)
            self.assertGreaterEqual(len(slept_durations), 1)
            self.assertAlmostEqual(slept_durations[0], 0.5, places=3)
            calls_after_first = fetcher.network_request_count

            # 3. Second fetch of same URL hits SHA-256 disk cache (0 additional network calls)
            res2 = fetcher.fetch("https://mock.example.com/public/swatch.png")
            self.assertTrue(res2.from_cache)
            self.assertEqual(fetcher.network_request_count, calls_after_first)

            # 4. Oversize payload rejected
            with self.assertRaises(PayloadTooLargeError):
                fetcher.fetch("https://mock.example.com/public/oversize.png", use_cache=False)

            # 5. Unsupported Content-Type rejected
            with self.assertRaises(FetchError):
                fetcher.fetch("https://mock.example.com/public/bad_type", use_cache=False)

    # ========================================================================
    # 3. Manual Segmentation: Highlight Hole Fill, Bit-Exact RGB & Multi-Click
    # ========================================================================

    def test_03_manual_segmentation_lossless_rgb_and_multiclick(self) -> None:
        _, rgb_u8 = _make_synthetic_disc_sheet_png(480, 360)

        # Single click on Disc 1 (140, 180): must fill specular highlight hole at (130, 160)
        single_res = segment_single_sample(
            rgb_u8,
            pos_points_xy=[(140, 180)],
            engine="grabcut",
            oklab_tol=0.08,
        )
        self.assertEqual(single_res["max_rgb_diff"], 0)
        self.assertTrue(single_res["validation"]["valid"])
        # Specular highlight island at (y=160, x=130) must be filled inside refined mask
        self.assertTrue(bool(single_res["refined_mask_bool"][160, 130]))

        # Verify saved RGBA PNG has bit-exact RGB inside opaque core
        crop_rgba = np.asarray(Image.open(io.BytesIO(single_res["crop_png_bytes"])).convert("RGBA"), dtype=np.uint8)
        x0, y0, x1, y1 = single_res["crop_bbox_xyxy"]
        orig_sub = rgb_u8[y0:y1, x0:x1]
        core = crop_rgba[:, :, 3] == 255
        self.assertGreater(int(np.count_nonzero(core)), 500)
        max_diff = int(np.max(np.abs(crop_rgba[:, :, :3][core].astype(np.int32) - orig_sub[core].astype(np.int32))))
        self.assertEqual(max_diff, 0)

        # Multi-click: Click 1 on Disc 1 (140, 180), Click 2 on Disc 2 (340, 180) -> 2 independent masks
        multi_res = segment_multiple_samples(
            rgb_u8,
            [
                {"pos": [[140, 180]], "neg": [[40, 40]]},
                {"pos": [[340, 180]], "neg": [[40, 40]]},
            ],
            engine="grabcut",
            oklab_tol=0.08,
        )
        self.assertEqual(len(multi_res["samples"]), 2)
        self.assertEqual(len(multi_res["overlap_warnings"]), 0)
        self.assertNotEqual(
            multi_res["samples"][0]["measured_srgb_hex"],
            multi_res["samples"][1]["measured_srgb_hex"],
        )

        # Validation checks: tiny mask (< 0.2%) and border-touching mask (> 2 borders) rejected
        tiny_mask = np.zeros((300, 300), dtype=bool)
        tiny_mask[150:153, 150:153] = True
        self.assertFalse(validate_mask_geometry(tiny_mask)["valid"])

        border_mask = np.zeros((300, 300), dtype=bool)
        border_mask[0:300, 0:100] = True  # touches top (0), bottom (299), and left (0) -> 3 borders
        self.assertFalse(validate_mask_geometry(border_mask)["valid"])

    # ========================================================================
    # 4. Gemini Client with Mocked Transport & Cross-Check (Sections 4.2, 4.3)
    # ========================================================================

    def test_04_gemini_detection_dedup_cache_crosscheck_and_failure(self) -> None:
        sheet_png, _ = _make_synthetic_disc_sheet_png(480, 360)

        # Normalized 0..1000 coordinates for Disc 1 (x: 75..205 -> 156..427, y: 115..245 -> 319..681)
        # Plus a duplicate overlapping box (IoU > 0.5) and a tiny < 0.2% box that must be dropped
        mock_response = {
            "image_width_hint": 480,
            "samples": [
                {
                    "index": 1,
                    "box_2d": [319, 156, 681, 427],
                    "point_2d": [500, 292],
                    "clean_points_2d": [[520, 292]],
                    "shape": "plaque",
                    "caption_text": "No Top Coat",
                    "inferred_topcoat": "none",
                    "inferred_undercoat": "gray",
                    "confidence": 0.95,
                },
                {
                    "index": 2,
                    "box_2d": [322, 160, 678, 425],  # Duplicate of sample 1 (IoU > 0.5)
                    "point_2d": [500, 292],
                    "clean_points_2d": [],
                    "shape": "plaque",
                    "caption_text": "Duplicate",
                    "inferred_topcoat": "none",
                    "inferred_undercoat": "gray",
                    "confidence": 0.70,
                },
                {
                    "index": 3,
                    "box_2d": [10, 10, 20, 20],  # Tiny box (< 0.2% area) -> dropped
                    "point_2d": [15, 15],
                    "clean_points_2d": [],
                    "shape": "other",
                    "caption_text": "",
                    "inferred_topcoat": "unknown",
                    "inferred_undercoat": "unknown",
                    "confidence": 0.50,
                },
                {
                    "index": 4,
                    "box_2d": [319, 573, 681, 844],
                    "point_2d": [500, 708],
                    "clean_points_2d": [[520, 708]],
                    "shape": "plaque",
                    "caption_text": "Gloss",
                    "inferred_topcoat": "gloss",
                    "inferred_undercoat": "gray",
                    "confidence": 0.93,
                },
            ],
            "sheet_text": "C3 Red vs C5 Blue Comparison",
            "notes": "",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = GeminiSampleDetector(
                cache_dir=Path(tmpdir) / "llm_cache",
                mock_transport=lambda model, sys_inst, prompt, img: mock_response,
            )

            res1 = detect_and_refine_samples_auto(sheet_png, detector=detector)
            self.assertEqual(res1["status"], "ok")
            self.assertFalse(res1["from_cache"])
            self.assertEqual(detector.network_call_count, 1)
            # Duplicate and tiny boxes dropped -> exactly 2 samples remain
            self.assertEqual(len(res1["samples"]), 2)

            # Second call hits disk cache -> 0 additional network calls
            res2 = detect_and_refine_samples_auto(sheet_png, detector=detector)
            self.assertTrue(res2["from_cache"])
            self.assertEqual(detector.network_call_count, 1)

            # Cross-check: huge box where mask covers < 40% sets needs_review=True
            mask_small = np.zeros((360, 480), dtype=bool)
            mask_small[160:190, 130:160] = True
            cc = cross_check_mask_against_box(mask_small, (50, 50, 400, 320))
            self.assertTrue(cc["needs_review"])

            # Cross-check: mask spilling > 15% outside tight box sets needs_review=True
            cc_spill = cross_check_mask_against_box(mask_small, (145, 160, 160, 190))
            self.assertTrue(cc_spill["needs_review"])

            # Layout hint mismatch (expected 5, found 2) sets needs_review=True
            lh = validate_layout_hint([(10, 10, 50, 50), (100, 10, 150, 50)], {"grid": "1x5", "expected_count": 5})
            self.assertTrue(lh["needs_review"])

            # API error / timeout -> vision_failed with zero fabricated boxes
            failing_det = GeminiSampleDetector(
                cache_dir=Path(tmpdir) / "llm_cache_fail",
                mock_transport=lambda m, s, p, i: (_ for _ in ()).throw(TimeoutError("504 Deadline Exceeded")),
            )
            fail_res = detect_and_refine_samples_auto(sheet_png, detector=failing_det, use_cache=False)
            self.assertEqual(fail_res["status"], "vision_failed")
            self.assertEqual(fail_res["samples"], [])

    # ========================================================================
    # 5. Taxonomy Rule Engine, Ambiguity & Isotonic Calibration (Section 7.1, 7.4)
    # ========================================================================

    def test_05_taxonomy_rules_ambiguity_and_isotonic_calibration(self) -> None:
        # Tamiya XF-1 without solvent hint has ambiguous solvent_type (0.60 vs 0.40, delta=0.20 <= 0.25)
        xf1 = classify_paint_sample("XF-1", "Flat Black", brand_hint="tamiya")
        self.assertEqual(xf1["solvent_type"], "acrylic_alcohol")
        self.assertIn("solvent_type", xf1["ambiguous_fields"])
        self.assertGreaterEqual(len(xf1["candidates"]["solvent_type"]), 2)

        # Passing sheet_text="Tamiya Enamel Paint" resolves the ambiguity to enamel
        xf1_enamel = classify_paint_sample("XF-1", "Flat Black", brand_hint="tamiya", sheet_text="Tamiya Enamel X-20")
        self.assertEqual(xf1_enamel["solvent_type"], "enamel")
        self.assertNotIn("solvent_type", xf1_enamel["ambiguous_fields"])

        # Leave-One-Paint-Out cross-validation for isotonic gloss calibration (Section 7.1)
        cv_report = evaluate_gloss_calibration_lopo_cv()
        self.assertGreaterEqual(cv_report["num_targets"], 3)
        self.assertTrue(cv_report["monotonic_preserved"])
        self.assertLess(cv_report["lopo_mae"], 0.18)

    # ========================================================================
    # 6. Batch Ingest, Secrets Isolation & FastAPI Server/WebSocket (Sections 5, 6, 7.5)
    # ========================================================================

    def test_06_batch_ingest_secrets_and_fastapi_websocket(self) -> None:
        sheet_png, _ = _make_synthetic_disc_sheet_png(360, 360)
        secret_api_key = "AIzaSySUPER_SECRET_TEST_KEY_999888777"
        secret_private_key = "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj\\n-----END PRIVATE KEY-----"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            secrets_file = tmp_path / "outside_repo" / "secrets.json"
            jobs_dir = tmp_path / "jobs"
            sample_root = tmp_path / "paint_samples"
            sources_file = tmp_path / "collector_sources.json"
            settings_file = tmp_path / "collector_settings.json"

            # 1. Batch ingest writes multiple samples atomically with monotonic IDs
            batch_rows = ingest_samples_batch(
                [
                    {
                        "image_bytes": sheet_png,
                        "paint_code": "C3",
                        "paint_name_en": "Red",
                        "paint_name_ko": "레드",
                        "source_url": "https://example.com/post",
                        "image_url": "https://example.com/c3.png",
                    },
                    {
                        "image_bytes": sheet_png,
                        "paint_code": "C5",
                        "paint_name_en": "Blue",
                        "paint_name_ko": "블루",
                        "source_url": "https://example.com/post",
                        "image_url": "https://example.com/c5.png",
                    },
                ],
                root_dir=sample_root,
            )
            self.assertEqual(len(batch_rows), 2)
            self.assertEqual(batch_rows[0]["sample_id"], "SMP_0001")
            self.assertEqual(batch_rows[1]["sample_id"], "SMP_0002")
            self.assertIn("https://example.com/c3.png", batch_rows[0]["notes"])
            opt0 = json.loads(batch_rows[0]["calibrated_optical_params"])
            self.assertIn("gloss_index", opt0)
            self.assertIn("alpha_ggx_calibrated", opt0)
            self.assertIn("gloss_margin_to_next_class", opt0)

            # 2. FastAPI + WebSocket + Secrets redaction test
            def mock_http(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/robots.txt":
                    return httpx.Response(200, text="User-agent: *\nAllow: /\n")
                if request.url.path == "/page":
                    return httpx.Response(
                        200,
                        text='<article><img src="https://mock.site/img1.png" /></article>',
                        headers={"content-type": "text/html"},
                    )
                if request.url.path == "/img1.png":
                    return httpx.Response(200, content=sheet_png, headers={"content-type": "image/png"})
                return httpx.Response(404)

            fetcher = PoliteFetcher(
                cache_dir=tmp_path / "http_cache",
                rate_limit_per_sec=100.0,
                transport=httpx.MockTransport(mock_http),
            )
            detector = GeminiSampleDetector(
                cache_dir=tmp_path / "llm_cache",
                secrets_path=secrets_file,
                mock_transport=lambda m, s, p, i: (_ for _ in ()).throw(
                    RuntimeError(f"Auth error with key {secret_api_key} and {secret_private_key}")
                ),
            )
            mgr = CollectorJobManager(
                jobs_dir=jobs_dir,
                sample_root_dir=sample_root,
                fetcher=fetcher,
                detector=detector,
                secrets_path=secrets_file,
            )
            app = create_collector_app(
                job_manager=mgr,
                source_registry=SourceRegistry(sources_file),
                settings_path=settings_file,
                secrets_path=secrets_file,
            )
            client = TestClient(app)

            # Save both API key and service-account JSON via PUT /api/secrets
            sa_payload = {
                "type": "service_account",
                "project_id": "paint-sim-test-proj",
                "private_key_id": "keyid987654321",
                "private_key": secret_private_key,
                "client_email": "vision-bot@paint-sim-test-proj.iam.gserviceaccount.com",
            }
            sec_resp = client.put(
                "/api/secrets",
                json={
                    "auth_mode": "vertex_service_account",
                    "api_key": secret_api_key,
                    "service_account_json": sa_payload,
                },
            )
            self.assertEqual(sec_resp.status_code, 200)
            sec_text = sec_resp.text
            self.assertNotIn(secret_api_key, sec_text)
            self.assertNotIn(secret_private_key, sec_text)
            self.assertIn("vision-bot@paint-sim-test-proj.iam.gserviceaccount.com", sec_text)

            # GET /api/settings must never leak secrets
            set_resp = client.get("/api/settings")
            self.assertEqual(set_resp.status_code, 200)
            self.assertNotIn(secret_api_key, set_resp.text)
            self.assertNotIn(secret_private_key, set_resp.text)

            # Create job, connect WebSocket, fetch, run failing auto vision (to test error log redaction), then manual segment & ingest
            create_resp = client.post(
                "/api/jobs",
                json={"urls": ["https://mock.site/page"], "defaults": {"paint_code": "C3", "paint_name_en": "Red"}},
            )
            self.assertEqual(create_resp.status_code, 200)
            job_id = create_resp.json()["job_id"]

            with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
                hello = ws.receive_json()
                self.assertEqual(hello["type"], "ws_connected")

                fetch_resp = client.post(f"/api/jobs/{job_id}/fetch")
                self.assertEqual(fetch_resp.status_code, 200)
                images = fetch_resp.json()["images"]
                self.assertEqual(len(images), 1)
                image_id = images[0]["image_id"]

                # Trigger auto vision failure containing secret strings -> must be redacted in response & job.json
                auto_resp = client.post(f"/api/images/{image_id}/segment/auto", json={})
                self.assertEqual(auto_resp.status_code, 200)
                self.assertEqual(auto_resp.json()["status"], "vision_failed")
                self.assertNotIn(secret_api_key, auto_resp.text)
                self.assertNotIn(secret_private_key, auto_resp.text)

                # Manual segmentation via REST
                man_resp = client.post(
                    f"/api/images/{image_id}/segment/manual",
                    json={"pos_points": [[140, 180]], "oklab_tol": 0.08},
                )
                self.assertEqual(man_resp.status_code, 200)
                sample_tmp_id = man_resp.json()["samples"][0]["sample_tmp_id"]

                # Accept sample via PATCH
                patch_resp = client.patch(
                    f"/api/samples/{sample_tmp_id}",
                    json={"status": "accepted"},
                )
                self.assertEqual(patch_resp.status_code, 200)

                # Ingest accepted sample
                ing_resp = client.post(f"/api/jobs/{job_id}/ingest", json={})
                self.assertEqual(ing_resp.status_code, 200)
                self.assertEqual(ing_resp.json()["ingested_count"], 1)

            # Verify job.json on disk contains zero secrets
            job_json_text = (jobs_dir / job_id / "job.json").read_text(encoding="utf-8")
            self.assertNotIn(secret_api_key, job_json_text)
            self.assertNotIn(secret_private_key, job_json_text)

    # ========================================================================
    # 7. Static Guard: No Hard-Coded Site URLs, Filename Hashes, or Crop %
    # ========================================================================

    def test_07_no_hardcoded_urls_hashes_or_crop_percentages_in_python(self) -> None:
        forbidden_hashes = (
            "7d2a8a5d380f463388d1c8d4512bef57",
            "40dcbe9d88c96297b642179a653642d8",
            "60275ab58ebd7cd921d15fb49998721b",
        )
        for py_file in (PYTHON_DIR / "paint_sim").rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for h in forbidden_hashes:
                self.assertNotIn(h, content, f"Hardcoded filename hash {h} found in {py_file}")
            self.assertNotIn("0.812", content, f"Hardcoded sheet crop percentage 0.812 found in {py_file}")

    # ========================================================================
    # 8. Regression: Re-collecting 3 HLP110 Sheets Reproduces 15 Samples (ΔE00 < 1.0)
    # ========================================================================

    def test_08_regression_recollect_hlp110_sheets_delta_e00_under_1(self) -> None:
        sample_root = ROOT_DIR / "sample" / "paint_samples"
        master_csv = sample_root / "paint_sample_master_index.csv"
        if not master_csv.exists():
            self.skipTest("Master index CSV not present.")

        with open(master_csv, "r", encoding="utf-8-sig", newline="") as f:
            master_rows = list(csv.DictReader(f))

        # First 15 rows are the 3 HLP110 sheets (gray: SMP_0001..0005, white: SMP_0006..0010, black: SMP_0011..0015)
        hlp110_rows = [r for r in master_rows if r["sample_id"] in {f"SMP_{i:04d}" for i in range(1, 16)}]
        self.assertEqual(len(hlp110_rows), 15)

        def hex_to_lin(h: str) -> List[float]:
            r = int(h[1:3], 16) / 255.0
            g = int(h[3:5], 16) / 255.0
            b = int(h[5:7], 16) / 255.0
            return self.bridge.srgb_to_linear(np.array([r, g, b], dtype=np.float32)).tolist()

        # Compose 3 synthetic 5-spoon sheets from the 15 isolated HLP110 spoon PNGs and run collector segmentation
        for sheet_idx in range(3):
            sheet_rows = hlp110_rows[sheet_idx * 5 : (sheet_idx + 1) * 5]
            spoon_imgs = [Image.open(sample_root / r["file_rel_path"]).convert("RGBA") for r in sheet_rows]
            col_w = max(im.width for im in spoon_imgs) + 40
            sheet_h = max(im.height for im in spoon_imgs) + 80
            sheet_w = col_w * 5

            # Composite onto a contrasting studio background (magenta reference mat)
            canvas = Image.new("RGB", (sheet_w, sheet_h), (210, 40, 190))
            prompts = []
            for idx, sp_img in enumerate(spoon_imgs):
                x_off = idx * col_w + 20
                y_off = 40
                alpha_np = np.asarray(sp_img.getchannel("A"), dtype=np.uint8) >= 250
                bin_mask_img = Image.fromarray((alpha_np.astype(np.uint8)) * 255, mode="L")
                canvas.paste(sp_img.convert("RGB"), (x_off, y_off), mask=bin_mask_img)
                # Find the thickest interior point of the spoon mask for a realistic click seed
                from scipy.ndimage import distance_transform_edt

                dist = distance_transform_edt(alpha_np)
                cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
                seed_x = x_off + int(cx)
                seed_y = y_off + int(cy)
                box = [x_off - 6, y_off - 6, x_off + sp_img.width + 6, y_off + sp_img.height + 6]
                prompts.append({"pos": [[seed_x, seed_y]], "neg": [[x_off - 4, y_off - 4]], "box": box})

            sheet_rgb = np.asarray(canvas, dtype=np.uint8)
            multi_res = segment_multiple_samples(sheet_rgb, prompts, engine="grabcut", oklab_tol=0.08, grabcut_iters=2)
            self.assertEqual(len(multi_res["samples"]), 5)

            for idx, (seg_smp, ref_row) in enumerate(zip(multi_res["samples"], sheet_rows)):
                lin_new = hex_to_lin(seg_smp["measured_srgb_hex"])
                lin_ref = hex_to_lin(ref_row["measured_srgb_hex"])
                de00 = self.bridge.delta_e00_linear_rgb(lin_new, lin_ref)
                self.assertLess(
                    de00,
                    1.0,
                    f"Sample {ref_row['sample_id']} (slot {idx}) ΔE00={de00:.3f} >= 1.0 "
                    f"(got {seg_smp['measured_srgb_hex']}, expected {ref_row['measured_srgb_hex']})",
                )


class TestHQSAM2InteractiveSegmenter(unittest.TestCase):
    def test_collector_settings_default_variant(self) -> None:
        cfg_path = ROOT_DIR / "config" / "collector_settings.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("sam2_model_variant"), "sam2.1_base_plus")
        self.assertTrue(cfg.get("sam2_checkpoint", "").endswith("sam2.1_hiera_base_plus.pt"))
        self.assertTrue(cfg.get("sam2_model_cfg", "").endswith("sam2.1_hiera_b+.yaml"))

    def test_variant_and_path_resolution(self) -> None:
        seg = SAM2PointBoxSegmenter(model_variant="sam2.1_base_plus")
        self.assertEqual(seg.checkpoint_path.name, "sam2.1_hiera_base_plus.pt")
        self.assertIn("sam2.1_hiera_b+.yaml", seg.model_cfg)
        self.assertAlmostEqual(seg.vram_footprint_gb, 2.0)

        seg_custom = SAM2PointBoxSegmenter(
            checkpoint_path="custom_chk.pt", model_cfg="custom_cfg.yaml", model_variant="sam2.1_small"
        )
        self.assertEqual(seg_custom.checkpoint_path.name, "custom_chk.pt")
        self.assertEqual(seg_custom.model_cfg, "custom_cfg.yaml")

    def test_hq_detection_and_downgrade(self) -> None:
        with mock.patch.object(SAM2PointBoxSegmenter, "_check_hq_support", return_value=False):
            seg = SAM2PointBoxSegmenter(model_variant="hq_sam2_large")
        self.assertEqual(seg.model_variant, "sam2.1_base_plus")
        self.assertEqual(seg.checkpoint_path.name, "sam2.1_hiera_base_plus.pt")

    def test_empty_prompt_rejected(self) -> None:
        seg = SAM2PointBoxSegmenter()
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "At least one positive point or bounding box is required."):
            seg.predict_interactive(rgb, pos_points_xy=[], neg_points_xy=[(10, 10)])

    def test_deterministic_fallback_when_checkpoint_missing(self) -> None:
        # Force fallback by specifying non-existent checkpoint path
        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        rgb = np.full((48, 48, 3), 220, dtype=np.uint8)
        rgb[10:38, 10:38] = [40, 100, 200]
        res = seg.predict_interactive(
            rgb, pos_points_xy=[(24, 24)], neg_points_xy=[], part_id="part_test_fallback", use_prev_mask=False
        )
        self.assertEqual(res["engine_used"], "oklab_grabcut_fallback")
        self.assertEqual(res["mask_bool"].shape, (48, 48))
        self.assertEqual(res["alpha_01"].shape, (48, 48))
        self.assertTrue(bool(res["mask_bool"][24, 24]))

    def test_iterative_negative_point_excludes_neighbor_part(self) -> None:
        rgb = np.full((64, 64, 3), 220, dtype=np.uint8)
        rgb[:, 29:31, :] = 18
        rgb[8:56, 6:28] = [235, 238, 242]
        rgb[8:56, 32:56] = [235, 238, 242]
        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        r1 = seg.predict_interactive(
            rgb, pos_points_xy=[(16, 32)], neg_points_xy=[], part_id="part_101", use_prev_mask=False
        )
        self.assertTrue(bool(r1["mask_bool"][32, 16]))
        self.assertEqual(r1["alpha_01"].dtype, np.float32)
        r2 = seg.predict_interactive(
            rgb, pos_points_xy=[(16, 32)], neg_points_xy=[(44, 32)], part_id="part_101", use_prev_mask=True
        )
        self.assertTrue(bool(r2["mask_bool"][32, 16]))
        self.assertFalse(bool(r2["mask_bool"][32, 44]))

    def test_iterative_logits_cache_lru_and_clear(self) -> None:
        seg = SAM2PointBoxSegmenter()
        dummy_sha = "aabbcc"
        seg._current_image_sha256 = dummy_sha
        for i in range(70):
            seg._cache_part_logits(dummy_sha, f"part_{i}", np.zeros((1, 256, 256), dtype=np.float32))
        self.assertEqual(len(seg._part_logits_cache), 64)
        self.assertNotIn((dummy_sha, "part_0"), seg._part_logits_cache)
        self.assertIn((dummy_sha, "part_69"), seg._part_logits_cache)

        seg.clear_part_logits("part_69")
        self.assertNotIn((dummy_sha, "part_69"), seg._part_logits_cache)

    def test_real_sam2_cuda_inference_when_checkpoint_exists(self) -> None:
        seg = SAM2PointBoxSegmenter(model_variant="sam2.1_base_plus")
        if not seg.is_available():
            self.skipTest("sam2.1_hiera_base_plus.pt not present on disk")
        rgb = np.full((64, 64, 3), 220, dtype=np.uint8)
        rgb[10:54, 10:54] = [30, 90, 210]
        res = seg.predict_interactive(rgb, pos_points_xy=[(32, 32)], part_id="part_cuda_real", use_prev_mask=False)
        self.assertEqual(res["engine_used"], "sam2.1_base_plus")
        self.assertTrue(bool(res["mask_bool"][32, 32]))

    def test_single_click_part_scale_selection_prefers_part_over_whole_body(self) -> None:
        h, w = 100, 100
        masks = np.zeros((3, h, w), dtype=np.float32)
        # Mask 0: small armor part (15x15 = 225 px out of 10000 = 2.25%)
        masks[0, 20:35, 20:35] = 1.0
        # Mask 1: medium subassembly (40x40 = 1600 px = 16%)
        masks[1, 10:50, 10:50] = 1.0
        # Mask 2: huge whole Gunpla body (85x85 = 7225 px = 72.25%)
        masks[2, 5:90, 5:90] = 1.0

        scores = np.array([0.82, 0.88, 0.92], dtype=np.float32)
        local_pos = [(25, 25)]

        best_idx = SAM2PointBoxSegmenter._select_best_mask_index(masks, scores, local_pos, multimask_output=True)
        self.assertEqual(best_idx, 0)

    def test_tolerance_shifts_mask_coverage_and_boundary_band(self) -> None:
        rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        rgb[20:44, 20:44] = [30, 80, 200]
        rgb[17:47, 17:47] = np.where(rgb[17:47, 17:47] == 200, [45, 95, 190], rgb[17:47, 17:47])

        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        res_low = seg.predict_interactive(
            rgb,
            pos_points_xy=[(32, 32)],
            part_id="part_tol_test",
            use_prev_mask=False,
            tolerance=2.5,
        )
        res_high = seg.predict_interactive(
            rgb,
            pos_points_xy=[(32, 32)],
            part_id="part_tol_test",
            use_prev_mask=False,
            tolerance=10.0,
        )
        cnt_low = np.count_nonzero(res_low["mask_bool"])
        cnt_high = np.count_nonzero(res_high["mask_bool"])
        self.assertGreater(cnt_high, cnt_low)

    def test_jpeg_chroma_noise_boundary_band_remains_smooth(self) -> None:
        import cv2
        from scipy.ndimage import binary_fill_holes

        h, w = 80, 80
        rgb = np.full((h, w, 3), 210, dtype=np.uint8)
        rgb[:, 14:16] = 20
        rgb[:, 64:66] = 20
        rgb[16:64, 16:64] = [30, 80, 200]
        rng = np.random.RandomState(42)
        for by in range(16, 64, 8):
            for bx in range(16, 64, 8):
                noise = rng.randint(-12, 13, size=3)
                block = rgb[by : by + 8, bx : bx + 8].astype(np.int16) + noise
                rgb[by : by + 8, bx : bx + 8] = np.clip(block, 0, 255).astype(np.uint8)

        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        res = seg.predict_interactive(
            rgb,
            pos_points_xy=[(40, 40)],
            tolerance=7.0,
            part_id="part_jpeg_noise",
            use_prev_mask=False,
        )

        mask = res["mask_bool"]
        self.assertTrue(np.any(mask))
        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        self.assertEqual(num_labels, 2, f"Expected 1 foreground island, got {num_labels - 1}")

        filled = binary_fill_holes(mask)
        self.assertTrue(np.array_equal(mask, filled), "Mask contains holes inside the part")

        if SAM2PointBoxSegmenter().is_available():
            seg_real = SAM2PointBoxSegmenter()
            res_real = seg_real.predict_interactive(
                rgb,
                pos_points_xy=[(40, 40)],
                tolerance=7.0,
                part_id="part_jpeg_noise_real",
                use_prev_mask=False,
            )
            mask_real = res_real["mask_bool"]
            self.assertTrue(np.any(mask_real))
            num_labels_real, _ = cv2.connectedComponents(mask_real.astype(np.uint8))
            self.assertEqual(num_labels_real, 2, f"Expected 1 foreground island in real SAM2, got {num_labels_real - 1}")
            filled_real = binary_fill_holes(mask_real)
            self.assertTrue(np.array_equal(mask_real, filled_real), "Real SAM2 mask contains holes inside the part")

    def test_warmup_session_image_and_foreground_object_detection(self) -> None:
        h, w = 120, 120
        rgb = np.full((h, w, 3), [60, 120, 70], dtype=np.uint8)
        rgb[25:86, 30:91] = [40, 50, 80]

        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        warm = seg.warmup_session_image(rgb)

        self.assertIn("object_mask", warm)
        self.assertIn("model_bbox_xyxy", warm)
        self.assertIn("roi_bbox_xyxy", warm)
        self.assertFalse(warm["object_mask"][0, 0])
        self.assertFalse(warm["object_mask"][h - 1, w - 1])
        self.assertTrue(warm["object_mask"][55, 60])

        x0, y0, x1, y1 = warm["model_bbox_xyxy"]
        self.assertGreaterEqual(x0, 25)
        self.assertGreaterEqual(y0, 20)
        self.assertLessEqual(x1, 95)
        self.assertLessEqual(y1, 90)

    def test_engine_bridge_dimension_validation(self) -> None:
        from paint_sim.physics.engine_bridge import NativePaintCoreBridge

        bridge = NativePaintCoreBridge()
        bad_arr_2d = np.zeros((32, 32), dtype=np.float32)
        bad_arr_4c = np.zeros((32, 32, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Expected HxWx3 linear_wb_rgb array"):
            bridge.compute_panel_lines(bad_arr_2d)
        with self.assertRaisesRegex(ValueError, "Expected HxWx3 linear_wb_rgb array"):
            bridge.compute_panel_lines(bad_arr_4c)
        with self.assertRaisesRegex(ValueError, "Expected HxWx3 linear_wb_rgb array"):
            bridge.snap_and_matte_part_mask(
                bad_arr_2d, np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "Expected HxWx3 linear_wb_rgb array"):
            bridge.snap_and_matte_part_mask(
                bad_arr_4c, np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=bool)
            )

    def test_refiner_apply_background_mode(self) -> None:
        from paint_sim.nano_banana.refiner import NanoBananaRefiner

        img = Image.new("RGB", (40, 40), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 30, 30], fill=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # full_bg / default
        res_full = NanoBananaRefiner.apply_background_mode(png_bytes, "full_bg")
        self.assertEqual(res_full, png_bytes)

        # object_transparent with mask
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:31, 10:31] = True
        res_trans = NanoBananaRefiner.apply_background_mode(png_bytes, "object_transparent", object_mask=mask)
        with Image.open(io.BytesIO(res_trans)) as out_img:
            self.assertEqual(out_img.mode, "RGBA")
            rgba = np.asarray(out_img)
            self.assertEqual(rgba[0, 0, 3], 0)
            self.assertEqual(rgba[20, 20, 3], 255)

        # object_crop with bbox
        res_crop = NanoBananaRefiner.apply_background_mode(
            png_bytes, "object_crop", model_bbox_xyxy=[10, 10, 30, 30]
        )
        with Image.open(io.BytesIO(res_crop)) as out_img:
            self.assertEqual(out_img.size, (21, 21))

    def test_roi_switch_invalidates_mismatched_logits_and_box_only_skips_stale_logits(self) -> None:
        import hashlib
        seg = SAM2PointBoxSegmenter()
        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_sha = hashlib.sha256(np.ascontiguousarray(dummy_rgb).tobytes()).hexdigest()

        # Initialize image state first so it does not clear part logits cache
        seg._ensure_image_state(dummy_rgb, img_sha)

        mock_pred = mock.MagicMock()
        mock_pred.predict.side_effect = lambda **kw: (
            np.ones((1, 51, 51) if seg._current_roi_bbox == (10, 10, 60, 60) else (1, 100, 100), dtype=bool),
            np.array([0.95], dtype=np.float32),
            np.zeros((1, 1, 256, 256), dtype=np.float32),
        )

        with mock.patch.object(seg, "is_available", return_value=True), \
             mock.patch.object(seg, "_ensure_predictor", return_value=mock_pred):

            logits_payload = np.ones((1, 256, 256), dtype=np.float32)
            seg._cache_part_logits(img_sha, "part_alpha", logits_payload, roi_bbox=(10, 10, 60, 60))

            # Case 1: active_roi is different from cached_roi -> mask_input must be None
            seg._cache_part_logits(img_sha, "part_alpha", logits_payload, roi_bbox=(10, 10, 60, 60))
            seg._current_roi_bbox = (0, 0, 99, 99)
            seg.predict_interactive(
                dummy_rgb, pos_points_xy=[(20, 20)], part_id="part_alpha", use_prev_mask=True
            )
            kwargs1 = mock_pred.predict.call_args[1]
            self.assertIsNone(kwargs1.get("mask_input"), "Mismatched ROI should invalidate cached logits")

            # Case 2: active_roi matches cached_roi AND positive point present -> mask_input must NOT be None
            seg._cache_part_logits(img_sha, "part_alpha", logits_payload, roi_bbox=(10, 10, 60, 60))
            seg._current_roi_bbox = (10, 10, 60, 60)
            seg.predict_interactive(
                dummy_rgb, pos_points_xy=[(20, 20)], part_id="part_alpha", use_prev_mask=True
            )
            kwargs2 = mock_pred.predict.call_args[1]
            self.assertIsNotNone(kwargs2.get("mask_input"), "Matching ROI should pass cached logits")
            np.testing.assert_array_equal(kwargs2["mask_input"], logits_payload)

            # Case 3: active_roi matches, but box-only prompt without any points -> mask_input must be None
            seg._cache_part_logits(img_sha, "part_alpha", logits_payload, roi_bbox=(10, 10, 60, 60))
            mock_pred.reset_mock()
            seg._current_roi_bbox = (10, 10, 60, 60)
            seg.predict_interactive(
                dummy_rgb, pos_points_xy=[], box_xyxy=(15, 15, 45, 45), part_id="part_alpha", use_prev_mask=True
            )
            kwargs3 = mock_pred.predict.call_args[1]
            self.assertIsNone(kwargs3.get("mask_input"), "Box-only prompt with zero points must skip stale logits")

    def test_negative_point_carves_exclusion_zone_not_readded_by_dilation(self) -> None:
        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        rgb[15:45, 15:45] = [30, 90, 210]

        # In (x, y) coordinates: pos=(25, 25), neg=(25, 30) -> [y, x] is [25, 25] and [30, 25]
        r1 = seg.predict_interactive(
            rgb,
            pos_points_xy=[(25, 25)],
            neg_points_xy=[],
            part_id="part_excl",
            tolerance=10.0,
            use_prev_mask=False,
            snap_to_panel_lines=True,
        )
        self.assertTrue(r1["mask_bool"][25, 25])
        self.assertTrue(r1["mask_bool"][30, 25])

        r2 = seg.predict_interactive(
            rgb,
            pos_points_xy=[(25, 25)],
            neg_points_xy=[(25, 30)],
            part_id="part_excl",
            tolerance=10.0,
            use_prev_mask=False,
            snap_to_panel_lines=True,
        )
        self.assertTrue(r2["mask_bool"][25, 25])
        # Negative point at (x=25, y=30) -> row=30, col=25
        self.assertFalse(r2["mask_bool"][30, 25])
        self.assertEqual(r2["alpha_01"][30, 25], 0.0)
        # Exclusion zone disk around (x=25, y=30) should not be re-added by dilation
        self.assertFalse(r2["mask_bool"][30, 24])
        self.assertEqual(r2["alpha_01"][30, 24], 0.0)
        self.assertFalse(r2["mask_bool"][30, 26])
        self.assertEqual(r2["alpha_01"][30, 26], 0.0)
        self.assertFalse(r2["mask_bool"][29, 25])
        self.assertEqual(r2["alpha_01"][29, 25], 0.0)
        self.assertFalse(r2["mask_bool"][31, 25])
        self.assertEqual(r2["alpha_01"][31, 25], 0.0)

        # Also verify negative-only prompt does not raise ValueError when use_prev_mask=True and part_id is present
        r3 = seg.predict_interactive(
            rgb,
            pos_points_xy=[],
            neg_points_xy=[(25, 30)],
            part_id="part_excl",
            use_prev_mask=True,
            snap_to_panel_lines=True,
        )
        self.assertIn("mask_bool", r3)

    def test_tiny_1px_mask_and_small_image_do_not_crash(self) -> None:
        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        # 1. Very small 2x2 image (smaller than 3x3 gaussian blur kernel)
        small_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        small_rgb[0, 0] = [220, 50, 50]
        res_small = seg.predict_interactive(
            small_rgb,
            pos_points_xy=[(0, 0)],
            tolerance=5.0,
            part_id="part_tiny_2x2",
            snap_to_panel_lines=True,
        )
        self.assertEqual(res_small["mask_bool"].shape, (2, 2))
        self.assertTrue(res_small["mask_bool"][0, 0])

        # 2. 1-pixel mask on larger image preserved through snap_and_matte_part_mask
        h, w = 16, 16
        iso_rgb = np.full((h, w, 3), 200, dtype=np.uint8)
        iso_rgb[8, 8] = [30, 40, 50]
        res_iso = seg.predict_interactive(
            iso_rgb,
            pos_points_xy=[(8, 8)],
            tolerance=1.0,
            part_id="part_tiny_1px",
            snap_to_panel_lines=True,
        )
        self.assertTrue(res_iso["mask_bool"][8, 8], "Tiny isolated pixel should be preserved")
        self.assertGreaterEqual(np.count_nonzero(res_iso["mask_bool"]), 1)

    def test_session_lru_eviction_and_safe_zlib_decoder_bounds(self) -> None:
        import base64
        import zlib
        from paint_sim.api.main import PaintSimAPIService, decode_alpha_bbox_zlib

        # Part A: decode_alpha_bbox_zlib safety & bounds
        # Inverted coordinates: x1 < x0 or y1 < y0
        idx, alpha = decode_alpha_bbox_zlib({"bbox_xyxy": [10, 10, 5, 5], "alpha_b64": "dummy"}, 100, 100)
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(alpha), 0)

        # Truncated payload: expected_len is 20x20 = 400, but raw decompressed is only 10 bytes
        short_raw = zlib.compress(b"x" * 10)
        b64_short = base64.b64encode(short_raw).decode("ascii")
        idx, alpha = decode_alpha_bbox_zlib({"bbox_xyxy": [0, 0, 19, 19], "alpha_b64": b64_short}, 100, 100)
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(alpha), 0)

        # Corrupt base64
        idx, alpha = decode_alpha_bbox_zlib({"bbox_xyxy": [0, 0, 10, 10], "alpha_b64": "not_valid_base64!!!"}, 100, 100)
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(alpha), 0)

        # Valid payload with extra trailing bytes: must slice expected_len and decode properly
        w, h = 10, 10
        raw_mask_u8 = np.zeros((4, 4), dtype=np.uint8)
        raw_mask_u8[1, 1] = 255
        raw_mask_u8[2, 2] = 128
        extra_bytes = raw_mask_u8.tobytes() + b"extra_trailing_data_padding"
        b64_extra = base64.b64encode(zlib.compress(extra_bytes)).decode("ascii")
        idx, alpha = decode_alpha_bbox_zlib({"bbox_xyxy": [2, 2, 5, 5], "alpha_b64": b64_extra}, w, h)
        self.assertEqual(len(idx), 2)
        self.assertEqual(len(alpha), 2)
        self.assertIn(33, idx)
        self.assertIn(44, idx)

        # Part B: LRU session eviction in PaintSimAPIService
        svc = PaintSimAPIService()
        svc.max_segment_sessions = 3
        sessions = []
        for i in range(4):
            img_arr = np.full((32, 32, 3), 50 + i * 10, dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_arr).save(buf, format="PNG")
            sess = svc.create_segment_session(buf.getvalue())
            sessions.append(sess["session_id"])

        self.assertEqual(len(svc.segment_sessions), 3)
        self.assertNotIn(sessions[0], svc.segment_sessions)
        self.assertIn(sessions[1], svc.segment_sessions)
        self.assertIn(sessions[2], svc.segment_sessions)
        self.assertIn(sessions[3], svc.segment_sessions)

        # Access s1 via clear_part_logits so it moves to MRU
        svc.clear_part_logits(sessions[1], "any_part")

        # Create session 4 -> should evict s2 (oldest), keeping s1 and s3
        img_arr = np.full((32, 32, 3), 150, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        s4 = svc.create_segment_session(buf.getvalue())

        self.assertEqual(len(svc.segment_sessions), 3)
        self.assertNotIn(sessions[2], svc.segment_sessions, "s2 should be evicted as LRU after s1 was touched")
        self.assertIn(sessions[1], svc.segment_sessions)
        self.assertIn(sessions[3], svc.segment_sessions)
        self.assertIn(s4["session_id"], svc.segment_sessions)

    def test_red_part_on_gray_background_includes_anti_aliased_fringe_at_high_tolerance(self) -> None:
        rgb = np.full((80, 80, 3), [195, 195, 200], dtype=np.uint8)
        rgb[18:63, 18:63] = [198, 145, 145]
        rgb[19:62, 19:62] = [200, 105, 105]
        rgb[20:61, 20:61] = [205, 35, 35]

        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        lin_rgb = NativePaintCoreBridge.srgb_to_linear(rgb.astype(np.float32) / 255.0)
        panel_mask = seg.bridge.compute_panel_lines(lin_rgb)

        # Step-edge borders of the red part must not be marked as > 100 panel lines
        self.assertLessEqual(int(panel_mask[19, 40]), 100)
        self.assertLessEqual(int(panel_mask[20, 40]), 100)
        self.assertLessEqual(int(np.max(panel_mask[18:22, 35:45])), 100)

        res = seg.predict_interactive(rgb, pos_points_xy=[(40, 40)], tolerance=15.0)
        mask_bool = res["mask_bool"]
        alpha_01 = res["alpha_01"]

        self.assertTrue(mask_bool[40, 40])
        self.assertTrue(bool(mask_bool[19, 40]))
        self.assertGreaterEqual(float(alpha_01[19, 40]), 0.85)

    def test_blue_part_on_gray_background_does_not_bleed_into_gray_at_max_tolerance(self) -> None:
        rgb = np.full((80, 80, 3), [185, 185, 190], dtype=np.uint8)
        rgb[19:62, 19:62] = [95, 120, 200]
        rgb[20:61, 20:61] = [15, 65, 210]

        seg = SAM2PointBoxSegmenter()
        res = seg.predict_interactive(rgb, pos_points_xy=[(40, 40)], tolerance=14.0)
        mask_bool = res["mask_bool"]
        alpha_01 = res["alpha_01"]

        self.assertTrue(bool(mask_bool[40, 40]))
        self.assertTrue(bool(mask_bool[19, 40]))
        self.assertFalse(bool(mask_bool[18, 40]))
        self.assertFalse(bool(mask_bool[17, 40]))
        self.assertEqual(float(alpha_01[17, 40]), 0.0)
        self.assertEqual(float(alpha_01[18, 40]), 0.0)

        if seg.is_available():
            seg_fb = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
            res_fb = seg_fb.predict_interactive(rgb, pos_points_xy=[(40, 40)], tolerance=14.0)
            self.assertTrue(bool(res_fb["mask_bool"][19, 40]))
            self.assertFalse(bool(res_fb["mask_bool"][18, 40]))
            self.assertFalse(bool(res_fb["mask_bool"][17, 40]))
            self.assertEqual(float(res_fb["alpha_01"][17, 40]), 0.0)
            self.assertEqual(float(res_fb["alpha_01"][18, 40]), 0.0)

    def test_boundary_luminance_ratio_absorption_and_core_alpha_fix1_fix6(self) -> None:
        # Part A: Real webp sample if present on disk
        sample_path = ROOT_DIR.parent / "sample" / "O1CN01mcVrYvKokfJ31Jg0_!!2207307885975.webp"
        if not sample_path.exists():
            sample_path = Path("sample/O1CN01mcVrYvKokfJ31Jg0_!!2207307885975.webp")
        if sample_path.exists():
            import cv2
            bgr = cv2.imread(str(sample_path))
            real_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            seg_real = SAM2PointBoxSegmenter()
            res_5 = seg_real.predict_interactive(
                real_rgb, pos_points_xy=[(484, 77)], tolerance=5.0, snap_to_panel_lines=True
            )
            # FIX-6: Interior core pixels x=503..507 on y=82 must have alpha_01 == 1.0 (not stuck at 0.88)
            for x in range(503, 508):
                self.assertAlmostEqual(
                    float(res_5["alpha_01"][82, x]),
                    1.0,
                    places=4,
                    msg=f"Core pixel x={x} alpha={res_5['alpha_01'][82, x]} stuck below 1.0",
                )
            # FIX-1: Anti-aliased boundary pixel x=508 gets 0.30 <= alpha_01 <= 0.70
            a_508 = float(res_5["alpha_01"][82, 508])
            self.assertTrue(
                0.30 <= a_508 <= 0.70,
                f"Boundary pixel x=508 alpha={a_508:.4f} expected in [0.30, 0.70]",
            )
            # Pure background bleed pixels x=510 and x=511 must have alpha_01 == 0.0 and mask_bool == False at tol=5.0
            self.assertFalse(bool(res_5["mask_bool"][82, 510]))
            self.assertEqual(float(res_5["alpha_01"][82, 510]), 0.0)
            self.assertFalse(bool(res_5["mask_bool"][82, 511]))
            self.assertEqual(float(res_5["alpha_01"][82, 511]), 0.0)

            # At tol=14.0 as well:
            res_14 = seg_real.predict_interactive(
                real_rgb, pos_points_xy=[(484, 77)], tolerance=14.0, snap_to_panel_lines=True
            )
            self.assertFalse(bool(res_14["mask_bool"][82, 510]))
            self.assertEqual(float(res_14["alpha_01"][82, 510]), 0.0)
            self.assertFalse(bool(res_14["mask_bool"][82, 511]))
            self.assertEqual(float(res_14["alpha_01"][82, 511]), 0.0)

        # Part B: Synthetic 1D horizontal boundary cross-section matching 07 handoff y=82, x=501..513
        h_syn, w_syn = 40, 60
        syn_rgb = np.full((h_syn, w_syn, 3), [192, 182, 185], dtype=np.uint8)
        for r in range(10, 30):
            syn_rgb[r, 10:31] = [165, 57, 48]
            syn_rgb[r, 31] = [176, 87, 88]
            syn_rgb[r, 32] = [191, 116, 118]
            syn_rgb[r, 33] = [199, 142, 143]   # boundary fringe (matches x=508)
            syn_rgb[r, 34] = [200, 157, 158]   # transition
            syn_rgb[r, 35] = [204, 171, 176]   # bleed (matches x=510)
            syn_rgb[r, 36] = [197, 176, 181]   # bleed (matches x=511)
            syn_rgb[r, 37:45] = [187, 177, 182] # pure bg
        seg_syn = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        res_syn = seg_syn.predict_interactive(
            syn_rgb, pos_points_xy=[(20, 20)], tolerance=5.0, snap_to_panel_lines=True
        )
        self.assertAlmostEqual(float(res_syn["alpha_01"][20, 25]), 1.0, places=4)
        self.assertAlmostEqual(float(res_syn["alpha_01"][20, 31]), 1.0, places=4)
        self.assertAlmostEqual(float(res_syn["alpha_01"][20, 32]), 1.0, places=4)
        a_syn_b = float(res_syn["alpha_01"][20, 33])
        self.assertTrue(0.30 <= a_syn_b <= 0.70, f"Synthetic fringe col 33 alpha={a_syn_b:.4f} expected in [0.30, 0.70]")
        self.assertFalse(bool(res_syn["mask_bool"][20, 35]))
        self.assertEqual(float(res_syn["alpha_01"][20, 35]), 0.0)
        self.assertFalse(bool(res_syn["mask_bool"][20, 36]))
        self.assertEqual(float(res_syn["alpha_01"][20, 36]), 0.0)

    def test_dual_contrast_neighbor_does_not_falsely_absorb_dark_inner_frame(self) -> None:
        h, w = 60, 60
        rgb = np.full((h, w, 3), [190, 185, 188], dtype=np.uint8)  # bright gray background (top/right)
        rgb[35:60, :] = [30, 32, 35]  # dark inner frame bottom
        rgb[:, 0:20] = [30, 32, 35]   # dark inner frame left
        rgb[15:35, 20:45] = [175, 50, 45]  # red part

        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        res = seg.predict_interactive(
            rgb, pos_points_xy=[(32, 25)], tolerance=5.0, snap_to_panel_lines=True
        )
        mask = res["mask_bool"]
        alpha = res["alpha_01"]

        self.assertTrue(mask[25, 32])
        self.assertAlmostEqual(float(alpha[25, 32]), 1.0, places=4)

        # Pure dark inner frame pixels must NOT be absorbed (alpha == 0.0, mask == False)
        self.assertFalse(bool(mask[40, 32]))
        self.assertEqual(float(alpha[40, 32]), 0.0)
        self.assertFalse(bool(mask[45, 32]))
        self.assertEqual(float(alpha[45, 32]), 0.0)
        self.assertFalse(bool(mask[25, 10]))
        self.assertEqual(float(alpha[25, 10]), 0.0)

    def test_multi_session_logits_isolation_and_inverted_box_normalization(self) -> None:
        seg = SAM2PointBoxSegmenter(checkpoint_path="checkpoints/non_existent.pt")
        img_a = np.full((64, 64, 3), [190, 190, 195], dtype=np.uint8)
        img_a[16:48, 16:48] = [195, 45, 45]
        # Inject bright specular glint at (32, 32) inside red part
        img_a[32, 32] = [250, 250, 252]

        img_b = np.full((64, 64, 3), [185, 185, 190], dtype=np.uint8)
        img_b[14:50, 14:50] = [30, 80, 205]

        # Seed logits in cache across two distinct images — switching images must NOT wipe earlier session logits
        sha_a = "sha_img_a_test"
        sha_b = "sha_img_b_test"
        seg._ensure_image_state(img_a, sha_a)
        seg._cache_part_logits(sha_a, "part_100", np.ones((16, 16), dtype=np.float32), (0, 0, 64, 64))
        seg._ensure_image_state(img_b, sha_b)
        seg._cache_part_logits(sha_b, "part_100", np.ones((16, 16), dtype=np.float32) * 2.0, (0, 0, 64, 64))
        self.assertIsNotNone(seg._part_logits_cache.get((sha_a, "part_100")))
        self.assertIsNotNone(seg._part_logits_cache.get((sha_b, "part_100")))

        # Scoped clear_part_logits removes only sha_a's part_100, preserving sha_b's part_100
        seg.clear_part_logits("part_100", image_sha256=sha_a)
        self.assertIsNone(seg._part_logits_cache.get((sha_a, "part_100")))
        self.assertIsNotNone(seg._part_logits_cache.get((sha_b, "part_100")))

        # Inverted drag box (x0 > x1, y0 > y1) + glint click must normalize cleanly without crashing
        res_inv = seg.predict_interactive(
            img_a,
            pos_points_xy=[(32, 32)],
            box_xyxy=(50, 50, 14, 14),
            tolerance=5.0,
        )
        self.assertTrue(bool(res_inv["mask_bool"][24, 24]))

    def test_ssrf_redirect_guard_multiline_pem_redaction_and_taxonomy_fixes(self) -> None:
        from paint_sim.collector.sources import normalize_url
        from paint_sim.collector.vision.gemini_detect import redact_secrets, save_secrets
        from paint_sim.collector.extractors.generic import extract_image_candidates_from_html
        from paint_sim.db.sample_ingestor import classify_paint_sample

        # 1. SSRF blocks loopback, localhost, link-local metadata, and redirect hop to 127.0.0.1
        for bad_url in (
            "http://127.0.0.1:8000/api/v1/health",
            "http://localhost:8765/api/settings",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:8000/",
        ):
            with self.assertRaises(ValueError):
                normalize_url(bad_url)

        def redir_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:8000/secret"})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            fetcher = PoliteFetcher(
                cache_dir=tmp_p / "c",
                respect_robots_txt=False,
                transport=httpx.MockTransport(redir_handler),
            )
            with self.assertRaises((FetchError, ValueError)):
                fetcher.fetch("https://mock.example.com/start")

            # 2. Multi-line PEM private key redaction inside json.dumps() payloads
            sec_path = tmp_p / "secrets.json"
            real_pem = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgw\n-----END PRIVATE KEY-----\n"
            save_secrets(
                {"auth_mode": "vertex_sa", "service_account_json": {"type": "service_account", "project_id": "p", "private_key": real_pem}},
                sec_path,
            )
            dumped = json.dumps({"err": f"Key leaked: {real_pem}"})
            redacted = redact_secrets(dumped, sec_path)
            self.assertNotIn("MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgw", redacted)

        # 3. Cloudflare comma-in-URL srcset preserved intact
        cands = extract_image_candidates_from_html(
            '<img src="https://cdn.example.com/cdn-cgi/image/w=320,q=80/swatch.jpg" '
            'srcset="https://cdn.example.com/cdn-cgi/image/w=320,q=80/swatch.jpg 320w, '
            'https://cdn.example.com/cdn-cgi/image/w=1600,q=90/swatch.jpg 1600w">',
            "https://cdn.example.com/post",
        )
        self.assertEqual(len(cands), 1)
        self.assertIn("w=1600,q=90/swatch.jpg", cands[0].image_url)

        # 4. Tamiya LP-7 lacquer vs XF-1 alcohol acrylic & GX100/LP-9 topcoat vs C47 candy clear
        lp7 = classify_paint_sample("LP-7", "Pure Red", brand_hint="tamiya")
        self.assertEqual(lp7["solvent_type"], "lacquer")
        self.assertFalse(lp7["needs_review"])
        gx100 = classify_paint_sample("GX100", "Super Clear III")
        self.assertEqual(gx100["paint_role"], "12_topcoat_varnish")
        self.assertFalse(gx100["needs_review"])
        lp9 = classify_paint_sample("LP-9", "Clear", brand_hint="tamiya")
        self.assertEqual(lp9["paint_role"], "12_topcoat_varnish")
        self.assertFalse(lp9["needs_review"])
        mig29 = classify_paint_sample("C308", "Gray FS36375 (MiG-29)")
        self.assertEqual(mig29["brand_id"], "gsi_creos")


if __name__ == "__main__":
    unittest.main(verbosity=2)
