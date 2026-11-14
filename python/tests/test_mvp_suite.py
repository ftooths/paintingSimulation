"""Phase 1 MVP Comprehensive Python Test Suite (Suites PAL, VIS, PERF, and Cross-Runtime Golden PHY)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

# Ensure `python/` is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from paint_sim.api.main import PaintSimAPIService, encode_rgb8_png
from paint_sim.archiver.gdrive_archiver import GoogleDriveArchiver
from paint_sim.config.prompt_loader import PromptConfigLoader
from paint_sim.db.repository import PaintDatabaseRepository
from paint_sim.db.sample_ingestor import (
    STANDARD_SAMPLE_DIRECTORY_TREE,
    VALID_BRAND_IDS,
    VALID_PAINT_ROLES,
    VALID_SOLVENT_TYPES,
    build_sample_filename,
    classify_paint_sample,
    evaluate_overcoat_compatibility,
    evaluate_solvent_solubility,
    ingest_sample_image,
    parse_sample_filename,
    resolve_highest_resolution_image_url,
)
from paint_sim.gemini.client import GeminiVisionClient
from paint_sim.nano_banana.refiner import NanoBananaRefiner
from paint_sim.palette.cluster_engine import AtomicRegion, PaletteAndPrecedenceEngine
from paint_sim.physics.engine_bridge import NativePaintCoreBridge
from paint_sim.vision.vram_manager import VRAMManager


class TestPhase1MVPSuite(unittest.TestCase):
    """Verifies TC-PAL-001..004, TC-VIS-001..004, TC-PERF-001/004, and Golden Stack Parity TC-PHY-006."""

    def setUp(self) -> None:
        self.bridge = NativePaintCoreBridge()
        self.db = PaintDatabaseRepository()

    # ========================================================================
    # Suite PAL: Palette-First Mode, OKLab k-Means++, Precedence & Protected Masks
    # ========================================================================

    def test_tc_pal_001_default_palette_first_mode_and_unified_db_catalog(self) -> None:
        engine = PaletteAndPrecedenceEngine(self.bridge)
        self.assertEqual(engine.active_selection_mode, "COLOR_GROUP")

        # Verify unified catalog contains all 58 paints across GSI Creos, Gaianotes, and Tamiya
        all_paints = self.db.list_paints()
        self.assertEqual(len(all_paints), 58)
        roles = {p["role"] for p in all_paints}
        for required_role in (
            "surfacer",
            "base_solid",
            "metallic",
            "mirror_chrome",
            "chameleon",
            "fluorescent",
            "fluorescent_clear",
            "clear_candy",
            "topcoat",
        ):
            self.assertIn(required_role, roles)

        # Verify surfacer grits #500, #1000, #1200, #1500 and Gloss Black Surfacer exist
        surfacers = self.db.list_paints(role="surfacer")
        grits = {s["surfacer_grit"] for s in surfacers}
        self.assertTrue({500, 1000, 1200, 1500}.issubset(grits))
        gloss_black_sf = self.db.get_paint("gsi_sf_gloss_black")
        self.assertIsNotNone(gloss_black_sf)
        self.assertAlmostEqual(gloss_black_sf["ggx_roughness"], 0.04, places=3)

        # Verify C47 Clear Red has unified Beer-Lambert absorption coefficients
        c47 = self.db.get_paint("gsi_c047")
        self.assertIsNotNone(c47)
        self.assertEqual(c47["beer_lambert_alpha"], [0.04, 1.65, 2.10])

    def test_tc_pal_002_precedence_hierarchy_region_part_color_group(self) -> None:
        engine = PaletteAndPrecedenceEngine(self.bridge)
        r_chest_white = AtomicRegion(1, "part_chest", "armor", "cg_white", [0.85, 0.85, 0.85])
        r_shoulder_main = AtomicRegion(2, "part_left_shoulder", "armor", "cg_white", [0.85, 0.85, 0.85])
        r_shoulder_sub = AtomicRegion(3, "part_left_shoulder", "armor", "cg_white", [0.85, 0.85, 0.85])

        # 1. Assign C1 White (Matte) to Color Group `cg_white`
        engine.color_group_stacks["cg_white"] = {"paint_id": "gsi_c001", "topcoat": "MATTE"}
        # 2. Assign C5 Blue (Gloss) to Part `part_left_shoulder`
        engine.part_stacks["part_left_shoulder"] = {"paint_id": "gsi_c005", "topcoat": "GLOSS"}
        # 3. Assign C9 Gold to Region 3 (`r_shoulder_sub`)
        engine.region_overrides[3] = {"paint_id": "gsi_c009", "topcoat": "GLOSS"}

        res1 = engine.resolve_region_stack(r_chest_white)
        res2 = engine.resolve_region_stack(r_shoulder_main)
        res3 = engine.resolve_region_stack(r_shoulder_sub)

        self.assertEqual(res1["paint_id"], "gsi_c001")
        self.assertEqual(res1["source"], "COLOR_GROUP_ASSIGNMENT")
        self.assertEqual(res2["paint_id"], "gsi_c005")
        self.assertEqual(res2["source"], "PART_ASSIGNMENT")
        self.assertEqual(res3["paint_id"], "gsi_c009")
        self.assertEqual(res3["source"], "REGION_OVERRIDE")

        # Clearing part assignment reverts main shoulder to C1 White while keeping sub-panel C9 Gold
        del engine.part_stacks["part_left_shoulder"]
        res2_after = engine.resolve_region_stack(r_shoulder_main)
        res3_after = engine.resolve_region_stack(r_shoulder_sub)
        self.assertEqual(res2_after["paint_id"], "gsi_c001")
        self.assertEqual(res3_after["paint_id"], "gsi_c009")

    def test_tc_pal_003_protected_region_immunity(self) -> None:
        engine = PaletteAndPrecedenceEngine(self.bridge)
        engine.color_group_stacks["cg_all"] = {"paint_id": "gsi_c002", "topcoat": "MATTE"}

        sensor = AtomicRegion(10, "part_eye", "clear_part", "cg_all", [0.1, 0.8, 0.2], is_protected=True)
        decal = AtomicRegion(11, "part_decal", "decal_sticker", "cg_all", [0.9, 0.1, 0.1], is_protected=True)
        bg = AtomicRegion(12, "part_bg", "background", "cg_all", [0.5, 0.5, 0.5], is_protected=True)

        for reg in (sensor, decal, bg):
            resolved = engine.resolve_region_stack(reg)
            self.assertFalse(resolved["is_modified"])
            self.assertTrue(resolved["is_protected"])

        # Unlocking explicitly allows painting
        sensor.unlocked_by_user = True
        resolved_unlocked = engine.resolve_region_stack(sensor)
        self.assertTrue(resolved_unlocked["is_modified"])
        self.assertEqual(resolved_unlocked["paint_id"], "gsi_c002")

    def test_tc_pal_004_eyedropper_delta_e00_and_oklab_kmeans_non_gundam(self) -> None:
        engine = PaletteAndPrecedenceEngine(self.bridge)
        r_seed = AtomicRegion(1, "p1", "armor", "cg_blue", [0.020, 0.070, 0.330])
        r_close = AtomicRegion(2, "p2", "armor", "cg_blue", [0.022, 0.074, 0.340])
        r_far = AtomicRegion(3, "p3", "armor", "cg_red", [0.570, 0.018, 0.026])
        r_prot = AtomicRegion(4, "p4", "clear_part", "cg_blue", [0.020, 0.070, 0.330], is_protected=True)

        matched = engine.select_regions_within_delta_e00(
            r_seed, [r_seed, r_close, r_far, r_prot], tolerance_delta_e00=3.5
        )
        self.assertEqual(matched, [1, 2])

        # Verify kit-agnostic OKLab k-Means++ clustering on a non-Gundam kit (Green Zaku + Purple + Dark Frame)
        synth_kit = np.zeros((48, 48, 3), dtype=np.float32)
        synth_kit[:20, :] = [0.06, 0.28, 0.07]   # Zaku Green armor
        synth_kit[20:36, :] = [0.22, 0.04, 0.38] # Eva Purple armor
        synth_kit[36:, :] = [0.04, 0.04, 0.05]   # Dark frame
        palette = engine.extract_oklab_kmeans_palette(synth_kit, k_init=5)
        group_ids = {item["color_group_id"] for item in palette}
        self.assertIn("cg_green", group_ids)
        self.assertIn("cg_purple", group_ids)

    def test_tc_pal_005_sample_ingestor_filename_and_excel_sync(self) -> None:
        fname = build_sample_filename(
            paint_code="C47",
            paint_name_en="Clear Red",
            surfacer_code="gsi_sf_gloss_black",
            undercoat_code="gsi_c009_gold",
            topcoat_code="gsi_gx100_gloss",
            coats=3,
            substrate_color="black",
            lighting_env="std",
            seq=1,
            ext="png",
        )
        self.assertNotIn(" ", fname)
        self.assertEqual(
            fname,
            "C47_clear_red__sf-gsi_sf_gloss_black+uc-gsi_c009_gold__tc-gsi_gx100_gloss__3c__sub-black_std_01.png",
        )
        parsed = parse_sample_filename(fname)
        self.assertEqual(parsed["paint_code"], "C47")
        self.assertEqual(parsed["surfacer_used"], "Y")
        self.assertEqual(parsed["surfacer_code"], "gsi_sf_gloss_black")
        self.assertEqual(parsed["undercoat_code"], "gsi_c009_gold")
        self.assertEqual(parsed["topcoat_used"], "Y")
        self.assertEqual(parsed["topcoat_code"], "gsi_gx100_gloss")
        self.assertEqual(parsed["main_coat_count"], 3.0)
        self.assertEqual(
            resolve_highest_resolution_image_url(
                "https://pm-paint.com/wp-content/uploads/2026/10/abc-320x180.jpg",
                "https://pm-paint.com/wp-content/uploads/2026/10/abc-320x180.jpg 320w, https://pm-paint.com/wp-content/uploads/2026/10/abc.jpg 2200w",
            ),
            "https://pm-paint.com/wp-content/uploads/2026/10/abc.jpg",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir) / "paint_samples"
            swatch_png = encode_rgb8_png(np.full((24, 24, 3), [210, 35, 45], dtype=np.uint8))
            # Verify automatic classification ("auto") resolves C47 Clear Red -> gsi_creos/lacquer/07_clear_candy/mr_color_c_clear
            row = ingest_sample_image(
                image_bytes=swatch_png,
                brand_id="auto",
                solvent_type="auto",
                paint_role="auto",
                product_line="auto",
                paint_code="C47",
                paint_name_en="Clear Red",
                paint_name_ko="C47 클리어 레드",
                surfacer_code="gsi_sf_gloss_black",
                undercoat_code="gsi_c009_gold",
                topcoat_code="gsi_gx100_gloss",
                coats=3.0,
                substrate_color="black",
                root_dir=tmp_root,
            )
            self.assertEqual(row["brand_id"], "gsi_creos")
            self.assertEqual(row["solvent_type"], "lacquer")
            self.assertEqual(row["paint_role"], "07_clear_candy")
            self.assertEqual(row["product_line"], "mr_color_c_clear")
            self.assertTrue((tmp_root / row["file_rel_path"]).exists())
            self.assertTrue((tmp_root / "paint_sample_master_index.csv").exists())
            self.assertTrue((tmp_root / "paint_sample_master_index.xlsx").exists())

        # Verify all 18 canonical brands, 7 solvent types, and 12 paint roles
        self.assertEqual(len(VALID_BRAND_IDS), 18)
        self.assertEqual(set(STANDARD_SAMPLE_DIRECTORY_TREE.keys()), set(VALID_BRAND_IDS))
        self.assertEqual(len(VALID_SOLVENT_TYPES), 7)
        self.assertEqual(len(VALID_PAINT_ROLES), 12)

        # Verify deterministic auto-classifier across specialty roles & brands
        self.assertEqual(classify_paint_sample("SM206", "Super Chrome Silver 2")["paint_role"], "04_mirror_chrome")
        self.assertEqual(classify_paint_sample("HLP110", "Stella Trillant Pearl")["product_line"], "aqueous_linkl_planet_color")
        self.assertEqual(classify_paint_sample("GM301", "Gundam Marker Pour Type Black")["paint_role"], "10_panel_wash_filter")
        self.assertEqual(classify_paint_sample("GP-08", "Premium Mirror Chrome", brand_hint="gaia_notes")["brand_id"], "gaia_notes")
        self.assertEqual(classify_paint_sample("XF-1", "Flat Black", brand_hint="tamiya")["solvent_type"], "acrylic_alcohol")

        # Regression guards for review findings (2026-10-26)
        self.assertNotEqual(classify_paint_sample("AK-123", "Chipping Effects")["brand_id"], "ipp")  # substring "ipp"
        chem = classify_paint_sample("71.001", "Chemical Green", brand_hint="vallejo")
        self.assertNotEqual(chem["paint_role"], "05_pearl_mica")  # substring "mica"
        self.assertTrue(classify_paint_sample("ZZ-1", "Unknown Brand Paint")["needs_review"])
        self.assertFalse(classify_paint_sample("C3", "Red")["needs_review"])
        self.assertEqual(parse_sample_filename(build_sample_filename("Ex 07", "Ex Silver"))["paint_code"], "EX-07")
        import io as _io
        from PIL import Image as _Image
        with tempfile.TemporaryDirectory() as tmpdir2:
            r2 = Path(tmpdir2)
            jpg = _io.BytesIO()
            _Image.new("RGB", (40, 40), (200, 30, 40)).save(jpg, "JPEG")
            row_a = ingest_sample_image(jpg.getvalue(), paint_code="C3", paint_name_en="Red", root_dir=r2)
            self.assertEqual((r2 / row_a["file_rel_path"]).read_bytes()[:4], b"\x89PNG")  # JPEG re-encoded
            png2 = encode_rgb8_png(np.full((40, 40, 3), [10, 200, 40], dtype=np.uint8))
            row_b = ingest_sample_image(png2, paint_code="C3", paint_name_en="Red", root_dir=r2)
            self.assertNotEqual(row_a["file_rel_path"], row_b["file_rel_path"])  # no silent overwrite
            self.assertNotEqual(row_a["sample_id"], row_b["sample_id"])
            with self.assertRaises(ValueError):
                ingest_sample_image(b"not an image", paint_code="C3", paint_name_en="Red", root_dir=r2)

        # Verify Solvent Solubility & Overcoating Compatibility Matrices (Section 3)
        std_gm = evaluate_solvent_solubility("lighter_fluid", "alcohol_marker_ink", is_pour_type_marker=False)
        pour_gm = evaluate_solvent_solubility("lighter_fluid", "alcohol_marker_ink", is_pour_type_marker=True)
        self.assertEqual(std_gm["symbol"], "X")
        self.assertEqual(pour_gm["symbol"], "◎")
        self.assertTrue(evaluate_overcoat_compatibility("lacquer", "enamel")["is_safe"])
        self.assertTrue(evaluate_overcoat_compatibility("enamel", "lacquer")["is_incompatible"])
        self.assertTrue(evaluate_overcoat_compatibility("mirror_chrome", "urethane_2k")["is_incompatible"])

        # Verify the 63 ingested pm-paint.com spoon samples & strict 15/15 GGX roughness monotonicity
        import csv as _csv
        sample_root = ROOT_DIR / "sample" / "paint_samples"
        master_csv = sample_root / "paint_sample_master_index.csv"
        if master_csv.exists():
            with open(master_csv, "r", encoding="utf-8-sig", newline="") as f:
                master_rows = list(_csv.DictReader(f))
            # Group by metadata, not row position, so future ingests do not break the test.
            pm_rows = [r for r in master_rows if r.get("source_url", "").startswith("https://pm-paint.com/")]
            self.assertGreaterEqual(len(pm_rows), 63)
            groups = {}
            for r in pm_rows:
                if r["surfacer_code"] != "none":
                    groups.setdefault((r["paint_code"], r["surfacer_code"]), {})[r["final_finish"]] = r
            self.assertGreaterEqual(len(groups), 15)
            for grp_map in groups.values():
                grp = [grp_map["pearl"], grp_map["gloss"], grp_map["semi_gloss"], grp_map["matte"]]
                alphas = [json.loads(r["calibrated_optical_params"])["estimated_alpha_ggx"] for r in grp]
                self.assertLess(alphas[1], alphas[2], f"Gloss must be smoother than Semi-Gloss in {grp[0]['sample_id']}")
                self.assertLess(alphas[2], alphas[3], f"Semi-Gloss must be smoother than Matte in {grp[0]['sample_id']}")

    # ========================================================================
    # Suite VIS: Gemini Budget, Sample-Point Gate, Nano Banana & Hot-Reload
    # ========================================================================

    def test_tc_vis_001_gemini_call_budget_and_caching(self) -> None:
        def mock_llm(sys_inst: str, prompt: str, img: bytes) -> dict:
            return {"subject_type": "gunpla", "parts": [], "color_palette": []}

        client = GeminiVisionClient(max_calls_per_image=2, mock_transport=mock_llm)
        sample_img = b"\x89PNG\r\n\x1a\nTEST_KIT_IMAGE_BYTES"

        # First run: Call 1 + Call 2 -> 2 network calls total
        client.call_1_scene_and_palette(sample_img)
        client.call_2_som_verification(sample_img, b"SOM_IMAGE")
        self.assertEqual(client.total_network_calls, 2)
        self.assertEqual(client.get_image_call_count(sample_img), 2)

        # Re-uploading or 50 interactive edits -> 0 additional network calls (hits SHA-256 cache)
        for _ in range(50):
            client.call_1_scene_and_palette(sample_img)
            client.call_2_som_verification(sample_img, b"SOM_IMAGE")
        self.assertEqual(client.total_network_calls, 2)

        # Verify offline client without API key or mock_transport explicitly reports stub status & 0 network calls
        offline_client = GeminiVisionClient(max_calls_per_image=2, api_key="", mock_transport=None)
        som_res = offline_client.call_2_som_verification(sample_img, b"SOM_IMAGE")
        self.assertEqual(som_res["status"], "stub_no_api_key")
        self.assertTrue(som_res.get("is_stub"))
        self.assertEqual(offline_client.total_network_calls, 0)

    def test_tc_vis_002_gemini_sample_point_validity_gate(self) -> None:
        engine = PaletteAndPrecedenceEngine(self.bridge)
        albedo = np.full((100, 100, 3), [0.02, 0.07, 0.33], dtype=np.float32)
        clip_mask = np.zeros((100, 100), dtype=np.uint8)
        panel_mask = np.zeros((100, 100), dtype=np.uint8)

        # Inject a clipped white highlight at (20, 20) and a dark panel line at (40, 40)
        albedo[20, 20] = [0.99, 0.99, 0.99]
        clip_mask[20, 20] = 1
        albedo[40, 40] = [0.005, 0.005, 0.005]
        panel_mask[40, 40] = 250

        # 4 sample points in normalized 0..1000 coords: 2 corrupted, 2 valid
        samples_norm = [[202, 202], [404, 404], [600, 600], [750, 750]]
        measured_rgb, valid_count = engine.fuse_with_gemini_samples(albedo, clip_mask, panel_mask, samples_norm)

        self.assertEqual(valid_count, 2)
        de = self.bridge.delta_e00_linear_rgb(measured_rgb, [0.02, 0.07, 0.33])
        self.assertLess(de, 0.5)

    def test_tc_vis_003_fault_tolerant_fallback_on_gemini_outage(self) -> None:
        def failing_transport(sys_inst: str, prompt: str, img: bytes) -> dict:
            raise TimeoutError("HTTP 503 Service Unavailable / Gemini Timeout")

        client = GeminiVisionClient(max_calls_per_image=2, mock_transport=failing_transport)
        res = client.call_1_scene_and_palette(b"TIMEOUT_TEST_IMAGE")
        self.assertEqual(res["status"], "completed (degraded_no_llm)")
        self.assertGreaterEqual(len(res["color_palette"]), 4)

    def test_tc_vis_004_vertex_ai_nano_banana_oklab_color_lock_and_hot_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_prompts = Path(tmpdir) / "prompts.json"
            shutil.copyfile(ROOT_DIR / "config" / "prompts.json", tmp_prompts)

            # Create valid 16x16 PNGs to test OKLab color-lock postprocessing (B6 fix)
            orig_png = encode_rgb8_png(np.full((16, 16, 3), [120, 120, 120], dtype=np.uint8))
            phys_png = encode_rgb8_png(np.full((16, 16, 3), [210, 30, 35], dtype=np.uint8))
            drifted_nb_png = encode_rgb8_png(np.full((16, 16, 3), [220, 95, 30], dtype=np.uint8))

            loader = PromptConfigLoader(tmp_prompts)
            refiner = NanoBananaRefiner(
                project_id="test-gcp-project",
                prompt_loader=loader,
                mock_vertex_transport=lambda payload: drifted_nb_png,
            )

            # 1. Modify `special_coating_modifiers.mirror_chrome` on disk without restarting
            time.sleep(0.02)
            cfg = json.loads(tmp_prompts.read_text(encoding="utf-8"))
            custom_marker = "CUSTOM_USER_EDITED_MIRROR_CHROME_DIRECTIVE_V99"
            cfg["vertex_ai_nano_banana"]["dual_image_refinement_prompt"]["special_coating_modifiers"][
                "mirror_chrome"
            ] = custom_marker
            tmp_prompts.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            res = refiner.refine_dual_image(
                orig_wb_png=orig_png,
                physics_sim_png=phys_png,
                recipe_context={
                    "workflow_summary": "Gloss Black Surfacer + SM206 Super Chrome Silver 2",
                    "finish_key": "intrinsic_paint_or_surfacer",
                    "special_coating_key": "mirror_chrome",
                    "lighting_env_key": "dark_room_uv_light",
                },
            )
            self.assertIn(custom_marker, res["prompt_used"])
            self.assertIsNotNone(refiner.last_request_payload)
            self.assertEqual(refiner.last_request_payload["primary_model"], "gemini-nano-banana-2.1")
            self.assertEqual(refiner.last_request_payload["fallback_model"], "gemini-3.1-flash-image")
            for forbidden_param in ("temperature", "top_p", "topP", "top_k", "topK", "seed"):
                self.assertNotIn(forbidden_param, refiner.last_request_payload)
            self.assertEqual(refiner.last_request_payload["image_1_orig_png"], orig_png)
            self.assertEqual(refiner.last_request_payload["image_2_physics_png"], phys_png)
            GoogleDriveArchiver.validate_image_payload("07_nano_banana_refined.png", res["refined_png_bytes"])

            # 2. Malformed JSON edit falls back gracefully without crashing
            time.sleep(0.02)
            tmp_prompts.write_text("{ INVALID JSON SYNTAX !!!", encoding="utf-8")
            fallback_cfg = loader.get_config()
            self.assertIn("vertex_ai_nano_banana", fallback_cfg)

    # ========================================================================
    # Suite PERF & Cross-Runtime Golden Parity (TC-PERF-001, TC-PERF-004, TC-PHY-006)
    # ========================================================================

    def test_tc_perf_001_rtx_3070_8gb_vram_ceiling_and_report(self) -> None:
        vram = VRAMManager(low_vram_mode=True, safety_ceiling_gb=7.0)
        for _ in range(10):
            with vram.acquire_stage("sam_segmentation", 1.8):
                self.assertLessEqual(vram.current_allocated_gb(), 6.5)
            with vram.acquire_stage("intrinsics_and_normals", 4.3):
                self.assertLessEqual(vram.current_allocated_gb(), 6.5)
        self.assertLessEqual(vram.peak_estimated_vram_gb, 5.2)
        report = vram.get_memory_report()
        self.assertEqual(report["mode"], "budget_estimation_only")
        self.assertIn("actual_cuda_allocated_gb", report)

    def test_tc_perf_004_google_drive_dataset_archiver_integrity_and_fake_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = PaintSimAPIService(archive_dir=Path(tmpdir))
            job_res = svc.create_job(b"\x89PNG\r\n\x1a\nRAW_GUNPLA")
            export_res = svc.export_job(
                job_id=job_res["job_id"],
                recipe={
                    "workflow_summary": "C4 Yellow + C175 Fluorescent Green",
                    "finish_key": "intrinsic_paint_or_surfacer",
                    "special_coating_key": "fluorescent",
                    "lighting_env_key": "dark_room_uv_light",
                },
                run_nano_banana=True,
                lighting_env="dark_uv_365nm",
            )
            bundle_dir = Path(export_res["archive_bundle_path"])
            for rel_file in GoogleDriveArchiver.REQUIRED_BUNDLE_FILES:
                full_path = bundle_dir / rel_file
                self.assertTrue(full_path.exists(), f"Missing archived file: {rel_file}")
                self.assertGreater(full_path.stat().st_size, 0)
                if rel_file.endswith((".png", ".jpg")):
                    GoogleDriveArchiver.validate_image_payload(rel_file, full_path.read_bytes())

            # Verify that fake placeholder bytes (e.g. missing IHDR/IEND) are rejected by GoogleDriveArchiver
            with self.assertRaises(ValueError):
                GoogleDriveArchiver.validate_image_payload("02_region_id_map.png", b"\x89PNG\r\n\x1a\nREGION_ID_16U")
            with self.assertRaises(ValueError):
                GoogleDriveArchiver.validate_image_payload("03_som_verification.jpg", b"\xff\xd8\xff\xe0SOM_OVERLAY")

    def test_tc_phy_006_cross_runtime_golden_stack_parity(self) -> None:
        # 1. Verify zero-copy pointer identity
        arr = np.zeros((128, 128, 3), dtype=np.float32)
        py_ptr, cpp_ptr = self.bridge.verify_zero_copy_pointer(arr)
        self.assertEqual(py_ptr, cpp_ptr)

        if not self.bridge.has_native_dll:
            return

        # 2. Scalar K-M channel parity
        for r_prev in (0.02, 0.18, 0.55, 0.88):
            for r_inf, h in ((0.94, 0.46), (0.57, 0.52), (0.935, 0.84), (0.015, 0.97)):
                a, b, sx = self.bridge.compute_km_constants(r_inf, h)
                cpp_val = self.bridge.lib.paintcore_evaluate_km_channel(r_prev, a, b, sx * 1.5)
                py_val = self.bridge.evaluate_km_channel_py(r_prev, a, b, sx * 1.5)
                self.assertAlmostEqual(cpp_val, py_val, places=5)

        # 3. Full Golden Pixel-Stack Parity (Delta E_00 < 0.05 across all 7 coating scenarios)
        golden_scenarios = [
            # Scenario A: Identity / Pure White Highlight Preservation (A1 Regression Guard)
            {
                "name": "identity_white_preservation",
                "orig_wb": [1.0, 1.0, 1.0],
                "base_albedo": [0.92, 0.92, 0.92],
                "shading_d": [1.0, 1.0, 1.0],
                "specular_s": [0.08, 0.08, 0.08],
                "is_modified": False,
                "layers": [],
                "topcoat_type": 0,
                "topcoat_roughness": 0.22,
                "env_mode": 0,
            },
            # Scenario B: Bare Plastic + Matte Topcoat Only (GX114)
            {
                "name": "bare_plastic_matte_topcoat",
                "orig_wb": [0.05, 0.12, 0.38],
                "base_albedo": [0.04, 0.10, 0.34],
                "shading_d": [0.95, 0.95, 0.95],
                "specular_s": [0.18, 0.18, 0.18],
                "is_modified": True,
                "layers": [],
                "topcoat_type": 3,
                "topcoat_roughness": 0.75,
                "env_mode": 0,
            },
            # Scenario C: Pink Surfacer (#1000) + C3 Red Solid Base
            {
                "name": "pink_surfacer_plus_c3_red",
                "orig_wb": [0.12, 0.12, 0.12],
                "base_albedo": [0.10, 0.10, 0.10],
                "shading_d": [0.92, 0.92, 0.92],
                "specular_s": [0.04, 0.04, 0.04],
                "is_modified": True,
                "layers": [
                    {"role": 0, "masstone_R_inf": [0.79, 0.34, 0.44], "hiding_power_h": [0.84, 0.88, 0.89], "intrinsic_roughness": 0.52, "coat_count": 1.5},
                    {"role": 1, "masstone_R_inf": [0.57, 0.018, 0.026], "hiding_power_h": [0.52, 0.90, 0.92], "intrinsic_roughness": 0.07, "coat_count": 2.0},
                ],
                "topcoat_type": 0,
                "topcoat_roughness": 0.07,
                "env_mode": 0,
            },
            # Scenario D: C9 Gold Metallic + C47 Clear Red Candy Overcoat
            {
                "name": "candy_ruby_gold_plus_c47",
                "orig_wb": [0.50, 0.50, 0.50],
                "base_albedo": [0.45, 0.45, 0.45],
                "shading_d": [1.05, 1.05, 1.05],
                "specular_s": [0.22, 0.22, 0.22],
                "normal_vec": [0.15, -0.20, 0.968],
                "px_x": 37,
                "px_y": 53,
                "is_modified": True,
                "layers": [
                    {"role": 2, "masstone_R_inf": [0.68, 0.43, 0.075], "hiding_power_h": [0.93, 0.93, 0.95], "intrinsic_roughness": 0.15, "coat_count": 2.5},
                    {"role": 4, "beer_alpha": [0.04, 1.65, 2.10], "intrinsic_roughness": 0.04, "coat_count": 1.2},
                ],
                "topcoat_type": 1,
                "topcoat_roughness": 0.04,
                "env_mode": 0,
            },
            # Scenario E: Gloss Black Surfacer + SM206 Mirror Chrome
            {
                "name": "mirror_chrome_over_gloss_black",
                "orig_wb": [0.20, 0.20, 0.20],
                "base_albedo": [0.18, 0.18, 0.18],
                "shading_d": [0.88, 0.88, 0.88],
                "specular_s": [0.12, 0.12, 0.12],
                "normal_vec": [0.10, -0.35, 0.931],
                "is_modified": True,
                "layers": [
                    {"role": 0, "masstone_R_inf": [0.008, 0.008, 0.009], "hiding_power_h": [0.98, 0.98, 0.98], "intrinsic_roughness": 0.04, "coat_count": 2.5},
                    {"role": 3, "intrinsic_roughness": 0.02, "chrome_f0": 0.95, "coat_count": 1.0},
                ],
                "topcoat_type": 0,
                "topcoat_roughness": 0.04,
                "env_mode": 0,
            },
            # Scenario F: CP-01 Chameleon at Grazing Angle
            {
                "name": "chameleon_cp01_grazing",
                "orig_wb": [0.15, 0.15, 0.15],
                "base_albedo": [0.02, 0.02, 0.02],
                "shading_d": [1.0, 1.0, 1.0],
                "specular_s": [0.05, 0.05, 0.05],
                "normal_vec": [0.0, 0.936, 0.35],
                "is_modified": True,
                "layers": [
                    {"role": 5, "chameleon_stops": [0.65, 0.18, 210.0, 0.52, 0.22, 265.0, 0.48, 0.24, 325.0], "intrinsic_roughness": 0.05, "coat_count": 2.0},
                ],
                "topcoat_type": 1,
                "topcoat_roughness": 0.04,
                "env_mode": 0,
            },
            # Scenario G: C175 Fluorescent Green under 365nm UV Blacklight
            {
                "name": "fluorescent_c175_uv_365nm",
                "orig_wb": [0.80, 0.80, 0.80],
                "base_albedo": [0.85, 0.85, 0.85],
                "shading_d": [0.90, 0.90, 0.90],
                "specular_s": [0.02, 0.02, 0.02],
                "is_modified": True,
                "layers": [
                    {
                        "role": 7,
                        "masstone_R_inf": [0.04, 0.92, 0.06],
                        "hiding_power_h": [0.78, 0.42, 0.85],
                        "intrinsic_roughness": 0.26,
                        "fluo_quantum_yield": 0.80,
                        "fluo_peak_rgb": [0.12, 1.00, 0.18],
                        "coat_count": 2.0,
                    },
                ],
                "topcoat_type": 0,
                "topcoat_roughness": 0.26,
                "env_mode": 2,
                "ambient_dim": 0.05,
                "uv_intensity": 1.8,
            },
        ]

        for sc in golden_scenarios:
            kwargs = {
                "orig_wb": sc["orig_wb"],
                "base_albedo": sc["base_albedo"],
                "shading_d": sc["shading_d"],
                "specular_s": sc["specular_s"],
                "normal_vec": sc.get("normal_vec", (0.0, 0.0, 1.0)),
                "is_modified": sc["is_modified"],
                "layers": sc["layers"],
                "topcoat_type": sc["topcoat_type"],
                "topcoat_roughness": sc["topcoat_roughness"],
                "env_mode": sc["env_mode"],
                "ambient_dim": sc.get("ambient_dim", 1.0),
                "uv_intensity": sc.get("uv_intensity", 0.0),
                "px_x": sc.get("px_x", 0),
                "px_y": sc.get("px_y", 0),
            }
            cpp_rgb, cpp_alpha = self.bridge.evaluate_pixel_stack_cpp(**kwargs)
            py_rgb, py_alpha = self.bridge.evaluate_pixel_stack_py(**kwargs)
            de00 = self.bridge.delta_e00_linear_rgb(cpp_rgb, py_rgb)
            self.assertLess(
                de00, 0.05,
                f"Golden stack parity failed on '{sc['name']}': C++={cpp_rgb} vs Py={py_rgb} (dE00={de00:.5f})"
            )
            self.assertAlmostEqual(cpp_alpha, py_alpha, places=4)



class TestHQSAM2Segmentation(unittest.TestCase):
    """Tests for HQ-SAM 2 / SAM 2.1 interactive segmentation and panel-line snapping."""

    def setUp(self) -> None:
        self.bridge = NativePaintCoreBridge()

    def test_bridge_panel_line_snap_and_guided_matte(self) -> None:
        h, w = 48, 48
        lin_rgb = np.full((h, w, 3), 0.75, dtype=np.float32)
        # Recessed dark vertical panel line at x = 24
        lin_rgb[:, 24, :] = 0.04
        panel_mask = self.bridge.compute_panel_lines(lin_rgb)
        self.assertEqual(panel_mask.shape, (h, w))
        self.assertEqual(panel_mask.dtype, np.uint8)
        self.assertGreater(int(panel_mask[24, 24]), 120)

        # Raw mask overshooting panel line to x = 26
        raw_mask = np.zeros((h, w), dtype=bool)
        raw_mask[:, :26] = True
        alpha = self.bridge.snap_and_matte_part_mask(
            lin_rgb, panel_mask, raw_mask, snap_radius=4, matte_radius=4, matte_eps=1e-4
        )
        self.assertEqual(alpha.shape, (h, w))
        self.assertEqual(alpha.dtype, np.float32)
        self.assertGreater(float(alpha[24, 10]), 0.95)
        self.assertLess(float(alpha[24, 35]), 0.05)
        # Overshoot pixel at x=25 must be snapped back across ridge
        self.assertLess(float(alpha[24, 25]), 0.50)

    def test_bridge_native_vs_fallback_parity(self) -> None:
        if not (self.bridge.lib and hasattr(self.bridge.lib, "paintcore_compute_panel_lines")):
            self.skipTest("Native paintcore_compute_panel_lines symbol not in current DLL; run .\\run_local_tests.ps1 to recompile")
        h, w = 32, 32
        lin_rgb = np.full((h, w, 3), 0.70, dtype=np.float32)
        lin_rgb[:, 16, :] = 0.05
        raw_mask = np.zeros((h, w), dtype=bool)
        raw_mask[:, :18] = True
        panel_mask = self.bridge.compute_panel_lines(lin_rgb)
        native_alpha = self.bridge.snap_and_matte_part_mask(lin_rgb, panel_mask, raw_mask, snap_radius=4, matte_radius=4)
        py_alpha = self.bridge._snap_and_matte_numpy_fallback(lin_rgb, panel_mask, raw_mask, snap_radius=4, matte_radius=4)
        max_diff = float(np.max(np.abs(native_alpha - py_alpha)))
        self.assertLess(max_diff, 0.10, f"Parity deviation too high: {max_diff:.4f}")

    def test_vram_manager_resident_lease_and_idle_offload(self) -> None:
        mgr = VRAMManager(low_vram_mode=True, safety_ceiling_gb=7.0)

        class FakeModule:
            def __init__(self) -> None:
                self.device = "cpu"
                self.calls: list[str] = []

            def to(self, device: str) -> "FakeModule":
                self.device = device
                self.calls.append(device)
                return self

        fake_model = FakeModule()
        mgr.register_model("sam_segmentation", fake_model)

        # 1. Non-resident acquire offloads on context exit
        with mgr.acquire_stage("sam_segmentation", vram_footprint_gb=2.0, keep_resident=False):
            self.assertIn("sam_segmentation", mgr.active_vram_models)
        self.assertNotIn("sam_segmentation", mgr.active_vram_models)
        self.assertEqual(fake_model.calls[-1], "cpu")

        # 2. Resident acquire preserves stage across context exit
        cpu_offloads_before = fake_model.calls.count("cpu")
        with mgr.acquire_stage("sam_segmentation", vram_footprint_gb=2.0, keep_resident=True):
            pass
        self.assertIn("sam_segmentation", mgr.active_vram_models)
        self.assertIn("sam_segmentation", mgr.resident_stages)
        self.assertEqual(fake_model.calls.count("cpu"), cpu_offloads_before)

        # 3. Acquiring a different stage in low_vram_mode evicts the resident stage
        with mgr.acquire_stage("nano_banana_refinement", vram_footprint_gb=3.0, keep_resident=False):
            self.assertNotIn("sam_segmentation", mgr.active_vram_models)
            self.assertNotIn("sam_segmentation", mgr.resident_stages)
            self.assertEqual(fake_model.calls.count("cpu"), cpu_offloads_before + 1)

        # 4. Idle timeout offload
        with mgr.acquire_stage("sam_segmentation", vram_footprint_gb=2.0, keep_resident=True):
            pass
        mgr.stage_last_used["sam_segmentation"] = time.time() - 200.0
        offloaded = mgr.offload_idle(max_idle_s=120.0)
        self.assertIn("sam_segmentation", offloaded)
        self.assertNotIn("sam_segmentation", mgr.active_vram_models)

    def test_api_session_and_segment_studio_part_pos_neg(self) -> None:
        svc = PaintSimAPIService()
        sess = svc.create_segment_session(b"synthetic_kit_raster")
        session_id = sess["session_id"]
        self.assertEqual(sess["width"], 32)
        self.assertEqual(sess["height"], 32)

        # 1. Positive click on left blue armor
        res_pos = svc.segment_studio_part(
            session_id=session_id,
            pos_points_xy=[(8, 8)],
            neg_points_xy=[],
            part_id="part_100",
            use_prev_mask=False,
        )
        self.assertGreater(res_pos["pixel_count"], 0)
        self.assertEqual(len(res_pos["mask_indices"]), res_pos["pixel_count"])
        self.assertEqual(len(res_pos["mask_alphas_u8"]), res_pos["pixel_count"])

        # 2. Add negative click on center hatch
        res_neg = svc.segment_studio_part(
            session_id=session_id,
            pos_points_xy=[(8, 8)],
            neg_points_xy=[(16, 16)],
            part_id="part_100",
            use_prev_mask=True,
        )
        center_idx = 16 * 32 + 16
        self.assertNotIn(center_idx, set(res_neg["mask_indices"]))

        # 3. Invalid image bytes raises ValueError
        with self.assertRaises(ValueError):
            svc.create_segment_session(b"invalid_corrupt_data_xyz")

    def test_zlib_bbox_alpha_roundtrip(self) -> None:
        from paint_sim.api.main import decode_alpha_bbox_zlib, encode_alpha_bbox_zlib

        w, h = 64, 64
        y_coords, x_coords = np.meshgrid(np.arange(15, 26), np.arange(10, 21), indexing="ij")
        indices = (y_coords.ravel() * w + x_coords.ravel()).astype(np.uint32)
        alphas = np.full(indices.shape, 240, dtype=np.uint8)

        encoded = encode_alpha_bbox_zlib(indices, alphas, w, h)
        self.assertEqual(encoded["alpha_encoding"], "zlib-u8-bbox")
        self.assertEqual(encoded["bbox_xyxy"], [10, 15, 20, 25])
        self.assertLess(len(encoded["alpha_b64"]), 200)

        dec_indices, dec_alphas = decode_alpha_bbox_zlib(encoded, w, h)
        np.testing.assert_array_equal(np.sort(dec_indices), np.sort(indices))
        self.assertEqual(len(dec_alphas), len(alphas))

    def test_fastapi_segment_session_routes(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from paint_sim.api.main import create_fastapi_app
        except ImportError:
            self.skipTest("FastAPI not installed in environment")

        app = create_fastapi_app()
        client = TestClient(app)

        # Health endpoint
        r_health = client.get("/api/v1/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertIn("segmentation_engine", r_health.json())

        # Create session
        r_sess = client.post("/api/v1/segment-sessions", content=b"synthetic_kit_raster")
        self.assertEqual(r_sess.status_code, 200)
        sess_json = r_sess.json()
        session_id = sess_json["session_id"]
        self.assertIn("model_bbox_xyxy", sess_json)
        self.assertIn("roi_bbox_xyxy", sess_json)
        self.assertIn("object_alpha_b64", sess_json)
        self.assertIn("engine_warmed", sess_json)

        # Segment part with tolerance parameter
        payload = {
            "pos_points_xy": [[8, 8]],
            "neg_points_xy": [],
            "part_id": "part_route_test",
            "use_prev_mask": False,
            "tolerance": 7.5,
        }
        r_seg = client.post(f"/api/v1/segment-sessions/{session_id}/segment-part", json=payload)
        self.assertEqual(r_seg.status_code, 200)
        data = r_seg.json()
        self.assertEqual(data["alpha_encoding"], "zlib-u8-bbox")
        self.assertIn("alpha_b64", data)
        self.assertAlmostEqual(float(data["tolerance"]), 7.5, places=2)

        # Delete logits
        r_del = client.delete(f"/api/v1/segment-sessions/{session_id}/parts/part_route_test/logits")
        self.assertEqual(r_del.status_code, 200)

    def test_studio_html_contains_ai_segmentation_and_pos_neg_refinement_controls(self) -> None:
        html = (ROOT_DIR / "frontend" / "local_test_studio.html").read_text(encoding="utf-8")
        self.assertIn('id="sel-seg-engine"', html)
        self.assertIn('id="btn-mask-edit-mode"', html)
        self.assertIn("ai_sam2", html)
        self.assertIn("local_oklab_flood", html)
        self.assertIn("isMaskEditMode", html)
        self.assertIn("buildMultiSeedMask", html)
        self.assertIn("ensureSegmentSession", html)
        self.assertIn("requestAISegmentPart", html)
        self.assertIn("DecompressionStream", html)
        self.assertIn("posPoints", html)
        self.assertIn("negPoints", html)
        # Phase 1-4 UI improvements
        self.assertIn("renderOverlayOnly", html)
        self.assertIn("AbortSignal.timeout(5000)", html)
        self.assertIn("buildPartContourPath", html)
        self.assertIn("rgba(0,0,0,0.85)", html)
        self.assertIn("#FFFFFF", html)
        self.assertNotIn('ctx.fillText("+",', html)
        self.assertNotIn('ctx.fillText("-",', html)
        self.assertIn("tolerance:", html)
        self.assertIn('id="sel-export-bg-mode"', html)
        self.assertIn("object_transparent", html)
        self.assertIn("object_crop", html)
        self.assertIn("quadraticCurveTo", html)
        self.assertIn('isMaskEditMode && selectionMode === "PART"', html)
        self.assertIn("partAlbR", html)
        # Contract 1-5 Frontend fixes
        self.assertIn("baseShadingMap", html)
        self.assertIn("baseAlbedoMap", html)
        self.assertIn("part._reqSeq", html)
        self.assertIn('typeof data.alpha_b64 === "string"', html)
        self.assertIn("renderFrame(true)", html)
        # Per-part boundary tolerance and edge rim halo elimination
        self.assertIn("part-tol-slider", html)
        self.assertIn("isBoundaryRim", html)

        # Verify zero JS syntax errors in all <script> tags via Node.js vm.Script if node is installed
        if shutil.which("node"):
            import subprocess

            check_js = (
                "const fs=require('fs'),vm=require('vm');"
                "const h=fs.readFileSync('frontend/local_test_studio.html','utf8');"
                "const s=[...h.matchAll(/<script(?:\\s[^>]*)?>([\\s\\S]*?)<\\/script>/g)];"
                "s.forEach((m,i)=>{if(m[1].trim())new vm.Script(m[1],{filename:'s_'+i+'.js'});});"
            )
            proc = subprocess.run(["node", "-e", check_js], cwd=str(ROOT_DIR), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"JS SyntaxError in local_test_studio.html: {proc.stderr}")

    def _run_studio_vm_json(self, js_test_code: str) -> dict:
        if not shutil.which("node"):
            self.skipTest("Node.js not installed in environment")
        import subprocess

        harness_js = f"""
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('frontend/local_test_studio.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\\s[^>]*)?>([\\s\\S]*?)<\\/script>/g)]
  .map(m => m[1])
  .filter(s => s.trim().length > 0);

const elCache = new Map();
function makeEl(id) {{
  if (elCache.has(id)) return elCache.get(id);
  const el = {{
    id,
    value: id === 'split-slider' ? '0' : (id === 'rng-tolerance' ? '5.0' : 'none'),
    checked: false,
    width: 64,
    height: 64,
    textContent: '',
    innerHTML: '',
    style: {{}},
    classList: {{ add(){{}}, remove(){{}}, toggle(){{}} }},
    addEventListener(){{}},
    appendChild(){{}},
    querySelector(){{ return null; }},
    querySelectorAll(){{ return []; }},
    getBoundingClientRect(){{ return {{ left: 0, top: 0, right: 64, bottom: 64, width: 64, height: 64 }}; }},
    getContext() {{
      return {{
        imageSmoothingEnabled: true,
        imageSmoothingQuality: 'high',
        createImageData: (w, h) => ({{ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }}),
        putImageData: (imgData) => {{ el._lastImageData = imgData; }},
        getImageData: (x, y, w, h) => el._mockSourceImageData || ({{ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }}),
        drawImage(){{}},
        save(){{}},
        restore(){{}},
        beginPath(){{}},
        stroke(){{}},
        setLineDash(){{}}
      }};
    }}
  }};
  elCache.set(id, el);
  return el;
}}

const sandbox = {{
  Math, Number, String, Boolean, Array, Object, Map, Set, JSON, Promise,
  Float32Array, Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array, Int32Array,
  parseFloat, parseInt, isNaN, isFinite, Infinity, NaN,
  console,
  performance: {{ now: () => 0 }},
  setTimeout: () => 0,
  clearTimeout: () => {{}},
  requestAnimationFrame: (cb) => {{ cb(); return 1; }},
  Path2D: class Path2D {{ moveTo(){{}} lineTo(){{}} quadraticCurveTo(){{}} closePath(){{}} }},
  alert: () => {{}},
  fetch: async () => ({{ ok: false }}),
  document: {{
    body: {{ appendChild(){{}} }},
    getElementById: (id) => makeEl(id),
    createElement: (tag) => makeEl(tag),
    addEventListener: () => {{}}
  }},
  window: {{
    __SKIP_AUTO_INIT__: true,
    innerWidth: 1280,
    innerHeight: 800,
    location: {{ protocol: 'http:', origin: 'http://127.0.0.1:8000' }},
    addEventListener: () => {{}}
  }}
}};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
for (let i = 0; i < scripts.length; i++) {{
  new vm.Script(scripts[i], {{ filename: 'studio_' + i + '.js' }}).runInContext(sandbox);
}}
const S = sandbox.__studioExports;
const result = (() => {{
{js_test_code}
}})();
process.stdout.write(JSON.stringify(result));
"""
        proc = subprocess.run(["node", "-e", harness_js], cwd=str(ROOT_DIR), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"Node VM execution failed:\n{proc.stderr}")
        return json.loads(proc.stdout)

    def test_studio_vm_shade_pixel_unpainted_identity(self) -> None:
        res = self._run_studio_vm_json(
            """
            const samples = [
              [12, 48, 210],
              [205, 38, 42],
              [242, 244, 246],
              [0, 0, 0],
              [255, 255, 255]
            ];
            const diffs = samples.map(([sr, sg, sb]) => {
              const orig = [S.srgb8ToLin(sr), S.srgb8ToLin(sg), S.srgb8ToLin(sb)];
              const px = {
                orig,
                a0: [orig[0] * 0.9, orig[1] * 0.9, orig[2] * 0.9],
                Sd: 0.95,
                Ss: 0.02,
                normal: [0.1, -0.2, 0.97],
                softAlpha: 0.88
              };
              const outNull = S.shadePixel(px, null, { lightingEnv: 'STANDARD' });
              const outDefault = S.shadePixel(px, S.defaultStack(), { lightingEnv: 'STANDARD' });
              const rgbNull = [S.linToSrgb8(outNull.r), S.linToSrgb8(outNull.g), S.linToSrgb8(outNull.b)];
              const rgbDef = [S.linToSrgb8(outDefault.r), S.linToSrgb8(outDefault.g), S.linToSrgb8(outDefault.b)];
              return Math.max(
                Math.abs(rgbNull[0] - sr), Math.abs(rgbNull[1] - sg), Math.abs(rgbNull[2] - sb),
                Math.abs(rgbDef[0] - sr), Math.abs(rgbDef[1] - sg), Math.abs(rgbDef[2] - sb)
              );
            });
            return { maxDiff: Math.max(...diffs) };
            """
        )
        self.assertLessEqual(res["maxDiff"], 1)

    def test_studio_vm_gloss_black_on_blue_no_rust_or_blotch(self) -> None:
        res = self._run_studio_vm_json(
            """
            // 1. Verify OK-1 normalizeShadingForMask on blue armor + zero-luminance NaN guard
            const w = 8, h = 8, N = w * h;
            const origLin = new Float32Array(3 * N);
            for (let i = 0; i < N; i++) {
              const scale = 0.7 + 0.4 * (i / N);
              origLin[3 * i + 0] = 0.024 * scale;
              origLin[3 * i + 1] = 0.085 * scale;
              origLin[3 * i + 2] = 0.322 * scale;
            }
            const maskIndices = Uint32Array.from({ length: N }, (_, i) => i);
            const maskAlphas = new Float32Array(N).fill(1.0);
            S.setState({
              imgW: w, imgH: h, isCustomPhoto: true,
              origLinMap: origLin,
              albedoMap: new Float32Array(3 * N),
              shadingMap: new Float32Array(N),
              specMap: new Float32Array(N),
              specBroadMap: new Float32Array(N),
              decalAlphaMap: new Float32Array(N)
            });
            S.normalizeShadingForMask(maskIndices, maskAlphas);
            const stAfterBlue = S.getState();
            const blueAlb = [stAfterBlue.albedoMap[0], stAfterBlue.albedoMap[1], stAfterBlue.albedoMap[2]];

            // Zero-luminance NaN safety check
            S.setState({ origLinMap: new Float32Array(3 * N) });
            S.normalizeShadingForMask(maskIndices, maskAlphas);
            const blackAlb = S.getState().albedoMap[0];

            // 2. Verify OK-2, OK-3 (0.010 Fresnel), FIX-6 (effAlpha=1 at 0.88), FIX-7 (2.4 coats)
            const st = { ...S.defaultStack(), surfacer: 'gsi_sf_gloss_black' };
            const pxA = {
              a0: [0.024, 0.085, 0.322],
              orig: [0.026, 0.087, 0.296],
              Sd: 1.0,
              Ss: 0.0,
              normal: [-0.24, -0.34, 0.91],
              softAlpha: 0.88
            };
            const pxB = { ...pxA, normal: [0.35, 0.42, 0.84] };
            const outA = S.shadePixel(pxA, st, { lightingEnv: 'STANDARD' });
            const outB = S.shadePixel(pxB, st, { lightingEnv: 'STANDARD' });
            const srgbA = [S.linToSrgb8(outA.r), S.linToSrgb8(outA.g), S.linToSrgb8(outA.b)];
            const srgbB = [S.linToSrgb8(outB.r), S.linToSrgb8(outB.g), S.linToSrgb8(outB.b)];

            // Also check clear overcoat blotch immunity (FIX-5)
            const stClear = { ...st, clear: 'gsi_gx101', clearCoats: 0.5 };
            const cA = S.shadePixel(pxA, stClear, { lightingEnv: 'STANDARD' });
            const cB = S.shadePixel(pxB, stClear, { lightingEnv: 'STANDARD' });
            const clearBlotchDiff = Math.max(
              Math.abs(S.linToSrgb8(cA.r) - S.linToSrgb8(cB.r)),
              Math.abs(S.linToSrgb8(cA.g) - S.linToSrgb8(cB.g)),
              Math.abs(S.linToSrgb8(cA.b) - S.linToSrgb8(cB.b))
            );

            return { blueAlb, blackAlbIsFinite: Number.isFinite(blackAlb), srgbA, srgbB, clearBlotchDiff };
            """
        )
        self.assertTrue(res["blackAlbIsFinite"])
        self.assertLess(res["blueAlb"][2], 0.48, f"Blue albedo blown up by /refLum: {res['blueAlb']}")
        r, g, b = res["srgbA"]
        self.assertLess(max(r, g, b), 40, f"Expected sRGB < 40 on gloss black, got {[r, g, b]}")
        self.assertLessEqual(abs(r - g), 6)
        self.assertLessEqual(abs(b - g), 8)
        self.assertLessEqual(r, max(g, b), f"Red channel must not dominate gloss black: {[r, g, b]}")
        blotch_diff = max(abs(res["srgbA"][c] - res["srgbB"][c]) for c in range(3))
        self.assertLessEqual(blotch_diff, 1, f"2D normal noise caused gray blotch: {res['srgbA']} vs {res['srgbB']}")
        self.assertLessEqual(res["clearBlotchDiff"], 2)

    def test_studio_vm_white_surfacer_shadow_lift_and_black_crease_preservation(self) -> None:
        res = self._run_studio_vm_json(
            """
            const pxShadowRed = {
              a0: [0.37, 0.04, 0.03],
              orig: [0.111, 0.012, 0.009],
              Sd: 0.30,
              Ss: 0.0,
              normal: [0.0, 0.0, 1.0],
              softAlpha: 1.0
            };
            const stWhite = { ...S.defaultStack(), surfacer: 'gaia_gs02' };
            const outW = S.shadePixel(pxShadowRed, stWhite, { lightingEnv: 'STANDARD' });
            const srgbW = [S.linToSrgb8(outW.r), S.linToSrgb8(outW.g), S.linToSrgb8(outW.b)];
            const lumW = 0.2126 * srgbW[0] + 0.7152 * srgbW[1] + 0.0722 * srgbW[2];

            // White surfacer + very thin clear coat must NOT lose effSd shadow lift
            const stWhiteClear = { ...stWhite, clear: 'gsi_c048', clearCoats: 0.05 };
            const outWC = S.shadePixel(pxShadowRed, stWhiteClear, { lightingEnv: 'STANDARD' });
            const lumWC = 0.2126 * S.linToSrgb8(outWC.r) + 0.7152 * S.linToSrgb8(outWC.g) + 0.0722 * S.linToSrgb8(outWC.b);

            // Deep crease (Sd=0.16) on black surfacer must NOT be washed out by a 0.18 ambient floor
            const pxCrease = { ...pxShadowRed, Sd: 0.16, orig: [0.059, 0.006, 0.005] };
            const stBlack = { ...S.defaultStack(), surfacer: 'gaia_gs03' };
            const outBlk = S.shadePixel(pxCrease, stBlack, { lightingEnv: 'STANDARD' });
            const maxBlkCrease = Math.max(S.linToSrgb8(outBlk.r), S.linToSrgb8(outBlk.g), S.linToSrgb8(outBlk.b));

            return { srgbW, lumW, lumWC, maxBlkCrease };
            """
        )
        self.assertGreaterEqual(res["lumW"], 185.0, f"White surfacer shadow too dark: {res}")
        self.assertGreaterEqual(res["lumWC"], 180.0, f"Clear coat discarded effSd white shadow lift: {res}")
        self.assertLessEqual(res["maxBlkCrease"], 26, f"Black surfacer crease washed out: {res}")

    def test_studio_vm_exterior_halo_chroma_bleed_decontamination(self) -> None:
        res = self._run_studio_vm_json(
            """
            const w = 64, h = 64, N = w * h;
            const canvas = sandbox.document.getElementById('sim-canvas');
            const raw = new Uint8ClampedArray(N * 4);

            function setPx(x, y, r, g, b) {
              const i = (y * w + x) * 4;
              raw[i] = r; raw[i + 1] = g; raw[i + 2] = b; raw[i + 3] = 255;
            }
            // Background: neutral white/light-gray armor [226, 226, 228]
            for (let y = 0; y < h; y++) {
              for (let x = 0; x < w; x++) setPx(x, y, 226, 226, 228);
            }
            // Upper red part (painted target): x=10..28, y=10..30 -> [185, 42, 42]
            for (let y = 10; y <= 30; y++) {
              for (let x = 10; x <= 28; x++) setPx(x, y, 185, 42, 42);
              // Simulate YUV 4:2:0 exterior 1..2px red chroma bleed on white neighbor (L~0.856, a~+0.053)
              setPx(29, y, 239, 194, 204);
              setPx(30, y, 232, 208, 213);
            }
            // Lower adjacent UNPAINTED red part: x=10..28, y=31..48 -> [180, 45, 45]
            for (let y = 31; y <= 48; y++) {
              for (let x = 10; x <= 28; x++) setPx(x, y, 180, 45, 45);
            }

            canvas._mockSourceImageData = { width: w, height: h, data: raw };
            sandbox.document.getElementById('sel-res-mode').value = '4096';
            S.decomposeUploadedPhoto({ width: w, height: h });

            // Build custom part mask for upper red part (x=10..28, y=10..30) BEFORE assigning paint stack
            const mIdx = [];
            const mAlpha = [];
            for (let y = 10; y <= 30; y++) {
              for (let x = 10; x <= 28; x++) {
                mIdx.push(y * w + x);
                mAlpha.push(1.0);
              }
            }
            const partObj = {
              id: 'part_100',
              regionId: 100,
              name: 'Upper Red Part',
              isCustom: true,
              maskIndices: Uint32Array.from(mIdx),
              maskAlphas: Float32Array.from(mAlpha)
            };
            const st0 = S.getState();
            st0.partsList.push(partObj);
            S.setState({ selectionMode: 'PART', selectedTargetId: 'part_100' });
            S.rebuildCompositeMaps();

            // Measure halo pixel (29, 20) and (30, 20) OKLab BEFORE painting
            const haloIdx1 = 20 * w + 29;
            const haloIdx2 = 20 * w + 30;
            const origLin = S.getState().origLinMap;
            const beforeLab1 = S.linToOklab(origLin[3 * haloIdx1], origLin[3 * haloIdx1 + 1], origLin[3 * haloIdx1 + 2]);
            const beforeLab2 = S.linToOklab(origLin[3 * haloIdx2], origLin[3 * haloIdx2 + 1], origLin[3 * haloIdx2 + 2]);

            // Now assign gray surfacer AFTER rebuildCompositeMaps (mirrors real UI dropdown/context-menu flow!)
            const pStacks = S.getState().partStacks;
            pStacks['part_100'] = { ...S.defaultStack(), surfacer: 'gsi_sf1000_gray' };
            S.renderFrame(true);

            const outData = S.getState().lastRenderedImageData.data;
            const afterLab1 = S.linToOklab(
              S.srgb8ToLin(outData[4 * haloIdx1]),
              S.srgb8ToLin(outData[4 * haloIdx1 + 1]),
              S.srgb8ToLin(outData[4 * haloIdx1 + 2])
            );
            const afterLab2 = S.linToOklab(
              S.srgb8ToLin(outData[4 * haloIdx2]),
              S.srgb8ToLin(outData[4 * haloIdx2 + 1]),
              S.srgb8ToLin(outData[4 * haloIdx2 + 2])
            );

            // Check unpainted adjacent red part border pixel (18, 31) — must NOT be bleached!
            const unpIdx = 31 * w + 18;
            const unpLab = S.linToOklab(
              S.srgb8ToLin(outData[4 * unpIdx]),
              S.srgb8ToLin(outData[4 * unpIdx + 1]),
              S.srgb8ToLin(outData[4 * unpIdx + 2])
            );

            return {
              beforeLab1,
              afterLab1,
              afterLab2,
              absAfterA1: Math.abs(afterLab1[1]),
              absAfterA2: Math.abs(afterLab2[1]),
              absDeltaL1: Math.abs(afterLab1[0] - beforeLab1[0]),
              absDeltaL2: Math.abs(afterLab2[0] - beforeLab2[0]),
              unpaintedRedA: unpLab[1]
            };
            """
        )
        self.assertLess(res["absAfterA1"], 0.01, f"1px Halo red chroma bleed remained: {res}")
        self.assertLess(res["absAfterA2"], 0.01, f"2px Halo red chroma bleed remained: {res}")
        self.assertLess(res["absDeltaL1"], 0.005, f"1px Halo luminance shifted: {res}")
        self.assertLess(res["absDeltaL2"], 0.005, f"2px Halo luminance shifted: {res}")
        self.assertGreater(res["unpaintedRedA"], 0.10, f"Unpainted adjacent red part was bleached: {res}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
