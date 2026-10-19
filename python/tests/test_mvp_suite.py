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


if __name__ == "__main__":
    unittest.main(verbosity=2)
