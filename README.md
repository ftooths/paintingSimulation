# 2D Plastic-Model Paint & Finish Simulator (`paint-sim`)

> 한국어 안내: [`README_KO.md`](./README_KO.md) | Master Documentation Index: [`docs/INDEX.md`](./docs/INDEX.md)

Physically grounded 2D plastic-model (Gunpla / scale kit) paint, surfacer, metallic, clear-candy, chameleon, fluorescent, and topcoat simulator with interactive **SAM 2.1** part segmentation, **C++20 `libpaintcore`** Kubelka-Munk + GGX optics, **Vertex AI Nano Banana** neural refinement, and a **Multi-Site Paint Sample Collector** pipeline.

- **Version:** `1.3.0` (`pyproject.toml` package `1.1.0`, FastAPI `1.2.0`)
- **Last Updated:** `2026-11-14`
- **Automated Verification Suite:** `67 / 67` tests passing (`14` C++20 native + `23` Python MVP/SAM2/Studio VM + `30` Sample Collector & boundary tests)

---

## 1. Key Capabilities

1. **Interactive AI Part & Color-Group Segmentation (`SAM 2.1 base_plus` + C++20 Guided Matting)**
   - Click or box-drag any part on an uploaded kit photo (`frontend/local_test_studio.html`) to segment individual armor pieces in `~55–92 ms` on an 8 GB RTX 3070 (`keep_resident=True` VRAM lease, `~0.46 GB` allocated).
   - Pure-C++20 4-axis valley panel-line snapping (`panel_snap_matting.hpp`), local linear-luminance boundary feathering (`FIX-1`), exact `1.0` core opacity (`FIX-6`), and 2-pass Chebyshev (`L_inf`) 1–3 px exterior YUV 4:2:0 chroma-bleed decontamination (`haloPackedMap`, `FIX-2`).
   - Full interactive mask refinement (`E` mask-edit mode, `+` / `-` points, box prompt, `Ctrl+Z` undo, per-part tolerance slider, W3C C40 dual-stroke high-contrast contour).
2. **Multi-Layer Kubelka-Munk & Microfacet Optical Stack (`libpaintcore` + `shadePixel`)**
   - Multi-coat stack: **Plastic Substrate -> Surfacer (`2.4` coats) -> Base / Metallic / Mirror Chrome / Chameleon / Fluorescent -> Clear Candy (Beer–Lambert) -> Topcoat (Gloss / Semi-Gloss / Matte)** under Standard Studio, Dark Room, and `365 nm` UV lighting.
   - Luminance-ratio intrinsic albedo regularization (`OK-1`), pigment-layer residual zeroing (`OK-2`), block-noise-free shading-driven dielectric Fresnel (`0.010 * Sd^2.2`, `OK-3`/`FIX-5`), and zero-floor albedo-dependent studio bounce fill (`ambientLift = 0.72 * coatLum^1.15`, `OK-4`).
   - Built-in **58-paint** starter catalog (`55` pigment/metallic/clear entries + `3` topcoats) across **GSI Creos Mr. Color**, **Gaianotes**, and **Tamiya Lacquer (LP)** (`python/paint_sim/db/seeds/mr_color_seed.json`).
3. **Dual-Image Neural Finish Refinement (`gemini-nano-banana-2.1`)**
   - Configurations hot-loaded from `config/prompts.json` (`v1.3.0`) with native 1:1 resolution preservation, OKLab chrominance color lock, and 3 background export modes (`full_bg`, `object_transparent`, `object_crop`).
4. **Multi-Site Paint Sample Collector (`port 8765`)**
   - Generic HTML extractor + site adapters (`config/site_adapters/`), SSRF guard, Gemini auto-sheet bbox detection, interactive SAM 2.1 / OKLab+GrabCut swatch editor (`frontend/sample_collector.html`), lossless 2 px EDT rim decontamination, mask-intrinsic optical extraction (`measure_swatch_optics`), 18-brand taxonomy auto-classifier, and 26-column CSV / 3-sheet XLSX synchronization.

---

## 2. Quick Start (Windows Local Studio)

### Prerequisites
- **Windows 10/11 x64** with **Visual Studio 2022 / 18 Build Tools (`cl.exe` C++20)**
- **Python 3.11+** with PyTorch CUDA & `sam2` (automatically uses `C:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe` when present, falling back to `python` on `PATH`)
- **Node.js 18+** (optional, used by headless `vm.Script` frontend optical unit tests in `test_mvp_suite.py`)

### Launch Interactive Paint Studio (`127.0.0.1:8000`)
```powershell
.\run_studio_server.bat
# Or via PowerShell with options:
powershell -ExecutionPolicy Bypass -File .\run_studio_server.ps1 -Port 8000
```
- Automatically compiles `build\paintcore.dll` if missing, verifies/downloads `checkpoints\sam2.1_hiera_base_plus.pt`, prints CUDA/VRAM diagnostics, and opens `http://127.0.0.1:8000/studio/local_test_studio.html`.

### Launch Paint Sample Collector (`127.0.0.1:8765`)
```powershell
# Interactive Web UI:
powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1 -Port 8765

# Headless CLI fetch + optional auto-ingest:
powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1 -Url "https://pm-paint.com/..." -Vision manual
```

### Run Full Build & Verification Suite (`67/67` Tests)
```powershell
powershell -ExecutionPolicy Bypass -File .\run_local_tests.ps1
```

---

## 3. Commands Reference

| Task | Command | Notes |
|---|---|---|
| Full C++20 build + all 3 test suites (`67` tests) | `powershell -ExecutionPolicy Bypass -File .\run_local_tests.ps1` | Compiles `paintcore.dll` & `test_paintcore.exe`, then runs C++ (`14`), MVP (`23`), and Collector (`30`) suites |
| Pre-flight GPU / SAM 2.1 / DLL check | `powershell -ExecutionPolicy Bypass -File .\run_studio_server.ps1 -CheckOnly` | Verifies CUDA, `sam2.1_base_plus`, and `paintcore.dll` without starting HTTP server |
| Download SAM 2.1 checkpoints | `powershell -ExecutionPolicy Bypass -File .\scripts\setup_hq_sam2_local.ps1 -Variant base_plus` | Supports `base_plus` (default), `small`, `tiny`, `hq_large` with byte-exact verification |
| Start Studio FastAPI server | `.\run_studio_server.bat` | Serves `/studio/local_test_studio.html` and `/api/v1/*` on `127.0.0.1:8000` |
| Start Sample Collector server | `powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1` | Serves `sample_collector.html`, REST `/api/*`, and `WS /ws/jobs/{id}` on `127.0.0.1:8765` |
| CMake cross-platform / Docker build | `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release` | Used by `docker-compose.yml` (`api-server`, `vision-worker`, `redis`) |

---

## 4. Repository Architecture

```text
paintingSimulation/
├── cpp/                        # Pure C++20 + OpenMP native optical & matting engine (libpaintcore)
│   ├── include/paintcore/      # types.hpp, color_spaces.hpp, km_brdf_solver.hpp, albedo_regularizer.hpp, panel_snap_matting.hpp
│   ├── src/paintcore_c_api.cpp # 11 extern "C" C-ABI exports compiled to build/paintcore.dll (.so)
│   └── tests/                  # test_paintcore_main.cpp (14 native verification tests: TC-CPP-001..006, TC-PHY-*)
├── python/
│   ├── paint_sim/
│   │   ├── api/main.py         # FastAPI Studio server (port 8000), SAM 2.1 session LRU, /api/v1/* routes
│   │   ├── physics/            # engine_bridge.py (zero-copy ctypes C-ABI bridge + NumPy parity fallback)
│   │   ├── vision/             # vram_manager.py (8 GB RTX 3070 resident lease & idle offload manager)
│   │   ├── collector/          # Multi-site Sample Collector (server.py port 8765, cli.py, vision/manual_segment.py, refine.py)
│   │   ├── db/                 # SQLite paint repository, 58-paint seed JSON, sample_ingestor.py (26-col CSV & 3-sheet XLSX)
│   │   ├── gemini/             # Gemini vision kit analysis client + OKLab k-Means++ fallback
│   │   ├── nano_banana/        # Vertex AI Nano Banana dual-image refiner + OKLab color lock & bg modes
│   │   └── config/             # Hot-reloadable PromptConfigLoader (config/prompts.json v1.3.0)
│   └── tests/                  # test_mvp_suite.py (23 tests) & test_collector_suite.py (30 tests)
├── frontend/
│   ├── local_test_studio.html  # Interactive 2D Paint Studio (SAM 2.1 UI + zero-alloc shadePixel + haloPackedMap)
│   ├── sample_collector.html   # Paint Sample Collector Web UI (manual SAM 2.1 / GrabCut & Gemini auto bbox)
│   └── src/                    # Collector JS modules & WebGL2 reference shader (km_compositor.frag)
├── config/                     # prompts.json, collector_settings.json, paint_taxonomy_rules.json, site_adapters/
├── scripts/                    # setup_hq_sam2_local.ps1 checkpoint manager
└── docs/                       # Full EN & KO engineering specifications, ADRs, guides, and implementation plans
```

---

## 5. Documentation Map (`docs/`)

> Note: `docs/` is listed in `.gitignore` for local specification management. See [`docs/INDEX.md`](./docs/INDEX.md) for the complete index.

| Document | English | Korean (`_KO`) | Scope |
|---|---|---|---|
| **Master Index** | [`docs/INDEX.md`](./docs/INDEX.md) | — | Full catalog of active specs, plans, and historical logs |
| **Product Requirements (PRD)** | [`docs/PRD.md`](./docs/PRD.md) | [`docs/PRD_KO.md`](./docs/PRD_KO.md) | Product vision, user workflows, FR/NFRs, design decisions `D1`–`D14` |
| **System & Technical Spec (SRS/TRD)** | [`docs/SRS_TRD.md`](./docs/SRS_TRD.md) | [`docs/SRS_TRD_KO.md`](./docs/SRS_TRD_KO.md) | Mathematical specs, REST/WS APIs (`8000` & `8765`), VRAM & latency SLAs |
| **Architecture & ADRs (SAD/ADR)** | [`docs/SAD_ADR.md`](./docs/SAD_ADR.md) | [`docs/SAD_ADR_KO.md`](./docs/SAD_ADR_KO.md) | System architecture, monorepo layout, `ADR-001` through `ADR-014` |
| **Software Design (SDD)** | [`docs/SDD.md`](./docs/SDD.md) | [`docs/SDD_KO.md`](./docs/SDD_KO.md) | C++20 C-ABI structs/exports, Python classes, Studio JS buffers, 58-paint seed table |
| **Image Processing Methodology** | [`docs/IMAGE_PROCESSING_METHODOLOGY.md`](./docs/IMAGE_PROCESSING_METHODOLOGY.md) | [`docs/IMAGE_PROCESSING_METHODOLOGY_KO.md`](./docs/IMAGE_PROCESSING_METHODOLOGY_KO.md) | Kubelka-Munk, Saunderson, GGX, `OK-1..4` bounce/Fresnel optics, `FIX-1/2/6` boundary & 4:2:0 halo math |
| **Test Plan & RTM** | [`docs/TEST_PLAN_SPEC.md`](./docs/TEST_PLAN_SPEC.md) | [`docs/TEST_PLAN_SPEC_KO.md`](./docs/TEST_PLAN_SPEC_KO.md) | All `67` automated test cases (`TC-CPP`, `TC-PHY`, `TC-PAL`, `TC-VIS`, `TC-SEG`, `TC-OPT`, `TC-COL`, `TC-SAM`) |
| **Sample Classification & Collector** | [`docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md`](./docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md) | [`docs/SAMPLE_COLLECTOR_USER_GUIDE_KO.md`](./docs/SAMPLE_COLLECTOR_USER_GUIDE_KO.md) | 18-brand taxonomy, canonical filenames, CSV/XLSX schema, Collector CLI & UI guide |
