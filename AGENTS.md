# Project Agent Instructions (`paintingSimulation`)

Concise repository rules, runtime paths, and architectural invariants for AI coding agents working in `C:\git\paintingSimulation`.

## 1. Build, Run & Verification Commands

- **Preferred Windows Python Runtime:**
  - `C:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe` (has CUDA + PyTorch + official `sam2` pre-installed).
  - Because embedded Python `._pth` files can ignore `PYTHONPATH`, always prepend `sys.path.insert(0, "python")` in inline `-c` one-liners or invoke via the repository `.ps1` launchers.
- **NEVER `pip install` HQ-SAM 2 into ComfyUI `python_embeded`:**
  - Installing `sam2` forks into ComfyUI's embedded Python shadows the official `sam2` package and breaks ComfyUI. Default segmentation uses official `sam2.1_base_plus` (`checkpoints/sam2.1_hiera_base_plus.pt`).
- **Full Build & Regression Suite (`67/67` tests — run before declaring any task done):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\run_local_tests.ps1
  ```
  1. Compiles `cpp/src/paintcore_c_api.cpp` -> `build/paintcore.dll` and `cpp/tests/test_paintcore_main.cpp` -> `build/test_paintcore.exe` via MSVC C++20 (`/std:c++20 /O2 /EHsc`).
  2. Runs `build/test_paintcore.exe` (`14` native C++20 tests: `TC-CPP-001..006`, `TC-PHY-001..005`, `TC-PHY-007..009`).
  3. Runs `python/tests/test_mvp_suite.py` (`23` tests, including `TC-PHY-006` C++/Python parity, FastAPI SAM 2.1 session routes, and headless Node.js `vm` Studio `shadePixel` & `haloPackedMap` tests).
  4. Runs `python/tests/test_collector_suite.py` (`30` tests covering multi-site Sample Collector, swatch optics, and `SAM2PointBoxSegmenter` `FIX-1`/`FIX-6` boundary invariants).
- **Local Servers:**
  - **Paint Studio (`127.0.0.1:8000`):** `.\run_studio_server.bat` or `powershell -ExecutionPolicy Bypass -File .\run_studio_server.ps1`
  - **Sample Collector (`127.0.0.1:8765`):** `powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1`

## 2. Critical Codebase Gotchas & Invariants

1. **`docs/` is in `.gitignore`:**
   - File globbing / ripgrep tools that respect `.gitignore` skip `docs/` by default. Always inspect files under `docs/` and `docs/superpowers/plans/` by explicit path (`read` tool) or `docs/INDEX.md`.
2. **Cross-Runtime Physics Parity (`TC-PHY-006`):**
   - `cpp/include/paintcore/km_brdf_solver.hpp` and `python/paint_sim/physics/engine_bridge.py` must stay mathematically locked (`dE00 < 0.05` in `TC-PHY-006`). Never change KM coat exponent (`pow(coats, 1.75)` for `coats > 1.0`) or Saunderson coefficients (`k1 = 0.04`, `k2 = 0.60`) in one runtime without updating and verifying all paired runtimes.
3. **Frontend Studio Pure Optical Harness (`frontend/local_test_studio.html`):**
   - Per-pixel optical math lives in pure `shadePixel(px, st, env, out)` and exterior YUV 4:2:0 chroma-bleed cleanup lives in `registerTargetHaloInMap()` + `decontaminateHaloChroma()` (zero heap allocations inside `renderFrame()`).
   - Exported on `globalThis.__studioExports` (with `window.__SKIP_AUTO_INIT__` guard) so `python/tests/test_mvp_suite.py` (`_run_studio_vm_json`) tests real Studio JS directly in Node.js `vm`.
   - `decomposeUploadedPhoto()` enforces a minimum canvas resolution of `64x64` (`Math.max(64, ...)`). Any synthetic image tested through `decomposeUploadedPhoto` in Node `vm` must be `>= 64x64`.
4. **Vertex AI Nano Banana Sampling Keys (`config/prompts.json` & `nano_banana/refiner.py`):**
   - `gemini-nano-banana-2.1` / `gemini-3.1-flash-image` reject `temperature`, `top_p`, `top_k`, and `seed`. `NanoBananaRefiner.UNSUPPORTED_SAMPLING_KEYS` strips them automatically — do not re-introduce mandatory sampling params.
5. **Dirty Working Tree Safety:**
   - Never run destructive git commands (`git reset --hard`, `git checkout --`, `git clean`) without explicit user confirmation.

## 3. Documentation Synchronization

- Core engineering documents in `docs/` are maintained as paired English (`*.md`) and Korean (`*_KO.md`) specifications (`PRD`, `SRS_TRD`, `SAD_ADR`, `SDD`, `IMAGE_PROCESSING_METHODOLOGY`, `TEST_PLAN_SPEC`), indexed in `docs/INDEX.md`. Keep EN and KO versions synchronized whenever changing APIs, schemas, ADRs, or optical formulas.
