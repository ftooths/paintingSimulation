# 2D 프라모델 도색 & 마감 시뮬레이터 (`paint-sim`)

> English README: [`README.md`](./README.md) | 전체 문서 색인: [`docs/INDEX.md`](./docs/INDEX.md)

업로드한 프라모델(건프라·스케일 키트) 사진 위에서 **대화형 SAM 2.1 부품 분할**, **C++20 `libpaintcore` Kubelka-Munk + GGX 다층 광학 연산**, **Vertex AI Nano Banana(`gemini-nano-banana-2.1`) 신경망 마감 정제**, **다중 사이트 도료 시편 수집기(Sample Collector)**를 통합 제공하는 물리 기반 2D 도색 시뮬레이터입니다.

- **문서/엔진 기준 버전:** `1.3.0` (`pyproject.toml` 패키지 `1.1.0`, FastAPI `1.2.0`)
- **최종 업데이트:** `2026-11-14`
- **자동화 검증 스위트:** 총 **`67 / 67` PASS** (`C++20` 네이티브 `14`개 + Python MVP·SAM2·Studio VM `23`개 + Sample Collector·경계 광학 `30`개)

---

## 1. 핵심 기능

1. **대화형 AI 부품·색상 그룹 분할 (`SAM 2.1 base_plus` + C++20 가이드 매팅)**
   - 키트 사진(`frontend/local_test_studio.html`)에서 부품을 클릭하거나 박스로 드래그하면 8 GB RTX 3070 기준 `~55~92 ms` 내에 개별 파츠 마스크를 추출합니다(`keep_resident=True` 상주 VRAM 리스, 실 점유 `~0.46 GB`).
   - 순수 C++20 4축 양방향 밸리 패널라인 스냅(`panel_snap_matting.hpp`), 국소 선형 휘도 비율 경계 페더링(`FIX-1`), 코어 불투명도 `1.0` 보장(`FIX-6`), 2-pass Chebyshev(`L_inf`) 거리 변환 기반 외부 1~3 px YUV 4:2:0 색번짐 제거(`haloPackedMap`, `FIX-2`)를 결합합니다.
   - `E` 마스크 편집 모드(`+`/`-` 포인트, 박스 프롬프트, `Ctrl+Z` 실행 취소, 부품별 허용치 슬라이더, W3C C40 이중 외곽선 표시)를 지원합니다.
2. **다층 Kubelka-Munk & 미세면 광학 스택 (`libpaintcore` + `shadePixel`)**
   - **사출 플라스틱 기판 → 서페이서(`2.4`회 도포 기준) → 기본색 / 메탈릭 / 미러 크롬 / 카멜레온 / 형광 → 클리어 캔디(Beer–Lambert) → 마감재(유광 / 반광 / 무광)** 다층 도막과 스튜디오 표준광·암실·`365 nm` UV 조명을 시뮬레이션합니다.
   - 휘도 비율 기반 알베도 정규화(`OK-1`), 유색 피그먼트 도포 시 기판 잔차 제거(`OK-2`), WebP/JPEG 블록 노이즈가 증폭되지 않는 매끄러운 프레넬 반사(`0.010 * Sd^2.2`, `OK-3`/`FIX-5`), 알베도 연동 스튜디오 간접 반사광 보정(`ambientLift = 0.72 * coatLum^1.15`, `OK-4`)이 반영되어 있습니다.
   - **GSI 크레오스(Mr. Color)**, **가이아노츠(Gaianotes)**, **타미야 라카(LP)** 3대 브랜드 **58종**(`PAINTS` `55`종 + 마감재 `3`종) 기본 도료 카탈로그를 내장합니다(`python/paint_sim/db/seeds/mr_color_seed.json`).
3. **듀얼 이미지 신경망 마감 정제 (`gemini-nano-banana-2.1`)**
   - `config/prompts.json`(`v1.3.0`) 핫리로드 구성, 원본 1:1 해상도 고정, OKLab 색차 고정(Color Lock) 후처리 및 3종 배경 내보내기 모드(`full_bg`, `object_transparent`, `object_crop`)를 제공합니다.
4. **다중 사이트 도료 시편 수집기 (`Sample Collector`, 포트 `8765`)**
   - 범용 HTML 추출기 + 사이트 어댑터(`config/site_adapters/`), SSRF 차단 가드, Gemini 시트 자동 바운딩 박스 검출, 대화형 SAM 2.1 / OKLab+GrabCut 시편 분할 UI(`frontend/sample_collector.html`), 무손실 2 px EDT 테두리 색번짐 정제, 마스크 내부 광학 파라미터 추출(`measure_swatch_optics`), 18개 브랜드 3축 자동 분류기, 26컬럼 CSV 및 3시트 XLSX 마스터 인덱스를 지원합니다.

---

## 2. 빠른 시작 (Windows 로컬 환경)

### 사전 요구사항
- **Windows 10/11 x64** + **Visual Studio 2022 / 18 Build Tools (`cl.exe` C++20)**
- **Python 3.11+** (PyTorch CUDA 및 `sam2` 포함 — `C:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe` 존재 시 자동 우선 선택)
- **Node.js 18+** (선택 사항 — `test_mvp_suite.py` 내 헤드리스 `vm.Script` 프론트엔드 광학 단위 테스트에 사용)

### 1) 페인트 스튜디오 서버 실행 (`127.0.0.1:8000`)
```powershell
.\run_studio_server.bat
# 또는 옵션 지정 실행:
powershell -ExecutionPolicy Bypass -File .\run_studio_server.ps1 -Port 8000
```
- `build\paintcore.dll` 미존재 시 자동 컴파일하고 `checkpoints\sam2.1_hiera_base_plus.pt` 체크포인트를 확인·다운로드한 뒤 `http://127.0.0.1:8000/studio/local_test_studio.html` 브라우저 창을 엽니다.

### 2) 도료 시편 수집기(Sample Collector) 실행 (`127.0.0.1:8765`)
```powershell
# 웹 UI 서버 실행:
powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1 -Port 8765

# 헤드리스 CLI 수집 실행:
powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1 -Url "https://pm-paint.com/..." -Vision manual
```

### 3) 전체 빌드 및 자동화 테스트 실행 (`67/67` PASS)
```powershell
powershell -ExecutionPolicy Bypass -File .\run_local_tests.ps1
```

---

## 3. 주요 명령어 요약

| 작업 | 실행 명령 | 설명 |
|---|---|---|
| C++20 빌드 + 전체 테스트(`67`개) | `powershell -ExecutionPolicy Bypass -File .\run_local_tests.ps1` | `paintcore.dll`·`test_paintcore.exe` 컴파일 후 C++(`14`), MVP(`23`), Collector(`30`) 연속 검증 |
| GPU / SAM 2.1 / DLL 사전 점검 | `powershell -ExecutionPolicy Bypass -File .\run_studio_server.ps1 -CheckOnly` | 서버 기동 없이 CUDA, `sam2.1_base_plus`, 네이티브 DLL 상태만 진단 |
| SAM 2.1 체크포인트 다운로드 | `powershell -ExecutionPolicy Bypass -File .\scripts\setup_hq_sam2_local.ps1 -Variant base_plus` | `base_plus`(기본), `small`, `tiny`, `hq_large` 다운로드 및 바이트 크기 무결성 검증 |
| 페인트 스튜디오 서버 기동 | `.\run_studio_server.bat` | `127.0.0.1:8000`에서 `/studio/local_test_studio.html` 및 `/api/v1/*` 제공 |
| 도료 시편 수집기 서버 기동 | `powershell -ExecutionPolicy Bypass -File .\run_sample_collector.ps1` | `127.0.0.1:8765`에서 `sample_collector.html`, `/api/*`, `WS /ws/jobs/{id}` 제공 |

---

## 4. 문서 안내 (`docs/`)

> 참고: `docs/` 디렉터리는 로컬 명세 관리를 위해 `.gitignore`에 등록되어 있습니다. 전체 문서 목록은 [`docs/INDEX.md`](./docs/INDEX.md)를 참조하세요.

| 분류 | 한국어 (`_KO`) | 영문 (`EN`) | 주요 내용 |
|---|---|---|---|
| **통합 문서 색인** | — | [`docs/INDEX.md`](./docs/INDEX.md) | 전체 설계 명세, 기능 플랜, 히스토리 로그 색인 |
| **제품 요구사항 정의서 (PRD)** | [`docs/PRD_KO.md`](./docs/PRD_KO.md) | [`docs/PRD.md`](./docs/PRD.md) | 제품 비전, 워크플로우, `FR-1..9`, `NFR-1..6`, 설계 결정 `D1`~`D14` |
| **소프트웨어·기술 요구사항 (SRS/TRD)** | [`docs/SRS_TRD_KO.md`](./docs/SRS_TRD_KO.md) | [`docs/SRS_TRD.md`](./docs/SRS_TRD.md) | 수식 요구사항, Studio(`8000`) & Collector(`8765`) API, SQLite 스키마, VRAM/지연시간 SLA |
| **아키텍처 및 의사결정 기록 (SAD/ADR)** | [`docs/SAD_ADR_KO.md`](./docs/SAD_ADR_KO.md) | [`docs/SAD_ADR.md`](./docs/SAD_ADR.md) | 순수 C++20/OpenMP `libpaintcore` + `ctypes` C-ABI 구조, 모노레포 트리, `ADR-001`~`ADR-014` |
| **소프트웨어 상세 설계서 (SDD)** | [`docs/SDD_KO.md`](./docs/SDD_KO.md) | [`docs/SDD.md`](./docs/SDD.md) | C++20 구조체·11개 C-ABI 함수, Python 클래스, `shadePixel`·`haloPackedMap`, 58종 도료 시드 표 |
| **영상처리 및 광학 방법론** | [`docs/IMAGE_PROCESSING_METHODOLOGY_KO.md`](./docs/IMAGE_PROCESSING_METHODOLOGY_KO.md) | [`docs/IMAGE_PROCESSING_METHODOLOGY.md`](./docs/IMAGE_PROCESSING_METHODOLOGY.md) | Kubelka-Munk, Saunderson, GGX, `OK-1..4` 반사/바운스 광학, `FIX-1/2/6` 경계·할로 제거 수식 |
| **통합 테스트 계획서 (RTM)** | [`docs/TEST_PLAN_SPEC_KO.md`](./docs/TEST_PLAN_SPEC_KO.md) | [`docs/TEST_PLAN_SPEC.md`](./docs/TEST_PLAN_SPEC.md) | 총 `67`개 자동화 검증 케이스(`TC-CPP`, `TC-PHY`, `TC-PAL`, `TC-VIS`, `TC-SEG`, `TC-OPT`, `TC-COL`, `TC-SAM`) |
| **시편 분류 및 수집기 가이드** | [`docs/SAMPLE_COLLECTOR_USER_GUIDE_KO.md`](./docs/SAMPLE_COLLECTOR_USER_GUIDE_KO.md) | [`docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md`](./docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md) | 18개 브랜드 3축 분류 체계, 표준 파일명, CSV/XLSX 스키마, 수집기 CLI/UI 사용법 |
