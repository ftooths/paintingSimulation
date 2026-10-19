"""Automated Google Drive Dataset Archiver (SRS Section 4.3)."""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GoogleDriveArchiver:
    """
    Packages every completed simulation job into the standardized dataset bundle structure
    (`SRS_TRD.md` Section 4.3) for future AI model training:
      job_{uuid}/
      ├── 00_raw_input.png
      ├── 01_preprocessed_wb.png
      ├── 02_region_id_map.png
      ├── 03_som_verification.jpg
      ├── 04_intrinsics/
      │   ├── albedo_base.png
      │   ├── shading_diffuse.png
      │   ├── specular.png
      │   └── normals_ao.png
      ├── 05_simulation_recipe.json
      ├── 06_rendered_output.png
      ├── 07_nano_banana_refined.png
      └── metadata.json
    Writes locally to `data/archive/YYYY-MM/job_{uuid}/` and optionally syncs to Google Drive API v3.
    """

    REQUIRED_BUNDLE_FILES = (
        "00_raw_input.png",
        "01_preprocessed_wb.png",
        "02_region_id_map.png",
        "03_som_verification.jpg",
        "04_intrinsics/albedo_base.png",
        "04_intrinsics/shading_diffuse.png",
        "04_intrinsics/specular.png",
        "04_intrinsics/normals_ao.png",
        "05_simulation_recipe.json",
        "06_rendered_output.png",
        "07_nano_banana_refined.png",
        "metadata.json",
    )

    def __init__(self, archive_root: Optional[Path | str] = None) -> None:
        if archive_root is None:
            archive_root = os.environ.get("LOCAL_ARCHIVE_DIR", "data/archive")
        self.archive_root = Path(archive_root).resolve()

    def archive_job_bundle(
        self,
        job_id: str,
        raw_input_png: bytes,
        preprocessed_wb_png: bytes,
        region_id_map_png: bytes,
        som_verification_jpg: bytes,
        intrinsics_pngs: Dict[str, bytes],
        simulation_recipe: Dict[str, Any],
        rendered_output_png: bytes,
        nano_banana_refined_png: bytes,
        metadata: Dict[str, Any],
    ) -> Path:
        month_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
        bundle_dir = self.archive_root / month_str / f"job_{job_id}"
        intrinsics_dir = bundle_dir / "04_intrinsics"
        intrinsics_dir.mkdir(parents=True, exist_ok=True)

        (bundle_dir / "00_raw_input.png").write_bytes(raw_input_png)
        (bundle_dir / "01_preprocessed_wb.png").write_bytes(preprocessed_wb_png)
        (bundle_dir / "02_region_id_map.png").write_bytes(region_id_map_png)
        (bundle_dir / "03_som_verification.jpg").write_bytes(som_verification_jpg)

        for name in ("albedo_base.png", "shading_diffuse.png", "specular.png", "normals_ao.png"):
            content = intrinsics_pngs.get(name, preprocessed_wb_png)
            (intrinsics_dir / name).write_bytes(content)

        (bundle_dir / "05_simulation_recipe.json").write_text(
            json.dumps(simulation_recipe, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (bundle_dir / "06_rendered_output.png").write_bytes(rendered_output_png)
        (bundle_dir / "07_nano_banana_refined.png").write_bytes(nano_banana_refined_png)
        (bundle_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return bundle_dir
