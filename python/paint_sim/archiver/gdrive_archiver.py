"""Automated Google Drive Dataset Archiver with Strict Payload Integrity Validation (SRS Section 4.3)."""
from __future__ import annotations

import datetime
import hashlib
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
    Strictly validates image headers/trailers and rejects fake placeholder bytes or duplicate
    intrinsic buffers so training datasets are never polluted (Cross-Review B2).
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

    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    PNG_IEND_MARKER = b"IEND\xaeB`\x82"
    JPEG_SOI_PREFIX = b"\xff\xd8\xff"
    JPEG_EOI_SUFFIX = b"\xff\xd9"

    def __init__(self, archive_root: Optional[Path | str] = None, strict_validation: bool = True) -> None:
        if archive_root is None:
            archive_root = os.environ.get("LOCAL_ARCHIVE_DIR", "data/archive")
        self.archive_root = Path(archive_root).resolve()
        self.strict_validation = strict_validation

    @classmethod
    def validate_image_payload(cls, rel_name: str, data: bytes) -> None:
        """
        Reject fake/stub byte strings (e.g., `b'\\x89PNG...REGION_ID_16U'` or `b'\\xff\\xd8\\xff\\xe0SOM_OVERLAY'`)
        that lack valid PNG IHDR+IEND chunks or JPEG SOI+EOI framing.
        """
        if not isinstance(data, (bytes, bytearray)) or len(data) < 24:
            raise ValueError(f"Archive payload '{rel_name}' is empty or too short ({len(data) if data else 0} bytes)")

        if rel_name.endswith(".png"):
            if not data.startswith(cls.PNG_SIGNATURE):
                raise ValueError(f"Archive payload '{rel_name}' does not start with a valid PNG signature")
            if b"IHDR" not in data[:32] or not data.endswith(cls.PNG_IEND_MARKER):
                raise ValueError(
                    f"Archive payload '{rel_name}' is a stub/corrupted PNG missing IHDR or IEND chunk"
                )
        elif rel_name.endswith(".jpg") or rel_name.endswith(".jpeg"):
            if not (data.startswith(cls.JPEG_SOI_PREFIX) and data.endswith(cls.JPEG_EOI_SUFFIX)):
                raise ValueError(
                    f"Archive payload '{rel_name}' is a stub/corrupted JPEG missing SOI (0xFFD8) or EOI (0xFFD9)"
                )

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
        if self.strict_validation:
            self.validate_image_payload("00_raw_input.png", raw_input_png)
            self.validate_image_payload("01_preprocessed_wb.png", preprocessed_wb_png)
            self.validate_image_payload("02_region_id_map.png", region_id_map_png)
            self.validate_image_payload("03_som_verification.jpg", som_verification_jpg)
            self.validate_image_payload("06_rendered_output.png", rendered_output_png)
            self.validate_image_payload("07_nano_banana_refined.png", nano_banana_refined_png)

            wb_hash = hashlib.sha256(preprocessed_wb_png).hexdigest()
            intrinsic_hashes = set()
            for name in ("albedo_base.png", "shading_diffuse.png", "specular.png", "normals_ao.png"):
                if name not in intrinsics_pngs:
                    raise ValueError(f"Missing required intrinsic map '{name}' in archive bundle")
                content = intrinsics_pngs[name]
                self.validate_image_payload(f"04_intrinsics/{name}", content)
                h = hashlib.sha256(content).hexdigest()
                if h == wb_hash:
                    raise ValueError(
                        f"Intrinsic map '04_intrinsics/{name}' is an un-decomposed duplicate of 01_preprocessed_wb.png"
                    )
                intrinsic_hashes.add(h)
            if len(intrinsic_hashes) < 4:
                raise ValueError("Intrinsic maps must be 4 distinct decomposed buffers, not duplicates of each other")

        month_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
        bundle_dir = self.archive_root / month_str / f"job_{job_id}"
        intrinsics_dir = bundle_dir / "04_intrinsics"
        intrinsics_dir.mkdir(parents=True, exist_ok=True)

        (bundle_dir / "00_raw_input.png").write_bytes(raw_input_png)
        (bundle_dir / "01_preprocessed_wb.png").write_bytes(preprocessed_wb_png)
        (bundle_dir / "02_region_id_map.png").write_bytes(region_id_map_png)
        (bundle_dir / "03_som_verification.jpg").write_bytes(som_verification_jpg)

        for name in ("albedo_base.png", "shading_diffuse.png", "specular.png", "normals_ao.png"):
            content = intrinsics_pngs[name]
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
