"""Brand-agnostic Relational Paint Database Repository (SQLite / PostgreSQL compatible)."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS brands (
    brand_id VARCHAR(32) PRIMARY KEY,
    name_en VARCHAR(128) NOT NULL,
    name_ko VARCHAR(128) NOT NULL,
    country VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS product_lines (
    line_id VARCHAR(64) PRIMARY KEY,
    brand_id VARCHAR(32) REFERENCES brands(brand_id),
    name_en VARCHAR(128) NOT NULL,
    binder_type VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS paints (
    paint_id VARCHAR(64) PRIMARY KEY,
    line_id VARCHAR(64) REFERENCES product_lines(line_id),
    code VARCHAR(32) NOT NULL,
    name_en VARCHAR(128) NOT NULL,
    name_ko VARCHAR(128) NOT NULL,
    role VARCHAR(32) NOT NULL,
    default_finish VARCHAR(32) NOT NULL,
    surfacer_grit INTEGER,
    srgb_hex CHAR(7) NOT NULL,
    masstone_lin_r REAL NOT NULL,
    masstone_lin_g REAL NOT NULL,
    masstone_lin_b REAL NOT NULL,
    hiding_power_r REAL NOT NULL,
    hiding_power_g REAL NOT NULL,
    hiding_power_b REAL NOT NULL,
    ggx_roughness REAL NOT NULL DEFAULT 0.08,
    metallic_ness REAL NOT NULL DEFAULT 0.0,
    metallic_f0_rgb_json TEXT,
    chrome_f0 REAL DEFAULT 0.95,
    beer_lambert_alpha_json TEXT,
    chameleon_stops_json TEXT,
    fluo_quantum_yield REAL DEFAULT 0.0,
    fluo_peak_rgb_json TEXT,
    is_calibrated BOOLEAN NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class PaintDatabaseRepository:
    """Manages the Gunze Mr. Color & Specialty Coating relational database."""

    _COLLECTOR_ROLE_MAP: Dict[str, str] = {
        "01_surfacer_primer": "surfacer",
        "02_base_solid": "base_solid",
        "03_metallic_flake": "metallic",
        "04_mirror_chrome": "mirror_chrome",
        "05_pearl_mica": "metallic",
        "06_chameleon_colorshift": "chameleon",
        "07_clear_candy": "clear_candy",
        "08_fluorescent_neon": "fluorescent",
        "09_phosphorescent_glow": "fluorescent",
        "12_topcoat_varnish": "topcoat",
    }

    def __init__(self, db_path: str | Path = ":memory:", seed_path: Optional[Path] = None) -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        if seed_path is None:
            seed_path = Path(__file__).parent / "seeds" / "mr_color_seed.json"
        if seed_path.exists():
            self.seed_from_json(seed_path)

    def _init_schema(self) -> None:
        with self._lock:
            with self.conn:
                self.conn.executescript(SCHEMA_SQL)

    def seed_from_json(self, seed_path: Path) -> None:
        with open(seed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            with self.conn:
                for b in data.get("brands", []):
                    self.conn.execute(
                        "INSERT OR REPLACE INTO brands (brand_id, name_en, name_ko, country) VALUES (?, ?, ?, ?)",
                        (b["brand_id"], b["name_en"], b["name_ko"], b.get("country", "")),
                    )
                for pl in data.get("product_lines", []):
                    self.conn.execute(
                        "INSERT OR REPLACE INTO product_lines (line_id, brand_id, name_en, binder_type) VALUES (?, ?, ?, ?)",
                        (pl["line_id"], pl["brand_id"], pl["name_en"], pl["binder_type"]),
                    )
                for p in data.get("paints", []):
                    r_inf = p["masstone_lin_rgb"]
                    h_rgb = p["hiding_power_rgb"]
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO paints (
                            paint_id, line_id, code, name_en, name_ko, role, default_finish,
                            surfacer_grit, srgb_hex,
                            masstone_lin_r, masstone_lin_g, masstone_lin_b,
                            hiding_power_r, hiding_power_g, hiding_power_b,
                            ggx_roughness, metallic_ness, metallic_f0_rgb_json, chrome_f0,
                            beer_lambert_alpha_json, chameleon_stops_json,
                            fluo_quantum_yield, fluo_peak_rgb_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            p["paint_id"],
                            p["line_id"],
                            p["code"],
                            p["name_en"],
                            p["name_ko"],
                            p["role"],
                            p["default_finish"],
                            p.get("surfacer_grit"),
                            p["srgb_hex"],
                            r_inf[0],
                            r_inf[1],
                            r_inf[2],
                            h_rgb[0],
                            h_rgb[1],
                            h_rgb[2],
                            p.get("ggx_roughness", 0.08),
                            p.get("metallic_ness", 0.0),
                            json.dumps(p["metallic_f0_rgb"]) if "metallic_f0_rgb" in p else None,
                            p.get("chrome_f0", 0.95),
                            json.dumps(p["beer_lambert_alpha"]) if "beer_lambert_alpha" in p else None,
                            json.dumps(p["chameleon_stops_oklch"]) if "chameleon_stops_oklch" in p else None,
                            p.get("fluo_quantum_yield", 0.0),
                            json.dumps(p["fluo_peak_rgb"]) if "fluo_peak_rgb" in p else None,
                        ),
                    )

    @staticmethod
    def _srgb_hex_to_linear_rgb(srgb_hex: str) -> List[float]:
        h = (srgb_hex or "").strip().lstrip("#")
        if len(h) != 6:
            return [0.5, 0.5, 0.5]
        try:
            vals = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
        except ValueError:
            return [0.5, 0.5, 0.5]
        return [
            round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 6)
            for c in vals
        ]

    def upsert_paint(self, paint: Dict[str, Any]) -> None:
        """Insert or update a paint record, ensuring brand and product_line FK rows exist."""
        role = str(paint.get("role", "base_solid"))
        line_id = str(paint["line_id"])
        brand_id = str(paint.get("brand_id") or "gsi_creos")
        binder_type = str(paint.get("binder_type") or "lacquer")
        line_name_en = str(paint.get("line_name_en") or line_id)
        srgb_hex = str(paint.get("srgb_hex") or "#808080")
        if not srgb_hex.startswith("#") or len(srgb_hex) != 7:
            srgb_hex = "#808080"

        r_inf = paint.get("masstone_lin_rgb") or self._srgb_hex_to_linear_rgb(srgb_hex)
        default_hiding = (
            [0.15, 0.15, 0.15]
            if role == "clear_candy"
            else ([0.05, 0.05, 0.05] if role == "topcoat" else ([0.94, 0.94, 0.95] if role == "surfacer" else [0.90, 0.90, 0.90]))
        )
        h_rgb = paint.get("hiding_power_rgb") or default_hiding

        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO brands (brand_id, name_en, name_ko, country) VALUES (?, ?, ?, ?)",
                    (brand_id, brand_id, brand_id, ""),
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO product_lines (line_id, brand_id, name_en, binder_type) VALUES (?, ?, ?, ?)",
                    (line_id, brand_id, line_name_en, binder_type),
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO paints (
                        paint_id, line_id, code, name_en, name_ko, role, default_finish,
                        surfacer_grit, srgb_hex,
                        masstone_lin_r, masstone_lin_g, masstone_lin_b,
                        hiding_power_r, hiding_power_g, hiding_power_b,
                        ggx_roughness, metallic_ness, metallic_f0_rgb_json, chrome_f0,
                        beer_lambert_alpha_json, chameleon_stops_json,
                        fluo_quantum_yield, fluo_peak_rgb_json, is_calibrated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(paint["paint_id"]),
                        line_id,
                        str(paint["code"]),
                        str(paint.get("name_en") or paint["code"]),
                        str(paint.get("name_ko") or paint.get("name_en") or paint["code"]),
                        role,
                        str(paint.get("default_finish") or "gloss"),
                        paint.get("surfacer_grit"),
                        srgb_hex,
                        float(r_inf[0]),
                        float(r_inf[1]),
                        float(r_inf[2]),
                        float(h_rgb[0]),
                        float(h_rgb[1]),
                        float(h_rgb[2]),
                        float(paint.get("ggx_roughness", 0.15)),
                        float(paint.get("metallic_ness", 1.0 if role in ("metallic", "mirror_chrome") else 0.0)),
                        json.dumps(paint["metallic_f0_rgb"]) if paint.get("metallic_f0_rgb") is not None else None,
                        float(paint.get("chrome_f0", 0.95)),
                        json.dumps(paint["beer_lambert_alpha"]) if paint.get("beer_lambert_alpha") is not None else None,
                        json.dumps(paint["chameleon_stops_oklch"]) if paint.get("chameleon_stops_oklch") is not None else None,
                        float(paint.get("fluo_quantum_yield", 0.0)),
                        json.dumps(paint["fluo_peak_rgb"]) if paint.get("fluo_peak_rgb") is not None else None,
                        1 if paint.get("is_calibrated", False) else 0,
                    ),
                )

    def sync_from_master_csv(self, csv_path: Path) -> int:
        """
        Ingest collected swatches from `paint_sample_master_index.csv` using canonical
        `col_<brand_id>_<paint_code_slug>` IDs, preferring pure reference swatches
        (`surfacer_used == "N"` or `topcoat_used == "N"`).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return 0

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        best_by_id: Dict[str, tuple[tuple[int, int], Dict[str, str]]] = {}
        for row in rows:
            collector_role = (row.get("paint_role") or "").strip()
            if collector_role not in self._COLLECTOR_ROLE_MAP:
                continue
            brand_id = (row.get("brand_id") or "gsi_creos").strip() or "gsi_creos"
            paint_code = (row.get("paint_code") or "").strip()
            if not paint_code:
                continue
            brand_slug = re.sub(r"[^a-z0-9]+", "_", brand_id.lower()).strip("_") or "unknown"
            code_slug = re.sub(r"[^a-z0-9]+", "_", paint_code.lower()).strip("_") or "sample"
            canonical_id = f"col_{brand_slug}_{code_slug}"

            sf_n = 1 if (row.get("surfacer_used") or "").strip().upper() == "N" else 0
            tc_n = 1 if (row.get("topcoat_used") or "").strip().upper() == "N" else 0
            has_hex = 1 if (row.get("measured_srgb_hex") or "").strip().startswith("#") else 0
            score = (sf_n * 2 + tc_n, has_hex)

            prev = best_by_id.get(canonical_id)
            if prev is None or score > prev[0]:
                best_by_id[canonical_id] = (score, row)

        count = 0
        with self._lock:
            for canonical_id, (_, row) in best_by_id.items():
                collector_role = (row.get("paint_role") or "").strip()
                role = self._COLLECTOR_ROLE_MAP[collector_role]
                brand_id = (row.get("brand_id") or "gsi_creos").strip() or "gsi_creos"
                line_id = (row.get("product_line") or f"{brand_id}_collected").strip() or f"{brand_id}_collected"
                binder_type = (row.get("solvent_type") or "lacquer").strip() or "lacquer"
                srgb_hex = (row.get("measured_srgb_hex") or "#808080").strip()
                r_inf = self._srgb_hex_to_linear_rgb(srgb_hex)

                alpha_ggx = 0.15
                is_calibrated = False
                raw_opt = (row.get("calibrated_optical_params") or "").strip()
                if raw_opt:
                    try:
                        opt = json.loads(raw_opt)
                        if isinstance(opt, dict):
                            cal_val = opt.get("alpha_ggx_calibrated")
                            est_val = opt.get("estimated_alpha_ggx")
                            if isinstance(cal_val, bool):
                                is_calibrated = cal_val
                            elif isinstance(cal_val, (int, float)):
                                alpha_ggx = float(cal_val)
                                is_calibrated = True
                            if not is_calibrated or isinstance(cal_val, bool):
                                if isinstance(est_val, (int, float)) and not isinstance(est_val, bool):
                                    alpha_ggx = float(est_val)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                alpha_ggx = max(0.01, min(1.0, float(alpha_ggx)))

                grit_raw = (row.get("surfacer_grit") or "").strip()
                surfacer_grit = int(grit_raw) if grit_raw.isdigit() else None

                self.upsert_paint(
                    {
                        "paint_id": canonical_id,
                        "line_id": line_id,
                        "brand_id": brand_id,
                        "binder_type": binder_type,
                        "code": (row.get("paint_code") or "").strip(),
                        "name_en": (row.get("paint_name_en") or row.get("paint_code") or "").strip(),
                        "name_ko": (row.get("paint_name_ko") or row.get("paint_name_en") or "").strip(),
                        "role": role,
                        "default_finish": (row.get("final_finish") or "gloss").strip() or "gloss",
                        "surfacer_grit": surfacer_grit,
                        "srgb_hex": srgb_hex,
                        "masstone_lin_rgb": r_inf,
                        "ggx_roughness": alpha_ggx,
                        "metallic_ness": 1.0 if role in ("metallic", "mirror_chrome") else 0.0,
                        "is_calibrated": is_calibrated,
                    }
                )
                count += 1
        return count

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["masstone_lin_rgb"] = [d.pop("masstone_lin_r"), d.pop("masstone_lin_g"), d.pop("masstone_lin_b")]
        d["hiding_power_rgb"] = [d.pop("hiding_power_r"), d.pop("hiding_power_g"), d.pop("hiding_power_b")]
        f0_json = d.pop("metallic_f0_rgb_json", None)
        r_inf = d["masstone_lin_rgb"]
        d["metallic_f0_rgb"] = json.loads(f0_json) if f0_json else [
            min(max(r_inf[0] * 1.45, 0.12), 1.0),
            min(max(r_inf[1] * 1.45, 0.12), 1.0),
            min(max(r_inf[2] * 1.45, 0.12), 1.0),
        ]
        beer_json = d.pop("beer_lambert_alpha_json", None)
        d["beer_lambert_alpha"] = json.loads(beer_json) if beer_json else [0.0, 0.0, 0.0]
        cham_json = d.pop("chameleon_stops_json", None)
        if cham_json:
            d["chameleon_stops_oklch"] = json.loads(cham_json)
        elif d.get("role") == "chameleon":
            d["chameleon_stops_oklch"] = [0.65, 0.18, 210.0, 0.52, 0.22, 265.0, 0.48, 0.24, 325.0]
        else:
            d["chameleon_stops_oklch"] = []
        fluo_json = d.pop("fluo_peak_rgb_json", None)
        d["fluo_peak_rgb"] = json.loads(fluo_json) if fluo_json else [0.0, 0.0, 0.0]
        return d

    def get_paint(self, paint_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM paints WHERE paint_id = ?", (paint_id,))
            row = cur.fetchone()
            return self._row_to_dict(row) if row else None

    def list_paints(
        self,
        brand: Optional[str] = None,
        line: Optional[str] = None,
        role: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT p.* FROM paints p
            JOIN product_lines pl ON p.line_id = pl.line_id
            WHERE 1=1
        """
        params: List[Any] = []
        if brand:
            query += " AND pl.brand_id = ?"
            params.append(brand)
        if line:
            query += " AND p.line_id = ?"
            params.append(line)
        if role:
            query += " AND p.role = ?"
            params.append(role)
        if search:
            query += " AND (p.code LIKE ? OR p.name_en LIKE ? OR p.name_ko LIKE ?)"
            q = f"%{search}%"
            params.extend([q, q, q])
        query += " ORDER BY p.role, p.code"
        with self._lock:
            cur = self.conn.execute(query, params)
            return [self._row_to_dict(r) for r in cur.fetchall()]
