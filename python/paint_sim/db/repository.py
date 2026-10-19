"""Brand-agnostic Relational Paint Database Repository (SQLite / PostgreSQL compatible)."""
from __future__ import annotations

import json
import sqlite3
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

    def __init__(self, db_path: str | Path = ":memory:", seed_path: Optional[Path] = None) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        if seed_path is None:
            seed_path = Path(__file__).parent / "seeds" / "mr_color_seed.json"
        if seed_path.exists():
            self.seed_from_json(seed_path)

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    def seed_from_json(self, seed_path: Path) -> None:
        with open(seed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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
                        ggx_roughness, metallic_ness, chrome_f0,
                        beer_lambert_alpha_json, chameleon_stops_json,
                        fluo_quantum_yield, fluo_peak_rgb_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        p.get("chrome_f0", 0.95),
                        json.dumps(p["beer_lambert_alpha"]) if "beer_lambert_alpha" in p else None,
                        json.dumps(p["chameleon_stops_oklch"]) if "chameleon_stops_oklch" in p else None,
                        p.get("fluo_quantum_yield", 0.0),
                        json.dumps(p["fluo_peak_rgb"]) if "fluo_peak_rgb" in p else None,
                    ),
                )

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["masstone_lin_rgb"] = [d.pop("masstone_lin_r"), d.pop("masstone_lin_g"), d.pop("masstone_lin_b")]
        d["hiding_power_rgb"] = [d.pop("hiding_power_r"), d.pop("hiding_power_g"), d.pop("hiding_power_b")]
        beer_json = d.pop("beer_lambert_alpha_json", None)
        d["beer_lambert_alpha"] = json.loads(beer_json) if beer_json else [0.0, 0.0, 0.0]
        cham_json = d.pop("chameleon_stops_json", None)
        d["chameleon_stops_oklch"] = json.loads(cham_json) if cham_json else [0.65, 0.18, 210.0, 0.52, 0.22, 265.0, 0.48, 0.24, 325.0]
        fluo_json = d.pop("fluo_peak_rgb_json", None)
        d["fluo_peak_rgb"] = json.loads(fluo_json) if fluo_json else [0.0, 0.0, 0.0]
        return d

    def get_paint(self, paint_id: str) -> Optional[Dict[str, Any]]:
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
        cur = self.conn.execute(query, params)
        return [self._row_to_dict(r) for r in cur.fetchall()]
