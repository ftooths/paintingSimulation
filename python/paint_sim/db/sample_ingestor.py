"""Automated Paint Sample Image Classifier, Directory Builder & Master CSV/Excel Indexer."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np

_PYTHON_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _PYTHON_ROOT.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from paint_sim.physics.engine_bridge import NativePaintCoreBridge

# ============================================================================
# Canonical Taxonomy Constants (No Spaces, English Snake_Case Only)
# ============================================================================

VALID_BRAND_IDS = (
    "gsi_creos",
    "gaia_notes",
    "tamiya",
    "ipp",
    "vallejo",
    "citadel",
    "ak_interactive",
    "ammo_mig",
    "finisheers",
    "zero_paints",
    "alclad_ii",
    "show_up",
    "mr_paint_mrp",
    "monument_pro_acryl",
    "gsw",
    "modo",
    "anchoret",
    "jumpwind",
)

VALID_SOLVENT_TYPES = (
    "lacquer",
    "acrylic_alcohol",
    "acrylic_water",
    "enamel",
    "urethane_2k",
    "alcohol_marker_ink",
    "oil_weathering",
)

VALID_PAINT_ROLES = (
    "01_surfacer_primer",
    "02_base_solid",
    "03_metallic_flake",
    "04_mirror_chrome",
    "05_pearl_mica",
    "06_chameleon_colorshift",
    "07_clear_candy",
    "08_fluorescent_neon",
    "09_phosphorescent_glow",
    "10_panel_wash_filter",
    "11_weathering_texture",
    "12_topcoat_varnish",
)

# Maps Level 3 paint_role folders 1:1 to C++20/WebGL2 `km_brdf_solver.hpp` LayerRole / solver targets
PAINT_ROLE_TO_SOLVER_MAPPING: Dict[str, Dict[str, Any]] = {
    "01_surfacer_primer": {"solver_target": "LayerRole.SURFACER", "layer_role_id": 0},
    "02_base_solid": {"solver_target": "LayerRole.BASE_SOLID", "layer_role_id": 1},
    "03_metallic_flake": {"solver_target": "LayerRole.METALLIC", "layer_role_id": 2},
    "04_mirror_chrome": {"solver_target": "LayerRole.MIRROR_CHROME", "layer_role_id": 3},
    "05_pearl_mica": {"solver_target": "LayerRole.PEARL", "layer_role_id": 6},
    "06_chameleon_colorshift": {"solver_target": "LayerRole.CHAMELEON", "layer_role_id": 5},
    "07_clear_candy": {"solver_target": "LayerRole.CLEAR_CANDY", "layer_role_id": 4},
    "08_fluorescent_neon": {"solver_target": "LayerRole.FLUORESCENT_SOLID", "layer_role_id": 7},
    "09_phosphorescent_glow": {"solver_target": "LayerRole.FLUORESCENT_SOLID", "layer_role_id": 7},
    "10_panel_wash_filter": {"solver_target": "RegionRenderStack.wash_params", "layer_role_id": None},
    "11_weathering_texture": {"solver_target": "RegionRenderStack.roughness_veil", "layer_role_id": None},
    "12_topcoat_varnish": {"solver_target": "TopcoatType", "layer_role_id": None},
}

# Pre-scaffolded standard directory tree for all 18 modeling brands in PAINT_SAMPLE_CLASSIFICATION_SPEC.md
STANDARD_SAMPLE_DIRECTORY_TREE: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "gsi_creos": {
        "lacquer": {
            "01_surfacer_primer": ["mr_surfacer", "mr_finishing_surfacer"],
            "02_base_solid": ["mr_color_c", "mr_color_gx", "mr_color_lascivus"],
            "03_metallic_flake": ["mr_color_c", "mr_color_gx_metal", "mr_super_metallic_2"],
            "04_mirror_chrome": ["mr_super_metallic_2"],
            "05_pearl_mica": ["mr_crystal_color", "mr_color_gx_pearl"],
            "06_chameleon_colorshift": ["mr_crystal_chameleon"],
            "07_clear_candy": ["mr_color_c_clear", "mr_color_gx_clear"],
            "08_fluorescent_neon": ["mr_color_c_fluorescent"],
            "12_topcoat_varnish": ["mr_super_clear", "mr_color_gx_clearcoat"],
        },
        "acrylic_alcohol": {
            "01_surfacer_primer": ["aqueous_surfacer"],
            "02_base_solid": ["aqueous_h", "aqueous_hobby_color_h", "aqueous_gundam_color"],
            "03_metallic_flake": ["aqueous_h", "aqueous_hobby_color_h"],
            "05_pearl_mica": ["aqueous_h", "aqueous_hobby_color_h", "aqueous_linkl_planet_color"],
            "07_clear_candy": ["aqueous_h", "aqueous_hobby_color_h"],
            "12_topcoat_varnish": ["aqueous_topcoat_h"],
        },
        "acrylic_water": {
            "01_surfacer_primer": ["acrysion_base"],
            "02_base_solid": ["acrysion_n"],
            "03_metallic_flake": ["acrysion_n"],
            "12_topcoat_varnish": ["acrysion_clear"],
        },
        "enamel": {
            "10_panel_wash_filter": ["mr_weathering_color"],
        },
        "alcohol_marker_ink": {
            "02_base_solid": ["gundam_marker"],
            "03_metallic_flake": ["gundam_marker_metallic"],
            "04_mirror_chrome": ["gundam_marker_ex"],
            "10_panel_wash_filter": ["gundam_marker_pour_type"],
        },
    },
    "gaia_notes": {
        "lacquer": {
            "01_surfacer_primer": ["surfacer_evo", "nazca", "nazca_surfacer"],
            "02_base_solid": ["gaia_basic", "gaia_ex", "nazca", "virtual_on", "evangelion_color"],
            "03_metallic_flake": ["star_bright_metallic", "gaia_ex_metallic"],
            "04_mirror_chrome": ["premium_color", "premium_mirror_chrome"],
            "05_pearl_mica": ["gaia_pearl", "premium_color"],
            "06_chameleon_colorshift": ["gaia_prism_polarized", "premium_color"],
            "07_clear_candy": ["gaia_clear"],
            "08_fluorescent_neon": ["fluorescent_clear", "gaia_fluorescent_clear"],
            "09_phosphorescent_glow": ["gaia_ex_glow"],
            "12_topcoat_varnish": ["gaia_ex_clear", "gaia_ex"],
        },
        "enamel": {
            "02_base_solid": ["gaia_enamel"],
            "10_panel_wash_filter": ["gaia_enamel_wash"],
        },
    },
    "tamiya": {
        "lacquer": {
            "01_surfacer_primer": ["surface_primer"],
            "02_base_solid": ["lacquer_lp", "spray_ts", "spray_as", "polycarbonate_ps"],
            "03_metallic_flake": ["lacquer_lp", "spray_ts", "polycarbonate_ps"],
            "05_pearl_mica": ["lacquer_lp", "spray_ts"],
            "07_clear_candy": ["lacquer_lp", "polycarbonate_ps"],
            "12_topcoat_varnish": ["lacquer_lp", "spray_ts"],
        },
        "acrylic_alcohol": {
            "02_base_solid": ["acrylic_x_gloss", "acrylic_xf_flat"],
            "03_metallic_flake": ["acrylic_x_gloss", "acrylic_xf_flat"],
            "07_clear_candy": ["acrylic_x_gloss"],
            "12_topcoat_varnish": ["acrylic_x_gloss", "acrylic_xf_flat"],
        },
        "enamel": {
            "02_base_solid": ["enamel_x_gloss", "enamel_xf_flat"],
            "03_metallic_flake": ["enamel_x_gloss", "enamel_xf_flat"],
            "10_panel_wash_filter": ["panel_line_accent"],
        },
    },
    "ipp": {
        "lacquer": {
            "01_surfacer_primer": ["ipp_surfacer"],
            "02_base_solid": ["ipp_basic_lacquer", "ipp_character_color"],
            "03_metallic_flake": ["ipp_super_fine_metallic"],
            "04_mirror_chrome": ["ipp_premium_chrome"],
            "05_pearl_mica": ["ipp_pearl_clear", "ipp_pearl"],
            "07_clear_candy": ["ipp_pearl_clear", "ipp_clear"],
            "08_fluorescent_neon": ["ipp_fluorescent"],
            "12_topcoat_varnish": ["ipp_super_clear"],
        },
        "urethane_2k": {
            "12_topcoat_varnish": ["ipp_urethane_2k", "ipp_urethane_clear_2k"],
        },
    },
    "vallejo": {
        "acrylic_water": {
            "01_surfacer_primer": ["surface_primer", "mecha_primer"],
            "02_base_solid": ["model_color", "model_air", "mecha_color", "game_color"],
            "03_metallic_flake": ["metal_color", "mecha_metallic"],
            "06_chameleon_colorshift": ["the_shifters"],
            "07_clear_candy": ["transparent_color"],
            "08_fluorescent_neon": ["fluorescent_color"],
            "10_panel_wash_filter": ["model_wash", "mecha_weathering"],
            "11_weathering_texture": ["diorama_fx", "pigments"],
            "12_topcoat_varnish": ["premium_varnish", "mecha_varnish"],
        },
    },
    "citadel": {
        "acrylic_water": {
            "02_base_solid": ["citadel_base", "citadel_layer", "citadel_air"],
            "03_metallic_flake": ["citadel_base", "citadel_layer", "citadel_base_metallic", "citadel_layer_metallic"],
            "07_clear_candy": ["citadel_contrast"],
            "10_panel_wash_filter": ["citadel_shade"],
            "11_weathering_texture": ["citadel_technical"],
            "12_topcoat_varnish": ["citadel_varnish", "citadel_technical"],
        },
    },
    "ak_interactive": {
        "acrylic_water": {
            "01_surfacer_primer": ["ak_3gen_primer"],
            "02_base_solid": ["ak_3gen_acrylic"],
            "03_metallic_flake": ["ak_3gen_metallic"],
        },
        "lacquer": {
            "02_base_solid": ["ak_real_colors_lacquer", "ak_real_colors"],
            "03_metallic_flake": ["ak_xtreme_metal"],
            "04_mirror_chrome": ["ak_xtreme_metal"],
        },
        "enamel": {
            "03_metallic_flake": ["ak_true_metal_wax"],
            "10_panel_wash_filter": ["ak_enamel_weathering", "ak_enamel_wash", "ak_panel_liner"],
            "11_weathering_texture": ["ak_enamel_weathering", "ak_enamel_effects"],
        },
    },
    "ammo_mig": {
        "acrylic_water": {
            "02_base_solid": ["ammo_acrylic"],
            "06_chameleon_colorshift": ["ammo_shifters"],
        },
        "lacquer": {
            "02_base_solid": ["ammo_a_stand_lacquer"],
            "03_metallic_flake": ["ammo_a_stand_lacquer"],
            "12_topcoat_varnish": ["ammo_a_stand_lacquer"],
        },
        "enamel": {
            "10_panel_wash_filter": ["ammo_enamel_wash"],
        },
        "oil_weathering": {
            "11_weathering_texture": ["ammo_oilbrusher"],
        },
    },
    "finisheers": {
        "lacquer": {
            "01_surfacer_primer": ["finisheers_foundation"],
            "02_base_solid": ["finisheers_auto_lacquer"],
            "03_metallic_flake": ["finisheers_metallic"],
        },
        "urethane_2k": {
            "12_topcoat_varnish": ["finisheers_gp1_gp2_urethane"],
        },
    },
    "zero_paints": {
        "lacquer": {
            "02_base_solid": ["zero_basecoat_lacquer"],
            "05_pearl_mica": ["zero_candy_pearl"],
            "07_clear_candy": ["zero_candy_pearl"],
        },
        "urethane_2k": {
            "12_topcoat_varnish": ["zero_2k_urethane_clear"],
        },
    },
    "alclad_ii": {
        "lacquer": {
            "01_surfacer_primer": ["alclad_gloss_base"],
            "03_metallic_flake": ["alclad_regular_metallic"],
            "04_mirror_chrome": ["alclad_high_shine_chrome"],
            "06_chameleon_colorshift": ["alclad_holomatic_spectral"],
            "07_clear_candy": ["alclad_transparent_candy"],
        },
        "acrylic_water": {
            "12_topcoat_varnish": ["alclad_aqua_gloss"],
        },
    },
    "show_up": {
        "lacquer": {
            "01_surfacer_primer": ["showup_base_primer"],
            "04_mirror_chrome": ["showup_terra_chrome"],
            "06_chameleon_colorshift": ["showup_maziora_chameleon"],
            "07_clear_candy": ["showup_onedrop_candy"],
        },
        "urethane_2k": {
            "12_topcoat_varnish": ["showup_urethane", "showup_urethane_clear"],
        },
    },
    "mr_paint_mrp": {
        "lacquer": {
            "01_surfacer_primer": ["mrp_fine_surfacer"],
            "02_base_solid": ["mrp_acrylic_lacquer"],
            "03_metallic_flake": ["mrp_super_metallic"],
            "12_topcoat_varnish": ["mrp_clear_varnish"],
        },
    },
    "monument_pro_acryl": {
        "acrylic_water": {
            "01_surfacer_primer": ["pro_acryl_prime"],
            "02_base_solid": ["pro_acryl_standard"],
            "03_metallic_flake": ["pro_acryl_metallic"],
            "07_clear_candy": ["pro_acryl_transparent"],
        },
    },
    "gsw": {
        "acrylic_water": {
            "04_mirror_chrome": ["gsw_chrome_metal"],
            "06_chameleon_colorshift": ["gsw_colorshift_chameleon"],
            "08_fluorescent_neon": ["gsw_fluorescent"],
            "11_weathering_texture": ["gsw_liquid_pigments"],
        },
    },
    "modo": {
        "lacquer": {
            "01_surfacer_primer": ["modo_surfacer"],
            "02_base_solid": ["modo_neo_lacquer"],
            "03_metallic_flake": ["modo_robot_metallic"],
            "06_chameleon_colorshift": ["modo_chameleon"],
        },
    },
    "anchoret": {
        "lacquer": {
            "01_surfacer_primer": ["anchoret_surfacer"],
            "02_base_solid": ["anchoret_mecha_lacquer"],
            "03_metallic_flake": ["anchoret_metallic"],
        },
    },
    "jumpwind": {
        "lacquer": {
            "01_surfacer_primer": ["jumpwind_prime"],
            "02_base_solid": ["jumpwind_mito_lacquer"],
            "03_metallic_flake": ["jumpwind_extra_metal", "jumpwind_prime_metal"],
        },
    },
}

# ============================================================================
# Section 3 Compatibility Matrices (Solvent Solubility & Layer Overcoating)
# ============================================================================

SOLVENT_SOLUBILITY_MATRIX: Dict[str, Dict[str, str]] = {
    "warm_water": {
        "acrylic_water": "△",
        "acrylic_alcohol": "△",
        "alcohol_marker_ink": "X",
        "enamel": "X",
        "lacquer": "X",
        "urethane_2k": "X",
    },
    "ethyl_alcohol": {
        "acrylic_water": "△",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "X",
        "lacquer": "X",
        "urethane_2k": "X",
    },
    "ipa_99": {
        "acrylic_water": "○",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "△",
        "lacquer": "△",
        "urethane_2k": "X",
    },
    "methyl_alcohol": {
        "acrylic_water": "△",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "◎",
        "lacquer": "◎",
        "urethane_2k": "X",
    },
    "acrylic_thinner_x20a": {
        "acrylic_water": "○",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "X",
        "lacquer": "X",
        "urethane_2k": "X",
    },
    "enamel_thinner": {
        "acrylic_water": "X",
        "acrylic_alcohol": "X",
        "alcohol_marker_ink": "X",  # Pour-type GM301..303 is "◎"
        "enamel": "◎",
        "lacquer": "X",
        "urethane_2k": "X",
    },
    "lighter_fluid": {
        "acrylic_water": "X",
        "acrylic_alcohol": "X",
        "alcohol_marker_ink": "X",  # Pour-type GM301..303 is "◎"
        "enamel": "◎",
        "lacquer": "X",
        "urethane_2k": "X",
    },
    "industrial_thinner": {
        "acrylic_water": "○",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "◎",
        "lacquer": "◎",
        "urethane_2k": "△",
    },
    "hobby_lacquer_thinner": {
        "acrylic_water": "X",  # Chart: X on light contact; peels if soaked/rubbed (△)
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "◎",
        "lacquer": "◎",
        "urethane_2k": "X",
    },
    "leveling_thinner": {
        "acrylic_water": "X",  # Chart: X on light contact; peels if soaked/rubbed (△)
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "◎",
        "lacquer": "◎",
        "urethane_2k": "X",
    },
    "acetone": {
        "acrylic_water": "◎",
        "acrylic_alcohol": "◎",
        "alcohol_marker_ink": "◎",
        "enamel": "◎",
        "lacquer": "◎",
        "urethane_2k": "△",
    },
}

OVERCOATING_COMPATIBILITY_MATRIX: Dict[str, Dict[str, str]] = {
    "bare_plastic": {
        "urethane_2k": "△",
        "lacquer": "○",
        "acrylic_alcohol": "◎",
        "enamel": "X",  # spec: "X ~ △" (cracks stressed PS/ABS) -> worst case kept for safety
        "acrylic_water": "△",
        "alcohol_marker_ink": "○",
    },
    "urethane_2k": {
        "urethane_2k": "◎",
        "lacquer": "○",
        "acrylic_alcohol": "○",
        "enamel": "◎",
        "acrylic_water": "○",
        "alcohol_marker_ink": "○",
    },
    "lacquer": {
        "urethane_2k": "◎",
        "lacquer": "◎",
        "acrylic_alcohol": "◎",
        "enamel": "◎",
        "acrylic_water": "◎",
        "alcohol_marker_ink": "△",
    },
    "acrylic_alcohol": {
        "urethane_2k": "△",
        "lacquer": "△",
        "acrylic_alcohol": "◎",
        "enamel": "◎",
        "acrylic_water": "◎",
        "alcohol_marker_ink": "X",
    },
    "acrylic_water": {
        "urethane_2k": "△",
        "lacquer": "△",
        "acrylic_alcohol": "○",
        "enamel": "◎",
        "acrylic_water": "◎",
        "alcohol_marker_ink": "△",
    },
    "enamel": {
        "urethane_2k": "X",
        "lacquer": "X",
        "acrylic_alcohol": "△",
        "enamel": "△",
        "acrylic_water": "◎",
        "alcohol_marker_ink": "X",
    },
    "alcohol_marker_ink": {
        "urethane_2k": "X",
        "lacquer": "X",
        "acrylic_alcohol": "X",
        "enamel": "◎",
        "acrylic_water": "◎",
        "alcohol_marker_ink": "X",
    },
    "mirror_chrome": {
        "urethane_2k": "X",
        "lacquer": "X",
        "acrylic_alcohol": "△",
        "enamel": "○",
        "acrylic_water": "○",
        "alcohol_marker_ink": "X",
    },
}


def evaluate_solvent_solubility(
    solvent_id: str,
    cured_film_type: str,
    is_pour_type_marker: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate solubility (`◎`, `○`, `△`, `X`) of a cured paint film in a given solvent
    per Section 3.1 of `docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md`.
    """
    solv = sanitize_token(solvent_id).lower()
    film = sanitize_token(cured_film_type).lower()
    if solv not in SOLVENT_SOLUBILITY_MATRIX:
        raise ValueError(f"Unknown solvent_id '{solv}'. Allowed: {tuple(SOLVENT_SOLUBILITY_MATRIX)}")
    if film not in SOLVENT_SOLUBILITY_MATRIX[solv]:
        raise ValueError(f"Unknown cured_film_type '{film}'. Allowed: {tuple(SOLVENT_SOLUBILITY_MATRIX[solv])}")

    symbol = SOLVENT_SOLUBILITY_MATRIX[solv][film]
    note = ""
    if film == "alcohol_marker_ink" and solv in ("enamel_thinner", "lighter_fluid") and is_pour_type_marker:
        symbol = "◎"
        note = "Pour-type panel line markers (GM301-GM303) use hydrocarbon ink and dissolve in enamel thinner / lighter fluid."
    elif film == "acrylic_water" and solv in ("hobby_lacquer_thinner", "leveling_thinner"):
        note = "Standard chart marks X on brief contact after 48h+ cure, but soaking or rubbing lifts/peels the emulsion film (△)."

    return {
        "solvent_id": solv,
        "cured_film_type": film,
        "symbol": symbol,
        "safe_for_panel_wipe": symbol == "X",
        "can_strip_or_thin": symbol in ("◎", "○"),
        "note": note,
    }


def evaluate_overcoat_compatibility(bottom_layer: str, top_layer: str) -> Dict[str, Any]:
    """
    Evaluate layer-over-layer overcoating safety (`◎`, `○`, `△`, `X`) per Section 3.2
    of `docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md`.
    """
    bot = sanitize_token(bottom_layer).lower()
    top = sanitize_token(top_layer).lower()
    if bot not in OVERCOATING_COMPATIBILITY_MATRIX:
        raise ValueError(f"Unknown bottom_layer '{bot}'. Allowed: {tuple(OVERCOATING_COMPATIBILITY_MATRIX)}")
    if top not in OVERCOATING_COMPATIBILITY_MATRIX[bot]:
        raise ValueError(f"Unknown top_layer '{top}'. Allowed: {tuple(OVERCOATING_COMPATIBILITY_MATRIX[bot])}")

    symbol = OVERCOATING_COMPATIBILITY_MATRIX[bot][top]
    return {
        "bottom_layer": bot,
        "top_layer": top,
        "symbol": symbol,
        "is_safe": symbol in ("◎", "○"),
        "requires_mist_coat": symbol == "△",
        "is_incompatible": symbol == "X",
    }


MASTER_CSV_COLUMNS = [
    "sample_id",
    "file_rel_path",
    "file_name",
    "source_type",
    "source_url",
    "brand_id",
    "solvent_type",
    "paint_role",
    "product_line",
    "paint_code",
    "paint_name_en",
    "paint_name_ko",
    "substrate_color",
    "surfacer_used",
    "surfacer_code",
    "surfacer_grit",
    "undercoat_code",
    "main_coat_count",
    "thinner_and_ratio",
    "topcoat_used",
    "topcoat_code",
    "final_finish",
    "lighting_env",
    "measured_srgb_hex",
    "calibrated_optical_params",
    "notes",
]


def sanitize_token(text: str) -> str:
    """Convert any identifier into lowercase ASCII snake_case with zero whitespace."""
    cleaned = re.sub(r"[^a-zA-Z0-9+.\-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "none"


def _has_kw(text: str, keywords: Sequence[str]) -> bool:
    """
    Token-boundary keyword match. Plain substring matching produced false positives
    (e.g. "chipping" -> "ipp" brand, "chemical" -> "mica" pearl role, "marigold" -> "gold").
    A boundary is only enforced on sides where the keyword itself starts/ends with [a-z0-9],
    so keywords such as "xf-" or Korean terms still match naturally.
    """
    for kw in keywords:
        k = kw.lower()
        lead = r"(?<![a-z0-9])" if k[:1].isascii() and k[:1].isalnum() else ""
        tail = r"(?![a-z0-9])" if k[-1:].isascii() and k[-1:].isalnum() else ""
        if re.search(lead + re.escape(k) + tail, text):
            return True
    return False


_TAXONOMY_RULES_CACHE: Dict[str, Any] = {"mtime": -1.0, "data": {}}


def load_taxonomy_rules(rules_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and hot-reload externalized taxonomy classification rules from `config/paint_taxonomy_rules.json`."""
    if rules_path is None:
        rules_path = _REPO_ROOT / "config" / "paint_taxonomy_rules.json"
    rules_path = Path(rules_path)
    try:
        mtime = rules_path.stat().st_mtime
        if mtime != _TAXONOMY_RULES_CACHE["mtime"] or not _TAXONOMY_RULES_CACHE["data"]:
            with open(rules_path, "r", encoding="utf-8") as f:
                _TAXONOMY_RULES_CACHE["data"] = json.load(f)
            _TAXONOMY_RULES_CACHE["mtime"] = mtime
    except Exception as exc:
        warnings.warn(f"Failed to load {rules_path} ({exc}); using cached/empty rules.")
    return _TAXONOMY_RULES_CACHE["data"]


def _rule_matches(rule: Dict[str, Any], combined_text: str, code_up: str, brand_id: str = "") -> bool:
    """Evaluate a single taxonomy rule from `config/paint_taxonomy_rules.json`."""
    match_brands = rule.get("match_brands")
    brand_ok = bool(match_brands and brand_id in match_brands)
    kws = rule.get("match_keywords") or []
    kw_ok = bool(kws and _has_kw(combined_text, kws))
    regex_pat = rule.get("match_code_regex")
    rx_ok = bool(regex_pat and re.match(regex_pat, code_up))

    if match_brands and (kws or regex_pat):
        # When both brand filter and code/keyword are given (e.g., Tamiya X/XF), require brand AND (kw OR regex)
        if not rule.get("kws_or_brand", False) and regex_pat and not kws:
            return brand_ok and rx_ok
        return brand_ok or kw_ok or rx_ok
    if match_brands:
        return brand_ok
    return kw_ok or rx_ok


def classify_paint_sample(
    paint_code: str,
    paint_name_en: str = "",
    paint_name_ko: str = "",
    brand_hint: str = "",
    solvent_hint: str = "",
    role_hint: str = "",
    product_line_hint: str = "",
    image_bytes: Optional[bytes] = None,
    sheet_text: str = "",
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Deterministic rule-engine classifier driven by `config/paint_taxonomy_rules.json` (Section 7.4).
    Resolves a paint catalog code + name (+ optional hints/defaults/sheet_text) into its canonical
    4-level taxonomy (`brand_id`, `solvent_type`, `paint_role`, `product_line`), simulator `LayerRole`
    mapping, `default_finish`, `surfacer_grit`, scored `candidates`, and `ambiguous_fields`.

    NOTE: classification is TEXT-based only. `image_bytes`, when given, is used solely to measure
    optical parameters (`measured_srgb_hex`, `calibrated_optical_params`); pixels never influence
    the taxonomy. Fields resolved by a default fallback (no rule matched) are listed in
    `review_reasons` and flagged with `needs_review=True`. When top two candidate scores for a field
    are within `ambiguity_margin` (0.25), the field is listed in `ambiguous_fields`.
    """
    defaults = defaults or {}
    brand_hint = brand_hint or defaults.get("brand_id", "")
    solvent_hint = solvent_hint or defaults.get("solvent_type", "")
    role_hint = role_hint or defaults.get("paint_role", "")
    product_line_hint = product_line_hint or defaults.get("product_line", "")

    rules_cfg = load_taxonomy_rules()
    ambiguity_margin = float(rules_cfg.get("ambiguity_margin", 0.25))

    review_reasons: List[str] = []
    ambiguous_fields: List[str] = []
    candidates: Dict[str, List[Dict[str, Any]]] = {
        "brand_id": [],
        "solvent_type": [],
        "paint_role": [],
    }

    raw_code = paint_code.strip()
    code_up = raw_code.upper().replace(" ", "")
    combined_text = f"{raw_code} {paint_name_en} {paint_name_ko} {brand_hint} {product_line_hint} {sheet_text}".lower()

    # 1. Resolve brand_id
    brand_id = sanitize_token(brand_hint).lower() if brand_hint.strip() else "none"
    if brand_id in VALID_BRAND_IDS:
        candidates["brand_id"] = [{"value": brand_id, "score": 1.0}]
    else:
        matched_brands: List[Dict[str, Any]] = []
        for rule in sorted(rules_cfg.get("brand_rules", []), key=lambda r: -int(r.get("priority", 0))):
            if _rule_matches(rule, combined_text, code_up):
                val = rule["value"]
                sc = float(rule.get("score", 0.9))
                if not any(c["value"] == val for c in matched_brands):
                    matched_brands.append({"value": val, "score": sc})
        if matched_brands:
            brand_id = matched_brands[0]["value"]
            candidates["brand_id"] = matched_brands
        else:
            brand_id = "gsi_creos"
            candidates["brand_id"] = [{"value": "gsi_creos", "score": 0.25}]
            review_reasons.append("brand_id: no rule matched, defaulted to gsi_creos")

    # 2. Resolve solvent_type
    solvent_type = sanitize_token(solvent_hint).lower() if solvent_hint.strip() else "none"
    if solvent_type in VALID_SOLVENT_TYPES:
        candidates["solvent_type"] = [{"value": solvent_type, "score": 1.0}]
    else:
        matched_solvents: List[Dict[str, Any]] = []
        for rule in sorted(rules_cfg.get("solvent_rules", []), key=lambda r: -int(r.get("priority", 0))):
            # For acrylic_water rule, match if brand in match_brands OR keywords/regex match
            m_brands = rule.get("match_brands")
            kws = rule.get("match_keywords") or []
            rx = rule.get("match_code_regex")
            if m_brands and (kws or rx):
                hit = (brand_id in m_brands) or (bool(kws) and _has_kw(combined_text, kws)) or (bool(rx) and bool(re.match(rx, code_up)))
            elif m_brands and rx and not kws:
                hit = (brand_id in m_brands) and bool(re.match(rx, code_up))
            else:
                hit = _rule_matches(rule, combined_text, code_up, brand_id=brand_id)
            if hit:
                val = rule["value"]
                sc = float(rule.get("score", 0.9))
                if not any(c["value"] == val for c in matched_solvents):
                    matched_solvents.append({"value": val, "score": sc})
                for sec in rule.get("secondary_candidates", []):
                    if not any(c["value"] == sec["value"] for c in matched_solvents):
                        matched_solvents.append({"value": sec["value"], "score": float(sec["score"])})
                break
        if matched_solvents:
            solvent_type = matched_solvents[0]["value"]
            candidates["solvent_type"] = matched_solvents
        else:
            solvent_type = "lacquer"
            candidates["solvent_type"] = [{"value": "lacquer", "score": 0.85}]

    # 3. Resolve paint_role (01..12)
    paint_role = sanitize_token(role_hint).lower() if role_hint.strip() else "none"
    if paint_role in VALID_PAINT_ROLES:
        candidates["paint_role"] = [{"value": paint_role, "score": 1.0}]
    else:
        matched_roles: List[Dict[str, Any]] = []
        for rule in sorted(rules_cfg.get("role_rules", []), key=lambda r: -int(r.get("priority", 0))):
            if _rule_matches(rule, combined_text, code_up, brand_id=brand_id):
                val = rule["value"]
                sc = float(rule.get("score", 0.9))
                if not any(c["value"] == val for c in matched_roles):
                    matched_roles.append({"value": val, "score": sc})
        if matched_roles:
            paint_role = matched_roles[0]["value"]
            candidates["paint_role"] = matched_roles
        else:
            paint_role = "02_base_solid"
            candidates["paint_role"] = [{"value": "02_base_solid", "score": 0.85}]

    # Check candidate score margins for ambiguity (Section 7.4)
    for field_name, cand_list in candidates.items():
        if len(cand_list) >= 2:
            top_diff = float(cand_list[0]["score"]) - float(cand_list[1]["score"])
            if top_diff <= ambiguity_margin:
                ambiguous_fields.append(field_name)

    # 4. Resolve canonical Level 4 product_line
    product_line = sanitize_token(product_line_hint).lower() if product_line_hint.strip() else "none"
    if product_line == "none":
        brand_tree = STANDARD_SAMPLE_DIRECTORY_TREE.get(brand_id, {})
        role_lines = brand_tree.get(solvent_type, {}).get(paint_role, [])
        if brand_id == "gsi_creos":
            if solvent_type == "acrylic_alcohol":
                if code_up.startswith("HLP") or "linkl planet" in combined_text:
                    product_line = "aqueous_linkl_planet_color"
                elif paint_role == "01_surfacer_primer":
                    product_line = "aqueous_surfacer"
                elif paint_role == "12_topcoat_varnish":
                    product_line = "aqueous_topcoat_h"
                elif "gundam" in combined_text or code_up.startswith("UG"):
                    product_line = "aqueous_gundam_color"
                else:
                    product_line = "aqueous_hobby_color_h"
            elif solvent_type == "acrylic_water":
                product_line = role_lines[0] if role_lines else "acrysion_n"
            elif solvent_type == "alcohol_marker_ink":
                product_line = role_lines[0] if role_lines else "gundam_marker"
            elif solvent_type == "enamel":
                product_line = "mr_weathering_color"
            else:
                if code_up.startswith("SM"):
                    product_line = "mr_super_metallic_2"
                elif code_up.startswith("XC"):
                    product_line = "mr_crystal_color"
                elif code_up.startswith("CP"):
                    product_line = "mr_crystal_chameleon"
                elif code_up.startswith("CL") or "lascivus" in combined_text:
                    product_line = "mr_color_lascivus"
                elif code_up.startswith("GX"):
                    if paint_role == "03_metallic_flake":
                        product_line = "mr_color_gx_metal"
                    elif paint_role == "05_pearl_mica":
                        product_line = "mr_color_gx_pearl"
                    elif paint_role == "07_clear_candy":
                        product_line = "mr_color_gx_clear"
                    elif paint_role == "12_topcoat_varnish":
                        product_line = "mr_color_gx_clearcoat"
                    else:
                        product_line = "mr_color_gx"
                elif paint_role == "01_surfacer_primer":
                    product_line = "mr_finishing_surfacer" if "1500" in combined_text else "mr_surfacer"
                elif paint_role == "07_clear_candy":
                    product_line = "mr_color_c_clear"
                elif paint_role == "08_fluorescent_neon":
                    product_line = "mr_color_c_fluorescent"
                elif paint_role == "12_topcoat_varnish":
                    product_line = "mr_super_clear"
                else:
                    product_line = "mr_color_c"
        elif brand_id == "gaia_notes":
            if "nazca" in combined_text or code_up.startswith("NP"):
                product_line = "nazca_surfacer" if paint_role == "01_surfacer_primer" else "nazca"
            elif code_up.startswith("GF") or paint_role == "08_fluorescent_neon":
                product_line = "gaia_fluorescent_clear"
            elif code_up.startswith("EX"):
                product_line = "gaia_ex_clear" if paint_role == "12_topcoat_varnish" else (
                    "gaia_ex_metallic" if paint_role == "03_metallic_flake" else "gaia_ex"
                )
            elif role_lines:
                product_line = role_lines[0]
            else:
                product_line = "gaia_basic"
        elif brand_id == "tamiya":
            if code_up.startswith("TS"):
                product_line = "spray_ts"
            elif code_up.startswith("AS"):
                product_line = "spray_as"
            elif code_up.startswith("PS"):
                product_line = "polycarbonate_ps"
            elif code_up.startswith("XF"):
                product_line = "enamel_xf_flat" if solvent_type == "enamel" else "acrylic_xf_flat"
            elif code_up.startswith("X-"):
                product_line = "enamel_x_gloss" if solvent_type == "enamel" else "acrylic_x_gloss"
            elif role_lines:
                product_line = role_lines[0]
            else:
                product_line = "lacquer_lp"
        else:
            if role_lines:
                product_line = role_lines[0]
            else:
                product_line = "unsorted"
                review_reasons.append(f"product_line: no canonical line for {brand_id}/{solvent_type}/{paint_role}")

    # 5. Infer surfacer_grit and default_finish
    surfacer_grit = "none"
    if paint_role == "01_surfacer_primer":
        for grit_cand in ("1500", "1200", "1000", "500"):
            if grit_cand in combined_text:
                surfacer_grit = grit_cand
                break
        if surfacer_grit == "none" and code_up == "GX2-B":
            surfacer_grit = "1500"

    if paint_role == "04_mirror_chrome":
        default_finish = "mirror_chrome"
    elif paint_role == "03_metallic_flake":
        default_finish = "metallic"
    elif paint_role == "05_pearl_mica":
        default_finish = "pearl"
    else:
        default_finish = "gloss"
        for f_rule in sorted(rules_cfg.get("finish_rules", []), key=lambda r: -int(r.get("priority", 0))):
            if _rule_matches(f_rule, combined_text, code_up, brand_id=brand_id):
                default_finish = f_rule["value"]
                break
        if default_finish == "gloss" and paint_role == "01_surfacer_primer":
            default_finish = "semi_gloss"

    measured_hex = None
    optical_params = None
    if image_bytes is not None:
        measured_hex, optical_params = measure_swatch_optics(image_bytes)

    solver_info = PAINT_ROLE_TO_SOLVER_MAPPING[paint_role]
    return {
        "brand_id": brand_id,
        "solvent_type": solvent_type,
        "paint_role": paint_role,
        "product_line": product_line,
        "paint_code": raw_code,
        "candidates": candidates,
        "ambiguous_fields": ambiguous_fields,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "solver_target": solver_info["solver_target"],
        "layer_role_id": solver_info["layer_role_id"],
        "surfacer_grit": surfacer_grit,
        "default_finish": default_finish,
        "measured_srgb_hex": measured_hex,
        "calibrated_optical_params": optical_params,
    }


def build_sample_filename(
    paint_code: str,
    paint_name_en: str,
    surfacer_code: str = "none",
    undercoat_code: str = "none",
    topcoat_code: str = "none",
    coats: float | int = 2,
    substrate_color: str = "white",
    lighting_env: str = "std",
    seq: int = 1,
    ext: str = "png",
) -> str:
    """
    Build a deterministic, zero-whitespace English filename per `docs/PAINT_SAMPLE_CLASSIFICATION_SPEC.md`:
      `{paint_code}_{paint_name}__sf-{surfacer_and_undercoat}__tc-{topcoat}__{coats}c__sub-{substrate}_{lighting}_{seq}.{ext}`
    """
    code_tok = sanitize_token(paint_code).upper().replace("_", "-")  # "_" is reserved as code/name separator
    name_tok = sanitize_token(paint_name_en).lower()
    sf_tok = sanitize_token(surfacer_code).lower()
    uc_tok = sanitize_token(undercoat_code).lower()
    tc_tok = sanitize_token(topcoat_code).lower()
    sub_tok = sanitize_token(substrate_color).lower().replace("_", "-")
    light_tok = sanitize_token(lighting_env).lower().replace("_", "-")
    ext_tok = sanitize_token(ext).lower().lstrip(".")

    sf_slot = f"sf-{sf_tok}"
    if uc_tok not in ("none", ""):
        sf_slot += f"+uc-{uc_tok}"

    coats_str = f"{int(coats)}c" if float(coats).is_integer() else f"{float(coats):.1f}c"
    filename = f"{code_tok}_{name_tok}__{sf_slot}__tc-{tc_tok}__{coats_str}__sub-{sub_tok}_{light_tok}_{int(seq):02d}.{ext_tok}"
    if " " in filename:
        raise ValueError(f"Generated filename must not contain spaces: {filename}")
    return filename


def parse_sample_filename(filename: str) -> Dict[str, Any]:
    """Parse a canonical sample filename back into structured layer metadata."""
    stem, _, ext = filename.rpartition(".")
    parts = stem.split("__")
    if len(parts) != 5:
        raise ValueError(
            f"Filename '{filename}' does not match 5-slot pattern "
            "`{code}_{name}__sf-{sf}__tc-{tc}__{coats}c__sub-{sub}_{light}_{seq}.{ext}`"
        )
    main_slot, sf_slot, tc_slot, coats_slot, cond_slot = parts
    paint_code, _, paint_name = main_slot.partition("_")

    sf_body = sf_slot[3:] if sf_slot.startswith("sf-") else sf_slot
    if "+uc-" in sf_body:
        surfacer_code, undercoat_code = sf_body.split("+uc-", 1)
    else:
        surfacer_code, undercoat_code = sf_body, "none"

    topcoat_code = tc_slot[3:] if tc_slot.startswith("tc-") else tc_slot
    coats_val = float(coats_slot.rstrip("c"))

    cond_tokens = cond_slot.rsplit("_", 2)
    raw_sub = cond_tokens[0]
    sub_color = raw_sub[4:] if raw_sub.startswith("sub-") else raw_sub
    lighting = cond_tokens[1] if len(cond_tokens) > 1 else "std"
    seq = int(cond_tokens[2]) if len(cond_tokens) > 2 and cond_tokens[2].isdigit() else 1

    return {
        "paint_code": paint_code,
        "paint_name_en": paint_name,
        "surfacer_used": "N" if surfacer_code == "none" else "Y",
        "surfacer_code": surfacer_code,
        "undercoat_code": undercoat_code,
        "topcoat_used": "N" if topcoat_code == "none" else "Y",
        "topcoat_code": topcoat_code,
        "main_coat_count": coats_val,
        "substrate_color": sub_color,
        "lighting_env": lighting,
        "seq": seq,
        "ext": ext,
    }


def resolve_highest_resolution_image_url(url: str, srcset: str = "") -> str:
    """
    Prevent resolution loss from WordPress thumbnail URLs (e.g. `-300x169.jpg`, `-768x432.jpg`)
    by selecting the highest `w` descriptor from `srcset` or stripping the `-<W>x<H>` suffix.
    """
    best_url = url.strip()
    if srcset:
        best_w = -1
        for entry in srcset.split(","):
            parts = entry.strip().split()
            if len(parts) >= 2 and parts[1].endswith("w") and parts[1][:-1].isdigit():
                w_val = int(parts[1][:-1])
                if w_val > best_w:
                    best_w = w_val
                    best_url = parts[0]
    # Strip WordPress auto-generated `-{width}x{height}` suffix before extension if present
    cleaned = re.sub(r"-\d{2,4}x\d{2,4}(?=\.(?:jpg|jpeg|png|webp)$)", "", best_url, flags=re.IGNORECASE)
    return cleaned


def load_image_normalized_srgb(image_bytes: bytes) -> Any:
    """
    Decode an image while preventing three sources of distortion:
      1. EXIF Orientation distortion (applies lossless `ImageOps.exif_transpose`)
      2. Wide-gamut ICC profile stripping (converts Display P3 / AdobeRGB to IEC 61966-2-1 sRGB
         via `ImageCms.profileToProfile` while preserving any alpha channel separately so
         `PyCMSError` never occurs on RGBA/LA/P images).
      3. Stale ICC profile tag retention (removes old `icc_profile` after sRGB conversion so
         downstream PNG encoders do not re-embed a wide-gamut profile over sRGB pixels).
    """
    from PIL import Image, ImageCms, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    icc_bytes = img.info.get("icc_profile")
    if icc_bytes:
        try:
            has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
            alpha_ch = img.convert("RGBA").getchannel("A") if has_alpha else None
            # Feed the profile its native colour space (CMYK / L / RGB). Converting to RGB first applies
            # naive un-profiled math and makes a CMYK or Gray input profile raise PyCMSError.
            if img.mode in ("CMYK", "L", "RGB"):
                src_img = img
            elif img.mode == "LA":
                src_img = img.convert("L")
            else:
                src_img = img.convert("RGB")
            src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            dst_prof = ImageCms.createProfile("sRGB")
            rgb_img = ImageCms.profileToProfile(
                src_img,
                src_prof,
                dst_prof,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,  # measurement: no gamut compression
                outputMode="RGB",
            )
            if rgb_img is not None:
                if alpha_ch is not None:
                    rgb_img.putalpha(alpha_ch)
                rgb_img.info.pop("icc_profile", None)
                img = rgb_img
        except Exception as exc:
            warnings.warn(f"ICC->sRGB conversion failed ({exc}); using embedded pixel values unconverted.")
    return img


# Reference 60-degree gloss meter (GU) calibration targets across multiple paint lines (Section 7.1)
DEFAULT_GLOSS_CALIBRATION_TARGETS: List[Dict[str, Any]] = [
    {"paint_code": "GX2", "paint_name": "Ueno Black (Gloss)", "finish": "gloss", "gu_60deg": 92.0, "gloss_index": 0.095, "alpha_ggx": 0.045},
    {"paint_code": "GX100", "paint_name": "Super Clear III (Gloss)", "finish": "gloss", "gu_60deg": 95.0, "gloss_index": 0.140, "alpha_ggx": 0.055},
    {"paint_code": "H30", "paint_name": "Aqueous Clear Gloss", "finish": "gloss", "gu_60deg": 86.0, "gloss_index": 0.215, "alpha_ggx": 0.075},
    {"paint_code": "B516", "paint_name": "Mr. Super Clear Semi-Gloss", "finish": "semi_gloss", "gu_60deg": 45.0, "gloss_index": 0.248, "alpha_ggx": 0.260},
    {"paint_code": "H110", "paint_name": "Aqueous Semi-Gloss Clear", "finish": "semi_gloss", "gu_60deg": 42.0, "gloss_index": 0.259, "alpha_ggx": 0.280},
    {"paint_code": "HSF01", "paint_name": "Aqueous Surfacer 1000 Gray", "finish": "semi_gloss", "gu_60deg": 28.0, "gloss_index": 0.374, "alpha_ggx": 0.420},
    {"paint_code": "GX114", "paint_name": "Super Smooth Clear Flat", "finish": "matte", "gu_60deg": 4.5, "gloss_index": 0.410, "alpha_ggx": 0.680},
    {"paint_code": "C33", "paint_name": "Flat Black", "finish": "matte", "gu_60deg": 3.0, "gloss_index": 0.506, "alpha_ggx": 0.740},
]

# Empirical class boundary thresholds on uncalibrated `gloss_index` (gloss | semi_gloss | matte)
GLOSS_INDEX_CLASS_BOUNDARIES: Tuple[float, float] = (0.250, 0.275)


def compute_gloss_margin_to_next_class(
    gloss_index: float,
    sibling_gloss_indices: Optional[Sequence[float]] = None,
) -> float:
    """
    Compute the distance from `gloss_index` to the nearest class boundary or adjacent finish sample
    (Section 7.1). Samples with margin < 0.01 (e.g. SMP_0008 / SMP_0009) are flagged `needs_review`.
    """
    boundary_margin = min(abs(float(gloss_index) - b) for b in GLOSS_INDEX_CLASS_BOUNDARIES)
    if sibling_gloss_indices:
        other_vals = [float(v) for v in sibling_gloss_indices if abs(float(v) - float(gloss_index)) > 1e-9]
        if other_vals:
            sib_margin = min(abs(float(gloss_index) - v) for v in other_vals)
            return round(min(boundary_margin, sib_margin), 4)
        if len(sibling_gloss_indices) > 1:
            return 0.0
    return round(boundary_margin, 4)


def fit_isotonic_gloss_calibration(
    targets: Optional[Sequence[Dict[str, Any]]] = None,
) -> Callable[[Sequence[float] | np.ndarray], np.ndarray]:
    """
    Fit a monotone non-decreasing mapping `gloss_index -> alpha_ggx` via exact
    Pool-Adjacent-Violators (PAV) isotonic regression on reference targets measured
    with a 60-degree gloss meter (Section 7.1). Avoids 1.1s cold-import overhead of sklearn.
    """
    ref = list(targets or DEFAULT_GLOSS_CALIBRATION_TARGETS)
    if len(ref) < 3:
        raise ValueError("At least 3 reference targets are required for isotonic gloss calibration.")
    xs = np.array([float(t["gloss_index"]) for t in ref], dtype=np.float64)
    ys = np.array([float(t["alpha_ggx"]) for t in ref], dtype=np.float64)
    order = np.argsort(xs, kind="mergesort")
    xs = xs[order]
    ys = ys[order]

    # Exact Pool-Adjacent-Violators (PAV) monotone non-decreasing solver
    blocks: List[List[float]] = [[float(y), 1.0] for y in ys]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = np.empty_like(ys)
    pos = 0
    for total, cnt in blocks:
        c_int = int(cnt)
        fitted[pos : pos + c_int] = total / cnt
        pos += c_int

    # Deduplicate tied x-coordinates for piecewise linear interpolation with out-of-bounds clipping
    unique_xs, inv = np.unique(xs, return_inverse=True)
    unique_ys = np.zeros_like(unique_xs)
    counts = np.zeros_like(unique_xs)
    np.add.at(unique_ys, inv, fitted)
    np.add.at(counts, inv, 1.0)
    unique_ys /= np.maximum(1.0, counts)

    return lambda q: np.interp(np.asarray(q, dtype=np.float64), unique_xs, unique_ys)


def evaluate_gloss_calibration_lopo_cv(
    targets: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate isotonic `gloss_index -> alpha_ggx` calibration using Leave-One-Paint-Out
    cross-validation (LOPO-CV) per Section 7.1.
    """
    ref = list(targets or DEFAULT_GLOSS_CALIBRATION_TARGETS)
    if len(ref) < 3:
        raise ValueError("At least 3 reference targets are required for LOPO-CV.")
    paint_codes = sorted({str(t["paint_code"]) for t in ref})
    if len(paint_codes) < 3:
        raise ValueError("At least 3 distinct paint_code targets are required for Leave-One-Paint-Out CV.")

    predictions: List[Dict[str, Any]] = []
    errors: List[float] = []
    for held_out_code in paint_codes:
        train_set = [t for t in ref if str(t["paint_code"]) != held_out_code]
        test_set = [t for t in ref if str(t["paint_code"]) == held_out_code]
        predictor = fit_isotonic_gloss_calibration(train_set)
        for item in test_set:
            pred_alpha = float(predictor([float(item["gloss_index"])])[0])
            true_alpha = float(item["alpha_ggx"])
            err = abs(pred_alpha - true_alpha)
            errors.append(err)
            predictions.append({
                "paint_code": held_out_code,
                "finish": item.get("finish", ""),
                "gloss_index": float(item["gloss_index"]),
                "true_alpha_ggx": round(true_alpha, 4),
                "pred_alpha_ggx": round(pred_alpha, 4),
                "abs_error": round(err, 4),
            })

    # Verify monotonicity of full calibration model across ordered gloss_index
    full_model = fit_isotonic_gloss_calibration(ref)
    sorted_gi = sorted(float(t["gloss_index"]) for t in ref)
    fitted_alphas = full_model(sorted_gi)
    monotonic_ok = bool(np.all(np.diff(fitted_alphas) >= -1e-9))

    return {
        "num_targets": len(ref),
        "num_paints": len(paint_codes),
        "lopo_mae": round(float(np.mean(errors)), 4),
        "lopo_max_error": round(float(np.max(errors)), 4),
        "monotonic_preserved": monotonic_ok,
        "predictions": predictions,
    }


def measure_swatch_optics(
    image_bytes: bytes | np.ndarray,
    clean_points_2d: Optional[Sequence[Sequence[int]]] = None,
    points_are_norm1000: bool = False,
    calibrate_alpha: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Geometry-free optical measurement inside the sample's own foreground mask (Sections 7.1, 7.2, 11.3):
      - Accepts either encoded `image_bytes` or a pre-decoded `HxWx4` uint8 RGBA array.
      - Computes equivalent diameter `equiv_diam` of the opaque mask (`alpha >= 250`) and erodes by
        `max(2.0, 0.04 * equiv_diam)` via Euclidean Distance Transform (`distance_transform_edt`),
        eliminating any hardcoded spoon bounding-box percentages.
      - When `clean_points_2d` are provided (either pixel `(x, y)` or Gemini `0..1000` `[y, x]` when
        `points_are_norm1000=True`), intersects `core_mask` with union disks of radius
        `max(6, int(0.08 * equiv_diam))` around those points to sample diffuse color while still
        measuring specular contrast across the full `core_mask`.
      - Reports both `estimated_alpha_ggx` (kept for compatibility) and `gloss_index`, plus
        `alpha_ggx_calibrated` and `gloss_margin_to_next_class`.
      - Raises `ValueError` on undecodable input instead of returning placeholder values.
    """
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    try:
        if isinstance(image_bytes, np.ndarray):
            if image_bytes.ndim != 3 or image_bytes.shape[2] not in (3, 4):
                raise ValueError(f"Expected HxWx3 or HxWx4 array, got shape {image_bytes.shape}")
            if image_bytes.shape[2] == 3:
                alpha_ch = np.full(image_bytes.shape[:2] + (1,), 255.0, dtype=np.float32)
                rgba_8u = np.concatenate([image_bytes.astype(np.float32), alpha_ch], axis=2)
            else:
                rgba_8u = image_bytes.astype(np.float32)
        else:
            img = load_image_normalized_srgb(image_bytes)
            rgba_8u = np.asarray(img.convert("RGBA"), dtype=np.float32)
        h, w, _ = rgba_8u.shape
        alpha_mask = rgba_8u[:, :, 3] >= 250.0
        has_alpha_mask = bool(np.any(rgba_8u[:, :, 3] < 240.0)) and int(np.count_nonzero(alpha_mask)) >= 32

        if has_alpha_mask:
            equiv_diam = float(np.sqrt(4.0 * np.count_nonzero(alpha_mask) / np.pi))
            erode_px = max(2.0, 0.04 * equiv_diam)
            dist_in = distance_transform_edt(alpha_mask)
            core_mask = dist_in >= erode_px
            if int(np.count_nonzero(core_mask)) < 64:
                core_mask = alpha_mask
        else:
            equiv_diam = float(np.sqrt(4.0 * (h * w) / np.pi))
            erode_px = max(2.0, 0.04 * equiv_diam)
            border_mask = np.ones((h, w), dtype=bool)
            if h > 4 and w > 4:
                border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = False
                dist_in = distance_transform_edt(border_mask)
                core_mask = dist_in >= erode_px
                if int(np.count_nonzero(core_mask)) < 64:
                    core_mask = np.ones((h, w), dtype=bool)
            else:
                core_mask = np.ones((h, w), dtype=bool)

        # Optional diffuse anchor disks from `clean_points_2d` (Section 11.3)
        diffuse_mask = core_mask
        if clean_points_2d:
            disk_radius = max(6, int(0.08 * equiv_diam))
            yy, xx = np.ogrid[:h, :w]
            disks = np.zeros((h, w), dtype=bool)
            for pt in clean_points_2d:
                if len(pt) < 2:
                    continue
                if points_are_norm1000:
                    py = max(0, min(h - 1, int(round(float(pt[0]) * h / 1000.0))))
                    px = max(0, min(w - 1, int(round(float(pt[1]) * w / 1000.0))))
                else:
                    px = max(0, min(w - 1, int(round(float(pt[0])))))
                    py = max(0, min(h - 1, int(round(float(pt[1])))))
                disks |= ((xx - px) ** 2 + (yy - py) ** 2) <= (disk_radius ** 2)
            cand_diffuse = core_mask & disks
            if int(np.count_nonzero(cand_diffuse)) >= 16:
                diffuse_mask = cand_diffuse

        ys, xs = np.where(core_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_rgba = rgba_8u[y0:y1, x0:x1]
        fg_mask_2d = core_mask[y0:y1, x0:x1]
        diff_mask_2d = diffuse_mask[y0:y1, x0:x1]

        sub_rgb_u8 = np.clip(sub_rgba[:, :, :3], 0, 255).astype(np.uint8)
        sub_lin_2d = NativePaintCoreBridge.srgb_u8_to_linear(sub_rgb_u8)
        lum_2d = 0.2126 * sub_lin_2d[:, :, 0] + 0.7152 * sub_lin_2d[:, :, 1] + 0.0722 * sub_lin_2d[:, :, 2]
        ach_2d = np.min(sub_lin_2d, axis=2)

        roi_lin = sub_lin_2d[fg_mask_2d]
        max_c = np.max(roi_lin, axis=1)
        lum = lum_2d[fg_mask_2d]
        ach = ach_2d[fg_mask_2d]

        diff_lin = sub_lin_2d[diff_mask_2d]
        diff_lum = lum_2d[diff_mask_2d]
        diff_max_c = np.max(diff_lin, axis=1)

        if diff_lin.shape[0] >= 64:
            p18, p82 = np.percentile(diff_lum, [18, 82])
            valid = (diff_lum >= p18) & (diff_lum <= p82) & (diff_max_c < 0.95)
        else:
            valid = (diff_max_c < 0.95) & (diff_max_c > 0.01)
        chosen_lin = diff_lin[valid] if np.any(valid) else diff_lin

        med_lin = np.median(chosen_lin, axis=0)
        med_srgb = np.clip(np.round(NativePaintCoreBridge.linear_to_srgb(med_lin) * 255.0), 0, 255).astype(int)
        hex_str = f"#{med_srgb[0]:02X}{med_srgb[1]:02X}{med_srgb[2]:02X}"

        # Headroom-normalized & resolution-invariant relative gloss index across full `core_mask`
        if roi_lin.shape[0] >= 64 and sub_rgba.shape[0] >= 16 and sub_rgba.shape[1] >= 16:
            lp995, lp90, lp85, lp78, lp50, lp25 = np.percentile(lum, [99.5, 90, 85, 78, 50, 25])
            ap995, ap85, ap50 = np.percentile(ach, [99.5, 85, 50])

            norm_lum = np.asarray(
                Image.fromarray(lum_2d.astype(np.float32)).resize((90, 145), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            norm_fg = (
                np.asarray(
                    Image.fromarray((fg_mask_2d.astype(np.uint8) * 255)).resize((90, 145), resample=Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
                > 128
            )
            gy, gx = np.gradient(norm_lum)
            grad_fg = np.hypot(gx, gy)[norm_fg] if np.any(norm_fg) else np.hypot(gx, gy).ravel()
            gp99 = float(np.percentile(grad_fg, 99))

            lum_headroom = max(0.22, 1.02 - float(lp50))
            lum_contrast_norm = float(np.clip((lp995 - lp50) / lum_headroom, 0.0, 1.0))
            lum_tail_norm = float(np.clip((lp995 - lp85) / (0.65 * lum_headroom), 0.0, 1.0))

            ach_headroom = max(0.22, 1.02 - float(ap50))
            ach_tail_norm = float(np.clip((ap995 - ap85) / (0.75 * ach_headroom), 0.0, 1.0))

            conc = float(np.clip((lp995 - lp85) / max(0.05, lp995 - lp25), 0.0, 1.0))
            grad_norm = float(np.clip(gp99 / max(0.05, 0.20 * lum_headroom), 0.0, 1.0))

            spec_score = (
                0.35 * lum_tail_norm
                + 0.20 * lum_contrast_norm
                + 0.15 * ach_tail_norm
                + 0.20 * conc
                + 0.10 * grad_norm
            )
            gloss_idx = round(float(np.clip(0.62 - 0.56 * spec_score, 0.04, 0.78)), 3)

            # Extract upper-shoulder pearl/interference tint (p78..p90)
            sh_mask = (lum >= lp78) & (lum <= lp90)
            sh_lin = np.median(roi_lin[sh_mask], axis=0) if np.any(sh_mask) else med_lin
        else:
            p98 = float(np.percentile(max_c, 98))
            p50 = float(np.percentile(max_c, 50))
            spec_contrast = max(0.0, p98 - p50)
            gloss_idx = round(float(np.clip(0.45 - spec_contrast * 0.85, 0.04, 0.78)), 3)
            sh_lin = med_lin

        if calibrate_alpha:
            iso_fn = fit_isotonic_gloss_calibration()
            est_alpha = round(float(iso_fn([gloss_idx])[0]), 3)
        else:
            est_alpha = gloss_idx

        margin = compute_gloss_margin_to_next_class(gloss_idx)

        return hex_str, {
            "r_inf": [round(float(med_lin[0]), 4), round(float(med_lin[1]), 4), round(float(med_lin[2]), 4)],
            "pearl_shoulder_rgb": [round(float(sh_lin[0]), 4), round(float(sh_lin[1]), 4), round(float(sh_lin[2]), 4)],
            "estimated_alpha_ggx": est_alpha,
            "gloss_index": gloss_idx,
            "alpha_ggx_calibrated": bool(calibrate_alpha),
            "gloss_margin_to_next_class": margin,
        }
    except Exception as exc:
        # Never write fabricated "#808080" measurements into the master index.
        raise ValueError(f"measure_swatch_optics failed: {exc}") from exc


def isolate_spoon_rgba_lossless(crop_rgb_img: Any, rembg_session: Any = None, model_name: str = "u2net") -> bytes:
    """
    Remove background from a single sample crop with ZERO interior RGB distortion:
      1. Extracts ONLY the alpha mask from `rembg` (`only_mask=True`), discarding `rembg`'s
         premultiplied RGB output (which otherwise darkens specular highlights by up to 49/255).
      2. Keeps the largest connected foreground component and closes/fills interior holes so
         bright specular glints inside the sample have `alpha = 255` (100% opaque).
      3. Preserves the untouched, bit-exact original `[R, G, B]` pixels inside `core_opaque`
         (`max_rgb_diff == 0`), while applying Euclidean Distance Transform edge color
         decontamination (`distance_transform_edt`) on the 2px anti-aliased boundary rim
         to eliminate background cutting-mat color halos.
      4. Crops on integer pixel boundaries (no resampling/interpolation) and encodes as
         lossless 32-bit RGBA PNG.
    """
    from PIL import Image
    from scipy import ndimage
    import rembg

    if rembg_session is None:
        rembg_session = rembg.new_session(model_name)

    orig_rgb = np.asarray(crop_rgb_img.convert("RGB"), dtype=np.uint8)
    mask_img = rembg.remove(crop_rgb_img, session=rembg_session, only_mask=True)
    raw_alpha = np.asarray(mask_img, dtype=np.uint8)

    fg = raw_alpha > 130
    if np.any(fg):
        labeled, num_features = ndimage.label(fg)
        if num_features > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            keep = labeled == int(np.argmax(counts))
        else:
            keep = fg

        solid_spoon = ndimage.binary_fill_holes(keep)
        solid_spoon = ndimage.binary_closing(solid_spoon, structure=np.ones((5, 5)))
        solid_spoon = ndimage.binary_fill_holes(solid_spoon)

        # Erode by 2px to define the 100% uncontaminated interior region
        core_opaque = ndimage.binary_erosion(solid_spoon, structure=np.ones((5, 5)))
        if not np.any(core_opaque):
            core_opaque = solid_spoon

        # Decontaminate the 2px anti-aliased boundary rim using nearest interior RGB
        _, nearest_idx = ndimage.distance_transform_edt(~core_opaque, return_indices=True)
        rim_rgb = orig_rgb[nearest_idx[0], nearest_idx[1]]
        final_rgb = np.where(core_opaque[:, :, None], orig_rgb, rim_rgb)

        clean_alpha = np.where(core_opaque, 255, np.where(solid_spoon, raw_alpha, 0)).astype(np.uint8)
        rgba = np.dstack([final_rgb, clean_alpha])

        ys, xs = np.where(clean_alpha > 16)
        pad_x = max(6, rgba.shape[1] // 25)
        pad_top = max(6, rgba.shape[0] // 32)
        ymin = max(0, int(np.min(ys)) - pad_top)
        ymax = min(rgba.shape[0], int(np.max(ys)) + 1)
        xmin = max(0, int(np.min(xs)) - pad_x)
        xmax = min(rgba.shape[1], int(np.max(xs)) + pad_x + 1)
        rgba = rgba[ymin:ymax, xmin:xmax]
    else:
        rgba = np.dstack([orig_rgb, raw_alpha])

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


_INITIALIZED_ROOT_DIRS: set[Path] = set()


def initialize_sample_directory_tree(
    root_dir: Optional[Path] = None,
    *,
    scaffold_all_brands: bool = True,
) -> Tuple[Path, Path]:
    """
    Create the complete `sample/paint_samples/{brand}/{solvent_type}/{paint_role}/{product_line}/`
    directory tree across all 18 canonical brands and initialize `paint_sample_master_index.csv`
    with UTF-8 BOM for Excel. Caches initialized root paths in-process to avoid redundant 105-folder walks.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[3] / "sample" / "paint_samples"
    root_dir = Path(root_dir).resolve()
    csv_path = root_dir / "paint_sample_master_index.csv"
    xlsx_path = root_dir / "paint_sample_master_index.xlsx"

    if root_dir in _INITIALIZED_ROOT_DIRS and csv_path.exists() and xlsx_path.exists():
        return root_dir, csv_path

    root_dir.mkdir(parents=True, exist_ok=True)
    if scaffold_all_brands:
        for brand, solvents in STANDARD_SAMPLE_DIRECTORY_TREE.items():
            for solvent, roles in solvents.items():
                for role, lines in roles.items():
                    for line in lines:
                        folder = root_dir / brand / solvent / role / line
                        folder.mkdir(parents=True, exist_ok=True)
                        gitkeep = folder / ".gitkeep"
                        if not gitkeep.exists():
                            gitkeep.write_text("", encoding="utf-8")
        _INITIALIZED_ROOT_DIRS.add(root_dir)

    if not csv_path.exists():
        _atomic_write_csv(csv_path, [])

    if not xlsx_path.exists():
        sync_csv_to_xlsx(csv_path, xlsx_path)

    return root_dir, csv_path


def _acquire_csv_lock(csv_path: Path):
    """Return a cross-process file lock for `paint_sample_master_index.csv` (Section 7.5)."""
    from filelock import FileLock

    lock_path = csv_path.with_suffix(".csv.lock")
    return FileLock(str(lock_path), timeout=30)


def _atomic_write_csv(csv_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Atomically write `paint_sample_master_index.csv` via temp file + `os.replace` (Section 7.5)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".master_idx_", suffix=".csv.tmp", dir=str(csv_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MASTER_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, csv_path)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise


def _col_letter(col_idx_1based: int) -> str:
    res = ""
    n = col_idx_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        res = chr(65 + rem) + res
    return res


def _build_worksheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        lines.append(f'    <row r="{r_idx}">')
        for c_idx, val in enumerate(row, start=1):
            cell_ref = f"{_col_letter(c_idx)}{r_idx}"
            text = xml_escape("" if val is None else str(val))
            lines.append(f'      <c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        lines.append("    </row>")
    lines.append("  </sheetData>")
    lines.append("</worksheet>")
    return "\n".join(lines)


def sync_csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> Path:
    """
    Generate a standards-compliant 3-sheet Microsoft Excel `.xlsx` workbook
    (`Sample_Master_Index`, `Solvent_Compatibility`, and `Overcoating_Compatibility`)
    directly from `paint_sample_master_index.csv`.
    """
    sheet1_rows: List[List[str]] = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                sheet1_rows.append(list(row))
    if not sheet1_rows:
        sheet1_rows.append(list(MASTER_CSV_COLUMNS))

    sheet2_rows: List[List[str]] = [
        [
            "Solvent / Thinner (용제)",
            "Emulsion Water Acrylic (바예호/아크리존 48h+)",
            "Alcohol Acrylic (타미야 X/XF, 군제 H)",
            "Gundam Marker / Namepen (건담마커/네임펜)",
            "Enamel Paint (에나멜)",
            "Lacquer Paint (락카 12h+)",
            "2K Urethane (우레탄 완전경화)",
        ],
        ["Warm Water (온수)", "△ (건조전 ◎)", "△ (건조전 ○)", "X", "X", "X", "X"],
        ["Ethyl Alcohol (에틸알콜/소독용)", "△", "◎", "◎", "X", "X", "X"],
        ["Isopropyl Alcohol (IPA 99%)", "○", "◎", "◎", "△", "△", "X"],
        ["Methyl Alcohol (메틸알콜)", "△", "◎", "◎", "◎", "◎", "X"],
        ["Acrylic Thinner (아크릴신너 X-20A)", "○", "◎", "◎", "X", "X", "X"],
        ["Enamel Thinner (에나멜신너/석유계)", "X", "X", "X (흘려넣기타입은 ◎)", "◎", "X", "X"],
        ["Lighter Fluid (라이터기름/지포)", "X", "X", "X (흘려넣기타입은 ◎)", "◎", "X", "X"],
        ["Industrial Thinner (공업용신너)", "△~○", "◎", "◎", "◎", "◎ (프라부식 주의)", "△"],
        ["Hobby Lacquer Thinner (모형용 락카신너)", "X (침적/문지름 시 △)", "◎", "◎", "◎", "◎", "X"],
        ["Leveling Thinner (레벨링신너)", "X (침적/문지름 시 △)", "◎", "◎", "◎", "◎", "X"],
        ["Acetone (아세톤)", "◎", "◎", "◎", "◎", "◎ (프라용해)", "△"],
    ]

    sheet3_rows: List[List[str]] = [
        [
            "Bottom Cured Layer (하도/중도) \\ Top Layer (상도)",
            "2K Urethane (우레탄)",
            "Lacquer (락카)",
            "Alcohol Acrylic (알콜계 아크릴)",
            "Enamel / Panel Wash (에나멜/먹선)",
            "Emulsion Water Acrylic (수성 아크릴)",
            "Gundam Marker (건담마커)",
        ],
        ["Bare Plastic (생 플라스틱 PS/ABS)", "△ (서페이서 권장)", "○ (ABS는 미스트코트)", "◎", "X~△ (크랙 위험!)", "△ (프라이머 권장)", "○ (흘려넣기 ABS 주의)"],
        ["2K Urethane (우레탄 완전경화)", "◎ (#1500 사포질 후)", "○", "○", "◎", "○", "○"],
        ["Lacquer (락카 완전건조 12h+)", "◎", "◎", "◎", "◎ (먹선/워싱 최적)", "◎", "△ (붓자국 녹음 주의)"],
        ["Alcohol Acrylic (알콜계 아크릴 48h+)", "△ (미스트코트 필수)", "△ (미스트코트 필수)", "◎", "◎", "◎", "X"],
        ["Emulsion Water Acrylic (에멀전 수성 48h+)", "△ (미스트코트 필수)", "△ (미스트코트 필수)", "○", "◎ (48h 완전건조 후)", "◎", "△"],
        ["Enamel (에나멜 완전건조)", "X (주름/크랙 발생)", "X (주름/크랙 발생)", "△ (에어브러시 박막만)", "△", "◎ (수성 바니시 안전)", "X"],
        ["Gundam Marker (건담마커 건조 후)", "X (번짐)", "X (번짐, 극박막 △)", "X", "◎", "◎ (수성 마감 권장)", "X (재도포 시 벗겨짐)"],
        ["Mirror Chrome (SM206 / 테라크롬)", "X (거울광 소실)", "X (거울광 소실)", "△ (H30/아쿠아글로스 박막)", "○", "○ (수성 유광 약 90% 유지)", "X"],
    ]

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

    root_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sample_Master_Index" sheetId="1" r:id="rId1"/>
    <sheet name="Solvent_Compatibility" sheetId="2" r:id="rId2"/>
    <sheet name="Overcoating_Compatibility" sheetId="3" r:id="rId3"/>
  </sheets>
</workbook>"""

    workbook_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
</Relationships>"""

    fd, tmp_xlsx = tempfile.mkstemp(prefix=".master_idx_", suffix=".xlsx.tmp", dir=str(xlsx_path.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml)
            zf.writestr("_rels/.rels", root_rels_xml)
            zf.writestr("xl/workbook.xml", workbook_xml)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            zf.writestr("xl/worksheets/sheet1.xml", _build_worksheet_xml(sheet1_rows))
            zf.writestr("xl/worksheets/sheet2.xml", _build_worksheet_xml(sheet2_rows))
            zf.writestr("xl/worksheets/sheet3.xml", _build_worksheet_xml(sheet3_rows))
        os.replace(tmp_xlsx, xlsx_path)
    except Exception:
        if os.path.exists(tmp_xlsx):
            try:
                os.unlink(tmp_xlsx)
            except OSError:
                pass
        raise

    return xlsx_path


def _prepare_single_ingest_item(
    item: Dict[str, Any],
    root_dir: Path,
) -> Dict[str, Any]:
    """Validate and write a single sample image to its 4-level directory, returning prepared row fields."""
    from PIL import Image

    image_bytes: bytes = item["image_bytes"]
    brand_id: str = item.get("brand_id", "auto")
    solvent_type: str = item.get("solvent_type", "auto")
    paint_role: str = item.get("paint_role", "auto")
    product_line: str = item.get("product_line", "auto")
    paint_code: str = item.get("paint_code", "UNKNOWN")
    paint_name_en: str = item.get("paint_name_en", "Sample Paint")
    paint_name_ko: str = item.get("paint_name_ko", "샘플 도료")
    surfacer_code: str = item.get("surfacer_code", "none")
    surfacer_grit: str = item.get("surfacer_grit", "none")
    undercoat_code: str = item.get("undercoat_code", "none")
    topcoat_code: str = item.get("topcoat_code", "none")
    final_finish: str = item.get("final_finish", "auto")
    coats: float = float(item.get("coats", 2.0))
    thinner_and_ratio: str = item.get("thinner_and_ratio", "standard")
    substrate_color: str = item.get("substrate_color", "white")
    lighting_env: str = item.get("lighting_env", "std")
    source_type: str = item.get("source_type", "direct_photo")
    source_url: str = item.get("source_url", "local_upload")
    image_url: str = item.get("image_url", "")
    notes: str = item.get("notes", "")
    seq: int = int(item.get("seq", 1))
    ext: str = item.get("ext", "png")
    overwrite: bool = bool(item.get("overwrite", False))
    clean_points_2d = item.get("clean_points_2d")
    points_are_norm1000 = bool(item.get("points_are_norm1000", False))

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            src_format = (probe.format or "").lower()
            probe.verify()
    except Exception as exc:
        raise ValueError(f"ingest_sample_image: image_bytes is not a decodable image ({exc})") from exc

    if sanitize_token(ext).lower().lstrip(".") == "png" and src_format != "png":
        norm = load_image_normalized_srgb(image_bytes)
        norm = norm.convert("RGBA" if "A" in norm.getbands() else "RGB")
        png_buf = io.BytesIO()
        norm.save(png_buf, format="PNG")
        image_bytes = png_buf.getvalue()

    auto_meta = classify_paint_sample(
        paint_code=paint_code,
        paint_name_en=paint_name_en,
        paint_name_ko=paint_name_ko,
        brand_hint="" if brand_id in ("auto", "") else brand_id,
        solvent_hint="" if solvent_type in ("auto", "") else solvent_type,
        role_hint="" if paint_role in ("auto", "") else paint_role,
        product_line_hint="" if product_line in ("auto", "") else product_line,
        sheet_text=str(item.get("sheet_text", "")),
    )

    brand_tok = auto_meta["brand_id"] if brand_id in ("auto", "") else sanitize_token(brand_id).lower()
    solvent_tok = auto_meta["solvent_type"] if solvent_type in ("auto", "") else sanitize_token(solvent_type).lower()
    role_tok = auto_meta["paint_role"] if paint_role in ("auto", "") else sanitize_token(paint_role).lower()
    line_tok = auto_meta["product_line"] if product_line in ("auto", "") else sanitize_token(product_line).lower()

    if brand_tok not in VALID_BRAND_IDS:
        raise ValueError(f"Invalid brand_id '{brand_tok}'. Allowed: {VALID_BRAND_IDS}")
    if solvent_tok not in VALID_SOLVENT_TYPES:
        raise ValueError(f"Invalid solvent_type '{solvent_tok}'. Allowed: {VALID_SOLVENT_TYPES}")
    if role_tok not in VALID_PAINT_ROLES:
        raise ValueError(f"Invalid paint_role '{role_tok}'. Allowed: {VALID_PAINT_ROLES}")

    resolved_grit = auto_meta["surfacer_grit"] if str(surfacer_grit) in ("auto", "") else str(surfacer_grit)
    resolved_finish = auto_meta["default_finish"] if final_finish in ("auto", "") else final_finish

    filename = build_sample_filename(
        paint_code=paint_code,
        paint_name_en=paint_name_en,
        surfacer_code=surfacer_code,
        undercoat_code=undercoat_code,
        topcoat_code=topcoat_code,
        coats=coats,
        substrate_color=substrate_color,
        lighting_env=lighting_env,
        seq=seq,
        ext=ext,
    )

    target_dir = root_dir / brand_tok / solvent_tok / role_tok / line_tok
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    while not overwrite and target_path.exists() and target_path.read_bytes() != image_bytes:
        seq += 1
        filename = build_sample_filename(
            paint_code=paint_code,
            paint_name_en=paint_name_en,
            surfacer_code=surfacer_code,
            undercoat_code=undercoat_code,
            topcoat_code=topcoat_code,
            coats=coats,
            substrate_color=substrate_color,
            lighting_env=lighting_env,
            seq=seq,
            ext=ext,
        )
        target_path = target_dir / filename
    target_path.write_bytes(image_bytes)

    auto_fields = {
        k
        for k, v in (("brand_id", brand_id), ("solvent_type", solvent_type), ("paint_role", paint_role), ("product_line", product_line))
        if v in ("auto", "")
    }
    review = [r for r in auto_meta["review_reasons"] if r.split(":", 1)[0] in auto_fields]
    if review:
        notes = (notes + " " if notes else "") + "[AUTO-CLASSIFY REVIEW: " + "; ".join(review) + "]"
    if image_url and image_url not in notes:
        notes = (notes + " " if notes else "") + f"[image_url: {image_url}]"

    rel_path = f"{brand_tok}/{solvent_tok}/{role_tok}/{line_tok}/{filename}"
    measured_hex, optical_params = measure_swatch_optics(
        image_bytes,
        clean_points_2d=clean_points_2d,
        points_are_norm1000=points_are_norm1000,
    )

    return {
        "file_rel_path": rel_path,
        "file_name": filename,
        "source_type": source_type,
        "source_url": source_url,
        "brand_id": brand_tok,
        "solvent_type": solvent_tok,
        "paint_role": role_tok,
        "product_line": line_tok,
        "paint_code": paint_code,
        "paint_name_en": paint_name_en,
        "paint_name_ko": paint_name_ko,
        "substrate_color": sanitize_token(substrate_color).lower(),
        "surfacer_used": "N" if sanitize_token(surfacer_code).lower() == "none" else "Y",
        "surfacer_code": sanitize_token(surfacer_code).lower(),
        "surfacer_grit": resolved_grit,
        "undercoat_code": sanitize_token(undercoat_code).lower(),
        "main_coat_count": f"{float(coats):.1f}",
        "thinner_and_ratio": thinner_and_ratio,
        "topcoat_used": "N" if sanitize_token(topcoat_code).lower() == "none" else "Y",
        "topcoat_code": sanitize_token(topcoat_code).lower(),
        "final_finish": resolved_finish,
        "lighting_env": sanitize_token(lighting_env).lower(),
        "measured_srgb_hex": measured_hex,
        "calibrated_optical_params": json.dumps(optical_params),
        "notes": notes,
    }


def ingest_samples_batch(
    items: Sequence[Dict[str, Any]],
    root_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Batch ingestion of multiple paint samples (Section 7.5):
      1. Initializes the directory tree ONCE.
      2. Acquires a cross-process file lock (`paint_sample_master_index.csv.lock`).
      3. Loads `paint_sample_master_index.csv` ONCE.
      4. Appends/updates all rows with monotonic `SMP_XXXX` IDs.
      5. Writes the CSV atomically (temp file + `os.replace`) and syncs `.xlsx` ONCE.
    """
    root_dir, csv_path = initialize_sample_directory_tree(root_dir, scaffold_all_brands=False)
    if not items:
        return []

    prepared_rows = [_prepare_single_ingest_item(item, root_dir) for item in items]

    with _acquire_csv_lock(csv_path):
        existing_rows: List[Dict[str, str]] = []
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                existing_rows = list(csv.DictReader(f))

        max_id = max(
            (
                int(r["sample_id"].split("_")[1])
                for r in existing_rows
                if r.get("sample_id", "").startswith("SMP_") and r["sample_id"].split("_")[1].isdigit()
            ),
            default=0,
        )

        result_rows: List[Dict[str, Any]] = []
        for prep in prepared_rows:
            rel_path = prep["file_rel_path"]
            matched_row = next((r for r in existing_rows if r.get("file_rel_path") == rel_path), None)
            if matched_row is not None:
                sample_id = matched_row["sample_id"]
            else:
                max_id += 1
                sample_id = f"SMP_{max_id:04d}"

            new_row = {"sample_id": sample_id, **prep}
            if matched_row is not None:
                idx = existing_rows.index(matched_row)
                existing_rows[idx] = new_row
            else:
                existing_rows.append(new_row)
            result_rows.append(new_row)

        _atomic_write_csv(csv_path, existing_rows)
        sync_csv_to_xlsx(csv_path, root_dir / "paint_sample_master_index.xlsx")

    return result_rows


def ingest_sample_image(
    image_bytes: bytes,
    brand_id: str = "auto",
    solvent_type: str = "auto",
    paint_role: str = "auto",
    product_line: str = "auto",
    paint_code: str = "UNKNOWN",
    paint_name_en: str = "Sample Paint",
    paint_name_ko: str = "샘플 도료",
    surfacer_code: str = "none",
    surfacer_grit: str = "none",
    undercoat_code: str = "none",
    topcoat_code: str = "none",
    final_finish: str = "auto",
    coats: float = 2.0,
    thinner_and_ratio: str = "standard",
    substrate_color: str = "white",
    lighting_env: str = "std",
    source_type: str = "direct_photo",
    source_url: str = "local_upload",
    notes: str = "",
    seq: int = 1,
    ext: str = "png",
    root_dir: Optional[Path] = None,
    overwrite: bool = False,
    image_url: str = "",
    clean_points_2d: Optional[Sequence[Sequence[int]]] = None,
    points_are_norm1000: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end automated ingestion for a single paint sample image, delegating to the
    locked atomic writer in `ingest_samples_batch`.
    """
    rows = ingest_samples_batch(
        [
            {
                "image_bytes": image_bytes,
                "brand_id": brand_id,
                "solvent_type": solvent_type,
                "paint_role": paint_role,
                "product_line": product_line,
                "paint_code": paint_code,
                "paint_name_en": paint_name_en,
                "paint_name_ko": paint_name_ko,
                "surfacer_code": surfacer_code,
                "surfacer_grit": surfacer_grit,
                "undercoat_code": undercoat_code,
                "topcoat_code": topcoat_code,
                "final_finish": final_finish,
                "coats": coats,
                "thinner_and_ratio": thinner_and_ratio,
                "substrate_color": substrate_color,
                "lighting_env": lighting_env,
                "source_type": source_type,
                "source_url": source_url,
                "image_url": image_url,
                "notes": notes,
                "seq": seq,
                "ext": ext,
                "overwrite": overwrite,
                "clean_points_2d": clean_points_2d,
                "points_are_norm1000": points_are_norm1000,
            }
        ],
        root_dir=root_dir,
    )
    return rows[0]


def recalibrate_master_index_optics(root_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Re-evaluate `measured_srgb_hex` and `calibrated_optical_params` across all existing sample PNGs
    in `paint_sample_master_index.csv` using the geometry-free mask optical analyzer and sync `.xlsx`.
    """
    root_dir, csv_path = initialize_sample_directory_tree(root_dir)
    if not csv_path.exists():
        return []

    with _acquire_csv_lock(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        parsed_opts: Dict[int, Dict[str, Any]] = {}
        group_gloss: Dict[Tuple[str, str], List[float]] = {}

        for idx, row in enumerate(rows):
            img_path = root_dir / row["file_rel_path"]
            if img_path.exists():
                try:
                    hex_str, opt_params = measure_swatch_optics(img_path.read_bytes())
                except ValueError as exc:
                    warnings.warn(f"Skipping {row.get('sample_id')}: {exc}")
                    continue
                row["measured_srgb_hex"] = hex_str
                parsed_opts[idx] = opt_params
                if row.get("surfacer_code", "none") != "none":
                    gkey = (row.get("paint_code", ""), row.get("surfacer_code", ""))
                    group_gloss.setdefault(gkey, []).append(float(opt_params["gloss_index"]))

        for idx, opt_params in parsed_opts.items():
            row = rows[idx]
            gkey = (row.get("paint_code", ""), row.get("surfacer_code", ""))
            siblings = group_gloss.get(gkey)
            opt_params["gloss_margin_to_next_class"] = compute_gloss_margin_to_next_class(
                float(opt_params["gloss_index"]),
                sibling_gloss_indices=siblings,
            )
            row["calibrated_optical_params"] = json.dumps(opt_params)

        _atomic_write_csv(csv_path, rows)
        sync_csv_to_xlsx(csv_path, root_dir / "paint_sample_master_index.xlsx")
    return rows


if __name__ == "__main__":
    root, csv_file = initialize_sample_directory_tree()
    if "--recalibrate-optics" in sys.argv:
        rows = recalibrate_master_index_optics(root)
        print(f"Recalibrated optical parameters for {len(rows)} samples in CSV & Excel.")
    else:
        print(f"Initialized sample directory tree at: {root}")
        print(f"Initialized master index CSV at: {csv_file}")
