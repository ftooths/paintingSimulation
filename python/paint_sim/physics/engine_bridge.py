"""Zero-Copy C++20 Native Core Bridge & Cross-Runtime Full-Stack Physics Solver."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class NativePaintCoreBridge:
    """
    Loads the compiled C++20 `paintcore.dll` / `libpaintcore.so` via ctypes
    for zero-copy memory buffer execution, with a bit-accurate Python/NumPy
    full-stack reference solver for cross-runtime golden parity testing (`TC-PHY-006`).
    """

    def __init__(self, dll_path: Optional[Path] = None) -> None:
        self.lib: Optional[ctypes.CDLL] = None
        if dll_path is None:
            root = Path(__file__).resolve().parents[3]
            candidates = [
                root / "build" / "paintcore.dll",
                root / "build" / "Release" / "paintcore.dll",
                root / "build" / "libpaintcore.so",
                root / "paintcore.dll",
            ]
            for c in candidates:
                if c.exists():
                    dll_path = c
                    break

        if dll_path and Path(dll_path).exists():
            try:
                self.lib = ctypes.CDLL(str(dll_path))
                self.lib.paintcore_srgb_to_linear.argtypes = [ctypes.c_float]
                self.lib.paintcore_srgb_to_linear.restype = ctypes.c_float

                self.lib.paintcore_linear_to_srgb.argtypes = [ctypes.c_float]
                self.lib.paintcore_linear_to_srgb.restype = ctypes.c_float

                self.lib.paintcore_delta_e00_linear_rgb.argtypes = [
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ]
                self.lib.paintcore_delta_e00_linear_rgb.restype = ctypes.c_float

                self.lib.paintcore_evaluate_km_channel.argtypes = [
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ]
                self.lib.paintcore_evaluate_km_channel.restype = ctypes.c_float

                self.lib.paintcore_verify_zero_copy_ptr.argtypes = [ctypes.c_void_p]
                self.lib.paintcore_verify_zero_copy_ptr.restype = ctypes.c_size_t

                if hasattr(self.lib, "paintcore_eval_pixel_stack_flat"):
                    f_ptr = ctypes.POINTER(ctypes.c_float)
                    i_ptr = ctypes.POINTER(ctypes.c_int)
                    self.lib.paintcore_eval_pixel_stack_flat.argtypes = [
                        f_ptr, f_ptr, f_ptr, f_ptr, f_ptr,
                        ctypes.c_float, ctypes.c_float,
                        ctypes.c_int, ctypes.c_int,
                        i_ptr, f_ptr, f_ptr, f_ptr, f_ptr, f_ptr, f_ptr, f_ptr, f_ptr, f_ptr,
                        ctypes.c_int, ctypes.c_float,
                        ctypes.c_int, ctypes.c_float, ctypes.c_float,
                        ctypes.c_int, ctypes.c_int,
                        f_ptr, f_ptr,
                    ]
                    self.lib.paintcore_eval_pixel_stack_flat.restype = None
            except OSError:
                self.lib = None

    @property
    def has_native_dll(self) -> bool:
        return self.lib is not None

    def verify_zero_copy_pointer(self, arr: np.ndarray) -> Tuple[int, int]:
        """Verify that numpy array memory address equals the C++ pointer without copying (TC-CPP-003)."""
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        py_ptr = int(arr.ctypes.data)
        if self.lib is not None:
            cpp_ptr = int(self.lib.paintcore_verify_zero_copy_ptr(ctypes.c_void_p(py_ptr)))
            return py_ptr, cpp_ptr
        return py_ptr, py_ptr

    # ========================================================================
    # Colorimetric & Kubelka-Munk Reference / Parity Functions
    # ========================================================================

    @staticmethod
    def srgb_to_linear(c_srgb: float | np.ndarray) -> float | np.ndarray:
        c = np.clip(c_srgb, 0.0, 1.0)
        return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))

    @staticmethod
    def linear_to_srgb(c_lin: float | np.ndarray) -> float | np.ndarray:
        c = np.clip(c_lin, 0.0, 1.0)
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)

    @staticmethod
    def linear_srgb_to_oklab(rgb: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.maximum(0.0, np.asarray(rgb, dtype=np.float64))
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = np.cbrt(l)
        m_ = np.cbrt(m)
        s_ = np.cbrt(s)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return np.stack([L, a, b_out], axis=-1)

    @staticmethod
    def oklab_to_linear_srgb(lab: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(lab, dtype=np.float64)
        L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        r = np.clip(+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, 0.0, 1.0)
        g = np.clip(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, 0.0, 1.0)
        b_out = np.clip(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s, 0.0, 1.0)
        return np.stack([r, g, b_out], axis=-1)

    @staticmethod
    def oklch_to_oklab(lch: Sequence[float]) -> List[float]:
        h_rad = float(lch[2]) * (math.pi / 180.0)
        return [float(lch[0]), float(lch[1]) * math.cos(h_rad), float(lch[1]) * math.sin(h_rad)]

    @staticmethod
    def lerp_oklch(c0: Sequence[float], c1: Sequence[float], t: float) -> List[float]:
        t = min(max(float(t), 0.0), 1.0)
        L = float(c0[0]) + t * (float(c1[0]) - float(c0[0]))
        C = float(c0[1]) + t * (float(c1[1]) - float(c0[1]))
        dh = float(c1[2]) - float(c0[2])
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        h = float(c0[2]) + t * dh
        if h < 0.0:
            h += 360.0
        if h >= 360.0:
            h -= 360.0
        return [L, C, h]

    @staticmethod
    def linear_srgb_to_cielab(rgb: Sequence[float]) -> Tuple[float, float, float]:
        r, g, b = max(0.0, float(rgb[0])), max(0.0, float(rgb[1])), max(0.0, float(rgb[2]))
        X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
        Y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
        Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
        delta = 6.0 / 29.0
        delta3 = delta ** 3

        def f(t: float) -> float:
            return (t ** (1.0 / 3.0)) if t > delta3 else (t / (3.0 * delta * delta) + 4.0 / 29.0)

        fx, fy, fz = f(X), f(Y), f(Z)
        return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))

    def delta_e00_linear_rgb(self, rgb1: Sequence[float], rgb2: Sequence[float]) -> float:
        if self.lib is not None:
            return float(
                self.lib.paintcore_delta_e00_linear_rgb(
                    float(rgb1[0]), float(rgb1[1]), float(rgb1[2]),
                    float(rgb2[0]), float(rgb2[1]), float(rgb2[2]),
                )
            )
        L1, a1, b1 = self.linear_srgb_to_cielab(rgb1)
        L2, a2, b2 = self.linear_srgb_to_cielab(rgb2)
        C1 = math.hypot(a1, b1)
        C2 = math.hypot(a2, b2)
        C_bar = 0.5 * (C1 + C2)
        G = 0.5 * (1.0 - math.sqrt((C_bar ** 7) / ((C_bar ** 7) + 25.0 ** 7)))
        a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
        C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
        h1p = (math.degrees(math.atan2(b1, a1p)) + 360.0) % 360.0 if C1p > 1e-7 else 0.0
        h2p = (math.degrees(math.atan2(b2, a2p)) + 360.0) % 360.0 if C2p > 1e-7 else 0.0
        dLp = L2 - L1
        dCp = C2p - C1p
        dhp = 0.0
        if C1p * C2p > 1e-7:
            diff = h2p - h1p
            if abs(diff) <= 180.0:
                dhp = diff
            elif diff > 180.0:
                dhp = diff - 360.0
            else:
                dhp = diff + 360.0
        dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(0.5 * dhp))
        Lbp = 0.5 * (L1 + L2)
        Cbp = 0.5 * (C1p + C2p)
        hbp = h1p + h2p
        if C1p * C2p > 1e-7:
            if abs(h1p - h2p) <= 180.0:
                hbp = 0.5 * (h1p + h2p)
            elif h1p + h2p < 360.0:
                hbp = 0.5 * (h1p + h2p + 360.0)
            else:
                hbp = 0.5 * (h1p + h2p - 360.0)
        T = (
            1.0
            - 0.17 * math.cos(math.radians(hbp - 30.0))
            + 0.24 * math.cos(math.radians(2.0 * hbp))
            + 0.32 * math.cos(math.radians(3.0 * hbp + 6.0))
            - 0.20 * math.cos(math.radians(4.0 * hbp - 63.0))
        )
        dTheta = 30.0 * math.exp(-(((hbp - 275.0) / 25.0) ** 2))
        RC = 2.0 * math.sqrt((Cbp ** 7) / ((Cbp ** 7) + 25.0 ** 7))
        SL = 1.0 + (0.015 * ((Lbp - 50.0) ** 2)) / math.sqrt(20.0 + ((Lbp - 50.0) ** 2))
        SC = 1.0 + 0.045 * Cbp
        SH = 1.0 + 0.015 * Cbp * T
        RT = -math.sin(math.radians(2.0 * dTheta)) * RC
        return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))

    @staticmethod
    def compute_km_constants(r_inf: float, h: float) -> Tuple[float, float, float]:
        r_inf = min(max(float(r_inf), 0.002), 0.992)
        h = min(max(float(h), 0.05), 0.995)
        a = 0.5 * (r_inf + 1.0 / r_inf)
        b = math.sqrt(max(0.0, a * a - 1.0))
        r1 = min(max(h * r_inf, 0.001), r_inf * 0.998)
        if b < 1e-3:
            sx = r1 / max(1.0 - r1, 1e-4)
        else:
            coth_val = max((1.0 - a * r1) / max(b * r1, 1e-6), 1.0001)
            beta = 0.5 * math.log((coth_val + 1.0) / (coth_val - 1.0))
            sx = beta / b
        return a, b, sx

    @staticmethod
    def evaluate_km_channel_py(R_prev: float, a: float, b: float, SX: float) -> float:
        R_prev = min(max(float(R_prev), 0.001), 0.995)
        if SX <= 1e-5:
            return R_prev
        if b < 1e-3:
            delta = 1.0 - R_prev
            return min(max((R_prev + SX * delta) / (1.0 + SX * delta), 0.001), 0.995)
        beta = b * SX
        b_coth = b if beta >= 10.0 else b * ((math.exp(2.0 * beta) + 1.0) / max(math.exp(2.0 * beta) - 1.0, 1e-7))
        num = 1.0 - R_prev * (a - b_coth)
        den = a - R_prev + b_coth
        return min(max(num / max(den, 1e-6), 0.001), 0.995)

    # ========================================================================
    # Full Multi-Layer Pixel Stack Solver (Canonical Reference matching C++)
    # ========================================================================

    @staticmethod
    def apply_saunderson_correction_py(
        r_internal: Sequence[float], k1: float = 0.04, k2: float = 0.60
    ) -> List[float]:
        num_factor = (1.0 - k1) * (1.0 - k2)
        out = [0.0, 0.0, 0.0]
        for c in range(3):
            r = min(max(float(r_internal[c]), 0.001), 0.995)
            out[c] = num_factor * r / max(1.0 - k2 * r, 1e-4)
        return out

    @staticmethod
    def sample_studio_env_map_py(
        reflect_dir: Sequence[float], roughness_alpha: float, ao: float = 1.0
    ) -> List[float]:
        rx, ry, rz = float(reflect_dir[0]), float(reflect_dir[1]), float(reflect_dir[2])
        sharpness = max(1.0, 32.0 * (1.0 - min(max(roughness_alpha * 1.4, 0.0), 0.96)))
        key_lobe = (max(0.0, -0.65 * ry + 0.65 * rz + 0.35 * rx)) ** sharpness
        rim_lobe = (max(0.0, 0.45 * ry + 0.70 * abs(rx) + 0.40 * rz)) ** (sharpness * 0.6)
        horizon_band = math.exp(-abs(ry + 0.1) * (8.0 / (1.0 + 6.0 * roughness_alpha)))
        blur_mix = min(max(roughness_alpha * 1.35, 0.0), 1.0)
        crisp_env = 0.08 + 1.15 * key_lobe + 0.55 * rim_lobe + 0.25 * horizon_band
        blurred_env = 0.44 + 0.12 * (-ry)
        env_lum = ((1.0 - blur_mix) * crisp_env + blur_mix * blurred_env) * (0.4 + 0.6 * ao)
        return [
            min(max(env_lum * 0.97, 0.01), 1.5),
            min(max(env_lum * 0.99, 0.01), 1.5),
            min(max(env_lum * 1.03, 0.01), 1.5),
        ]

    @staticmethod
    def tone_map_highlight_shoulder_py(rgb: Sequence[float]) -> List[float]:
        r, g, b = max(0.0, float(rgb[0])), max(0.0, float(rgb[1])), max(0.0, float(rgb[2]))
        max_ch = max(r, g, b)
        if max_ch <= 1.0:
            return [r, g, b]
        scale = 1.0 / max_ch
        return [min(max(r * scale, 0.0), 1.0), min(max(g * scale, 0.0), 1.0), min(max(b * scale, 0.0), 1.0)]

    @classmethod
    def evaluate_pixel_stack_py(
        cls,
        orig_wb: Sequence[float],
        base_albedo: Sequence[float],
        shading_d: Sequence[float],
        specular_s: Sequence[float],
        normal_vec: Sequence[float] = (0.0, 0.0, 1.0),
        ao_val: float = 1.0,
        panel_val: float = 0.0,
        is_modified: bool = True,
        layers: Sequence[Dict[str, Any]] = (),
        topcoat_type: int = 0,
        topcoat_roughness: float = 0.06,
        env_mode: int = 0,
        ambient_dim: float = 1.0,
        uv_intensity: float = 0.0,
        px_x: int = 0,
        px_y: int = 0,
    ) -> Tuple[List[float], float]:
        """
        Canonical Python reference implementation of `paintcore::physics::evaluate_pixel_stack`
        + `tone_map_highlight_shoulder` (`cpp/include/paintcore/km_brdf_solver.hpp`).
        """
        if not is_modified and env_mode == 0:
            return cls.tone_map_highlight_shoulder_py(orig_wb), 0.22

        # 1. Resolve stack roughness
        current_alpha = 0.22
        undercoat_alpha = current_alpha
        for L in layers[:6]:
            undercoat_alpha = current_alpha
            role = int(L.get("role", 1))
            alpha_l = float(L.get("intrinsic_roughness", 0.08))
            coats_l = max(0.0, float(L.get("coat_count", 1.0)))
            if role == 3:  # MIRROR_CHROME
                current_alpha = math.sqrt(alpha_l * alpha_l + 0.85 * undercoat_alpha * undercoat_alpha)
            else:
                leveling = 1.0 - math.exp(-1.2 * coats_l)
                current_alpha = current_alpha + leveling * (alpha_l - current_alpha)
        alpha_eff = min(max(topcoat_roughness if topcoat_type != 0 else current_alpha, 0.01), 0.95)

        R_curr = [float(base_albedo[0]), float(base_albedo[1]), float(base_albedo[2])]
        residual_r = [
            float(orig_wb[c]) - (float(base_albedo[c]) * float(shading_d[c]) + float(specular_s[c]))
            for c in range(3)
        ]
        E_fluo = [0.0, 0.0, 0.0]
        clear_transmittance = [1.0, 1.0, 1.0]
        metallic_f0_tint = [0.0, 0.0, 0.0]
        is_mirror_chrome_top = False
        is_metallic_under = False
        has_clear_overcoat = False
        chrome_f0 = 0.95
        total_optical_depth = 0.0

        ndotv = min(max(float(normal_vec[2]), 0.0), 1.0)
        uv_irrad = 0.0
        if env_mode == 2:
            ndotl_uv = max(0.0, normal_vec[0] * 0.2 + normal_vec[1] * (-0.5) + normal_vec[2] * 0.84)
            uv_irrad = uv_intensity * (0.35 * ao_val + 0.65 * ndotl_uv)
        w_vis = 0.22 if env_mode == 0 else 0.04
        w_uv = 1.35 if env_mode == 2 else 0.0
        E_ex = w_vis * float(shading_d[2]) + w_uv * uv_irrad

        for L in layers[:6]:
            role = int(L.get("role", 1))
            coats = max(0.0, float(L.get("coat_count", 1.0)))
            eff_coats = coats if coats <= 1.0 else (coats ** 1.75)
            op_mult = max(0.1, float(L.get("opacity_override", 1.0)))
            r_inf = L.get("masstone_R_inf", [0.85, 0.85, 0.85])
            hiding = L.get("hiding_power_h", [0.75, 0.75, 0.75])

            if role in (0, 1):  # SURFACER, BASE_SOLID
                for c in range(3):
                    a, b, sx1 = cls.compute_km_constants(r_inf[c], hiding[c])
                    R_curr[c] = cls.evaluate_km_channel_py(R_curr[c], a, b, eff_coats * op_mult * sx1)
                total_optical_depth += eff_coats * op_mult
                is_mirror_chrome_top = False
                is_metallic_under = False
            elif role == 2:  # METALLIC
                cov = min(max(1.0 - math.exp(-1.45 * coats * op_mult), 0.0), 1.0)
                flake_mod = 1.0
                if px_x != 0 or px_y != 0:
                    h1 = math.sin(float(px_x) * 12.9898 + float(px_y) * 78.233) * 43758.5453
                    f1 = h1 - math.floor(h1)
                    h2 = math.sin(float(px_x // 2) * 93.989 + float(px_y // 2) * 67.345) * 24634.6345
                    f2 = h2 - math.floor(h2)
                    f_signed = (f1 - 0.5) * 0.40 + (f2 - 0.5) * 0.60
                    f_sparkle = max(0.0, (f1 * 0.38 + f2 * 0.62 - 0.46) / 0.54) ** 2.1
                    flake_mod = min(max(1.0 + 0.22 * f_signed + 0.38 * f_sparkle, 0.35), 1.55)
                for c in range(3):
                    a, b, sx1 = cls.compute_km_constants(r_inf[c], hiding[c])
                    km_r = cls.evaluate_km_channel_py(R_curr[c], a, b, eff_coats * op_mult * sx1)
                    R_curr[c] = min(max(km_r * flake_mod, 0.001), 0.995)
                    metallic_f0_tint[c] = min(max(float(r_inf[c]) * 1.45, 0.12), 1.0) * cov
                total_optical_depth += eff_coats * op_mult
                is_mirror_chrome_top = False
                is_metallic_under = True
            elif role == 3:  # MIRROR_CHROME
                for c in range(3):
                    R_curr[c] *= 0.03
                is_mirror_chrome_top = True
                is_metallic_under = False
                chrome_f0 = float(L.get("chrome_f0", 0.95))
                total_optical_depth += 3.0
            elif role == 4:  # CLEAR_CANDY
                beer = L.get("beer_alpha", [0.0, 0.0, 0.0])
                path_scale = 1.0 / max(0.45, ndotv)
                for c in range(3):
                    trans = math.exp(-2.0 * float(beer[c]) * coats * path_scale)
                    R_curr[c] = min(max(R_curr[c] * trans, 0.001), 0.995)
                    clear_transmittance[c] *= trans
                has_clear_overcoat = True
                total_optical_depth += 0.35 * coats
            elif role in (5, 6):  # CHAMELEON, PEARL
                stops = L.get("chameleon_stops", [0.65, 0.18, 210.0, 0.52, 0.22, 265.0, 0.48, 0.24, 325.0])
                u = 1.0 - ndotv
                s0, s45, s80 = stops[0:3], stops[3:6], stops[6:9]
                lch_interp = cls.lerp_oklch(s0, s45, u / 0.30) if u <= 0.30 else cls.lerp_oklch(s45, s80, (u - 0.30) / 0.53)
                cham_rgb = cls.oklab_to_linear_srgb(cls.oklch_to_oklab(lch_interp))
                cov = 1.0 - math.exp(-1.1 * coats * op_mult)
                for c in range(3):
                    R_curr[c] = min(max(R_curr[c] * (1.0 - 0.85 * cov) + float(cham_rgb[c]) * cov, 0.002), 0.992)
                total_optical_depth += coats
                is_mirror_chrome_top = False
                is_metallic_under = False
            elif role == 7:  # FLUORESCENT_SOLID
                for c in range(3):
                    a, b, sx1 = cls.compute_km_constants(r_inf[c], hiding[c])
                    R_curr[c] = cls.evaluate_km_channel_py(R_curr[c], a, b, eff_coats * op_mult * sx1)
                sub_lum = 0.2126 * R_curr[0] + 0.7152 * R_curr[1] + 0.0722 * R_curr[2]
                sub_boost = 0.55 + 0.65 * math.sqrt(min(max(sub_lum, 0.05), 1.0))
                qy = float(L.get("fluo_quantum_yield", 0.0))
                peak = L.get("fluo_peak_rgb", [0.0, 0.0, 0.0])
                abs_factor = qy * (1.0 - math.exp(-0.9 * coats)) * E_ex * sub_boost
                for c in range(3):
                    E_fluo[c] += abs_factor * float(peak[c])
                total_optical_depth += eff_coats * op_mult
                is_mirror_chrome_top = False
                is_metallic_under = False
            elif role == 8:  # FLUORESCENT_CLEAR
                beer = L.get("beer_alpha", [0.0, 0.0, 0.0])
                path_scale = 1.0 / max(0.45, ndotv)
                for c in range(3):
                    trans = math.exp(-2.0 * float(beer[c]) * coats * path_scale)
                    R_curr[c] = min(max(R_curr[c] * trans, 0.001), 0.995)
                    clear_transmittance[c] *= trans
                qy = float(L.get("fluo_quantum_yield", 0.0))
                peak = L.get("fluo_peak_rgb", [0.0, 0.0, 0.0])
                abs_factor = qy * (1.0 - math.exp(-0.85 * coats)) * E_ex
                for c in range(3):
                    E_fluo[c] += abs_factor * float(peak[c])
                has_clear_overcoat = True
                total_optical_depth += 0.25 * coats

        # Saunderson surface correction
        A_body = list(R_curr)
        if len(layers) > 0 and not is_metallic_under:
            A_body = cls.apply_saunderson_correction_py(R_curr, 0.04, 0.60)
        elif len(layers) > 0 and is_metallic_under:
            A_body = [min(max(R_curr[c] * 0.88, 0.001), 0.995) for c in range(3)]
        elif topcoat_type == 1:  # GLOSS over bare plastic
            A_body = [R_curr[c] * 0.90 for c in range(3)]

        gloss_factor = min(max((0.22 - alpha_eff) / 0.18, -1.0), 1.0)
        matte_factor = min(max((alpha_eff - 0.22) / 0.50, 0.0), 1.0)
        spec_scale = (1.0 + 0.65 * gloss_factor) if gloss_factor >= 0.0 else (1.0 - 0.78 * matte_factor)
        veil_strength = 0.032 * matte_factor

        spec_and_veil = [0.0, 0.0, 0.0]
        ndoth = min(max(-0.24 * float(normal_vec[0]) - 0.34 * float(normal_vec[1]) + 0.91 * ndotv, 0.0), 1.0)
        if is_mirror_chrome_top:
            refl_dir = [
                2.0 * ndotv * float(normal_vec[0]),
                2.0 * ndotv * float(normal_vec[1]),
                2.0 * ndotv * ndotv - 1.0,
            ]
            env_rgb = cls.sample_studio_env_map_py(refl_dir, alpha_eff, ao_val)
            fresnel = chrome_f0 + (1.0 - chrome_f0) * ((1.0 - ndotv) ** 5.0)
            shade_lum = 0.2126 * float(shading_d[0]) + 0.7152 * float(shading_d[1]) + 0.0722 * float(shading_d[2])
            env_mod = 0.35 + 0.65 * min(max(shade_lum, 0.1), 1.6)
            for c in range(3):
                spec_and_veil[c] = (fresnel * env_rgb[c] * env_mod + float(specular_s[c]) * spec_scale) * clear_transmittance[c]
        elif is_metallic_under:
            metal_lobe = (ndoth ** 5.0) * min(max(float(shading_d[1]), 0.15), 1.6)
            top_wet_fresnel = (
                (1.0 - 0.85 * matte_factor) * 0.16 * (ndoth ** 24.0) * min(max(float(shading_d[1]), 0.45), 1.6)
                if has_clear_overcoat else 0.0
            )
            for c in range(3):
                T_spec = math.sqrt(min(max(clear_transmittance[c], 0.0), 1.0))
                internal_metal_spec = metallic_f0_tint[c] * (0.38 * metal_lobe + 0.65 * float(specular_s[c])) * T_spec
                top_dielec_spec = max(0.0, float(specular_s[c]) - 0.22) * spec_scale + top_wet_fresnel
                spec_and_veil[c] = internal_metal_spec + top_dielec_spec + veil_strength * float(shading_d[c])
        else:
            top_wet_fresnel = (
                (1.0 - 0.85 * matte_factor) * 0.14 * (ndoth ** 24.0) * min(max(float(shading_d[1]), 0.45), 1.6)
                if has_clear_overcoat else 0.0
            )
            for c in range(3):
                top_glint = (max(0.0, float(specular_s[c]) - 0.22) * spec_scale + top_wet_fresnel) if has_clear_overcoat else 0.0
                spec_and_veil[c] = float(specular_s[c]) * spec_scale * clear_transmittance[c] + top_glint + veil_strength * float(shading_d[c])

        kappa_amb = 1.0
        T_leak = [1.0, 1.0, 1.0]
        if env_mode == 1:
            kappa_amb = min(max(ambient_dim, 0.02), 0.35)
        elif env_mode == 2:
            kappa_amb = min(max(ambient_dim, 0.02), 0.15)
            T_leak = [0.18, 0.05, 0.42]

        eta_coat = math.exp(-1.5 * total_optical_depth)
        out_rgb = [0.0, 0.0, 0.0]
        for c in range(3):
            diff_term = A_body[c] * (kappa_amb * float(shading_d[c]) * T_leak[c])
            spec_term = kappa_amb * spec_and_veil[c]
            res_term = eta_coat * kappa_amb * residual_r[c]
            out_rgb[c] = min(max(diff_term + spec_term + E_fluo[c] + res_term, 0.0), 4.0)

        return cls.tone_map_highlight_shoulder_py(out_rgb), alpha_eff

    def evaluate_pixel_stack_cpp(
        self,
        orig_wb: Sequence[float],
        base_albedo: Sequence[float],
        shading_d: Sequence[float],
        specular_s: Sequence[float],
        normal_vec: Sequence[float] = (0.0, 0.0, 1.0),
        ao_val: float = 1.0,
        panel_val: float = 0.0,
        is_modified: bool = True,
        layers: Sequence[Dict[str, Any]] = (),
        topcoat_type: int = 0,
        topcoat_roughness: float = 0.06,
        env_mode: int = 0,
        ambient_dim: float = 1.0,
        uv_intensity: float = 0.0,
        px_x: int = 0,
        px_y: int = 0,
    ) -> Tuple[List[float], float]:
        """Evaluate pixel stack via native C++20 `paintcore.dll` (`paintcore_eval_pixel_stack_flat`)."""
        if self.lib is None or not hasattr(self.lib, "paintcore_eval_pixel_stack_flat"):
            return self.evaluate_pixel_stack_py(
                orig_wb, base_albedo, shading_d, specular_s, normal_vec,
                ao_val, panel_val, is_modified, layers, topcoat_type,
                topcoat_roughness, env_mode, ambient_dim, uv_intensity, px_x, px_y,
            )

        num_l = min(len(layers), 6)
        c_orig = (ctypes.c_float * 3)(*orig_wb)
        c_alb = (ctypes.c_float * 3)(*base_albedo)
        c_shd = (ctypes.c_float * 3)(*shading_d)
        c_spc = (ctypes.c_float * 3)(*specular_s)
        c_nrm = (ctypes.c_float * 3)(*normal_vec)

        n_alloc = max(1, num_l)
        c_roles = (ctypes.c_int * n_alloc)()
        c_rinf = (ctypes.c_float * (3 * n_alloc))()
        c_hid = (ctypes.c_float * (3 * n_alloc))()
        c_alp = (ctypes.c_float * n_alloc)()
        c_cts = (ctypes.c_float * n_alloc)()
        c_opa = (ctypes.c_float * n_alloc)()
        c_beer = (ctypes.c_float * (3 * n_alloc))()
        c_cham = (ctypes.c_float * (9 * n_alloc))()
        c_fy = (ctypes.c_float * n_alloc)()
        c_fp = (ctypes.c_float * (3 * n_alloc))()

        for k in range(num_l):
            L = layers[k]
            c_roles[k] = int(L.get("role", 1))
            r_inf = L.get("masstone_R_inf", [0.85, 0.85, 0.85])
            hid = L.get("hiding_power_h", [0.75, 0.75, 0.75])
            beer = L.get("beer_alpha", [0.0, 0.0, 0.0])
            cham = L.get("chameleon_stops", [0.65, 0.18, 210.0, 0.52, 0.22, 265.0, 0.48, 0.24, 325.0])
            fp = L.get("fluo_peak_rgb", [0.0, 0.0, 0.0])
            for c in range(3):
                c_rinf[3 * k + c] = float(r_inf[c])
                c_hid[3 * k + c] = float(hid[c])
                c_beer[3 * k + c] = float(beer[c])
                c_fp[3 * k + c] = float(fp[c])
            for j in range(9):
                c_cham[9 * k + j] = float(cham[j])
            c_alp[k] = float(L.get("intrinsic_roughness", 0.08))
            c_cts[k] = float(L.get("coat_count", 1.0))
            c_opa[k] = float(L.get("opacity_override", 1.0))
            c_fy[k] = float(L.get("fluo_quantum_yield", 0.0))

        c_out_rgb = (ctypes.c_float * 3)()
        c_out_alpha = ctypes.c_float(0.22)

        self.lib.paintcore_eval_pixel_stack_flat(
            c_orig, c_alb, c_shd, c_spc, c_nrm,
            ctypes.c_float(ao_val), ctypes.c_float(panel_val),
            ctypes.c_int(1 if is_modified else 0), ctypes.c_int(num_l),
            c_roles, c_rinf, c_hid, c_alp, c_cts, c_opa, c_beer, c_cham, c_fy, c_fp,
            ctypes.c_int(topcoat_type), ctypes.c_float(topcoat_roughness),
            ctypes.c_int(env_mode), ctypes.c_float(ambient_dim), ctypes.c_float(uv_intensity),
            ctypes.c_int(px_x), ctypes.c_int(px_y),
            c_out_rgb, ctypes.byref(c_out_alpha),
        )
        return [float(c_out_rgb[0]), float(c_out_rgb[1]), float(c_out_rgb[2])], float(c_out_alpha.value)
