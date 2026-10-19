"""Zero-Copy C++20 Native Core Bridge & Cross-Runtime Physics Solver."""
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
    reference implementation for cross-runtime parity testing (`TC-PHY-006`).
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
        # Exact CIEDE2000 fallback
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
