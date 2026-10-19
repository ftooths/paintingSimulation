#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <array>
#include <vector>

namespace paintcore::color {

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// ============================================================================
// 1. IEC 61966-2-1 sRGB <-> Scene-Linear RGB (Exact Analytical + 64K LUT)
// ============================================================================

inline float srgb_to_linear_exact(float c_srgb) noexcept {
    c_srgb = std::clamp(c_srgb, 0.0f, 1.0f);
    if (c_srgb <= 0.04045f) {
        return c_srgb / 12.92f;
    }
    return std::pow((c_srgb + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb_exact(float c_lin) noexcept {
    c_lin = std::clamp(c_lin, 0.0f, 1.0f);
    if (c_lin <= 0.0031308f) {
        return 12.92f * c_lin;
    }
    return 1.055f * std::pow(c_lin, 1.0f / 2.4f) - 0.055f;
}

class SrgbLut64K {
public:
    static const SrgbLut64K& instance() {
        static SrgbLut64K lut;
        return lut;
    }

    inline float to_linear_u16(uint16_t val) const noexcept {
        return srgb16_to_lin_[val];
    }

    inline float to_linear_u8(uint8_t val) const noexcept {
        return srgb8_to_lin_[val];
    }

    inline float to_linear_float(float srgb_01) const noexcept {
        const float scaled = std::clamp(srgb_01, 0.0f, 1.0f) * 65535.0f;
        const uint16_t idx = static_cast<uint16_t>(scaled + 0.5f);
        return srgb16_to_lin_[idx];
    }

    inline uint8_t to_srgb_u8(float lin_01) const noexcept {
        const float scaled = std::clamp(lin_01, 0.0f, 1.0f) * 65535.0f;
        const uint16_t idx = static_cast<uint16_t>(scaled + 0.5f);
        return lin16_to_srgb8_[idx];
    }

private:
    std::array<float, 65536> srgb16_to_lin_{};
    std::array<float, 256> srgb8_to_lin_{};
    std::array<uint8_t, 65536> lin16_to_srgb8_{};

    SrgbLut64K() {
        for (uint32_t i = 0; i < 65536; ++i) {
            const float x = static_cast<float>(i) / 65535.0f;
            srgb16_to_lin_[i] = srgb_to_linear_exact(x);
            const float s = linear_to_srgb_exact(x);
            lin16_to_srgb8_[i] = static_cast<uint8_t>(std::clamp(s * 255.0f + 0.5f, 0.0f, 255.0f));
        }
        for (uint32_t i = 0; i < 256; ++i) {
            const float x = static_cast<float>(i) / 255.0f;
            srgb8_to_lin_[i] = srgb_to_linear_exact(x);
        }
    }
};

inline std::array<float, 3> srgb_to_linear_rgb(const std::array<float, 3>& srgb) noexcept {
    return {
        srgb_to_linear_exact(srgb[0]),
        srgb_to_linear_exact(srgb[1]),
        srgb_to_linear_exact(srgb[2])
    };
}

inline std::array<float, 3> linear_to_srgb_rgb(const std::array<float, 3>& lin) noexcept {
    return {
        linear_to_srgb_exact(lin[0]),
        linear_to_srgb_exact(lin[1]),
        linear_to_srgb_exact(lin[2])
    };
}

// ============================================================================
// 2. Linear sRGB <-> OKLab <-> OKLCh
// ============================================================================

inline std::array<float, 3> linear_srgb_to_oklab(const std::array<float, 3>& rgb) noexcept {
    const float r = std::max(0.0f, rgb[0]);
    const float g = std::max(0.0f, rgb[1]);
    const float b = std::max(0.0f, rgb[2]);

    const float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    const float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    const float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

    const float l_ = std::cbrt(l);
    const float m_ = std::cbrt(m);
    const float s_ = std::cbrt(s);

    return {
        0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
        1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
        0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_
    };
}

inline std::array<float, 3> oklab_to_linear_srgb(const std::array<float, 3>& lab) noexcept {
    const float L = lab[0];
    const float a = lab[1];
    const float b = lab[2];

    const float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    const float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    const float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    const float l = l_ * l_ * l_;
    const float m = m_ * m_ * m_;
    const float s = s_ * s_ * s_;

    return {
        std::clamp(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s, 0.0f, 1.0f),
        std::clamp(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s, 0.0f, 1.0f),
        std::clamp(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s, 0.0f, 1.0f)
    };
}

// OKLab (L, a, b) -> OKLCh (L, C, h_deg in [0, 360))
inline std::array<float, 3> oklab_to_oklch(const std::array<float, 3>& lab) noexcept {
    const float C = std::sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
    float h_deg = std::atan2(lab[2], lab[1]) * RAD_TO_DEG;
    if (h_deg < 0.0f) h_deg += 360.0f;
    return {lab[0], C, h_deg};
}

// OKLCh (L, C, h_deg) -> OKLab (L, a, b)
inline std::array<float, 3> oklch_to_oklab(const std::array<float, 3>& lch) noexcept {
    const float h_rad = lch[2] * DEG_TO_RAD;
    return {
        lch[0],
        lch[1] * std::cos(h_rad),
        lch[1] * std::sin(h_rad)
    };
}

// Shortest-arc interpolation in OKLCh for Chameleon / Color-Shift coatings
inline std::array<float, 3> lerp_oklch(
    const std::array<float, 3>& c0,
    const std::array<float, 3>& c1,
    float t
) noexcept {
    t = std::clamp(t, 0.0f, 1.0f);
    const float L = c0[0] + t * (c1[0] - c0[0]);
    const float C = c0[1] + t * (c1[1] - c0[1]);
    float dh = c1[2] - c0[2];
    if (dh > 180.0f) dh -= 360.0f;
    else if (dh < -180.0f) dh += 360.0f;
    float h = c0[2] + t * dh;
    if (h < 0.0f) h += 360.0f;
    if (h >= 360.0f) h -= 360.0f;
    return {L, C, h};
}

// ============================================================================
// 3. Linear sRGB -> CIE XYZ (D65) -> CIELAB (L*, a*, b*) & CIEDE2000 (Delta E00)
// ============================================================================

inline std::array<float, 3> linear_srgb_to_cielab(const std::array<float, 3>& rgb) noexcept {
    const float r = std::max(0.0f, rgb[0]);
    const float g = std::max(0.0f, rgb[1]);
    const float b = std::max(0.0f, rgb[2]);

    // D65 reference white normalized XYZ
    const float X = (0.4124564f * r + 0.3575761f * g + 0.1804375f * b) / 0.95047f;
    const float Y = (0.2126729f * r + 0.7151522f * g + 0.0721750f * b) / 1.00000f;
    const float Z = (0.0193339f * r + 0.1191920f * g + 0.9503041f * b) / 1.08883f;

    auto f = [](float t) noexcept -> float {
        constexpr float delta = 6.0f / 29.0f;
        constexpr float delta3 = delta * delta * delta;
        if (t > delta3) return std::cbrt(t);
        return t / (3.0f * delta * delta) + (4.0f / 29.0f);
    };

    const float fx = f(X);
    const float fy = f(Y);
    const float fz = f(Z);

    return {
        116.0f * fy - 16.0f,
        500.0f * (fx - fy),
        200.0f * (fy - fz)
    };
}

// Exact CIEDE2000 perceptual color difference (Delta E_00)
inline float ciede2000(const std::array<float, 3>& lab1, const std::array<float, 3>& lab2) noexcept {
    const float L1 = lab1[0], a1 = lab1[1], b1 = lab1[2];
    const float L2 = lab2[0], a2 = lab2[1], b2 = lab2[2];

    const float C1 = std::sqrt(a1 * a1 + b1 * b1);
    const float C2 = std::sqrt(a2 * a2 + b2 * b2);
    const float C_bar = 0.5f * (C1 + C2);

    const float C_bar7 = std::pow(C_bar, 7.0f);
    const float G = 0.5f * (1.0f - std::sqrt(C_bar7 / (C_bar7 + 6103515625.0f))); // 25^7

    const float a1_prime = (1.0f + G) * a1;
    const float a2_prime = (1.0f + G) * a2;

    const float C1_prime = std::sqrt(a1_prime * a1_prime + b1 * b1);
    const float C2_prime = std::sqrt(a2_prime * a2_prime + b2 * b2);

    auto compute_h_prime = [](float a_p, float b_val) noexcept -> float {
        if (std::abs(a_p) < 1e-7f && std::abs(b_val) < 1e-7f) return 0.0f;
        float h = std::atan2(b_val, a_p) * RAD_TO_DEG;
        return (h < 0.0f) ? (h + 360.0f) : h;
    };

    const float h1_prime = compute_h_prime(a1_prime, b1);
    const float h2_prime = compute_h_prime(a2_prime, b2);

    const float dL_prime = L2 - L1;
    const float dC_prime = C2_prime - C1_prime;

    float dh_prime = 0.0f;
    if (C1_prime * C2_prime > 1e-7f) {
        const float diff = h2_prime - h1_prime;
        if (std::abs(diff) <= 180.0f) dh_prime = diff;
        else if (diff > 180.0f) dh_prime = diff - 360.0f;
        else dh_prime = diff + 360.0f;
    }
    const float dH_prime = 2.0f * std::sqrt(C1_prime * C2_prime) * std::sin(0.5f * dh_prime * DEG_TO_RAD);

    const float L_bar_prime = 0.5f * (L1 + L2);
    const float C_bar_prime = 0.5f * (C1_prime + C2_prime);

    float h_bar_prime = h1_prime + h2_prime;
    if (C1_prime * C2_prime > 1e-7f) {
        if (std::abs(h1_prime - h2_prime) <= 180.0f) {
            h_bar_prime = 0.5f * (h1_prime + h2_prime);
        } else if (h1_prime + h2_prime < 360.0f) {
            h_bar_prime = 0.5f * (h1_prime + h2_prime + 360.0f);
        } else {
            h_bar_prime = 0.5f * (h1_prime + h2_prime - 360.0f);
        }
    }

    const float T = 1.0f
        - 0.17f * std::cos((h_bar_prime - 30.0f) * DEG_TO_RAD)
        + 0.24f * std::cos((2.0f * h_bar_prime) * DEG_TO_RAD)
        + 0.32f * std::cos((3.0f * h_bar_prime + 6.0f) * DEG_TO_RAD)
        - 0.20f * std::cos((4.0f * h_bar_prime - 63.0f) * DEG_TO_RAD);

    const float dTheta = 30.0f * std::exp(-((h_bar_prime - 275.0f) / 25.0f) * ((h_bar_prime - 275.0f) / 25.0f));
    const float C_bar_prime7 = std::pow(C_bar_prime, 7.0f);
    const float R_C = 2.0f * std::sqrt(C_bar_prime7 / (C_bar_prime7 + 6103515625.0f));

    const float L_term = L_bar_prime - 50.0f;
    const float S_L = 1.0f + (0.015f * L_term * L_term) / std::sqrt(20.0f + L_term * L_term);
    const float S_C = 1.0f + 0.045f * C_bar_prime;
    const float S_H = 1.0f + 0.015f * C_bar_prime * T;
    const float R_T = -std::sin(2.0f * dTheta * DEG_TO_RAD) * R_C;

    const float term_L = dL_prime / S_L;
    const float term_C = dC_prime / S_C;
    const float term_H = dH_prime / S_H;

    return std::sqrt(term_L * term_L + term_C * term_C + term_H * term_H + R_T * term_C * term_H);
}

inline float delta_e00_linear_rgb(const std::array<float, 3>& rgb1, const std::array<float, 3>& rgb2) noexcept {
    return ciede2000(linear_srgb_to_cielab(rgb1), linear_srgb_to_cielab(rgb2));
}

// ============================================================================
// 4. Shades-of-Gray White Balance & Clipping Detection (Module 1)
// ============================================================================

inline std::array<float, 3> estimate_shades_of_gray_illuminant(
    const float* linear_rgb_hwc,
    size_t num_pixels,
    uint8_t* out_clip_mask = nullptr,
    float p_norm = 6.0f
) noexcept {
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    size_t valid_count = 0;

    for (size_t i = 0; i < num_pixels; ++i) {
        const float r = linear_rgb_hwc[3 * i + 0];
        const float g = linear_rgb_hwc[3 * i + 1];
        const float b = linear_rgb_hwc[3 * i + 2];
        const bool is_clipped = (std::max({r, g, b}) >= 0.98f);
        if (out_clip_mask) {
            out_clip_mask[i] = is_clipped ? 1 : 0;
        }
        if (!is_clipped && std::max({r, g, b}) > 0.01f) {
            sum_r += std::pow(static_cast<double>(r), p_norm);
            sum_g += std::pow(static_cast<double>(g), p_norm);
            sum_b += std::pow(static_cast<double>(b), p_norm);
            ++valid_count;
        }
    }

    if (valid_count == 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    const double inv_n = 1.0 / static_cast<double>(valid_count);
    const double inv_p = 1.0 / static_cast<double>(p_norm);
    const float er = static_cast<float>(std::pow(sum_r * inv_n, inv_p));
    const float eg = static_cast<float>(std::pow(sum_g * inv_n, inv_p));
    const float eb = static_cast<float>(std::pow(sum_b * inv_n, inv_p));
    return {std::max(er, 1e-4f), std::max(eg, 1e-4f), std::max(eb, 1e-4f)};
}

inline void apply_von_kries_wb(
    const float* in_linear_rgb,
    float* out_wb_rgb,
    size_t num_pixels,
    const std::array<float, 3>& illuminant
) noexcept {
    const float gain_r = illuminant[1] / std::max(illuminant[0], 1e-4f);
    const float gain_g = 1.0f;
    const float gain_b = illuminant[1] / std::max(illuminant[2], 1e-4f);

    for (size_t i = 0; i < num_pixels; ++i) {
        out_wb_rgb[3 * i + 0] = std::clamp(in_linear_rgb[3 * i + 0] * gain_r, 0.0f, 1.0f);
        out_wb_rgb[3 * i + 1] = std::clamp(in_linear_rgb[3 * i + 1] * gain_g, 0.0f, 1.0f);
        out_wb_rgb[3 * i + 2] = std::clamp(in_linear_rgb[3 * i + 2] * gain_b, 0.0f, 1.0f);
    }
}

} // namespace paintcore::color
