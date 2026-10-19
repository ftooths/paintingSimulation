#pragma once
#include "paintcore/color_spaces.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>

namespace paintcore::intrinsics {

struct IntrinsicDecompositionResult {
    std::vector<float> albedo_base_hwc;     // H x W x 3: Piecewise-constant bare plastic albedo A_base(x)
    std::vector<float> shading_diffuse_hwc; // H x W x 3: Consistent diffuse shading S_d(x)
    std::vector<float> specular_hwc;        // H x W x 3: Specular highlight S_s(x)
    std::vector<float> residual_hwc;        // H x W x 3: Exact identity residual R(x)
    std::vector<std::array<float, 3>> region_median_albedos; // Per-region linear RGB median albedo
};

// ============================================================================
// Piecewise-Constant Plastic Albedo Regularizer & Zero-Loss Identity Residual
// ============================================================================

inline IntrinsicDecompositionResult regularize_plastic_albedo_and_shading(
    const float* orig_wb_hwc,
    const float* raw_shading_hwc,     // Initial diffuse shading estimate (or nullptr for Track A dichromatic)
    const float* raw_specular_hwc,    // Initial specular estimate (or nullptr)
    const uint16_t* region_id_map_hw,
    const uint8_t* clip_mask_hw,
    const uint8_t* panel_mask_hw,
    int width,
    int height,
    uint16_t num_regions
) {
    const size_t n_pix = static_cast<size_t>(width) * height;
    IntrinsicDecompositionResult out;
    out.albedo_base_hwc.resize(3 * n_pix, 0.5f);
    out.shading_diffuse_hwc.resize(3 * n_pix, 1.0f);
    out.specular_hwc.resize(3 * n_pix, 0.0f);
    out.residual_hwc.resize(3 * n_pix, 0.0f);
    out.region_median_albedos.resize(num_regions, {0.5f, 0.5f, 0.5f});

    // Step 1: Populate initial specular S_s(x) and initial shading S_d(x)
    for (size_t i = 0; i < n_pix; ++i) {
        const float r = orig_wb_hwc[3 * i + 0];
        const float g = orig_wb_hwc[3 * i + 1];
        const float b = orig_wb_hwc[3 * i + 2];

        if (raw_specular_hwc) {
            out.specular_hwc[3 * i + 0] = raw_specular_hwc[3 * i + 0];
            out.specular_hwc[3 * i + 1] = raw_specular_hwc[3 * i + 1];
            out.specular_hwc[3 * i + 2] = raw_specular_hwc[3 * i + 2];
        } else {
            // Classical Track A neutral specular highlight extraction
            const float min_c = std::min({r, g, b});
            const float max_c = std::max({r, g, b});
            const float spec_est = (max_c > 0.65f && (max_c - min_c) < 0.18f)
                ? std::max(0.0f, (min_c - 0.55f) * 0.85f)
                : 0.0f;
            out.specular_hwc[3 * i + 0] = spec_est;
            out.specular_hwc[3 * i + 1] = spec_est;
            out.specular_hwc[3 * i + 2] = spec_est;
        }

        if (raw_shading_hwc) {
            out.shading_diffuse_hwc[3 * i + 0] = std::max(0.05f, raw_shading_hwc[3 * i + 0]);
            out.shading_diffuse_hwc[3 * i + 1] = std::max(0.05f, raw_shading_hwc[3 * i + 1]);
            out.shading_diffuse_hwc[3 * i + 2] = std::max(0.05f, raw_shading_hwc[3 * i + 2]);
        } else {
            const float diff_r = std::max(0.005f, r - out.specular_hwc[3 * i + 0]);
            const float diff_g = std::max(0.005f, g - out.specular_hwc[3 * i + 1]);
            const float diff_b = std::max(0.005f, b - out.specular_hwc[3 * i + 2]);
            const float lum = std::max(0.05f, 0.2126f * diff_r + 0.7152f * diff_g + 0.0722f * diff_b);
            out.shading_diffuse_hwc[3 * i + 0] = lum;
            out.shading_diffuse_hwc[3 * i + 1] = lum;
            out.shading_diffuse_hwc[3 * i + 2] = lum;
        }
    }

    // Step 2: Gather valid diffuse pixels per region in OKLab and compute O(N) median via std::nth_element
    std::vector<std::vector<float>> reg_L(num_regions), reg_a(num_regions), reg_b(num_regions);
    std::vector<std::vector<float>> fallback_L(num_regions), fallback_a(num_regions), fallback_b(num_regions);

    for (size_t i = 0; i < n_pix; ++i) {
        const uint16_t rid = region_id_map_hw ? region_id_map_hw[i] : 0;
        if (rid >= num_regions) continue;

        const float sr = out.specular_hwc[3 * i + 0];
        const float sg = out.specular_hwc[3 * i + 1];
        const float sb = out.specular_hwc[3 * i + 2];
        const float spec_inf = std::max({sr, sg, sb});

        const float sdr = out.shading_diffuse_hwc[3 * i + 0];
        const float sdg = out.shading_diffuse_hwc[3 * i + 1];
        const float sdb = out.shading_diffuse_hwc[3 * i + 2];
        const float sd_norm = std::sqrt(sdr * sdr + sdg * sdg + sdb * sdb);

        const std::array<float, 3> cand_rgb = {
            std::clamp((orig_wb_hwc[3 * i + 0] - sr) / std::max(sdr, 0.08f), 0.015f, 0.96f),
            std::clamp((orig_wb_hwc[3 * i + 1] - sg) / std::max(sdg, 0.08f), 0.015f, 0.96f),
            std::clamp((orig_wb_hwc[3 * i + 2] - sb) / std::max(sdb, 0.08f), 0.015f, 0.96f)
        };
        const auto lab = color::linear_srgb_to_oklab(cand_rgb);

        fallback_L[rid].push_back(lab[0]);
        fallback_a[rid].push_back(lab[1]);
        fallback_b[rid].push_back(lab[2]);

        const bool not_clipped = (!clip_mask_hw || clip_mask_hw[i] == 0);
        const bool not_panel   = (!panel_mask_hw || panel_mask_hw[i] < 45);
        if (not_clipped && not_panel && spec_inf < 0.15f && sd_norm > 0.15f) {
            reg_L[rid].push_back(lab[0]);
            reg_a[rid].push_back(lab[1]);
            reg_b[rid].push_back(lab[2]);
        }
    }

    auto compute_median = [](std::vector<float>& v) -> float {
        if (v.empty()) return 0.5f;
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    };

    for (uint16_t r = 0; r < num_regions; ++r) {
        auto& vL = reg_L[r].empty() ? fallback_L[r] : reg_L[r];
        auto& va = reg_a[r].empty() ? fallback_a[r] : reg_a[r];
        auto& vb = reg_b[r].empty() ? fallback_b[r] : reg_b[r];
        if (!vL.empty()) {
            const std::array<float, 3> med_lab = {
                compute_median(vL),
                compute_median(va),
                compute_median(vb)
            };
            const auto lin_rgb = color::oklab_to_linear_srgb(med_lab);
            out.region_median_albedos[r] = {
                std::clamp(lin_rgb[0], 0.015f, 0.95f),
                std::clamp(lin_rgb[1], 0.015f, 0.95f),
                std::clamp(lin_rgb[2], 0.015f, 0.95f)
            };
        }
    }

    // Step 3: Assign piecewise-constant plastic albedo A_base(x), decompose diffuse shading S_d(x)
    // into low-frequency 3D form shading S_d_low(x) + high-frequency mold/panel relief S_d_high(x)
    // (suppressing within-region chromatic/decal outliers into R(x) so opaque paint never inherits blotches),
    // and compute exact zero-loss identity residual R(x) = I_wb(x) - (A_base(x) * S_d(x) + S_s(x)).
    std::vector<float> raw_sd_scalar(n_pix, 1.0f);
    std::vector<float> chroma_outlier_weight(n_pix, 0.0f);

    for (size_t i = 0; i < n_pix; ++i) {
        const uint16_t rid = (region_id_map_hw && region_id_map_hw[i] < num_regions) ? region_id_map_hw[i] : 0;
        const auto& A_r = out.region_median_albedos[rid];
        const float Y_base = std::max(0.015f, 0.2126f * A_r[0] + 0.7152f * A_r[1] + 0.0722f * A_r[2]);
        const float max_base = std::max({A_r[0], A_r[1], A_r[2], 0.015f});

        const float dr = std::max(0.0f, orig_wb_hwc[3 * i + 0] - out.specular_hwc[3 * i + 0]);
        const float dg = std::max(0.0f, orig_wb_hwc[3 * i + 1] - out.specular_hwc[3 * i + 1]);
        const float db = std::max(0.0f, orig_wb_hwc[3 * i + 2] - out.specular_hwc[3 * i + 2]);
        const float Y_diff = 0.2126f * dr + 0.7152f * dg + 0.0722f * db;
        const float max_diff = std::max({dr, dg, db});

        const float sd_est = std::clamp(
            0.65f * (Y_diff / Y_base) + 0.35f * (max_diff / max_base),
            0.02f, 2.40f
        );
        raw_sd_scalar[i] = sd_est;

        // Measure chromaticity divergence from region median albedo (e.g. decal markings or color blemishes)
        const std::array<float, 3> norm_px = {
            std::clamp(dr / std::max(sd_est, 0.08f), 0.005f, 1.0f),
            std::clamp(dg / std::max(sd_est, 0.08f), 0.005f, 1.0f),
            std::clamp(db / std::max(sd_est, 0.08f), 0.005f, 1.0f)
        };
        const auto lab_px = color::linear_srgb_to_oklab(norm_px);
        const auto lab_reg = color::linear_srgb_to_oklab(A_r);
        const float da = lab_px[1] - lab_reg[1];
        const float db_lab = lab_px[2] - lab_reg[2];
        const float chroma_dist = std::sqrt(da * da + db_lab * db_lab);
        // High chroma_dist (> 0.045 in OKLab) indicates a decal/stain rather than geometric mold shading
        chroma_outlier_weight[i] = std::clamp((chroma_dist - 0.045f) / 0.08f, 0.0f, 1.0f);
    }

    // Region-aware 5x5 edge-preserving low-frequency shading filter
    std::vector<float> smooth_sd_scalar(n_pix, 1.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            const uint16_t self_rid = region_id_map_hw ? region_id_map_hw[idx] : 0;
            const float center_sd = raw_sd_scalar[idx];
            float sum_w = 0.0f;
            float sum_sd = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                const int ny = std::clamp(y + dy, 0, height - 1);
                for (int dx = -2; dx <= 2; ++dx) {
                    const int nx = std::clamp(x + dx, 0, width - 1);
                    const size_t n_idx = static_cast<size_t>(ny) * width + nx;
                    if (region_id_map_hw && region_id_map_hw[n_idx] != self_rid) continue;
                    const float w_chroma = 1.0f - 0.85f * chroma_outlier_weight[n_idx];
                    const float diff = std::abs(raw_sd_scalar[n_idx] - center_sd);
                    const float w_range = std::exp(-diff * diff * 6.0f);
                    const float w = w_chroma * w_range;
                    sum_sd += w * raw_sd_scalar[n_idx];
                    sum_w += w;
                }
            }
            smooth_sd_scalar[idx] = (sum_w > 1e-5f) ? (sum_sd / sum_w) : center_sd;
        }
    }

    for (size_t i = 0; i < n_pix; ++i) {
        const uint16_t rid = (region_id_map_hw && region_id_map_hw[i] < num_regions) ? region_id_map_hw[i] : 0;
        const auto& A_r = out.region_median_albedos[rid];

        // Separate low-frequency form shading and high-frequency mold detail (panel lines, bevels)
        // High-frequency detail is preserved for true geometric relief but damped on chromatic decals/stains
        const float sd_low = smooth_sd_scalar[i];
        const float sd_high = raw_sd_scalar[i] - sd_low;
        const float detail_keep = 1.0f - 0.90f * chroma_outlier_weight[i];
        const float Sd = std::clamp(sd_low + detail_keep * sd_high, 0.02f, 2.5f);

        for (int c = 0; c < 3; ++c) {
            const float I_c  = orig_wb_hwc[3 * i + c];
            const float Ss_c = out.specular_hwc[3 * i + c];
            const float Ab_c = A_r[c];

            out.albedo_base_hwc[3 * i + c] = Ab_c;
            out.shading_diffuse_hwc[3 * i + c] = Sd;
            out.residual_hwc[3 * i + c] = I_c - (Ab_c * Sd + Ss_c);
        }
    }

    return out;
}

} // namespace paintcore::intrinsics
