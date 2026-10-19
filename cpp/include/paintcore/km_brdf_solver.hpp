#pragma once
#include "paintcore/types.hpp"
#include "paintcore/color_spaces.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace paintcore::physics {

// ============================================================================
// 1. Precomputation of Kubelka-Munk Parameters from Masstone & Hiding Power
// ============================================================================

inline void populate_km_parameters(PaintLayerParams& layer) noexcept {
    for (int c = 0; c < 3; ++c) {
        const float R_inf = std::clamp(layer.masstone_R_inf[c], 0.002f, 0.992f);
        const float h = std::clamp(layer.hiding_power_h[c], 0.05f, 0.995f);

        const float a = 0.5f * (R_inf + (1.0f / R_inf));
        const float b = std::sqrt(std::max(0.0f, a * a - 1.0f));
        layer.km_a[c] = a;
        layer.km_b[c] = b;

        // Contrast reflectance over perfectly black substrate (R_0 = 0): R_1 = h * R_inf
        const float R_1 = std::clamp(h * R_inf, 0.001f, R_inf * 0.998f);
        if (b < 1e-3f) {
            // Pure white limit (R_inf -> 1, b -> 0): R_1 = SX / (1 + SX) => SX = R_1 / (1 - R_1)
            layer.km_SX_1coat[c] = R_1 / std::max(1.0f - R_1, 1e-4f);
        } else {
            // coth(b * SX) = (1 - a * R_1) / (b * R_1)
            const float coth_val = std::max((1.0f - a * R_1) / std::max(b * R_1, 1e-6f), 1.0001f);
            // arcoth(x) = 0.5 * ln((x + 1) / (x - 1))
            const float beta = 0.5f * std::log((coth_val + 1.0f) / (coth_val - 1.0f));
            layer.km_SX_1coat[c] = beta / b;
        }
    }
}

// ============================================================================
// 2. Stabilized Per-Channel Kubelka-Munk Solver
// ============================================================================

inline float evaluate_km_channel(
    float R_prev,     // Underlying substrate reflectance R_{k-1} in [0.001, 0.995]
    float a,          // KM parameter a >= 1.0
    float b,          // KM parameter b = sqrt(a^2 - 1) >= 0.0
    float SX          // Effective optical scattering thickness S * X >= 0.0
) noexcept {
    R_prev = std::clamp(R_prev, 0.001f, 0.995f);
    if (SX <= 1e-5f) {
        return R_prev; // Zero film thickness -> underlying substrate unchanged
    }
    if (b < 1e-3f) {
        // Analytical Taylor limit as R_inf -> 1 (b -> 0)
        const float delta = 1.0f - R_prev;
        return std::clamp((R_prev + SX * delta) / (1.0f + SX * delta), 0.001f, 0.995f);
    }
    const float beta = b * SX;
    const float b_coth_beta = (beta >= 10.0f)
        ? b
        : b * ((std::exp(2.0f * beta) + 1.0f) / std::max(std::exp(2.0f * beta) - 1.0f, 1e-7f));
    const float num = 1.0f - R_prev * (a - b_coth_beta);
    const float den = a - R_prev + b_coth_beta;
    return std::clamp(num / std::max(den, 1e-6f), 0.001f, 0.995f);
}

// ============================================================================
// 3. Intrinsic Roughness Inheritance & Undercoat Coupling (Topcoat == NONE)
// ============================================================================

inline float resolve_layer_stack_roughness(
    const RegionRenderStack& stack,
    float& out_undercoat_alpha
) noexcept {
    float current_alpha = stack.bare_plastic_roughness; // ~0.22f for bare PS/ABS plastic
    out_undercoat_alpha = current_alpha;

    for (uint8_t k = 0; k < stack.num_active_layers && k < MAX_STACK_LAYERS; ++k) {
        const auto& L = stack.layers[k];
        if (!L.enabled) continue;
        out_undercoat_alpha = current_alpha; // Save substrate roughness before applying layer k

        if (L.role == LayerRole::MIRROR_CHROME) {
            // Mirror Chrome inherits 85% of the undercoat's roughness variance:
            // Requires Gloss Black Surfacer/Base (alpha_under <= 0.06) for mirror reflection!
            constexpr float k_transfer = 0.85f;
            current_alpha = std::sqrt(
                L.intrinsic_roughness * L.intrinsic_roughness +
                k_transfer * out_undercoat_alpha * out_undercoat_alpha
            );
        } else {
            // Thin coats (< 1.5 passes) over a coarse surfacer (e.g., #500) partially retain substrate texture
            const float leveling = 1.0f - std::exp(-1.2f * std::max(0.0f, L.coat_count));
            current_alpha = current_alpha + leveling * (L.intrinsic_roughness - current_alpha);
        }
    }

    // If user selected an explicit Topcoat (Gloss / Semi-Gloss / Matte), override final surface roughness;
    // If Topcoat == NONE_INHERIT, return the topmost layer's intrinsic roughness!
    if (stack.finish.topcoat_type != TopcoatType::NONE_INHERIT) {
        return std::clamp(stack.finish.topcoat_roughness, 0.01f, 0.95f);
    }
    return std::clamp(current_alpha, 0.01f, 0.95f);
}

// ============================================================================
// 4. Studio Environment Reflection & Specialty Coating Evaluators
// ============================================================================

// Analytical multi-lobe studio softbox environment map (with roughness mip pre-filtering)
inline std::array<float, 3> sample_studio_env_map(
    const std::array<float, 3>& reflect_dir,
    float roughness_alpha,
    float ao = 1.0f
) noexcept {
    const float rx = reflect_dir[0];
    const float ry = reflect_dir[1]; // Up/down (-1 top softbox, +1 floor)
    const float rz = reflect_dir[2];

    // Smooth sharpness controlled by roughness_alpha in [0.01, 0.95]
    const float sharpness = std::max(1.0f, 32.0f * (1.0f - std::clamp(roughness_alpha * 1.4f, 0.0f, 0.96f)));

    // Key overhead softbox + rim light strips + dark studio floor horizon
    const float key_lobe = std::pow(std::max(0.0f, -0.65f * ry + 0.65f * rz + 0.35f * rx), sharpness);
    const float rim_lobe = std::pow(std::max(0.0f, 0.45f * ry + 0.70f * std::abs(rx) + 0.40f * rz), sharpness * 0.6f);
    const float horizon_band = std::exp(-std::abs(ry + 0.1f) * (8.0f / (1.0f + 6.0f * roughness_alpha)));

    // Ambient diffuse average when roughness is high (e.g., over #1000 matte surfacer)
    const float blur_mix = std::clamp(roughness_alpha * 1.35f, 0.0f, 1.0f);
    const float crisp_env = 0.08f + 1.15f * key_lobe + 0.55f * rim_lobe + 0.25f * horizon_band;
    const float blurred_env = 0.44f + 0.12f * (-ry);
    const float env_lum = ((1.0f - blur_mix) * crisp_env + blur_mix * blurred_env) * (0.4f + 0.6f * ao);

    return {
        std::clamp(env_lum * 0.97f, 0.01f, 1.5f),
        std::clamp(env_lum * 0.99f, 0.01f, 1.5f),
        std::clamp(env_lum * 1.03f, 0.01f, 1.5f)
    };
}

// Saunderson surface boundary correction for dielectric binder (n ~ 1.5)
inline std::array<float, 3> apply_saunderson_correction(
    const std::array<float, 3>& R_internal,
    float k1 = 0.04f,
    float k2 = 0.60f
) noexcept {
    std::array<float, 3> out{};
    const float num_factor = (1.0f - k1) * (1.0f - k2);
    for (int c = 0; c < 3; ++c) {
        const float r = std::clamp(R_internal[c], 0.001f, 0.995f);
        out[c] = num_factor * r / std::max(1.0f - k2 * r, 1e-4f);
    }
    return out;
}

// ============================================================================
// 5. Single-Pixel Full Layer Stack + Specialty + Dark/UV Evaluator
// ============================================================================

struct PixelRenderResult {
    std::array<float, 3> linear_rgb;      // Pre-bloom linear RGB pixel output
    std::array<float, 3> fluo_emission;   // Isolated Stokes emission E_fluo(x) for HDR bloom pass
    float effective_roughness;            // Resolved surface GGX alpha_eff
    float undercoat_roughness;            // Substrate roughness under topmost layer
};

inline std::array<float, 3> tone_map_highlight_shoulder(const std::array<float, 3>& rgb) noexcept {
    const float r = std::max(0.0f, rgb[0]);
    const float g = std::max(0.0f, rgb[1]);
    const float b = std::max(0.0f, rgb[2]);
    const float max_ch = std::max({r, g, b});
    if (max_ch <= 1.0f) {
        return {r, g, b}; // In-gamut values untouched (keeps identity reconstruction exact)
    }
    const float scale = 1.0f / max_ch;
    return {
        std::clamp(r * scale, 0.0f, 1.0f),
        std::clamp(g * scale, 0.0f, 1.0f),
        std::clamp(b * scale, 0.0f, 1.0f)
    };
}

inline PixelRenderResult evaluate_pixel_stack(
    const std::array<float, 3>& orig_wb,      // I_wb(x)
    const std::array<float, 3>& base_albedo,  // A_base(x)
    const std::array<float, 3>& shading_d,    // S_d(x)
    const std::array<float, 3>& specular_s,   // S_s(x)
    const std::array<float, 3>& residual_r,   // R(x) = I_wb - (A_base * S_d + S_s)
    const std::array<float, 3>& normal_vec,   // Unit normal N(x)
    float ao_val,                             // Ambient occlusion AO(x) in [0, 1]
    float panel_line_strength,                // P_panel(x) in [0, 1]
    const RegionRenderStack& stack,
    const LightingEnvParams& env_params,
    int px_x = 0,
    int px_y = 0
) noexcept {
    // Fast identity path: if region is protected or unmodified in Standard Studio mode
    if (stack.is_protected || (!stack.is_modified && env_params.mode == LightingEnvMode::STANDARD_STUDIO)) {
        return {orig_wb, {0.0f, 0.0f, 0.0f}, stack.bare_plastic_roughness, stack.bare_plastic_roughness};
    }

    float undercoat_alpha = stack.bare_plastic_roughness;
    const float alpha_eff = resolve_layer_stack_roughness(stack, undercoat_alpha);

    // Step 0: Initialize with bare plastic albedo R^(0) = A_base(x)
    std::array<float, 3> R_curr = base_albedo;
    std::array<float, 3> E_fluo = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> clear_transmittance = {1.0f, 1.0f, 1.0f};
    std::array<float, 3> metallic_f0_tint = {0.0f, 0.0f, 0.0f};
    bool is_mirror_chrome_top = false;
    bool is_metallic_under = false;
    bool has_clear_overcoat = false;
    float chrome_f0 = 0.95f;
    float total_optical_depth = 0.0f;

    // Viewing vector V = (0, 0, 1) in camera space; cos_theta_v = max(0, N . V)
    const float ndotv = std::clamp(normal_vec[2], 0.0f, 1.0f);

    // Compute excitation irradiance for fluorescent coatings:
    // E_ex(x) = w_vis * S_{d,B}(x) + w_uv * I_uv(x)
    float uv_irradiance = 0.0f;
    if (env_params.mode == LightingEnvMode::DARK_UV_365NM) {
        const float ndotl_uv = std::max(0.0f,
            normal_vec[0] * env_params.uv_light_dir[0] +
            normal_vec[1] * env_params.uv_light_dir[1] +
            normal_vec[2] * env_params.uv_light_dir[2]
        );
        uv_irradiance = env_params.uv_intensity * (0.35f * ao_val + 0.65f * ndotl_uv);
    }
    const float w_vis = (env_params.mode == LightingEnvMode::STANDARD_STUDIO) ? 0.22f : 0.04f;
    const float w_uv  = (env_params.mode == LightingEnvMode::DARK_UV_365NM) ? 1.35f : 0.0f;
    const float E_ex  = w_vis * shading_d[2] + w_uv * uv_irradiance;

    // Step k = 1 .. K: Evaluate active coating layers in order
    for (uint8_t k = 0; k < stack.num_active_layers && k < MAX_STACK_LAYERS; ++k) {
        const auto& L = stack.layers[k];
        if (!L.enabled) continue;

        const float coats = std::max(0.0f, L.coat_count);
        // Multi-pass airbrush leveling: at 1 coat factor is 1.0; multiple passes build full hiding coverage
        const float eff_coats = (coats <= 1.0f) ? coats : std::pow(coats, 1.75f);
        const float op_mult = std::max(0.1f, L.opacity_override);

        switch (L.role) {
            case LayerRole::SURFACER:
            case LayerRole::BASE_SOLID: {
                for (int c = 0; c < 3; ++c) {
                    const float SX = eff_coats * op_mult * L.km_SX_1coat[c];
                    R_curr[c] = evaluate_km_channel(R_curr[c], L.km_a[c], L.km_b[c], SX);
                }
                total_optical_depth += eff_coats * op_mult;
                is_mirror_chrome_top = false;
                is_metallic_under = false;
                break;
            }
            case LayerRole::METALLIC: {
                const float cov = std::clamp(1.0f - std::exp(-1.45f * coats * op_mult), 0.0f, 1.0f);
                float flake_mod = 1.0f;
                if (px_x != 0 || px_y != 0) {
                    const float h1 = std::sin(static_cast<float>(px_x) * 12.9898f + static_cast<float>(px_y) * 78.233f) * 43758.5453f;
                    const float f1 = h1 - std::floor(h1);
                    const float h2 = std::sin(static_cast<float>(px_x / 2) * 93.989f + static_cast<float>(px_y / 2) * 67.345f) * 24634.6345f;
                    const float f2 = h2 - std::floor(h2);
                    const float f_signed = (f1 - 0.5f) * 0.40f + (f2 - 0.5f) * 0.60f;
                    const float f_sparkle = std::pow(std::max(0.0f, (f1 * 0.38f + f2 * 0.62f - 0.46f) / 0.54f), 2.1f);
                    flake_mod = std::clamp(1.0f + 0.22f * f_signed + 0.38f * f_sparkle, 0.35f, 1.55f);
                }
                for (int c = 0; c < 3; ++c) {
                    const float SX = eff_coats * op_mult * L.km_SX_1coat[c];
                    const float km_r = evaluate_km_channel(R_curr[c], L.km_a[c], L.km_b[c], SX);
                    R_curr[c] = std::clamp(km_r * flake_mod, 0.001f, 0.995f);
                    metallic_f0_tint[c] = std::clamp(L.masstone_R_inf[c] * 1.45f, 0.12f, 1.0f) * cov;
                }
                total_optical_depth += eff_coats * op_mult;
                is_mirror_chrome_top = false;
                is_metallic_under = true;
                break;
            }
            case LayerRole::MIRROR_CHROME: {
                // Mirror Chrome suppresses diffuse body color and switches to conductor env reflection
                for (int c = 0; c < 3; ++c) {
                    R_curr[c] *= 0.03f;
                }
                is_mirror_chrome_top = true;
                is_metallic_under = false;
                chrome_f0 = L.chrome_f0;
                total_optical_depth += 3.0f;
                break;
            }
            case LayerRole::CLEAR_CANDY: {
                // Beer-Lambert exponential transmission: R = R_prev * exp(-2 * alpha * coats)
                const float path_scale = 1.0f / std::max(0.45f, ndotv);
                for (int c = 0; c < 3; ++c) {
                    const float trans = std::exp(-2.0f * L.beer_alpha[c] * coats * path_scale);
                    R_curr[c] = std::clamp(R_curr[c] * trans, 0.001f, 0.995f);
                    clear_transmittance[c] *= trans;
                }
                has_clear_overcoat = true;
                total_optical_depth += 0.35f * coats;
                break;
            }
            case LayerRole::CHAMELEON:
            case LayerRole::PEARL: {
                // View-angle (N.V) 3-stop OKLCh color shift
                const float u = 1.0f - ndotv;
                const std::array<float, 3> stop0  = {L.chameleon_stops[0], L.chameleon_stops[1], L.chameleon_stops[2]};
                const std::array<float, 3> stop45 = {L.chameleon_stops[3], L.chameleon_stops[4], L.chameleon_stops[5]};
                const std::array<float, 3> stop80 = {L.chameleon_stops[6], L.chameleon_stops[7], L.chameleon_stops[8]};

                std::array<float, 3> lch_interp{};
                if (u <= 0.30f) {
                    lch_interp = color::lerp_oklch(stop0, stop45, u / 0.30f);
                } else {
                    lch_interp = color::lerp_oklch(stop45, stop80, (u - 0.30f) / 0.53f);
                }
                const auto cham_rgb = color::oklab_to_linear_srgb(color::oklch_to_oklab(lch_interp));
                const float cov = 1.0f - std::exp(-1.1f * coats * op_mult);
                // Over dark bases (e.g. Gloss Black Surfacer), interference color dominates vividly
                for (int c = 0; c < 3; ++c) {
                    R_curr[c] = std::clamp(R_curr[c] * (1.0f - 0.85f * cov) + cham_rgb[c] * cov, 0.002f, 0.992f);
                }
                total_optical_depth += coats;
                is_mirror_chrome_top = false;
                is_metallic_under = false;
                break;
            }
            case LayerRole::FLUORESCENT_SOLID: {
                for (int c = 0; c < 3; ++c) {
                    const float SX = eff_coats * op_mult * L.km_SX_1coat[c];
                    R_curr[c] = evaluate_km_channel(R_curr[c], L.km_a[c], L.km_b[c], SX);
                }
                // Stokes emission: boosted when sprayed over bright white/pink substrate
                const float sub_lum = 0.2126f * R_curr[0] + 0.7152f * R_curr[1] + 0.0722f * R_curr[2];
                const float sub_boost = 0.55f + 0.65f * std::sqrt(std::clamp(sub_lum, 0.05f, 1.0f));
                const float abs_factor = L.fluo_quantum_yield * (1.0f - std::exp(-0.9f * coats)) * E_ex * sub_boost;
                for (int c = 0; c < 3; ++c) {
                    E_fluo[c] += abs_factor * L.fluo_peak_rgb[c];
                }
                total_optical_depth += eff_coats * op_mult;
                is_mirror_chrome_top = false;
                is_metallic_under = false;
                break;
            }
            case LayerRole::FLUORESCENT_CLEAR: {
                // Beer-Lambert transparent tint + Stokes fluorescence emission
                const float path_scale = 1.0f / std::max(0.45f, ndotv);
                for (int c = 0; c < 3; ++c) {
                    const float trans = std::exp(-2.0f * L.beer_alpha[c] * coats * path_scale);
                    R_curr[c] = std::clamp(R_curr[c] * trans, 0.001f, 0.995f);
                    clear_transmittance[c] *= trans;
                }
                const float abs_factor = L.fluo_quantum_yield * (1.0f - std::exp(-0.85f * coats)) * E_ex;
                for (int c = 0; c < 3; ++c) {
                    E_fluo[c] += abs_factor * L.fluo_peak_rgb[c];
                }
                has_clear_overcoat = true;
                total_optical_depth += 0.25f * coats;
                break;
            }
        }
    }

    // Optional Panel-Line Wash (Standard or UV-Fluorescent)
    if (stack.wash_enabled && panel_line_strength > 0.01f && stack.wash_intensity > 0.0f) {
        const float w_alpha = std::clamp(panel_line_strength * stack.wash_intensity, 0.0f, 1.0f);
        for (int c = 0; c < 3; ++c) {
            R_curr[c] = (1.0f - w_alpha) * R_curr[c] + w_alpha * stack.wash_rgb[c];
        }
        if (stack.wash_is_fluorescent) {
            for (int c = 0; c < 3; ++c) {
                E_fluo[c] += w_alpha * 0.85f * E_ex * stack.wash_rgb[c];
            }
        }
    }

    // Saunderson Surface Correction
    std::array<float, 3> A_body = R_curr;
    if (stack.num_active_layers > 0 && !is_metallic_under) {
        A_body = apply_saunderson_correction(R_curr, stack.finish.saunderson_k1, stack.finish.saunderson_k2);
    } else if (stack.num_active_layers > 0 && is_metallic_under) {
        // Metallic coats preserve higher conductor reflectance
        for (int c = 0; c < 3; ++c) {
            A_body[c] = std::clamp(R_curr[c] * 0.88f, 0.001f, 0.995f);
        }
    } else if (stack.finish.topcoat_type == TopcoatType::GLOSS) {
        // Bare plastic + Clear Gloss Topcoat only: optical wetting deepens body color by ~8% (dL* in [-1.0, -2.5])
        for (int c = 0; c < 3; ++c) {
            A_body[c] = R_curr[c] * 0.90f;
        }
    }

    // Specular & Diffuse Surface Veil Redistribution based on effective roughness alpha_eff
    // Bare plastic baseline roughness is ~0.22.
    // - Matte (alpha_eff ~ 0.72): specular peak drops by >= 65%, diffuse veil V(x) scatters ~3.2% surface reflection
    // - Gloss (alpha_eff ~ 0.05): specular peak sharpens/boosts, diffuse veil -> 0
    const float gloss_factor = std::clamp((0.22f - alpha_eff) / 0.18f, -1.0f, 1.0f);
    const float matte_factor = std::clamp((alpha_eff - 0.22f) / 0.50f, 0.0f, 1.0f);

    const float spec_scale = (gloss_factor >= 0.0f)
        ? (1.0f + 0.65f * gloss_factor)
        : (1.0f - 0.78f * matte_factor); // At alpha=0.75 (Matte), spec_scale = 0.22 (78% reduction!)

    // Microfacet diffuse surface scattering veil V(x) for matte/coarse surfaces
    const float veil_strength = 0.032f * matte_factor;

    std::array<float, 3> spec_and_veil{};
    if (is_mirror_chrome_top) {
        // Mirror Chrome reflection: reflect(-V, N) = (2*N.z*N.x, 2*N.z*N.y, 2*N.z^2 - 1) for V=(0,0,1)
        const std::array<float, 3> refl_dir = {
            2.0f * ndotv * normal_vec[0],
            2.0f * ndotv * normal_vec[1],
            2.0f * ndotv * ndotv - 1.0f
        };
        const auto env_rgb = sample_studio_env_map(refl_dir, alpha_eff, ao_val);
        // Schlick conductor Fresnel F(N.V, F_0)
        const float fresnel = chrome_f0 + (1.0f - chrome_f0) * std::pow(1.0f - ndotv, 5.0f);
        const float shade_lum = 0.2126f * shading_d[0] + 0.7152f * shading_d[1] + 0.0722f * shading_d[2];
        const float env_mod = 0.35f + 0.65f * std::clamp(shade_lum, 0.1f, 1.6f);
        for (int c = 0; c < 3; ++c) {
            spec_and_veil[c] = (fresnel * env_rgb[c] * env_mod + specular_s[c] * spec_scale) * clear_transmittance[c];
        }
    } else if (is_metallic_under) {
        // Metallic conductor reflection + micro-flake glint transmitted through optional clear overcoat + top clear gloss
        const float ndoth = std::clamp(-0.24f * normal_vec[0] - 0.34f * normal_vec[1] + 0.91f * ndotv, 0.0f, 1.0f);
        const float metal_lobe = std::pow(ndoth, 5.0f) * std::clamp(shading_d[1], 0.15f, 1.6f);
        const float top_wet_fresnel = has_clear_overcoat
            ? ((1.0f - 0.85f * matte_factor) * 0.16f * std::pow(ndoth, 24.0f) * std::clamp(shading_d[1], 0.45f, 1.6f))
            : 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float T_spec = std::sqrt(std::clamp(clear_transmittance[c], 0.0f, 1.0f));
            const float internal_metal_spec = metallic_f0_tint[c] * (0.38f * metal_lobe + 0.65f * specular_s[c]) * T_spec;
            const float top_dielec_spec = std::max(0.0f, specular_s[c] - 0.22f) * spec_scale + top_wet_fresnel;
            spec_and_veil[c] = internal_metal_spec + top_dielec_spec + veil_strength * shading_d[c];
        }
    } else {
        const float ndoth = std::clamp(-0.24f * normal_vec[0] - 0.34f * normal_vec[1] + 0.91f * ndotv, 0.0f, 1.0f);
        const float top_wet_fresnel = has_clear_overcoat
            ? ((1.0f - 0.85f * matte_factor) * 0.14f * std::pow(ndoth, 24.0f) * std::clamp(shading_d[1], 0.45f, 1.6f))
            : 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float top_glint = has_clear_overcoat ? (std::max(0.0f, specular_s[c] - 0.22f) * spec_scale + top_wet_fresnel) : 0.0f;
            spec_and_veil[c] = specular_s[c] * spec_scale * clear_transmittance[c] + top_glint + veil_strength * shading_d[c];
        }
    }

    // Lighting Environment Scaling (Standard Studio vs. Dark Room vs. Dark Room + 365nm UV)
    float kappa_amb = 1.0f;
    std::array<float, 3> T_leak = {1.0f, 1.0f, 1.0f};
    if (env_params.mode == LightingEnvMode::DARK_ROOM) {
        kappa_amb = std::clamp(env_params.ambient_dim_factor, 0.02f, 0.35f);
    } else if (env_params.mode == LightingEnvMode::DARK_UV_365NM) {
        kappa_amb = std::clamp(env_params.ambient_dim_factor, 0.02f, 0.15f);
        T_leak = {0.18f, 0.05f, 0.42f}; // Subtle 365nm deep-violet visible leak tint
    }

    // Attenuate bare-plastic residual R(x) as opaque paint thickness increases
    const float eta_coat = std::exp(-1.5f * total_optical_depth);

    std::array<float, 3> out_rgb{};
    for (int c = 0; c < 3; ++c) {
        const float diffuse_term = A_body[c] * (kappa_amb * shading_d[c] * T_leak[c]);
        const float spec_term    = kappa_amb * spec_and_veil[c];
        const float res_term     = eta_coat * kappa_amb * residual_r[c];
        out_rgb[c] = std::clamp(diffuse_term + spec_term + E_fluo[c] + res_term, 0.0f, 4.0f);
    }

    return {out_rgb, E_fluo, alpha_eff, undercoat_alpha};
}

// ============================================================================
// 6. Multi-Scale Separable Gaussian HDR Bloom for 365nm UV Fluorescence
// ============================================================================

inline void apply_separable_gaussian_1d(
    const float* src,
    float* dst,
    int width,
    int height,
    int sigma_px,
    bool horizontal
) noexcept {
    const int radius = std::min(sigma_px * 2, 64);
    std::vector<float> kernel(2 * radius + 1);
    float sum_w = 0.0f;
    const float inv_2sigma2 = 1.0f / (2.0f * static_cast<float>(sigma_px * sigma_px));
    for (int k = -radius; k <= radius; ++k) {
        const float w = std::exp(-static_cast<float>(k * k) * inv_2sigma2);
        kernel[k + radius] = w;
        sum_w += w;
    }
    for (float& w : kernel) w /= sum_w;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int nx = horizontal ? std::clamp(x + k, 0, width - 1) : x;
                const int ny = horizontal ? y : std::clamp(y + k, 0, height - 1);
                const size_t idx = 3 * (static_cast<size_t>(ny) * width + nx);
                const float w = kernel[k + radius];
                acc_r += src[idx + 0] * w;
                acc_g += src[idx + 1] * w;
                acc_b += src[idx + 2] * w;
            }
            const size_t out_idx = 3 * (static_cast<size_t>(y) * width + x);
            dst[out_idx + 0] = acc_r;
            dst[out_idx + 1] = acc_g;
            dst[out_idx + 2] = acc_b;
        }
    }
}

inline void add_multiscale_uv_bloom(
    float* io_linear_rgb,
    const float* fluo_emission_rgb,
    int width,
    int height,
    float bloom_strength = 0.65f,
    float glow_threshold = 0.05f
) {
    const size_t n_pix = static_cast<size_t>(width) * height;
    std::vector<float> bright_pass(3 * n_pix, 0.0f);
    bool has_glow = false;

    for (size_t i = 0; i < n_pix; ++i) {
        const float r = std::max(0.0f, fluo_emission_rgb[3 * i + 0] - glow_threshold);
        const float g = std::max(0.0f, fluo_emission_rgb[3 * i + 1] - glow_threshold);
        const float b = std::max(0.0f, fluo_emission_rgb[3 * i + 2] - glow_threshold);
        if (r > 0.0f || g > 0.0f || b > 0.0f) {
            bright_pass[3 * i + 0] = r;
            bright_pass[3 * i + 1] = g;
            bright_pass[3 * i + 2] = b;
            has_glow = true;
        }
    }
    if (!has_glow || bloom_strength <= 0.0f) return;

    std::vector<float> temp_buf(3 * n_pix, 0.0f);
    std::vector<float> blur_out(3 * n_pix, 0.0f);

    // 3-scale Gaussian bloom (scaled appropriately for image dimensions)
    const int scale_div = std::max(1, 2048 / std::max(width, height));
    const std::array<int, 3> sigmas = {
        std::max(2, 8 / scale_div),
        std::max(4, 24 / scale_div),
        std::max(8, 64 / scale_div)
    };
    const std::array<float, 3> weights = {0.50f, 0.32f, 0.18f};

    for (size_t m = 0; m < 3; ++m) {
        apply_separable_gaussian_1d(bright_pass.data(), temp_buf.data(), width, height, sigmas[m], true);
        apply_separable_gaussian_1d(temp_buf.data(), blur_out.data(), width, height, sigmas[m], false);
        const float wm = bloom_strength * weights[m];
        for (size_t i = 0; i < 3 * n_pix; ++i) {
            io_linear_rgb[i] = std::clamp(io_linear_rgb[i] + wm * blur_out[i], 0.0f, 1.0f);
        }
    }
}

// ============================================================================
// 7. Full-Image Tiled Renderer (OpenMP / Multi-Core CPU)
// ============================================================================

inline void render_stack_tiled_omp(
    const float* orig_wb_hwc,
    const float* albedo_base_hwc,
    const float* shading_d_hwc,
    const float* specular_s_hwc,
    const float* normals_ao_hwc4,      // [nx, ny, nz, ao] per pixel (or nullptr for default normal)
    const uint8_t* panel_mask_hw,      // [0..255] per pixel (or nullptr)
    const uint16_t* region_id_map_hw,  // region_id per pixel
    int width,
    int height,
    const std::vector<RegionRenderStack>& region_stacks,
    const LightingEnvParams& env_params,
    float* out_linear_rgb_hwc
) {
    const size_t n_pix = static_cast<size_t>(width) * height;
    std::vector<float> fluo_buf(3 * n_pix, 0.0f);
    const size_t num_regions = region_stacks.size();

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const size_t i3 = 3 * i;
            const uint16_t rid = region_id_map_hw ? region_id_map_hw[i] : 0;

            const std::array<float, 3> orig_wb = {orig_wb_hwc[i3 + 0], orig_wb_hwc[i3 + 1], orig_wb_hwc[i3 + 2]};
            if (rid >= num_regions) {
                out_linear_rgb_hwc[i3 + 0] = orig_wb[0];
                out_linear_rgb_hwc[i3 + 1] = orig_wb[1];
                out_linear_rgb_hwc[i3 + 2] = orig_wb[2];
                continue;
            }

            const std::array<float, 3> albedo  = {albedo_base_hwc[i3 + 0], albedo_base_hwc[i3 + 1], albedo_base_hwc[i3 + 2]};
            const std::array<float, 3> shading = {shading_d_hwc[i3 + 0], shading_d_hwc[i3 + 1], shading_d_hwc[i3 + 2]};
            const std::array<float, 3> spec    = {specular_s_hwc[i3 + 0], specular_s_hwc[i3 + 1], specular_s_hwc[i3 + 2]};
            // Exact identity residual R(x) = I_wb(x) - (A_base(x) * S_d(x) + S_s(x))
            const std::array<float, 3> residual = {
                orig_wb[0] - (albedo[0] * shading[0] + spec[0]),
                orig_wb[1] - (albedo[1] * shading[1] + spec[1]),
                orig_wb[2] - (albedo[2] * shading[2] + spec[2])
            };

            std::array<float, 3> normal_vec = {0.0f, 0.0f, 1.0f};
            float ao_val = 1.0f;
            if (normals_ao_hwc4) {
                const size_t i4 = 4 * i;
                normal_vec = {normals_ao_hwc4[i4 + 0], normals_ao_hwc4[i4 + 1], normals_ao_hwc4[i4 + 2]};
                ao_val = normals_ao_hwc4[i4 + 3];
            }
            const float panel_val = panel_mask_hw ? (static_cast<float>(panel_mask_hw[i]) / 255.0f) : 0.0f;

            const auto res = evaluate_pixel_stack(
                orig_wb, albedo, shading, spec, residual,
                normal_vec, ao_val, panel_val,
                region_stacks[rid], env_params,
                x, y
            );

            const auto mapped_rgb = tone_map_highlight_shoulder(res.linear_rgb);
            out_linear_rgb_hwc[i3 + 0] = mapped_rgb[0];
            out_linear_rgb_hwc[i3 + 1] = mapped_rgb[1];
            out_linear_rgb_hwc[i3 + 2] = mapped_rgb[2];

            fluo_buf[i3 + 0] = res.fluo_emission[0];
            fluo_buf[i3 + 1] = res.fluo_emission[1];
            fluo_buf[i3 + 2] = res.fluo_emission[2];
        }
    }

    if (env_params.mode == LightingEnvMode::DARK_UV_365NM && env_params.uv_bloom_strength > 0.0f) {
        add_multiscale_uv_bloom(out_linear_rgb_hwc, fluo_buf.data(), width, height, env_params.uv_bloom_strength);
    }
}

} // namespace paintcore::physics
