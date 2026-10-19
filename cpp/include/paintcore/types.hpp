#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>

namespace paintcore {

enum class LayerRole : uint8_t {
    SURFACER          = 0, // #500, #1000, #1200, #1500 or Gloss Black Surfacer
    BASE_SOLID        = 1, // Solid pigment paint (natively Gloss, Semi-Gloss, or Matte)
    METALLIC          = 2, // Standard metallic flake (e.g., C8 Silver, C9 Gold)
    MIRROR_CHROME     = 3, // Ultra-reflective conductor (e.g., SM206), coupled to undercoat alpha
    CLEAR_CANDY       = 4, // Beer-Lambert transparent dye coat (e.g., C47 Clear Red)
    CHAMELEON         = 5, // View-angle (N.V) 3-stop interference color shift
    PEARL             = 6, // Translucent interference mica overcoat
    FLUORESCENT_SOLID = 7, // K-M diffuse + Stokes UV/blue fluorescence emission
    FLUORESCENT_CLEAR = 8  // Beer-Lambert clear tint + Stokes UV/blue fluorescence emission
};

enum class TopcoatType : uint8_t {
    NONE_INHERIT     = 0, // No separate topcoat -> inherit topmost active layer's intrinsic roughness!
    GLOSS            = 1, // Clear Gloss topcoat override (alpha ~ 0.05, veil ~ 0.0)
    SEMI_GLOSS       = 2, // Clear Semi-Gloss topcoat override (alpha ~ 0.28, veil ~ 0.25)
    MATTE_FLAT       = 3  // Clear Matte/Flat topcoat override (alpha ~ 0.75, veil ~ 1.0)
};

enum class LightingEnvMode : uint8_t {
    STANDARD_STUDIO  = 0, // Full white-balanced studio illumination
    DARK_ROOM        = 1, // Dimmed ambient illumination (kappa_amb in [0.05, 0.25])
    DARK_UV_365NM    = 2  // Dark room + 365nm UV Blacklight excitation + multi-scale HDR bloom
};

// Precomputed optical constants for a single coating layer
struct alignas(32) PaintLayerParams {
    std::array<float, 3> masstone_R_inf = {0.85f, 0.85f, 0.85f}; // R_infinity in Linear RGB [0.002, 0.992]
    std::array<float, 3> hiding_power_h = {0.75f, 0.75f, 0.75f}; // Contrast ratio at 1 coat [0.05, 0.995]
    std::array<float, 3> km_a           = {1.0f, 1.0f, 1.0f};    // a = 0.5 * (R_inf + 1/R_inf)
    std::array<float, 3> km_b           = {0.0f, 0.0f, 0.0f};    // b = sqrt(a^2 - 1)
    std::array<float, 3> km_SX_1coat    = {1.0f, 1.0f, 1.0f};    // Optical scattering thickness S * X for 1 coat
    std::array<float, 3> beer_alpha     = {0.0f, 0.0f, 0.0f};    // Absorption coeff for CLEAR_CANDY / FLUORESCENT_CLEAR

    // Intrinsic Surface Sheen (Used when TopcoatType == NONE_INHERIT or as undercoat for Chrome)
    float intrinsic_roughness = 0.08f;     // Native or user-adjusted GGX alpha in [0.03, 0.90]
    uint16_t surfacer_grit    = 0;         // 500, 1000, 1200, 1500 (0 if non-surfacer)

    // Specialty Coating Parameters
    float chrome_f0           = 0.95f;     // Conductor Fresnel F_0 for MIRROR_CHROME
    std::array<float, 9> chameleon_stops = {
        0.65f, 0.18f, 210.0f,  // 0 deg (normal incidence) OKLCh: L, C, h_deg
        0.52f, 0.22f, 265.0f,  // 45 deg (mid angle) OKLCh: L, C, h_deg
        0.48f, 0.24f, 325.0f   // 80 deg (grazing angle) OKLCh: L, C, h_deg
    };
    float fluo_quantum_yield  = 0.0f;      // Phi_F in [0.0, 1.0] for FLUORESCENT_*
    std::array<float, 3> fluo_peak_rgb = {0.0f, 0.0f, 0.0f}; // Normalized Stokes emission color C_peak

    float coat_count          = 1.0f;      // User multiplier n_coats in [0.1, 5.0]
    float opacity_override    = 1.0f;      // Optional user multiplier on SX_1coat [0.2, 2.0]
    LayerRole role            = LayerRole::BASE_SOLID;
    bool enabled              = false;
};

// Topcoat & Surface Reflectance Parameters
struct alignas(32) FinishParams {
    TopcoatType topcoat_type = TopcoatType::NONE_INHERIT;
    float topcoat_roughness  = 0.06f;      // Used when topcoat_type != NONE_INHERIT
    float saunderson_k1      = 0.04f;      // External Fresnel surface reflection (~4%)
    float saunderson_k2      = 0.60f;      // Internal diffuse reflection factor (~60%)
    float metallic_ness      = 0.0f;       // [0.0, 1.0]
    std::array<float, 3> metallic_f0 = {0.9f, 0.9f, 0.9f};
};

// Global Viewing & Lighting Environment Parameters
struct alignas(32) LightingEnvParams {
    LightingEnvMode mode     = LightingEnvMode::STANDARD_STUDIO;
    float ambient_dim_factor = 1.0f;       // kappa_amb: 1.0 (Standard), 0.12 (Dark), 0.05 (UV)
    float uv_intensity       = 0.0f;       // I_uv,0 in [0.0, 2.5] when mode == DARK_UV_365NM
    float uv_bloom_strength  = 0.65f;      // HDR bloom halo multiplier
    std::array<float, 3> uv_light_dir = {0.2f, -0.5f, 0.84f};
};

// Resolved stack for a single atomic region r = (part_id, color_group_id)
constexpr size_t MAX_STACK_LAYERS = 6;

struct alignas(32) RegionRenderStack {
    uint16_t region_id          = 0;
    uint16_t part_id            = 0;
    uint16_t color_group_id     = 0;
    bool is_protected           = false;
    bool is_modified            = false;   // False -> fast path (exact identity in Standard mode)
    uint8_t num_active_layers   = 0;       // 0 if only Topcoat is applied over bare plastic!
    float bare_plastic_roughness = 0.22f;  // Default satin roughness of bare injection plastic
    std::array<PaintLayerParams, MAX_STACK_LAYERS> layers{};
    FinishParams finish{};
    bool wash_enabled           = false;
    bool wash_is_fluorescent    = false;   // True for UV-reactive enamel panel-line wash
    std::array<float, 3> wash_rgb = {0.03f, 0.03f, 0.03f};
    float wash_intensity        = 0.0f;
};

} // namespace paintcore
