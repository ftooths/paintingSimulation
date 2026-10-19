#include "paintcore/types.hpp"
#include "paintcore/color_spaces.hpp"
#include "paintcore/km_brdf_solver.hpp"
#include "paintcore/panel_snap_matting.hpp"
#include "paintcore/albedo_regularizer.hpp"

#if defined(_WIN32)
#define PAINTCORE_EXPORT extern "C" __declspec(dllexport)
#else
#define PAINTCORE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using namespace paintcore;

PAINTCORE_EXPORT float paintcore_srgb_to_linear(float c_srgb) {
    return color::srgb_to_linear_exact(c_srgb);
}

PAINTCORE_EXPORT float paintcore_linear_to_srgb(float c_lin) {
    return color::linear_to_srgb_exact(c_lin);
}

PAINTCORE_EXPORT float paintcore_delta_e00_linear_rgb(
    float r1, float g1, float b1,
    float r2, float g2, float b2
) {
    return color::delta_e00_linear_rgb({r1, g1, b1}, {r2, g2, b2});
}

PAINTCORE_EXPORT float paintcore_evaluate_km_channel(
    float R_prev, float a, float b, float SX
) {
    return physics::evaluate_km_channel(R_prev, a, b, SX);
}

PAINTCORE_EXPORT uintptr_t paintcore_verify_zero_copy_ptr(float* buffer_ptr) {
    if (buffer_ptr) {
        // Perform an in-place volatile read-write barrier to prove zero-copy shared memory access
        const float orig = buffer_ptr[0];
        buffer_ptr[0] = orig;
    }
    return reinterpret_cast<uintptr_t>(buffer_ptr);
}

PAINTCORE_EXPORT void paintcore_populate_km_params(PaintLayerParams* layer) {
    if (layer) {
        physics::populate_km_parameters(*layer);
    }
}

PAINTCORE_EXPORT float paintcore_resolve_stack_roughness(
    const RegionRenderStack* stack,
    float* out_undercoat_alpha
) {
    if (!stack) return 0.22f;
    float under = 0.22f;
    const float eff = physics::resolve_layer_stack_roughness(*stack, under);
    if (out_undercoat_alpha) *out_undercoat_alpha = under;
    return eff;
}

PAINTCORE_EXPORT void paintcore_render_image(
    const float* orig_wb_hwc,
    const float* albedo_base_hwc,
    const float* shading_d_hwc,
    const float* specular_s_hwc,
    const float* normals_ao_hwc4,
    const uint8_t* panel_mask_hw,
    const uint16_t* region_id_map_hw,
    int width,
    int height,
    const RegionRenderStack* stacks_array,
    int num_regions,
    const LightingEnvParams* env_params,
    float* out_linear_rgb_hwc
) {
    std::vector<RegionRenderStack> stacks(stacks_array, stacks_array + num_regions);
    for (auto& st : stacks) {
        for (uint8_t k = 0; k < st.num_active_layers && k < MAX_STACK_LAYERS; ++k) {
            if (st.layers[k].enabled) {
                physics::populate_km_parameters(st.layers[k]);
            }
        }
    }
    LightingEnvParams env = env_params ? *env_params : LightingEnvParams{};
    physics::render_stack_tiled_omp(
        orig_wb_hwc, albedo_base_hwc, shading_d_hwc, specular_s_hwc,
        normals_ao_hwc4, panel_mask_hw, region_id_map_hw,
        width, height, stacks, env, out_linear_rgb_hwc
    );
}

PAINTCORE_EXPORT void paintcore_eval_pixel_stack_flat(
    const float* orig_wb_3,
    const float* albedo_3,
    const float* shading_3,
    const float* spec_3,
    const float* normal_3,
    float ao_val,
    float panel_val,
    int is_modified,
    int num_layers,
    const int* roles,
    const float* r_inf_flat,
    const float* hiding_flat,
    const float* alpha_flat,
    const float* coats_flat,
    const float* opacity_flat,
    const float* beer_flat,
    const float* cham_flat,
    const float* fluo_yield_flat,
    const float* fluo_peak_flat,
    int topcoat_type,
    float topcoat_roughness,
    int env_mode,
    float ambient_dim,
    float uv_intensity,
    int px_x,
    int px_y,
    float* out_linear_rgb_3,
    float* out_effective_roughness
) {
    RegionRenderStack st{};
    st.is_modified = (is_modified != 0);
    st.num_active_layers = static_cast<uint8_t>(std::clamp(num_layers, 0, static_cast<int>(MAX_STACK_LAYERS)));
    st.finish.topcoat_type = static_cast<TopcoatType>(std::clamp(topcoat_type, 0, 3));
    st.finish.topcoat_roughness = topcoat_roughness;

    for (uint8_t k = 0; k < st.num_active_layers; ++k) {
        auto& L = st.layers[k];
        L.enabled = true;
        L.role = static_cast<LayerRole>(roles ? roles[k] : 1);
        if (r_inf_flat) {
            L.masstone_R_inf = {r_inf_flat[3 * k + 0], r_inf_flat[3 * k + 1], r_inf_flat[3 * k + 2]};
        }
        if (hiding_flat) {
            L.hiding_power_h = {hiding_flat[3 * k + 0], hiding_flat[3 * k + 1], hiding_flat[3 * k + 2]};
        }
        L.intrinsic_roughness = alpha_flat ? alpha_flat[k] : 0.08f;
        L.coat_count = coats_flat ? coats_flat[k] : 1.0f;
        L.opacity_override = opacity_flat ? opacity_flat[k] : 1.0f;
        if (beer_flat) {
            L.beer_alpha = {beer_flat[3 * k + 0], beer_flat[3 * k + 1], beer_flat[3 * k + 2]};
        }
        if (cham_flat) {
            for (int j = 0; j < 9; ++j) L.chameleon_stops[j] = cham_flat[9 * k + j];
        }
        L.fluo_quantum_yield = fluo_yield_flat ? fluo_yield_flat[k] : 0.0f;
        if (fluo_peak_flat) {
            L.fluo_peak_rgb = {fluo_peak_flat[3 * k + 0], fluo_peak_flat[3 * k + 1], fluo_peak_flat[3 * k + 2]};
        }
        physics::populate_km_parameters(L);
    }

    LightingEnvParams env{};
    env.mode = static_cast<LightingEnvMode>(std::clamp(env_mode, 0, 2));
    env.ambient_dim_factor = ambient_dim;
    env.uv_intensity = uv_intensity;

    const std::array<float, 3> orig_wb = {orig_wb_3[0], orig_wb_3[1], orig_wb_3[2]};
    const std::array<float, 3> albedo  = {albedo_3[0], albedo_3[1], albedo_3[2]};
    const std::array<float, 3> shading = {shading_3[0], shading_3[1], shading_3[2]};
    const std::array<float, 3> spec    = {spec_3[0], spec_3[1], spec_3[2]};
    const std::array<float, 3> residual = {
        orig_wb[0] - (albedo[0] * shading[0] + spec[0]),
        orig_wb[1] - (albedo[1] * shading[1] + spec[1]),
        orig_wb[2] - (albedo[2] * shading[2] + spec[2])
    };
    const std::array<float, 3> normal_v = normal_3
        ? std::array<float, 3>{normal_3[0], normal_3[1], normal_3[2]}
        : std::array<float, 3>{0.0f, 0.0f, 1.0f};

    const auto res = physics::evaluate_pixel_stack(
        orig_wb, albedo, shading, spec, residual,
        normal_v, ao_val, panel_val, st, env, px_x, px_y
    );
    const auto mapped = physics::tone_map_highlight_shoulder(res.linear_rgb);
    if (out_linear_rgb_3) {
        out_linear_rgb_3[0] = mapped[0];
        out_linear_rgb_3[1] = mapped[1];
        out_linear_rgb_3[2] = mapped[2];
    }
    if (out_effective_roughness) {
        *out_effective_roughness = res.effective_roughness;
    }
}
