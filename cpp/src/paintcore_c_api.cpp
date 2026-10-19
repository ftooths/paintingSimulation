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

PAINTCORE_EXPORT uintptr_t paintcore_verify_zero_copy_ptr(const float* buffer_ptr) {
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
