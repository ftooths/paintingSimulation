#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

/**
 * WebGL2 Real-Time Kubelka-Munk, Intrinsic Gloss, Specialty Coating & 365nm UV Compositor
 * Specification: SDD Section 4.1 & SRS Section 3.6
 * Guarantees < 0.10 Delta E_00 numerical parity with C++20 `libpaintcore` (`km_brdf_solver.hpp`).
 */

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_orig_wb_tex;       // TEXTURE5: I_wb(x)
uniform sampler2D u_albedo_base_tex;   // TEXTURE0: A_base(x)
uniform sampler2D u_shading_tex;       // TEXTURE1: .rgb = 0.5 * S_d(x), .a = P_panel(x)
uniform sampler2D u_specular_tex;      // TEXTURE2: .rgb = S_s(x)
uniform sampler2D u_normals_ao_tex;    // TEXTURE3: .rgb = 0.5 * N(x) + 0.5, .a = AO(x)
uniform usampler2D u_region_id_tex;    // TEXTURE4: .r = region_id
uniform sampler2D u_region_params_tex; // TEXTURE6: 24 x N_regions RGBA32F parameter lookup texture

uniform int u_lighting_mode;           // 0 = STANDARD_STUDIO, 1 = DARK_ROOM, 2 = DARK_UV_365NM
uniform float u_ambient_dim;           // kappa_amb
uniform float u_uv_intensity;          // I_uv,0
uniform vec3 u_uv_light_dir;

float evaluate_km_channel(float R_prev, float a, float b, float SX) {
    R_prev = clamp(R_prev, 0.001, 0.995);
    if (SX <= 1e-5) return R_prev;
    if (b < 1e-3) {
        float delta = 1.0 - R_prev;
        return clamp((R_prev + SX * delta) / (1.0 + SX * delta), 0.001, 0.995);
    }
    float beta = b * SX;
    float b_coth_beta = (beta >= 10.0)
        ? b
        : b * ((exp(2.0 * beta) + 1.0) / max(exp(2.0 * beta) - 1.0, 1e-7));
    float num = 1.0 - R_prev * (a - b_coth_beta);
    float den = a - R_prev + b_coth_beta;
    return clamp(num / max(den, 1e-6), 0.001, 0.995);
}

vec3 apply_saunderson(vec3 R_in, float k1, float k2) {
    vec3 r = clamp(R_in, vec3(0.001), vec3(0.995));
    return ((1.0 - k1) * (1.0 - k2) * r) / max(vec3(1.0) - k2 * r, vec3(1e-4));
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, vec3(0.0), vec3(1.0));
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, lessThanEqual(c, vec3(0.0031308)));
}

void main() {
    vec3 orig_wb = texture(u_orig_wb_tex, v_uv).rgb;
    uint rid = texture(u_region_id_tex, v_uv).r;
    int r_row = int(rid);

    // Fetch Texel (0, r): [is_modified, num_active_layers, topcoat_type, effective_ggx_roughness]
    vec4 t0 = texelFetch(u_region_params_tex, ivec2(0, r_row), 0);
    bool is_modified = (t0.r > 0.5);
    int num_layers = int(t0.g + 0.5);
    int topcoat_type = int(t0.b + 0.5);
    float alpha_eff = t0.a;

    if (!is_modified && u_lighting_mode == 0) {
        fragColor = vec4(linear_to_srgb(orig_wb), 1.0);
        return;
    }

    vec3 albedo_base = texture(u_albedo_base_tex, v_uv).rgb;
    vec4 sh_pack = texture(u_shading_tex, v_uv);
    vec3 shading_d = sh_pack.rgb * 2.0;
    vec3 spec_s = texture(u_specular_tex, v_uv).rgb;
    vec4 norm_ao = texture(u_normals_ao_tex, v_uv);
    vec3 normal_v = normalize(norm_ao.rgb * 2.0 - 1.0);
    float ao_val = norm_ao.a;

    vec3 residual_r = orig_wb - (albedo_base * shading_d + spec_s);
    vec3 R_curr = albedo_base;
    vec3 E_fluo = vec3(0.0);

    float ndotl_uv = max(0.0, dot(normal_v, u_uv_light_dir));
    float uv_irrad = (u_lighting_mode == 2) ? (u_uv_intensity * (0.35 * ao_val + 0.65 * ndotl_uv)) : 0.0;
    float w_vis = (u_lighting_mode == 0) ? 0.22 : 0.04;
    float w_uv  = (u_lighting_mode == 2) ? 1.35 : 0.0;
    float E_ex  = w_vis * shading_d.b + w_uv * uv_irrad;

    for (int k = 0; k < 6; ++k) {
        if (k >= num_layers) break;
        vec4 texA = texelFetch(u_region_params_tex, ivec2(3 + 3 * k, r_row), 0);
        vec4 texB = texelFetch(u_region_params_tex, ivec2(4 + 3 * k, r_row), 0);
        vec4 texC = texelFetch(u_region_params_tex, ivec2(5 + 3 * k, r_row), 0);
        int role = int(texA.a + 0.5);
        if (role == 4 || role == 8) {
            // Beer-Lambert Clear / Fluorescent Clear
            R_curr = clamp(R_curr * exp(-2.0 * texC.rgb), vec3(0.001), vec3(0.995));
        } else {
            R_curr.r = evaluate_km_channel(R_curr.r, texA.r, texB.r, texC.r);
            R_curr.g = evaluate_km_channel(R_curr.g, texA.g, texB.g, texC.g);
            R_curr.b = evaluate_km_channel(R_curr.b, texA.b, texB.b, texC.b);
        }
        if (role == 7 || role == 8) {
            vec3 fluo_peak = texelFetch(u_region_params_tex, ivec2(23, r_row), 0).rgb;
            E_fluo += texC.a * E_ex * fluo_peak;
        }
    }

    vec4 t1 = texelFetch(u_region_params_tex, ivec2(1, r_row), 0);
    vec3 A_body = (num_layers > 0 || topcoat_type != 0)
        ? apply_saunderson(R_curr, t1.r, t1.g)
        : R_curr;

    float gloss_factor = clamp((0.22 - alpha_eff) / 0.18, -1.0, 1.0);
    float matte_factor = clamp((alpha_eff - 0.22) / 0.50, 0.0, 1.0);
    float spec_scale = (gloss_factor >= 0.0) ? (1.0 + 0.65 * gloss_factor) : (1.0 - 0.78 * matte_factor);
    float veil_strength = 0.032 * matte_factor;

    float kappa_amb = (u_lighting_mode == 0) ? 1.0 : u_ambient_dim;
    vec3 T_leak = (u_lighting_mode == 2) ? vec3(0.18, 0.05, 0.42) : vec3(1.0);

    vec3 out_lin = A_body * (kappa_amb * shading_d * T_leak)
                 + kappa_amb * (spec_s * spec_scale + veil_strength * shading_d)
                 + E_fluo
                 + (num_layers > 0 ? 0.05 : 1.0) * kappa_amb * residual_r;

    fragColor = vec4(linear_to_srgb(clamp(out_lin, vec3(0.0), vec3(1.0))), 1.0);
}
