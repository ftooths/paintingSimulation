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

const float DEG_TO_RAD = 3.14159265358979323846 / 180.0;

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

vec3 oklch_to_linear_srgb(vec3 lch) {
    float h_rad = lch.z * DEG_TO_RAD;
    float L = lch.x;
    float a = lch.y * cos(h_rad);
    float b = lch.y * sin(h_rad);
    float l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    float m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    float s_ = L - 0.0894841775 * a - 1.2914855480 * b;
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;
    return clamp(vec3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    ), vec3(0.0), vec3(1.0));
}

vec3 lerp_oklch(vec3 c0, vec3 c1, float t) {
    t = clamp(t, 0.0, 1.0);
    float dh = c1.z - c0.z;
    if (dh > 180.0) dh -= 360.0;
    else if (dh < -180.0) dh += 360.0;
    float h = c0.z + t * dh;
    if (h < 0.0) h += 360.0;
    if (h >= 360.0) h -= 360.0;
    return vec3(mix(c0.xy, c1.xy, t), h);
}

vec3 sample_studio_env_map(vec3 reflect_dir, float roughness_alpha, float ao) {
    float rx = reflect_dir.x;
    float ry = reflect_dir.y;
    float rz = reflect_dir.z;
    float sharpness = max(1.0, 32.0 * (1.0 - clamp(roughness_alpha * 1.4, 0.0, 0.96)));
    float key_lobe = pow(max(0.0, -0.65 * ry + 0.65 * rz + 0.35 * rx), sharpness);
    float rim_lobe = pow(max(0.0, 0.45 * ry + 0.70 * abs(rx) + 0.40 * rz), sharpness * 0.6);
    float horizon_band = exp(-abs(ry + 0.1) * (8.0 / (1.0 + 6.0 * roughness_alpha)));
    float blur_mix = clamp(roughness_alpha * 1.35, 0.0, 1.0);
    float crisp_env = 0.08 + 1.15 * key_lobe + 0.55 * rim_lobe + 0.25 * horizon_band;
    float blurred_env = 0.44 + 0.12 * (-ry);
    float env_lum = ((1.0 - blur_mix) * crisp_env + blur_mix * blurred_env) * (0.4 + 0.6 * ao);
    return clamp(vec3(env_lum * 0.97, env_lum * 0.99, env_lum * 1.03), vec3(0.01), vec3(1.5));
}

vec3 tone_map_highlight_shoulder(vec3 rgb) {
    vec3 c = max(vec3(0.0), rgb);
    float max_ch = max(c.r, max(c.g, c.b));
    if (max_ch <= 1.0) return c;
    return clamp(c / max_ch, vec3(0.0), vec3(1.0));
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
    float ndotv = clamp(normal_v.z, 0.0, 1.0);

    vec3 residual_r = orig_wb - (albedo_base * shading_d + spec_s);
    vec3 R_curr = albedo_base;
    vec3 E_fluo = vec3(0.0);
    vec3 clear_transmittance = vec3(1.0);
    vec3 metallic_f0_tint = vec3(0.0);
    bool is_mirror_chrome_top = false;
    bool is_metallic_under = false;
    bool has_clear_overcoat = false;
    float chrome_f0 = 0.95;
    float total_optical_depth = 0.0;

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
        float eff_coats = max(0.0, texB.a);

        if (role == 0 || role == 1) {
            // SURFACER or BASE_SOLID
            R_curr.r = evaluate_km_channel(R_curr.r, texA.r, texB.r, texC.r);
            R_curr.g = evaluate_km_channel(R_curr.g, texA.g, texB.g, texC.g);
            R_curr.b = evaluate_km_channel(R_curr.b, texA.b, texB.b, texC.b);
            total_optical_depth += max(eff_coats, 1.0);
            is_mirror_chrome_top = false;
            is_metallic_under = false;
        } else if (role == 2) {
            // METALLIC
            R_curr.r = evaluate_km_channel(R_curr.r, texA.r, texB.r, texC.r);
            R_curr.g = evaluate_km_channel(R_curr.g, texA.g, texB.g, texC.g);
            R_curr.b = evaluate_km_channel(R_curr.b, texA.b, texB.b, texC.b);
            vec4 t_f0 = texelFetch(u_region_params_tex, ivec2(2, r_row), 0);
            metallic_f0_tint = t_f0.rgb;
            total_optical_depth += max(eff_coats, 1.0);
            is_mirror_chrome_top = false;
            is_metallic_under = true;
        } else if (role == 3) {
            // MIRROR_CHROME
            R_curr *= 0.03;
            is_mirror_chrome_top = true;
            is_metallic_under = false;
            chrome_f0 = (texC.a > 0.0) ? texC.a : 0.95;
            total_optical_depth += 3.0;
        } else if (role == 4 || role == 8) {
            // CLEAR_CANDY or FLUORESCENT_CLEAR (Beer-Lambert)
            float path_scale = 1.0 / max(0.45, ndotv);
            vec3 trans = exp(-2.0 * texC.rgb * path_scale);
            R_curr = clamp(R_curr * trans, vec3(0.001), vec3(0.995));
            clear_transmittance *= trans;
            has_clear_overcoat = true;
            total_optical_depth += (role == 4 ? 0.35 : 0.25) * max(eff_coats, 1.0);
            if (role == 8) {
                vec3 fluo_peak = texelFetch(u_region_params_tex, ivec2(23, r_row), 0).rgb;
                E_fluo += texC.a * E_ex * fluo_peak;
            }
        } else if (role == 5 || role == 6) {
            // CHAMELEON or PEARL (OKLCh 3-stop view-angle shift)
            float u = 1.0 - ndotv;
            vec3 stop0  = texA.rgb;
            vec3 stop45 = texB.rgb;
            vec3 stop80 = texC.rgb;
            vec3 lch_interp = (u <= 0.30)
                ? lerp_oklch(stop0, stop45, u / 0.30)
                : lerp_oklch(stop45, stop80, (u - 0.30) / 0.53);
            vec3 cham_rgb = oklch_to_linear_srgb(lch_interp);
            float cov = clamp(texC.a, 0.0, 1.0);
            R_curr = clamp(R_curr * (1.0 - 0.85 * cov) + cham_rgb * cov, vec3(0.002), vec3(0.992));
            total_optical_depth += max(eff_coats, 1.0);
            is_mirror_chrome_top = false;
            is_metallic_under = false;
        } else if (role == 7) {
            // FLUORESCENT_SOLID
            R_curr.r = evaluate_km_channel(R_curr.r, texA.r, texB.r, texC.r);
            R_curr.g = evaluate_km_channel(R_curr.g, texA.g, texB.g, texC.g);
            R_curr.b = evaluate_km_channel(R_curr.b, texA.b, texB.b, texC.b);
            float sub_lum = dot(R_curr, vec3(0.2126, 0.7152, 0.0722));
            float sub_boost = 0.55 + 0.65 * sqrt(clamp(sub_lum, 0.05, 1.0));
            vec3 fluo_peak = texelFetch(u_region_params_tex, ivec2(23, r_row), 0).rgb;
            E_fluo += texC.a * E_ex * sub_boost * fluo_peak;
            total_optical_depth += max(eff_coats, 1.0);
            is_mirror_chrome_top = false;
            is_metallic_under = false;
        }
    }

    vec4 t1 = texelFetch(u_region_params_tex, ivec2(1, r_row), 0);
    vec3 A_body = R_curr;
    if (num_layers > 0 && !is_metallic_under) {
        A_body = apply_saunderson(R_curr, t1.r, t1.g);
    } else if (num_layers > 0 && is_metallic_under) {
        A_body = clamp(R_curr * 0.88, vec3(0.001), vec3(0.995));
    } else if (topcoat_type == 1) {
        A_body = R_curr * 0.90;
    }

    float gloss_factor = clamp((0.22 - alpha_eff) / 0.18, -1.0, 1.0);
    float matte_factor = clamp((alpha_eff - 0.22) / 0.50, 0.0, 1.0);
    float spec_scale = (gloss_factor >= 0.0) ? (1.0 + 0.65 * gloss_factor) : (1.0 - 0.78 * matte_factor);
    float veil_strength = 0.032 * matte_factor;

    vec3 spec_and_veil = vec3(0.0);
    float ndoth = clamp(-0.24 * normal_v.x - 0.34 * normal_v.y + 0.91 * ndotv, 0.0, 1.0);
    if (is_mirror_chrome_top) {
        vec3 refl_dir = vec3(2.0 * ndotv * normal_v.x, 2.0 * ndotv * normal_v.y, 2.0 * ndotv * ndotv - 1.0);
        vec3 env_rgb = sample_studio_env_map(refl_dir, alpha_eff, ao_val);
        float fresnel = chrome_f0 + (1.0 - chrome_f0) * pow(1.0 - ndotv, 5.0);
        float shade_lum = dot(shading_d, vec3(0.2126, 0.7152, 0.0722));
        float env_mod = 0.35 + 0.65 * clamp(shade_lum, 0.1, 1.6);
        spec_and_veil = (fresnel * env_rgb * env_mod + spec_s * spec_scale) * clear_transmittance;
    } else if (is_metallic_under) {
        float metal_lobe = pow(ndoth, 5.0) * clamp(shading_d.g, 0.15, 1.6);
        float top_wet_fresnel = has_clear_overcoat
            ? ((1.0 - 0.85 * matte_factor) * 0.16 * pow(ndoth, 24.0) * clamp(shading_d.g, 0.45, 1.6))
            : 0.0;
        vec3 T_spec = sqrt(clamp(clear_transmittance, vec3(0.0), vec3(1.0)));
        vec3 internal_metal_spec = metallic_f0_tint * (0.38 * metal_lobe + 0.65 * spec_s) * T_spec;
        vec3 top_dielec_spec = max(vec3(0.0), spec_s - vec3(0.22)) * spec_scale + vec3(top_wet_fresnel);
        spec_and_veil = internal_metal_spec + top_dielec_spec + veil_strength * shading_d;
    } else {
        float top_wet_fresnel = has_clear_overcoat
            ? ((1.0 - 0.85 * matte_factor) * 0.14 * pow(ndoth, 24.0) * clamp(shading_d.g, 0.45, 1.6))
            : 0.0;
        vec3 top_glint = has_clear_overcoat ? (max(vec3(0.0), spec_s - vec3(0.22)) * spec_scale + vec3(top_wet_fresnel)) : vec3(0.0);
        spec_and_veil = spec_s * spec_scale * clear_transmittance + top_glint + veil_strength * shading_d;
    }

    float kappa_amb = (u_lighting_mode == 0) ? 1.0 : u_ambient_dim;
    vec3 T_leak = (u_lighting_mode == 2) ? vec3(0.18, 0.05, 0.42) : vec3(1.0);
    float eta_coat = exp(-1.5 * total_optical_depth);

    vec3 out_lin = clamp(
        A_body * (kappa_amb * shading_d * T_leak)
        + kappa_amb * spec_and_veil
        + E_fluo
        + eta_coat * kappa_amb * residual_r,
        vec3(0.0),
        vec3(4.0)
    );

    fragColor = vec4(linear_to_srgb(tone_map_highlight_shoulder(out_lin)), 1.0);
}
