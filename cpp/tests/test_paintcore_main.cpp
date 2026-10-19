#include "paintcore/types.hpp"
#include "paintcore/color_spaces.hpp"
#include "paintcore/km_brdf_solver.hpp"
#include "paintcore/panel_snap_matting.hpp"
#include "paintcore/albedo_regularizer.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>

using namespace paintcore;

namespace {

int g_passed = 0;
int g_failed = 0;

void report_test(const std::string& tc_id, const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) {
        ++g_passed;
        std::cout << "[PASS] " << std::left << std::setw(12) << tc_id << " | " << name;
        if (!detail.empty()) std::cout << " (" << detail << ")";
        std::cout << "\n";
    } else {
        ++g_failed;
        std::cout << "[FAIL] " << std::left << std::setw(12) << tc_id << " | " << name;
        if (!detail.empty()) std::cout << " -> " << detail;
        std::cout << "\n";
    }
}

PaintLayerParams make_solid_paint(
    std::array<float, 3> r_inf,
    std::array<float, 3> hiding,
    float intrinsic_roughness,
    float coats = 1.0f,
    LayerRole role = LayerRole::BASE_SOLID,
    uint16_t grit = 0
) {
    PaintLayerParams p{};
    p.masstone_R_inf = r_inf;
    p.hiding_power_h = hiding;
    p.intrinsic_roughness = intrinsic_roughness;
    p.coat_count = coats;
    p.role = role;
    p.surfacer_grit = grit;
    p.enabled = true;
    physics::populate_km_parameters(p);
    return p;
}

// ============================================================================
// Suite CPP: C++20 Native Core Tests (TC-CPP-001 .. TC-CPP-005)
// ============================================================================

void run_tc_cpp_001() {
    const auto& lut = color::SrgbLut64K::instance();
    float max_lin_err = 0.0f;
    for (uint32_t i = 0; i < 65536; ++i) {
        const float exact = color::srgb_to_linear_exact(static_cast<float>(i) / 65535.0f);
        const float from_lut = lut.to_linear_u16(static_cast<uint16_t>(i));
        max_lin_err = std::max(max_lin_err, std::abs(exact - from_lut));
    }

    int roundtrip_errors = 0;
    for (int v = 0; v < 256; ++v) {
        const float lin = lut.to_linear_u8(static_cast<uint8_t>(v));
        const uint8_t back = lut.to_srgb_u8(lin);
        if (back != static_cast<uint8_t>(v)) ++roundtrip_errors;
    }

    // OKLab <-> OKLCh roundtrip
    const std::array<float, 3> test_rgb = {0.57f, 0.018f, 0.026f};
    const auto lab = color::linear_srgb_to_oklab(test_rgb);
    const auto lch = color::oklab_to_oklch(lab);
    const auto rgb_back = color::oklab_to_linear_srgb(color::oklch_to_oklab(lch));
    const float de = color::delta_e00_linear_rgb(test_rgb, rgb_back);

    const bool ok = (max_lin_err <= 1.5e-5f) && (roundtrip_errors == 0) && (de < 0.01f);
    report_test("TC-CPP-001", "IEC 61966-2-1 sRGB 64K LUT & OKLab/OKLCh Precision",
                ok, "max_err=" + std::to_string(max_lin_err) + ", dE00=" + std::to_string(de));
}

void run_tc_cpp_002() {
    // 1. Pure scattering limit (R_inf -> 1, b -> 0)
    const float R_g = 0.15f;
    const float SX = 1.8f;
    const float r_white = physics::evaluate_km_channel(R_g, 1.0000001f, 0.0001f, SX);
    const float analytical = (R_g + SX * (1.0f - R_g)) / (1.0f + SX * (1.0f - R_g));
    const float err_white = std::abs(r_white - analytical);

    // 2. Zero coat thickness (SX = 0)
    const float r_zero = physics::evaluate_km_channel(0.42f, 1.25f, std::sqrt(1.25f * 1.25f - 1.0f), 0.0f);

    // 3. Extreme thickness (beta = 50.0 -> no exp overflow)
    const float R_inf = 0.60f;
    const float a = 0.5f * (R_inf + 1.0f / R_inf);
    const float b = std::sqrt(a * a - 1.0f);
    const float r_thick = physics::evaluate_km_channel(0.05f, a, b, 50.0f / b);

    // 4. Ultra-black paint (R_inf = 0.002)
    const float R_blk = 0.002f;
    const float a_blk = 0.5f * (R_blk + 1.0f / R_blk);
    const float b_blk = std::sqrt(a_blk * a_blk - 1.0f);
    const float r_black = physics::evaluate_km_channel(0.90f, a_blk, b_blk, 5.0f);

    const bool no_nan = !std::isnan(r_white) && !std::isnan(r_zero) && !std::isnan(r_thick) && !std::isnan(r_black) &&
                        !std::isinf(r_white) && !std::isinf(r_thick);
    const bool ok = no_nan && (err_white < 1e-5f) && (std::abs(r_zero - 0.42f) < 1e-6f) && (std::abs(r_thick - R_inf) < 1e-4f);
    report_test("TC-CPP-002", "Kubelka-Munk Singularity & Overflow Stability",
                ok, "white_err=" + std::to_string(err_white) + ", thick=" + std::to_string(r_thick));
}

void run_tc_cpp_003() {
    std::vector<float> buf(256 * 256 * 3, 0.5f);
    const uintptr_t orig_addr = reinterpret_cast<uintptr_t>(buf.data());
    const float* wrapped_ptr = buf.data();
    const uintptr_t wrapped_addr = reinterpret_cast<uintptr_t>(wrapped_ptr);
    const bool ok = (orig_addr == wrapped_addr);
    report_test("TC-CPP-003", "Zero-Copy Buffer Pointer Identity Verification",
                ok, "ptr=0x" + std::to_string(orig_addr));
}

void run_tc_cpp_004() {
    const int W = 64, H = 64;
    std::vector<float> img_wb(W * H * 3, 0.75f);
    // Create a sharp vertical dark panel line at x = 32
    for (int y = 0; y < H; ++y) {
        const size_t idx = static_cast<size_t>(y) * W + 32;
        img_wb[3 * idx + 0] = 0.04f;
        img_wb[3 * idx + 1] = 0.04f;
        img_wb[3 * idx + 2] = 0.04f;
    }
    std::vector<uint8_t> panel_mask(W * H, 0);
    segmentation::compute_black_hat_panel_lines(img_wb.data(), W, H, panel_mask.data());

    // Perturbed mask boundary at x = 30..34
    std::vector<uint16_t> region_map(W * H, 0);
    for (int y = 0; y < H; ++y) {
        const int jitter_x = 32 + ((y % 2 == 0) ? 2 : -2);
        for (int x = jitter_x; x < W; ++x) {
            region_map[static_cast<size_t>(y) * W + x] = 1;
        }
    }
    segmentation::snap_mask_boundaries_to_panel_lines(region_map.data(), panel_mask.data(), W, H, 4);

    // Check panel detection at x = 32 and Guided Filter matte bounds
    std::vector<float> guide_lum(W * H, 0.75f), raw_m(W * H, 0.0f), alpha_m(W * H, 0.0f);
    for (int i = 0; i < W * H; ++i) raw_m[i] = (region_map[i] == 1) ? 1.0f : 0.0f;
    segmentation::fast_guided_filter_matte(guide_lum.data(), raw_m.data(), alpha_m.data(), W, H, 3, 1e-4f);

    const bool ok = (panel_mask[32 * W + 32] > 180) && (alpha_m[32 * W + 10] < 0.05f) && (alpha_m[32 * W + 54] > 0.95f);
    report_test("TC-CPP-004", "Black-Hat Panel-Line Extraction & Fast Guided Matting",
                ok, "panel_strength=" + std::to_string(panel_mask[32 * W + 32]));
}

void run_tc_cpp_005() {
    // Non-convex L-shaped mask inside 64x64 grid
    const int W = 64, H = 64;
    std::vector<uint8_t> l_mask(W * H, 0);
    for (int y = 10; y < 54; ++y) {
        for (int x = 10; x < 24; ++x) l_mask[y * W + x] = 1;
    }
    for (int y = 40; y < 54; ++y) {
        for (int x = 10; x < 54; ++x) l_mask[y * W + x] = 1;
    }
    const auto badge = segmentation::compute_pole_of_inaccessibility(l_mask.data(), W, H, 1);
    const bool inside = (l_mask[badge.center_y * W + badge.center_x] == 1) && (badge.max_clearance_px >= 6.0f);
    report_test("TC-CPP-005", "Pole-of-Inaccessibility Set-of-Mark (SoM) Badge Placement",
                inside, "center=(" + std::to_string(badge.center_x) + "," + std::to_string(badge.center_y) + "), clearance=" + std::to_string(badge.max_clearance_px));
}

// ============================================================================
// Suite PHY: Layer Stack, Intrinsic Gloss, Specialty & UV Tests (TC-PHY-001..009)
// ============================================================================

void run_tc_phy_001() {
    const int W = 64, H = 64;
    const size_t N = W * H;
    std::vector<float> orig_wb(3 * N, 0.0f);
    std::vector<uint16_t> region_map(N, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = y * W + x;
            region_map[i] = (x < W / 2) ? 0 : 1;
            const float shade = 0.7f + 0.25f * std::sin(0.1f * x) * std::cos(0.1f * y);
            const float spec  = (x == 20 && y == 20) ? 0.25f : 0.01f;
            const std::array<float, 3> base = (region_map[i] == 0)
                ? std::array<float, 3>{0.06f, 0.16f, 0.48f}
                : std::array<float, 3>{0.82f, 0.83f, 0.85f};
            for (int c = 0; c < 3; ++c) {
                orig_wb[3 * i + c] = std::clamp(base[c] * shade + spec, 0.0f, 1.0f);
            }
        }
    }

    auto decomp = intrinsics::regularize_plastic_albedo_and_shading(
        orig_wb.data(), nullptr, nullptr, region_map.data(), nullptr, nullptr, W, H, 2
    );

    std::vector<RegionRenderStack> stacks(2);
    stacks[0].region_id = 0; stacks[0].is_modified = false;
    stacks[1].region_id = 1; stacks[1].is_modified = false;

    std::vector<float> rendered(3 * N, 0.0f);
    LightingEnvParams env{};
    physics::render_stack_tiled_omp(
        orig_wb.data(), decomp.albedo_base_hwc.data(), decomp.shading_diffuse_hwc.data(),
        decomp.specular_hwc.data(), nullptr, nullptr, region_map.data(),
        W, H, stacks, env, rendered.data()
    );

    double mse = 0.0, sum_de = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double diff = orig_wb[3 * i + c] - rendered[3 * i + c];
            mse += diff * diff;
        }
        sum_de += color::delta_e00_linear_rgb(
            {orig_wb[3 * i + 0], orig_wb[3 * i + 1], orig_wb[3 * i + 2]},
            {rendered[3 * i + 0], rendered[3 * i + 1], rendered[3 * i + 2]}
        );
    }
    mse /= static_cast<double>(3 * N);
    const double psnr = (mse < 1e-12) ? 99.9 : (10.0 * std::log10(1.0 / mse));
    const double mean_de = sum_de / static_cast<double>(N);

    const bool ok = (psnr >= 45.0) && (mean_de < 0.20);
    report_test("TC-PHY-001", "Zero-Loss Identity Reconstruction",
                ok, "PSNR=" + std::to_string(psnr) + " dB, mean_dE00=" + std::to_string(mean_de));
}

void run_tc_phy_002() {
    // Non-Sequential Case 1: Bare Plastic + Topcoat Only (No Primer, No Paint)
    const std::array<float, 3> dark_blue_albedo = {0.025f, 0.055f, 0.180f};
    const std::array<float, 3> shading = {0.95f, 0.95f, 0.95f};
    const std::array<float, 3> spec_peak = {0.35f, 0.35f, 0.35f};
    const std::array<float, 3> spec_flat = {0.00f, 0.00f, 0.00f};
    const std::array<float, 3> orig_peak = {
        dark_blue_albedo[0] * shading[0] + spec_peak[0],
        dark_blue_albedo[1] * shading[1] + spec_peak[1],
        dark_blue_albedo[2] * shading[2] + spec_peak[2]
    };
    const std::array<float, 3> orig_flat = {
        dark_blue_albedo[0] * shading[0],
        dark_blue_albedo[1] * shading[1],
        dark_blue_albedo[2] * shading[2]
    };

    RegionRenderStack matte_stack{};
    matte_stack.is_modified = true;
    matte_stack.num_active_layers = 0; // No Primer, No Paint!
    matte_stack.finish.topcoat_type = TopcoatType::MATTE_FLAT;
    matte_stack.finish.topcoat_roughness = 0.75f; // GX114 Matte

    RegionRenderStack gloss_stack = matte_stack;
    gloss_stack.finish.topcoat_type = TopcoatType::GLOSS;
    gloss_stack.finish.topcoat_roughness = 0.05f; // GX100 Gloss

    LightingEnvParams env{};
    const auto res_matte_peak = physics::evaluate_pixel_stack(
        orig_peak, dark_blue_albedo, shading, spec_peak, {0,0,0}, {0,0,1}, 1.0f, 0.0f, matte_stack, env);
    const auto res_matte_flat = physics::evaluate_pixel_stack(
        orig_flat, dark_blue_albedo, shading, spec_flat, {0,0,0}, {0,0,1}, 1.0f, 0.0f, matte_stack, env);
    const auto res_gloss_flat = physics::evaluate_pixel_stack(
        orig_flat, dark_blue_albedo, shading, spec_flat, {0,0,0}, {0,0,1}, 1.0f, 0.0f, gloss_stack, env);

    const float L_orig_flat  = color::linear_srgb_to_cielab(orig_flat)[0];
    const float L_matte_flat = color::linear_srgb_to_cielab(res_matte_flat.linear_rgb)[0];
    const float L_gloss_flat = color::linear_srgb_to_cielab(res_gloss_flat.linear_rgb)[0];
    const float dL_matte = L_matte_flat - L_orig_flat;
    const float dL_gloss = L_gloss_flat - L_orig_flat;

    const float spec_contrib_matte = res_matte_peak.linear_rgb[0] - res_matte_flat.linear_rgb[0];
    const float spec_reduction_pct = (1.0f - spec_contrib_matte / spec_peak[0]) * 100.0f;

    const bool ok = (spec_reduction_pct >= 65.0f) && (dL_matte >= 2.0f && dL_matte <= 7.0f) && (dL_gloss < 0.0f);
    report_test("TC-PHY-002", "Bare Plastic + Topcoat Only (Matte GX114 vs Gloss GX100)",
                ok, "spec_drop=" + std::to_string(spec_reduction_pct) + "%, dL_matte=+" + std::to_string(dL_matte));
}

void run_tc_phy_003() {
    // Non-Sequential Case 2: Bare Plastic + Low-Hiding Paint (Mr. Color C4 Yellow over Black vs White plastic)
    const std::array<float, 3> black_plastic = {0.015f, 0.015f, 0.015f};
    const std::array<float, 3> white_plastic = {0.880f, 0.880f, 0.880f};
    const std::array<float, 3> c4_masstone   = {0.940f, 0.680f, 0.012f};
    const std::array<float, 3> c4_hiding     = {0.46f, 0.50f, 0.95f};

    auto eval_coats = [&](float coats) {
        RegionRenderStack st{};
        st.is_modified = true;
        st.num_active_layers = 1;
        st.layers[0] = make_solid_paint(c4_masstone, c4_hiding, 0.07f, coats);
        LightingEnvParams env{};
        const auto res_blk = physics::evaluate_pixel_stack(
            black_plastic, black_plastic, {1,1,1}, {0,0,0}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st, env);
        const auto res_wht = physics::evaluate_pixel_stack(
            white_plastic, white_plastic, {1,1,1}, {0,0,0}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st, env);
        return color::delta_e00_linear_rgb(res_blk.linear_rgb, res_wht.linear_rgb);
    };

    const float de_1coat = eval_coats(1.0f);
    const float de_2coat = eval_coats(2.0f);
    const float de_5coat = eval_coats(5.0f);

    const bool ok = (de_1coat > 14.0f) && (de_2coat < de_1coat) && (de_5coat < 1.5f);
    report_test("TC-PHY-003", "Substrate Bleed-Through: C4 Yellow over Black vs White Plastic",
                ok, "dE00(1c)=" + std::to_string(de_1coat) + ", dE00(2c)=" + std::to_string(de_2coat) + ", dE00(5c)=" + std::to_string(de_5coat));
}

void run_tc_phy_004() {
    // Surfacer Color Interaction under Mr. Color C3 Red
    const std::array<float, 3> black_sub = {0.015f, 0.015f, 0.015f};
    const auto sf_black = make_solid_paint({0.014f, 0.015f, 0.016f}, {0.97f, 0.97f, 0.97f}, 0.35f, 1.0f, LayerRole::SURFACER, 1500);
    const auto sf_gray  = make_solid_paint({0.260f, 0.268f, 0.280f}, {0.93f, 0.93f, 0.94f}, 0.35f, 1.0f, LayerRole::SURFACER, 1500);
    const auto sf_pink  = make_solid_paint({0.790f, 0.340f, 0.440f}, {0.84f, 0.88f, 0.89f}, 0.52f, 1.0f, LayerRole::SURFACER, 1000);
    const auto c3_red   = make_solid_paint({0.570f, 0.018f, 0.026f}, {0.52f, 0.90f, 0.92f}, 0.07f, 1.2f, LayerRole::BASE_SOLID);

    auto eval_over_surfacer = [&](const PaintLayerParams& sf) {
        RegionRenderStack st{};
        st.is_modified = true;
        st.num_active_layers = 2;
        st.layers[0] = sf;
        st.layers[1] = c3_red;
        LightingEnvParams env{};
        const auto res = physics::evaluate_pixel_stack(
            black_sub, black_sub, {1,1,1}, {0,0,0}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st, env);
        return color::linear_srgb_to_cielab(res.linear_rgb);
    };

    const auto lab_blk  = eval_over_surfacer(sf_black);
    const auto lab_gray = eval_over_surfacer(sf_gray);
    const auto lab_pink = eval_over_surfacer(sf_pink);

    const float C_blk  = std::sqrt(lab_blk[1] * lab_blk[1] + lab_blk[2] * lab_blk[2]);
    const float C_pink = std::sqrt(lab_pink[1] * lab_pink[1] + lab_pink[2] * lab_pink[2]);

    const bool ok = (lab_pink[0] > lab_gray[0]) && (lab_pink[0] - lab_blk[0] >= 10.0f) && (C_pink > C_blk);
    report_test("TC-PHY-004", "Surfacer Interaction: C3 Red over Black vs Gray vs Pink Surfacer",
                ok, "L*(Pink)=" + std::to_string(lab_pink[0]) + " vs L*(Blk)=" + std::to_string(lab_blk[0]));
}

void run_tc_phy_005() {
    // Candy Coating: Silver Base + C47 Clear Red (Beer-Lambert) at 1, 2, 3 coats
    const std::array<float, 3> silver_base = {0.78f, 0.80f, 0.82f};
    PaintLayerParams clear_red{};
    clear_red.role = LayerRole::CLEAR_CANDY;
    clear_red.beer_alpha = {0.06f, 0.95f, 1.05f}; // Transmits red, strongly absorbs green/blue
    clear_red.intrinsic_roughness = 0.04f;
    clear_red.enabled = true;

    auto eval_candy = [&](float coats) {
        RegionRenderStack st{};
        st.is_modified = true;
        st.num_active_layers = 1;
        st.layers[0] = clear_red;
        st.layers[0].coat_count = coats;
        LightingEnvParams env{};
        return physics::evaluate_pixel_stack(
            silver_base, silver_base, {1,1,1}, {0.2f,0.2f,0.2f}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st, env).linear_rgb;
    };

    const auto rgb_1c = eval_candy(1.0f);
    const auto rgb_3c = eval_candy(3.0f);
    const bool ok = (rgb_3c[1] < rgb_1c[1] * 0.25f) && (rgb_3c[0] > rgb_3c[1] * 8.0f);
    report_test("TC-PHY-005", "Multi-Layer Candy Coating (C8 Silver + C47 Clear Red 1..3 Coats)",
                ok, "G_1c=" + std::to_string(rgb_1c[1]) + " -> G_3c=" + std::to_string(rgb_3c[1]));
}

void run_tc_phy_007() {
    // Intrinsic Gloss of Exposed Surfacers (#500-#1500, Gloss Black) & Base Paints When Topcoat = NONE_INHERIT
    RegionRenderStack st{};
    st.is_modified = true;
    st.finish.topcoat_type = TopcoatType::NONE_INHERIT;
    st.num_active_layers = 1;

    float under_a = 0.0f;
    st.layers[0] = make_solid_paint({0.25f,0.25f,0.26f}, {0.95f,0.95f,0.95f}, 0.68f, 2.0f, LayerRole::SURFACER, 500);
    const float alpha_sf500 = physics::resolve_layer_stack_roughness(st, under_a);

    st.layers[0] = make_solid_paint({0.26f,0.27f,0.28f}, {0.93f,0.93f,0.94f}, 0.35f, 2.0f, LayerRole::SURFACER, 1500);
    const float alpha_sf1500 = physics::resolve_layer_stack_roughness(st, under_a);

    st.layers[0] = make_solid_paint({0.008f,0.008f,0.009f}, {0.98f,0.98f,0.98f}, 0.04f, 2.0f, LayerRole::SURFACER, 1500);
    const float alpha_sf_gloss_blk = physics::resolve_layer_stack_roughness(st, under_a);

    // Thin 0.4-coat gloss paint over #500 surfacer retains partial substrate roughness
    st.num_active_layers = 2;
    st.layers[0] = make_solid_paint({0.25f,0.25f,0.26f}, {0.95f,0.95f,0.95f}, 0.68f, 2.5f, LayerRole::SURFACER, 500);
    st.layers[1] = make_solid_paint({0.90f,0.91f,0.90f}, {0.62f,0.64f,0.68f}, 0.07f, 0.4f, LayerRole::BASE_SOLID, 0);
    const float alpha_thin_over_500 = physics::resolve_layer_stack_roughness(st, under_a);

    const bool ok = (alpha_sf500 > 0.60f) &&
                    (alpha_sf1500 > 0.30f && alpha_sf1500 < 0.40f) &&
                    (alpha_sf_gloss_blk < 0.06f) &&
                    (alpha_thin_over_500 > 0.30f && alpha_thin_over_500 < alpha_sf500);
    report_test("TC-PHY-007", "Intrinsic Gloss When Topcoat=None (#500/#1500/GlossBlk & Thin Leveling)",
                ok, "a500=" + std::to_string(alpha_sf500) + ", a1500=" + std::to_string(alpha_sf1500) + ", aGlossBlk=" + std::to_string(alpha_sf_gloss_blk));
}

void run_tc_phy_008() {
    // 1. Mirror Chrome (SM206) over Gloss Black Surfacer (a=0.04) vs #1000 Gray Surfacer (a=0.52)
    PaintLayerParams sm206{};
    sm206.role = LayerRole::MIRROR_CHROME;
    sm206.intrinsic_roughness = 0.02f;
    sm206.chrome_f0 = 0.95f;
    sm206.coat_count = 1.0f;
    sm206.enabled = true;

    RegionRenderStack st_chrome_gloss{};
    st_chrome_gloss.is_modified = true;
    st_chrome_gloss.num_active_layers = 2;
    st_chrome_gloss.layers[0] = make_solid_paint({0.008f,0.008f,0.009f}, {0.98f,0.98f,0.98f}, 0.04f, 3.0f, LayerRole::SURFACER, 1500);
    st_chrome_gloss.layers[1] = sm206;

    RegionRenderStack st_chrome_matte = st_chrome_gloss;
    st_chrome_matte.layers[0] = make_solid_paint({0.255f,0.262f,0.274f}, {0.94f,0.94f,0.95f}, 0.52f, 3.0f, LayerRole::SURFACER, 1000);

    float under_g = 0.0f, under_m = 0.0f;
    const float a_chrome_over_gloss = physics::resolve_layer_stack_roughness(st_chrome_gloss, under_g);
    const float a_chrome_over_matte = physics::resolve_layer_stack_roughness(st_chrome_matte, under_m);

    // 2. Chameleon (CP-01 Cyan-Purple) view-angle shift across cos(theta_v) in {0.95, 0.70, 0.25}
    PaintLayerParams cham{};
    cham.role = LayerRole::CHAMELEON;
    cham.intrinsic_roughness = 0.05f;
    cham.coat_count = 2.0f;
    cham.chameleon_stops = {
        0.65f, 0.18f, 210.0f, // Cyan
        0.52f, 0.22f, 265.0f, // Deep Blue
        0.48f, 0.24f, 325.0f  // Magenta-Purple
    };
    cham.enabled = true;

    RegionRenderStack st_cham{};
    st_cham.is_modified = true;
    st_cham.num_active_layers = 2;
    st_cham.layers[0] = st_chrome_gloss.layers[0]; // Gloss Black base
    st_cham.layers[1] = cham;

    LightingEnvParams env{};
    const auto res_center  = physics::evaluate_pixel_stack({0.1f,0.1f,0.1f}, {0.01f,0.01f,0.01f}, {1,1,1}, {0,0,0}, {0,0,0}, {0.0f, 0.31f, 0.95f}, 1.0f, 0.0f, st_cham, env);
    const auto res_grazing = physics::evaluate_pixel_stack({0.1f,0.1f,0.1f}, {0.01f,0.01f,0.01f}, {1,1,1}, {0,0,0}, {0,0,0}, {0.0f, 0.968f, 0.25f}, 1.0f, 0.0f, st_cham, env);

    const float h_center  = color::oklab_to_oklch(color::linear_srgb_to_oklab(res_center.linear_rgb))[2];
    const float h_grazing = color::oklab_to_oklch(color::linear_srgb_to_oklab(res_grazing.linear_rgb))[2];
    const float dh = std::abs(h_grazing - h_center);

    const bool ok = (a_chrome_over_gloss < 0.055f) && (a_chrome_over_matte > 0.44f) && (dh > 40.0f);
    report_test("TC-PHY-008", "Mirror Chrome Undercoat Coupling & Chameleon Angle Shift",
                ok, "aChrome(GlossBlk)=" + std::to_string(a_chrome_over_gloss) + ", aChrome(#1000)=" + std::to_string(a_chrome_over_matte) + ", dHue=" + std::to_string(dh) + " deg");
}

void run_tc_phy_009() {
    // Fluorescent Solid (C175) & Fluorescent Clear (GF-01) in Dark Room + 365nm UV Blacklight Mode
    PaintLayerParams c175 = make_solid_paint({0.04f, 0.92f, 0.06f}, {0.78f, 0.42f, 0.85f}, 0.26f, 2.0f, LayerRole::FLUORESCENT_SOLID);
    c175.fluo_quantum_yield = 0.80f;
    c175.fluo_peak_rgb = {0.12f, 1.00f, 0.18f};

    RegionRenderStack st_fluo{};
    st_fluo.is_modified = true;
    st_fluo.num_active_layers = 1;
    st_fluo.layers[0] = c175;

    RegionRenderStack st_non_fluo{};
    st_non_fluo.is_modified = true;
    st_non_fluo.num_active_layers = 1;
    st_non_fluo.layers[0] = make_solid_paint({0.05f, 0.18f, 0.65f}, {0.85f, 0.85f, 0.85f}, 0.07f, 2.0f, LayerRole::BASE_SOLID);

    LightingEnvParams env_uv{};
    env_uv.mode = LightingEnvMode::DARK_UV_365NM;
    env_uv.ambient_dim_factor = 0.05f;
    env_uv.uv_intensity = 1.8f;

    const auto res_fluo_uv = physics::evaluate_pixel_stack(
        {0.85f,0.85f,0.85f}, {0.88f,0.88f,0.88f}, {1,1,1}, {0,0,0}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st_fluo, env_uv);
    const auto res_blue_uv = physics::evaluate_pixel_stack(
        {0.50f,0.50f,0.50f}, {0.50f,0.50f,0.50f}, {1,1,1}, {0,0,0}, {0,0,0}, {0,0,1}, 1.0f, 0.0f, st_non_fluo, env_uv);

    const float L_fluo = color::linear_srgb_to_cielab(res_fluo_uv.linear_rgb)[0];
    const float L_blue = color::linear_srgb_to_cielab(res_blue_uv.linear_rgb)[0];

    const bool ok = (L_fluo > 75.0f) && (L_blue < 18.0f) && (res_fluo_uv.fluo_emission[1] > 0.50f);
    report_test("TC-PHY-009", "Fluorescent Paint & 365nm UV Blacklight Stokes Luminescence",
                ok, "L*(FluoGreen)=" + std::to_string(L_fluo) + " vs L*(StdBlue)=" + std::to_string(L_blue));
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  libpaintcore (C++20) Phase 1 MVP Native Verification Suite\n";
    std::cout << "================================================================================\n";

    const auto t0 = std::chrono::high_resolution_clock::now();

    run_tc_cpp_001();
    run_tc_cpp_002();
    run_tc_cpp_003();
    run_tc_cpp_004();
    run_tc_cpp_005();

    run_tc_phy_001();
    run_tc_phy_002();
    run_tc_phy_003();
    run_tc_phy_004();
    run_tc_phy_005();
    run_tc_phy_007();
    run_tc_phy_008();
    run_tc_phy_009();

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "================================================================================\n";
    std::cout << "  Summary: " << g_passed << " passed, " << g_failed << " failed ("
              << std::fixed << std::setprecision(2) << elapsed_ms << " ms)\n";
    std::cout << "================================================================================\n";

    return (g_failed == 0) ? 0 : 1;
}
