#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>

namespace paintcore::segmentation {

struct BadgePosition {
    uint16_t mask_id;
    int center_x;
    int center_y;
    float max_clearance_px;
};

// ============================================================================
// 1. Multi-Scale Morphological Black-Hat Panel-Line Extractor
// ============================================================================

inline void compute_black_hat_panel_lines(
    const float* linear_wb_rgb,
    int width,
    int height,
    uint8_t* out_panel_mask_8u
) {
    const size_t n_pix = static_cast<size_t>(width) * height;
    std::vector<float> lum(n_pix, 0.0f);
    for (size_t i = 0; i < n_pix; ++i) {
        lum[i] = 0.2126f * linear_wb_rgb[3 * i + 0] +
                 0.7152f * linear_wb_rgb[3 * i + 1] +
                 0.0722f * linear_wb_rgb[3 * i + 2];
    }

    // Morphological closing (dilation followed by erosion) across radii r in {1, 2, 3}
    std::vector<float> dilated(n_pix, 0.0f);
    std::vector<float> closed(n_pix, 0.0f);
    std::vector<float> black_hat_max(n_pix, 0.0f);

    for (int r = 1; r <= 3; ++r) {
        // Dilation
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float max_v = 0.0f;
                for (int dy = -r; dy <= r; ++dy) {
                    const int ny = std::clamp(y + dy, 0, height - 1);
                    for (int dx = -r; dx <= r; ++dx) {
                        if (dx * dx + dy * dy > r * r + 1) continue;
                        const int nx = std::clamp(x + dx, 0, width - 1);
                        max_v = std::max(max_v, lum[static_cast<size_t>(ny) * width + nx]);
                    }
                }
                dilated[static_cast<size_t>(y) * width + x] = max_v;
            }
        }
        // Erosion
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float min_v = 1e9f;
                for (int dy = -r; dy <= r; ++dy) {
                    const int ny = std::clamp(y + dy, 0, height - 1);
                    for (int dx = -r; dx <= r; ++dx) {
                        if (dx * dx + dy * dy > r * r + 1) continue;
                        const int nx = std::clamp(x + dx, 0, width - 1);
                        min_v = std::min(min_v, dilated[static_cast<size_t>(ny) * width + nx]);
                    }
                }
                const size_t idx = static_cast<size_t>(y) * width + x;
                closed[idx] = min_v;
                black_hat_max[idx] = std::max(black_hat_max[idx], std::max(0.0f, closed[idx] - lum[idx]));
            }
        }
    }

    for (size_t i = 0; i < n_pix; ++i) {
        const float scaled = std::clamp(black_hat_max[i] * 6.0f, 0.0f, 1.0f);
        out_panel_mask_8u[i] = static_cast<uint8_t>(scaled * 255.0f + 0.5f);
    }
}

// ============================================================================
// 2. Active-Contour Boundary Snapping to Panel Lines (TC-CPP-004)
// ============================================================================

inline void snap_mask_boundaries_to_panel_lines(
    uint16_t* io_region_id_map,
    const uint8_t* panel_mask_8u,
    int width,
    int height,
    int search_radius = 4
) {
    std::vector<uint16_t> orig(io_region_id_map, io_region_id_map + static_cast<size_t>(width) * height);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            const uint16_t self_id = orig[idx];

            // Check if pixel is near a boundary
            bool near_boundary = false;
            uint16_t neighbor_id = self_id;
            for (int dy = -search_radius; dy <= search_radius && !near_boundary; ++dy) {
                const int ny = std::clamp(y + dy, 0, height - 1);
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    const int nx = std::clamp(x + dx, 0, width - 1);
                    const uint16_t nid = orig[static_cast<size_t>(ny) * width + nx];
                    if (nid != self_id) {
                        near_boundary = true;
                        neighbor_id = nid;
                        break;
                    }
                }
            }
            if (!near_boundary) continue;

            // Find peak panel-line ridge between current pixel and neighbor
            int best_px = x, best_py = y;
            uint8_t max_panel = panel_mask_8u[idx];
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                const int ny = std::clamp(y + dy, 0, height - 1);
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    const int nx = std::clamp(x + dx, 0, width - 1);
                    const uint8_t pval = panel_mask_8u[static_cast<size_t>(ny) * width + nx];
                    if (pval > max_panel) {
                        max_panel = pval;
                        best_px = nx;
                        best_py = ny;
                    }
                }
            }

            // If a strong panel line exists within search_radius, align region assignment across ridge
            if (max_panel > 35) {
                // Determine which side of the panel ridge (best_px, best_py) this pixel lies on
                // by sampling one step further away from the ridge
                const int step_x = std::clamp(x + (x - best_px), 0, width - 1);
                const int step_y = std::clamp(y + (y - best_py), 0, height - 1);
                const uint16_t anchor_id = orig[static_cast<size_t>(step_y) * width + step_x];
                if (x == best_px && y == best_py) {
                    io_region_id_map[idx] = std::min(self_id, neighbor_id);
                } else {
                    io_region_id_map[idx] = anchor_id;
                }
            }
        }
    }
}

// ============================================================================
// 3. Fast Guided Filter Soft Alpha Matting (O(N) Box-Filter Implementation)
// ============================================================================

inline void box_filter_mean(
    const float* src,
    float* dst,
    int width,
    int height,
    int radius
) {
    std::vector<float> temp(static_cast<size_t>(width) * height, 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            int cnt = 0;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int nx = std::clamp(x + dx, 0, width - 1);
                sum += src[static_cast<size_t>(y) * width + nx];
                ++cnt;
            }
            temp[static_cast<size_t>(y) * width + x] = sum / static_cast<float>(cnt);
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            int cnt = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int ny = std::clamp(y + dy, 0, height - 1);
                sum += temp[static_cast<size_t>(ny) * width + x];
                ++cnt;
            }
            dst[static_cast<size_t>(y) * width + x] = sum / static_cast<float>(cnt);
        }
    }
}

inline void fast_guided_filter_matte(
    const float* guide_lum,
    const float* raw_mask_01,
    float* out_alpha_01,
    int width,
    int height,
    int radius = 4,
    float eps = 1e-4f
) {
    const size_t n = static_cast<size_t>(width) * height;
    std::vector<float> mean_I(n), mean_p(n), corr_I(n), corr_Ip(n);
    std::vector<float> II(n), Ip(n), a(n), b(n), mean_a(n), mean_b(n);

    for (size_t i = 0; i < n; ++i) {
        II[i] = guide_lum[i] * guide_lum[i];
        Ip[i] = guide_lum[i] * raw_mask_01[i];
    }

    box_filter_mean(guide_lum, mean_I.data(), width, height, radius);
    box_filter_mean(raw_mask_01, mean_p.data(), width, height, radius);
    box_filter_mean(II.data(), corr_I.data(), width, height, radius);
    box_filter_mean(Ip.data(), corr_Ip.data(), width, height, radius);

    for (size_t i = 0; i < n; ++i) {
        const float var_I = std::max(0.0f, corr_I[i] - mean_I[i] * mean_I[i]);
        const float cov_Ip = corr_Ip[i] - mean_I[i] * mean_p[i];
        a[i] = cov_Ip / (var_I + eps);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    box_filter_mean(a.data(), mean_a.data(), width, height, radius);
    box_filter_mean(b.data(), mean_b.data(), width, height, radius);

    for (size_t i = 0; i < n; ++i) {
        out_alpha_01[i] = std::clamp(mean_a[i] * guide_lum[i] + mean_b[i], 0.0f, 1.0f);
    }
}

// ============================================================================
// 4. Pole-of-Inaccessibility Set-of-Mark (SoM) Badge Placement (TC-CPP-005)
// ============================================================================

inline BadgePosition compute_pole_of_inaccessibility(
    const uint8_t* binary_mask,
    int width,
    int height,
    uint16_t mask_id
) {
    const size_t n = static_cast<size_t>(width) * height;
    std::vector<float> dist(n, 0.0f);
    constexpr float INF_D = 1e6f;

    // Two-pass Chamfer / Euclidean approximation distance transform inside mask
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            if (binary_mask[idx] == 0) {
                dist[idx] = 0.0f;
            } else if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                dist[idx] = 1.0f;
            } else {
                float d = INF_D;
                d = std::min(d, dist[static_cast<size_t>(y - 1) * width + x] + 1.0f);
                d = std::min(d, dist[static_cast<size_t>(y) * width + (x - 1)] + 1.0f);
                d = std::min(d, dist[static_cast<size_t>(y - 1) * width + (x - 1)] + 1.41421356f);
                d = std::min(d, dist[static_cast<size_t>(y - 1) * width + (x + 1)] + 1.41421356f);
                dist[idx] = d;
            }
        }
    }

    int best_x = width / 2, best_y = height / 2;
    float max_d = -1.0f;

    for (int y = height - 1; y >= 0; --y) {
        for (int x = width - 1; x >= 0; --x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            if (binary_mask[idx] == 0) continue;
            float d = dist[idx];
            if (x < width - 1 && y < height - 1 && x > 0) {
                d = std::min(d, dist[static_cast<size_t>(y + 1) * width + x] + 1.0f);
                d = std::min(d, dist[static_cast<size_t>(y) * width + (x + 1)] + 1.0f);
                d = std::min(d, dist[static_cast<size_t>(y + 1) * width + (x + 1)] + 1.41421356f);
                d = std::min(d, dist[static_cast<size_t>(y + 1) * width + (x - 1)] + 1.41421356f);
            }
            dist[idx] = d;
            if (d > max_d) {
                max_d = d;
                best_x = x;
                best_y = y;
            }
        }
    }

    return {mask_id, best_x, best_y, std::max(0.0f, max_d)};
}

} // namespace paintcore::segmentation
