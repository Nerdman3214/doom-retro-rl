
#include <cstdint>
#include <algorithm>

extern "C" {

// Packs RGB HWC uint8 image into CHW uint8 output using nearest-neighbor resize.
// src: H x W x 3
// dst: 3 x out_h x out_w
void pack_rgb_hwc_to_chw_u8(
    const uint8_t* src,
    int src_h,
    int src_w,
    uint8_t* dst,
    int out_h,
    int out_w
) {
    if (!src || !dst || src_h <= 0 || src_w <= 0 || out_h <= 0 || out_w <= 0) {
        return;
    }

    for (int oy = 0; oy < out_h; ++oy) {
        int sy = (oy * src_h) / out_h;
        sy = std::min(std::max(sy, 0), src_h - 1);

        for (int ox = 0; ox < out_w; ++ox) {
            int sx = (ox * src_w) / out_w;
            sx = std::min(std::max(sx, 0), src_w - 1);

            int src_idx = (sy * src_w + sx) * 3;

            uint8_t r = src[src_idx + 0];
            uint8_t g = src[src_idx + 1];
            uint8_t b = src[src_idx + 2];

            int pixel = oy * out_w + ox;

            dst[0 * out_h * out_w + pixel] = r;
            dst[1 * out_h * out_w + pixel] = g;
            dst[2 * out_h * out_w + pixel] = b;
        }
    }
}


// Packs grayscale H x W uint8 into 1 x out_h x out_w.
void pack_gray_u8(
    const uint8_t* src,
    int src_h,
    int src_w,
    uint8_t* dst,
    int out_h,
    int out_w
) {
    if (!src || !dst || src_h <= 0 || src_w <= 0 || out_h <= 0 || out_w <= 0) {
        return;
    }

    for (int oy = 0; oy < out_h; ++oy) {
        int sy = (oy * src_h) / out_h;
        sy = std::min(std::max(sy, 0), src_h - 1);

        for (int ox = 0; ox < out_w; ++ox) {
            int sx = (ox * src_w) / out_w;
            sx = std::min(std::max(sx, 0), src_w - 1);

            dst[oy * out_w + ox] = src[sy * src_w + sx];
        }
    }
}


// Normalizes float32 depth to uint8 and resizes to 1 x out_h x out_w.
// Lower depth = closer, higher depth = farther after normalization.
void pack_depth_f32_to_u8(
    const float* src,
    int src_h,
    int src_w,
    uint8_t* dst,
    int out_h,
    int out_w
) {
    if (!src || !dst || src_h <= 0 || src_w <= 0 || out_h <= 0 || out_w <= 0) {
        return;
    }

    float min_v = src[0];
    float max_v = src[0];

    int total = src_h * src_w;

    for (int i = 0; i < total; ++i) {
        float v = src[i];

        if (v < min_v) {
            min_v = v;
        }

        if (v > max_v) {
            max_v = v;
        }
    }

    float denom = max_v - min_v;

    if (denom <= 1e-6f) {
        for (int i = 0; i < out_h * out_w; ++i) {
            dst[i] = 0;
        }
        return;
    }

    for (int oy = 0; oy < out_h; ++oy) {
        int sy = (oy * src_h) / out_h;
        sy = std::min(std::max(sy, 0), src_h - 1);

        for (int ox = 0; ox < out_w; ++ox) {
            int sx = (ox * src_w) / out_w;
            sx = std::min(std::max(sx, 0), src_w - 1);

            float v = src[sy * src_w + sx];
            float norm = (v - min_v) / denom;

            int out = static_cast<int>(norm * 255.0f);

            if (out < 0) {
                out = 0;
            }

            if (out > 255) {
                out = 255;
            }

            dst[oy * out_w + ox] = static_cast<uint8_t>(out);
        }
    }
}


// Packs RGB + depth into one compact tensor:
// output shape = 4 x out_h x out_w
// channels:
// 0 R
// 1 G
// 2 B
// 3 normalized depth
void pack_rgb_depth_compact(
    const uint8_t* rgb,
    int rgb_h,
    int rgb_w,
    const float* depth,
    int depth_h,
    int depth_w,
    uint8_t* dst,
    int out_h,
    int out_w
) {
    if (!dst) {
        return;
    }

    pack_rgb_hwc_to_chw_u8(
        rgb,
        rgb_h,
        rgb_w,
        dst,
        out_h,
        out_w
    );

    pack_depth_f32_to_u8(
        depth,
        depth_h,
        depth_w,
        dst + (3 * out_h * out_w),
        out_h,
        out_w
    );
}

}