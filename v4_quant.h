#ifndef V4_QUANT_H
#define V4_QUANT_H

#include <math.h>
#include <stdint.h>

static const float v4_fp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float v4_fp4_code_value(uint8_t code) {
    return v4_fp4_values[code & 0x0f];
}

static inline float v4_fp4_packed_value(const uint8_t *packed, int64_t index) {
    uint8_t byte = packed[index >> 1];
    uint8_t code = (index & 1) ? (byte >> 4) : (byte & 0x0f);
    return v4_fp4_code_value(code);
}

static inline float v4_e8m0_to_f32(uint8_t code) {
    if (code == 255) return NAN;
    return ldexpf(1.0f, (int)code - 127);
}

static inline void v4_fp4_dequant_row(float *out, const uint8_t *packed,
                                      const uint8_t *scales, int count) {
    for (int i = 0; i < count; i++) {
        float scale = v4_e8m0_to_f32(scales[i / 32]);
        out[i] = v4_fp4_packed_value(packed, i) * scale;
    }
}

#endif
