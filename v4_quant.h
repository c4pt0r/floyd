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

static inline float v4_e4m3_to_f32(uint8_t code) {
    int sign = (code & 0x80) ? -1 : 1;
    int exponent = (code >> 3) & 0x0f;
    int mantissa = code & 0x07;
    if (exponent == 0x0f && mantissa == 0x07) return NAN;
    float value;
    if (exponent == 0)
        value = ldexpf((float)mantissa / 8.0f, -6);
    else
        value = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign * value;
}

static inline void v4_fp4_dequant_row(float *out, const uint8_t *packed,
                                      const uint8_t *scales, int count) {
    for (int i = 0; i < count; i++) {
        float scale = v4_e8m0_to_f32(scales[i / 32]);
        out[i] = v4_fp4_packed_value(packed, i) * scale;
    }
}

static inline void v4_fp8_dequant_matrix(float *out, const uint8_t *weights,
                                         const uint8_t *scales, int rows,
                                         int columns, int block_size) {
    int scale_columns = (columns + block_size - 1) / block_size;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            int scale_index = (row / block_size) * scale_columns + column / block_size;
            float scale = v4_e8m0_to_f32(scales[scale_index]);
            out[(int64_t)row * columns + column] =
                v4_e4m3_to_f32(weights[(int64_t)row * columns + column]) * scale;
        }
    }
}

#endif
