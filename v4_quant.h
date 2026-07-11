#ifndef V4_QUANT_H
#define V4_QUANT_H

#include <math.h>
#include <stdint.h>

#ifdef FLOYD_METAL
#include "backend_metal.h"
#endif

typedef struct {
    uint64_t metal_calls;
    uint64_t cpu_fallbacks;
} V4QuantBackendStats;

static int v4_quant_metal_enabled;
static int v4_quant_metal_min_batch = 4;
static V4QuantBackendStats v4_quant_stats;

static inline int v4_quant_backend_enable_metal(int min_batch) {
#ifdef FLOYD_METAL
    if (min_batch < 2 || !fm_init()) return 0;
    v4_quant_metal_min_batch = min_batch;
    v4_quant_metal_enabled = 1;
    return 1;
#else
    (void)min_batch;
    return 0;
#endif
}

static inline void v4_quant_backend_reset_stats(void) {
    v4_quant_stats = (V4QuantBackendStats){0, 0};
}

static inline V4QuantBackendStats v4_quant_backend_stats(void) {
    return v4_quant_stats;
}

static inline const char *v4_quant_backend_name(void) {
    return v4_quant_metal_enabled ? "metal" : "cpu";
}

static inline int v4_quant_backend_min_batch(void) {
    return v4_quant_metal_min_batch;
}

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

static inline int v4_fp4_matmul_f32(float *output, const float *input,
                                    int n_inputs, const uint8_t *weights,
                                    const uint8_t *scales, int rows,
                                    int columns) {
    if (!output || !input || !weights || !scales || n_inputs <= 0 ||
        rows <= 0 || columns <= 0 || columns % 32 != 0)
        return 0;
#ifdef FLOYD_METAL
    if (v4_quant_metal_enabled && n_inputs >= v4_quant_metal_min_batch) {
        if (fm_matmul_v4_fp4(output, input, weights, scales,
                             rows, columns, n_inputs)) {
            v4_quant_stats.metal_calls++;
            return 1;
        }
        v4_quant_stats.cpu_fallbacks++;
    } else if (v4_quant_metal_enabled) {
        v4_quant_stats.cpu_fallbacks++;
    }
#endif
    int row_bytes = columns / 2;
    int scale_columns = columns / 32;
    for (int input_row = 0; input_row < n_inputs; input_row++) {
        const float *x = input + (int64_t)input_row * columns;
        for (int row = 0; row < rows; row++) {
            const uint8_t *packed = weights + (int64_t)row * row_bytes;
            const uint8_t *row_scales = scales + (int64_t)row * scale_columns;
            float sum = 0.0f;
            for (int column = 0; column < columns; column++) {
                float scale = v4_e8m0_to_f32(row_scales[column / 32]);
                float value = v4_fp4_packed_value(packed, column) * scale;
                sum += x[column] * value;
            }
            output[(int64_t)input_row * rows + row] = sum;
        }
    }
    return 1;
}

static inline int v4_fp8_matmul_f32(float *output, const float *input,
                                    int n_inputs, const uint8_t *weights,
                                    const uint8_t *scales, int rows,
                                    int columns, int block_size) {
    if (!output || !input || !weights || !scales || n_inputs <= 0 ||
        rows <= 0 || columns <= 0 || block_size <= 0 ||
        rows % block_size != 0 || columns % block_size != 0)
        return 0;
#ifdef FLOYD_METAL
    if (v4_quant_metal_enabled && n_inputs >= v4_quant_metal_min_batch) {
        if (fm_matmul_v4_fp8(output, input, weights, scales, rows, columns,
                             n_inputs, block_size)) {
            v4_quant_stats.metal_calls++;
            return 1;
        }
        v4_quant_stats.cpu_fallbacks++;
    } else if (v4_quant_metal_enabled) {
        v4_quant_stats.cpu_fallbacks++;
    }
#endif
    int scale_columns = columns / block_size;
    for (int input_row = 0; input_row < n_inputs; input_row++) {
        const float *x = input + (int64_t)input_row * columns;
        for (int row = 0; row < rows; row++) {
            const uint8_t *weight = weights + (int64_t)row * columns;
            const uint8_t *row_scales = scales
                + (int64_t)(row / block_size) * scale_columns;
            float sum = 0.0f;
            for (int column = 0; column < columns; column++) {
                float scale = v4_e8m0_to_f32(row_scales[column / block_size]);
                float value = v4_e4m3_to_f32(weight[column]) * scale;
                sum += x[column] * value;
            }
            output[(int64_t)input_row * rows + row] = sum;
        }
    }
    return 1;
}

#endif
