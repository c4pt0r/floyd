#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_quant.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void make_inputs(float *input, int n_inputs, int columns) {
    for (int row = 0; row < n_inputs; row++)
        for (int column = 0; column < columns; column++)
            input[(int64_t)row * columns + column] =
                0.25f * sinf((float)((row + 1) * (column + 3)) * 0.013f)
              + 0.05f * cosf((float)(column + 1) * 0.031f);
}

static void reference_matmul(float *output, const float *input,
                             const float *weight, int n_inputs,
                             int rows, int columns) {
    for (int input_row = 0; input_row < n_inputs; input_row++) {
        const float *x = input + (int64_t)input_row * columns;
        for (int row = 0; row < rows; row++) {
            const float *w = weight + (int64_t)row * columns;
            float sum = 0.0f;
            for (int column = 0; column < columns; column++)
                sum += x[column] * w[column];
            output[(int64_t)input_row * rows + row] = sum;
        }
    }
}

static float max_error(const float *actual, const float *expected, int count) {
    float maximum = 0.0f;
    for (int i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
    }
    return maximum;
}

static int top1(const float *values, int count) {
    int best = 0;
    for (int i = 1; i < count; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

static int test_official_fp8(shards *source, float *max_abs, int *top1_matches) {
    enum { ROWS = 256, COLUMNS = 256, BLOCK = 128, INPUTS = 4 };
    const char *weight_name = "layers.0.attn.wq_a.weight";
    const char *scale_name = "layers.0.attn.wq_a.scale";
    st_tensor *weight_tensor = st_find(source, weight_name);
    st_tensor *scale_tensor = st_find(source, scale_name);
    CHECK(weight_tensor && weight_tensor->dtype == ST_DTYPE_F8_E4M3);
    CHECK(scale_tensor && scale_tensor->dtype == ST_DTYPE_F8_E8M0);
    CHECK(weight_tensor->numel == 1024LL * 4096);
    CHECK(scale_tensor->numel == 8 * 32);

    uint8_t *weight = malloc(ROWS * COLUMNS);
    uint8_t scales[4];
    float *dequant = malloc((size_t)ROWS * COLUMNS * sizeof(float));
    float input[INPUTS * COLUMNS], reference[INPUTS * ROWS], actual[INPUTS * ROWS];
    CHECK(weight && dequant);
    for (int row = 0; row < ROWS; row++)
        st_read_slice_raw(source, weight_name, (int64_t)row * 4096,
                          COLUMNS, weight + row * COLUMNS, 0);
    for (int block_row = 0; block_row < ROWS / BLOCK; block_row++)
        st_read_slice_raw(source, scale_name, (int64_t)block_row * 32,
                          COLUMNS / BLOCK,
                          scales + block_row * (COLUMNS / BLOCK), 0);

    make_inputs(input, INPUTS, COLUMNS);
    v4_fp8_dequant_matrix(dequant, weight, scales, ROWS, COLUMNS, BLOCK);
    reference_matmul(reference, input, dequant, INPUTS, ROWS, COLUMNS);
    CHECK(v4_fp8_matmul_f32(actual, input, INPUTS, weight, scales,
                            ROWS, COLUMNS, BLOCK));
    *max_abs = max_error(actual, reference, INPUTS * ROWS);
    *top1_matches = 0;
    for (int i = 0; i < INPUTS; i++)
        if (top1(actual + i * ROWS, ROWS) == top1(reference + i * ROWS, ROWS))
            (*top1_matches)++;
    free(weight);
    free(dequant);
    return 0;
}

static int test_official_fp4(shards *source, float *max_abs, int *top1_matches) {
    enum { ROWS = 16, COLUMNS = 256, SCALE_BLOCK = 32, INPUTS = 4 };
    const char *weight_name = "layers.0.ffn.experts.0.w1.weight";
    const char *scale_name = "layers.0.ffn.experts.0.w1.scale";
    st_tensor *weight_tensor = st_find(source, weight_name);
    st_tensor *scale_tensor = st_find(source, scale_name);
    CHECK(weight_tensor && weight_tensor->dtype == ST_DTYPE_U8);
    CHECK(scale_tensor && scale_tensor->dtype == ST_DTYPE_F8_E8M0);
    CHECK(weight_tensor->numel == 2048LL * 2048);
    CHECK(scale_tensor->numel == 2048LL * 128);

    uint8_t weight[ROWS * (COLUMNS / 2)];
    uint8_t scales[ROWS * (COLUMNS / SCALE_BLOCK)];
    float dequant[ROWS * COLUMNS];
    float input[INPUTS * COLUMNS], reference[INPUTS * ROWS], actual[INPUTS * ROWS];
    for (int row = 0; row < ROWS; row++) {
        st_read_slice_raw(source, weight_name, (int64_t)row * 2048,
                          COLUMNS / 2, weight + row * (COLUMNS / 2), 0);
        st_read_slice_raw(source, scale_name, (int64_t)row * 128,
                          COLUMNS / SCALE_BLOCK,
                          scales + row * (COLUMNS / SCALE_BLOCK), 0);
        v4_fp4_dequant_row(dequant + row * COLUMNS,
                           weight + row * (COLUMNS / 2),
                           scales + row * (COLUMNS / SCALE_BLOCK), COLUMNS);
    }

    make_inputs(input, INPUTS, COLUMNS);
    reference_matmul(reference, input, dequant, INPUTS, ROWS, COLUMNS);
    CHECK(v4_fp4_matmul_f32(actual, input, INPUTS, weight, scales,
                            ROWS, COLUMNS));
    *max_abs = max_error(actual, reference, INPUTS * ROWS);
    *top1_matches = 0;
    for (int i = 0; i < INPUTS; i++)
        if (top1(actual + i * ROWS, ROWS) == top1(reference + i * ROWS, ROWS))
            (*top1_matches)++;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark>\n", argv[0]);
        return 2;
    }
    shards source;
    st_init(&source, argv[1]);
#ifdef FLOYD_METAL
    CHECK(v4_quant_backend_enable_metal(4));
    v4_quant_backend_reset_stats();
#endif
    float fp4_error, fp8_error;
    int fp4_top1, fp8_top1;
    CHECK(test_official_fp4(&source, &fp4_error, &fp4_top1) == 0);
    CHECK(test_official_fp8(&source, &fp8_error, &fp8_top1) == 0);
    printf("v4 native FP4: max_abs=%.9g top1=%d/4\n", fp4_error, fp4_top1);
    printf("v4 native FP8: max_abs=%.9g top1=%d/4\n", fp8_error, fp8_top1);
    CHECK(fp4_error < 3e-5f && fp4_top1 == 4);
    CHECK(fp8_error < 3e-5f && fp8_top1 == 4);
#ifdef FLOYD_METAL
    V4QuantBackendStats stats = v4_quant_backend_stats();
    printf("v4 native Metal: calls=%llu cpu_fallbacks=%llu\n",
           (unsigned long long)stats.metal_calls,
           (unsigned long long)stats.cpu_fallbacks);
    CHECK(stats.metal_calls == 2);
    CHECK(stats.cpu_fallbacks == 0);
#endif
    puts("v4 native quantized matmul tests: ok");
    return 0;
}
