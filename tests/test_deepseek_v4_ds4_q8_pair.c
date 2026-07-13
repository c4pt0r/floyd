#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4_gpu.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void fill_q8_0(uint8_t *weights, uint32_t in_dim,
                      uint32_t out_dim, uint32_t seed) {
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    for (uint32_t row = 0; row < out_dim; row++) {
        uint8_t *dst = weights + (uint64_t)row * row_bytes;
        for (uint32_t block = 0; block < in_dim / 32u; block++) {
            const uint16_t one_f16 = 0x3c00u;
            memcpy(dst + (uint64_t)block * 34u, &one_f16,
                   sizeof(one_f16));
            int8_t *qs = (int8_t *)(dst + (uint64_t)block * 34u + 2u);
            for (uint32_t i = 0; i < 32u; i++) {
                qs[i] = (int8_t)(((row * 17u + block * 11u + i * 7u + seed) % 31u) - 15);
            }
        }
    }
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float error = fabsf(a[i] - b[i]);
        if (error > max_abs) max_abs = error;
    }
    return max_abs;
}

int main(void) {
    enum { IN_DIM = 256, OUT0_DIM = 96, OUT1_DIM = 32 };
    const uint64_t row_bytes = (IN_DIM / 32u) * 34u;
    const uint64_t weight0_bytes = OUT0_DIM * row_bytes;
    const uint64_t weight1_bytes = OUT1_DIM * row_bytes;
    const uint64_t model_bytes = weight0_bytes + weight1_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t allocation = (model_bytes + page - 1u) & ~(page - 1u);

    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)allocation) == 0);
    memset(model, 0, (size_t)allocation);
    fill_q8_0(model, IN_DIM, OUT0_DIM, 3);
    fill_q8_0((uint8_t *)model + weight0_bytes, IN_DIM, OUT1_DIM, 19);

    float x_host[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++) {
        x_host[i] = ((int)(i % 29u) - 14) / 32.0f;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_host));
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc(OUT0_DIM * sizeof(float));
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc(OUT1_DIM * sizeof(float));
    ds4_gpu_tensor *pair0 = ds4_gpu_tensor_alloc(OUT0_DIM * sizeof(float));
    ds4_gpu_tensor *pair1 = ds4_gpu_tensor_alloc(OUT1_DIM * sizeof(float));
    CHECK(x && ref0 && ref1 && pair0 && pair1);
    CHECK(ds4_gpu_tensor_write(x, 0, x_host, sizeof(x_host)) != 0);
    CHECK(ds4_gpu_set_model_map(model, allocation) != 0);

    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_matmul_q8_0_tensor(ref0, model, allocation, 0,
                                     IN_DIM, OUT0_DIM, x, 1) != 0);
    CHECK(ds4_gpu_matmul_q8_0_tensor(ref1, model, allocation, weight0_bytes,
                                     IN_DIM, OUT1_DIM, x, 1) != 0);
    CHECK(ds4_gpu_end_commands() != 0);

    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(pair0, pair1, model, allocation,
                                          0, weight0_bytes,
                                          IN_DIM, OUT0_DIM, OUT1_DIM,
                                          x, 1) != 0);
    CHECK(ds4_gpu_end_commands() != 0);

    float ref0_host[OUT0_DIM], ref1_host[OUT1_DIM];
    float pair0_host[OUT0_DIM], pair1_host[OUT1_DIM];
    CHECK(ds4_gpu_tensor_read(ref0, 0, ref0_host, sizeof(ref0_host)) != 0);
    CHECK(ds4_gpu_tensor_read(ref1, 0, ref1_host, sizeof(ref1_host)) != 0);
    CHECK(ds4_gpu_tensor_read(pair0, 0, pair0_host, sizeof(pair0_host)) != 0);
    CHECK(ds4_gpu_tensor_read(pair1, 0, pair1_host, sizeof(pair1_host)) != 0);

    const float max0 = max_abs_diff(ref0_host, pair0_host, OUT0_DIM);
    const float max1 = max_abs_diff(ref1_host, pair1_host, OUT1_DIM);
    printf("DeepSeek V4 Q8 pair parity: out0_max_abs=%.9g out1_max_abs=%.9g\n",
           max0, max1);
    CHECK(max0 == 0.0f);
    CHECK(max1 == 0.0f);

    ds4_gpu_tensor_free(pair1);
    ds4_gpu_tensor_free(pair0);
    ds4_gpu_tensor_free(ref1);
    ds4_gpu_tensor_free(ref0);
    ds4_gpu_tensor_free(x);
    free(model);
    return 0;
}
