#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_compress.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float *load_f32(shards *source, const char *name, int64_t expected) {
    int64_t count = st_numel(source, name);
    if (count != expected) {
        fprintf(stderr, "%s: elements=%lld expected=%lld\n", name,
                (long long)count, (long long)expected);
        exit(1);
    }
    float *data = malloc((size_t)count * sizeof(float));
    st_read_f32(source, name, data, 0);
    return data;
}

static int run_case(shards *source, const char *mode, int ratio,
                    int n_tokens, int overlap) {
    enum { HEAD_DIM = 16, ROPE_DIM = 4 };
    int input_width = overlap ? 2 * HEAD_DIM : HEAD_DIM;
    int n_windows = n_tokens / ratio;
    char name[128];

    snprintf(name, sizeof(name), "compress.%s.kv", mode);
    float *kv = load_f32(source, name, (int64_t)n_tokens * input_width);
    snprintf(name, sizeof(name), "compress.%s.gate", mode);
    float *gate = load_f32(source, name, (int64_t)n_tokens * input_width);
    snprintf(name, sizeof(name), "compress.%s.ape", mode);
    float *ape = load_f32(source, name, (int64_t)ratio * input_width);
    snprintf(name, sizeof(name), "compress.%s.norm", mode);
    float *norm = load_f32(source, name, HEAD_DIM);
    snprintf(name, sizeof(name), "compress.%s.output", mode);
    float *expected = load_f32(source, name, (int64_t)n_windows * HEAD_DIM);
    float *actual = malloc((size_t)n_windows * HEAD_DIM * sizeof(float));

    V4PrefillCompressorF32 model = {
        .ratio = ratio, .head_dim = HEAD_DIM, .rope_dim = ROPE_DIM,
        .overlap = overlap, .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = ape, .norm_weight = norm,
    };
    CHECK(v4_compress_prefill_f32(&model, kv, gate, n_tokens, actual));
    CHECK(!v4_compress_prefill_f32(&model, kv, gate, n_tokens - 1, actual));
    CHECK(!v4_compress_prefill_f32(&model, kv, gate, 0, actual));
    V4PrefillCompressorF32 invalid = model;
    invalid.overlap = 2;
    CHECK(!v4_compress_prefill_f32(&invalid, kv, gate, n_tokens, actual));

    float max_abs = 0.0f;
    for (int64_t i = 0; i < (int64_t)n_windows * HEAD_DIM; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) max_abs = error;
    }
    printf("v4 %s compressor: max_abs=%.9g\n", mode, max_abs);
    CHECK(max_abs < 3e-5f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_v4>\n", argv[0]);
        return 2;
    }

    shards source;
    st_init(&source, argv[1]);
    CHECK(run_case(&source, "hca", 128, 256, 0) == 0);
    CHECK(run_case(&source, "csa", 4, 12, 1) == 0);
    puts("v4 prefill compressor fixture tests: ok");
    return 0;
}
