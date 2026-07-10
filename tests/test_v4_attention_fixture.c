#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_attention.h"

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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_v4>\n", argv[0]);
        return 2;
    }

    enum { D = 64, H = 4, HD = 16, QR = 32, GROUPS = 2, OR = 16 };
    shards source;
    st_init(&source, argv[1]);
    int64_t n_tokens = st_numel(&source, "attn.0.input") / D;
    CHECK(n_tokens > 0);

    V4SlidingAttentionF32 model = {
        .hidden = D, .heads = H, .head_dim = HD, .q_rank = QR,
        .rope_dim = 4, .o_groups = GROUPS, .o_rank = OR,
        .window = 8, .norm_eps = 1e-6f, .rope_theta = 10000.0f,
        .q_a = load_f32(&source, "model.layers.0.self_attn.q_a_proj.weight", QR * D),
        .q_a_norm = load_f32(&source, "model.layers.0.self_attn.q_a_norm.weight", QR),
        .q_b = load_f32(&source, "model.layers.0.self_attn.q_b_proj.weight", H * HD * QR),
        .kv = load_f32(&source, "model.layers.0.self_attn.kv_proj.weight", HD * D),
        .kv_norm = load_f32(&source, "model.layers.0.self_attn.kv_norm.weight", HD),
        .sinks = load_f32(&source, "model.layers.0.self_attn.sinks", H),
        .o_a = load_f32(&source, "model.layers.0.self_attn.o_a_proj.weight", GROUPS * OR * (H * HD / GROUPS)),
        .o_b = load_f32(&source, "model.layers.0.self_attn.o_b_proj.weight", D * GROUPS * OR),
    };
    float *input = load_f32(&source, "attn.0.input", n_tokens * D);
    float *expected = load_f32(&source, "attn.0.output", n_tokens * D);
    float *actual = malloc((size_t)n_tokens * D * sizeof(float));

    CHECK(v4_sliding_attention_forward(&model, input, (int)n_tokens, actual));

    float max_abs = 0.0f;
    for (int64_t i = 0; i < n_tokens * D; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) max_abs = error;
    }
    printf("v4 sliding attention: max_abs=%.9g\n", max_abs);
    CHECK(max_abs < 3e-5f);
    puts("v4 sliding attention fixture tests: ok");
    return 0;
}
