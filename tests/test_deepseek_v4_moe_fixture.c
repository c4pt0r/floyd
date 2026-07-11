#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../moe_exec.h"
#include "../st.h"

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

static int64_t *load_i64(shards *source, const char *name, int64_t expected) {
    st_tensor *tensor = st_find(source, name);
    if (!tensor || tensor->dtype != ST_DTYPE_I64 || tensor->numel != expected) {
        fprintf(stderr, "%s: missing or invalid I64 tensor\n", name);
        exit(1);
    }
    int64_t *data = malloc((size_t)expected * sizeof(int64_t));
    st_read_raw(source, name, data, 0);
    return data;
}

static MoeExpertF32 load_expert(shards *source, int layer, int expert) {
    char name[256];
    MoeExpertF32 result;
    snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%d.w1.weight", layer, expert);
    result.w1 = load_f32(source, name, 32 * 64);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%d.w2.weight", layer, expert);
    result.w2 = load_f32(source, name, 64 * 32);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%d.w3.weight", layer, expert);
    result.w3 = load_f32(source, name, 32 * 64);
    return result;
}

static MoeExpertF32 load_shared(shards *source, int layer) {
    char name[256];
    MoeExpertF32 result;
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.gate_proj.weight", layer);
    result.w1 = load_f32(source, name, 32 * 64);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.down_proj.weight", layer);
    result.w2 = load_f32(source, name, 64 * 32);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.up_proj.weight", layer);
    result.w3 = load_f32(source, name, 32 * 64);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_deepseek_v4>\n", argv[0]);
        return 2;
    }

    enum { D = 64, I = 32, E = 8, K = 2, V = 128 };
    shards source;
    st_init(&source, argv[1]);
    int64_t token_count = st_numel(&source, "input_ids");
    CHECK(token_count > 0);
    int64_t *input_ids_i64 = load_i64(&source, "input_ids", token_count);
    int *input_ids = malloc((size_t)token_count * sizeof(int));
    for (int64_t i = 0; i < token_count; i++) input_ids[i] = (int)input_ids_i64[i];

    for (int layer = 0; layer < 3; layer++) {
        char name[256];
        MoeExpertF32 experts[E];
        for (int expert = 0; expert < E; expert++) experts[expert] = load_expert(&source, layer, expert);
        MoeExpertF32 shared = load_shared(&source, layer);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer);
        float *router = load_f32(&source, name, E * D);
        float *bias = NULL;
        int64_t *fixed = NULL;
        if (layer == 0) {
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.tid2eid", layer);
            fixed = load_i64(&source, name, V * K);
        } else {
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.e_score_correction_bias", layer);
            bias = load_f32(&source, name, E);
        }

        snprintf(name, sizeof(name), "moe.%d.input", layer);
        float *input = load_f32(&source, name, token_count * D);
        snprintf(name, sizeof(name), "moe.%d.output", layer);
        float *expected = load_f32(&source, name, token_count * D);
        float *actual = malloc((size_t)token_count * D * sizeof(float));

        MoeF32 model = {
            .hidden = D, .intermediate = I, .n_experts = E, .top_k = K,
            .vocab_size = V, .score_fn = MOE_SCORE_SQRT_SOFTPLUS,
            .normalize = 1, .route_scale = 1.5f, .swiglu_limit = 10.0f,
            .router = router, .selection_bias = bias, .fixed_experts = fixed,
            .experts = experts, .shared_experts = &shared, .n_shared_experts = 1,
        };
        CHECK(moe_f32_forward(&model, input, input_ids, (int)token_count, actual));

        float max_abs = 0.0f;
        for (int64_t i = 0; i < token_count * D; i++) {
            float error = fabsf(actual[i] - expected[i]);
            if (error > max_abs) max_abs = error;
        }
        printf("DeepSeek V4 moe layer %d: max_abs=%.9g\n", layer, max_abs);
        CHECK(max_abs < 2e-5f);
    }
    puts("DeepSeek V4 moe fixture tests: ok");
    return 0;
}
