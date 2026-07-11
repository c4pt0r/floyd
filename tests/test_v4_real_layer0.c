#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_real_layer0.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float *load_f32(shards *source, const char *name, int64_t count) {
    if (st_numel(source, name) != count) return NULL;
    float *data = malloc((size_t)count * sizeof(float));
    st_read_f32(source, name, data, 0);
    return data;
}

static float max_error(const float *actual, const float *expected, int count) {
    float maximum = 0.0f;
    for (int i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
    }
    return maximum;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark> <oracle-dir>\n", argv[0]);
        return 2;
    }
    enum { D = 4096, HC = 4, EXPERTS = 256, TOP_K = 6 };
    shards model, oracle;
    st_init(&model, argv[1]);
    st_init(&oracle, argv[2]);
    float input[HC * D], attn_post[HC], attn_comb[HC * HC], attn_collapsed[D];
    float attn_norm[D], attn_output[D], after_attn[HC * D];
    float ffn_post[HC], ffn_comb[HC * HC], ffn_collapsed[D], ffn_norm[D];
    float router_scores[EXPERTS], router_weights[TOP_K], moe_output[D], output[HC * D];
    int64_t router_indices[TOP_K];
    V4RealLayer0Capture capture = {
        .input = input, .attn_post = attn_post, .attn_comb = attn_comb,
        .attn_collapsed = attn_collapsed, .attn_norm = attn_norm,
        .attn_output = attn_output, .after_attn = after_attn,
        .ffn_post = ffn_post, .ffn_comb = ffn_comb,
        .ffn_collapsed = ffn_collapsed, .ffn_norm = ffn_norm,
        .router_scores = router_scores, .router_weights = router_weights,
        .router_indices = router_indices, .moe_output = moe_output, .output = output,
    };
    CHECK(v4_real_layer0_forward(&model, 3, &capture));

    float attn_error = 0.0f, ffn_error = 0.0f;
#define ACCUM_ERROR(group, field, count) do { \
    float *expected = load_f32(&oracle, field, count); CHECK(expected); \
    float error = max_error(capture.group, expected, count); \
    if (error > group##_error) group##_error = error; free(expected); \
} while (0)
    ACCUM_ERROR(attn, "attn.post", HC);
    ACCUM_ERROR(attn, "attn.comb", HC * HC);
    ACCUM_ERROR(attn, "attn.collapsed", D);
    ACCUM_ERROR(attn, "attn.norm", D);
    ACCUM_ERROR(attn, "attn.output", D);
    ACCUM_ERROR(attn, "after_attn", HC * D);
    ACCUM_ERROR(ffn, "ffn.post", HC);
    ACCUM_ERROR(ffn, "ffn.comb", HC * HC);
    ACCUM_ERROR(ffn, "ffn.collapsed", D);
    ACCUM_ERROR(ffn, "ffn.norm", D);
    ACCUM_ERROR(ffn, "moe.output", D);
    ACCUM_ERROR(ffn, "output", HC * D);
#undef ACCUM_ERROR
    float *expected_scores = load_f32(&oracle, "router.scores", EXPERTS);
    float *expected_weights = load_f32(&oracle, "router.weights", TOP_K);
    CHECK(expected_scores && expected_weights);
    float router_error = max_error(router_scores, expected_scores, EXPERTS);
    float weight_error = max_error(router_weights, expected_weights, TOP_K);
    int64_t expected_indices[TOP_K];
    st_read_raw(&oracle, "router.indices", expected_indices, 0);
    int route_hits = 0;
    for (int i = 0; i < TOP_K; i++) if (router_indices[i] == expected_indices[i]) route_hits++;
    printf("v4 real layer0: attn=%.9g ffn=%.9g router=%.9g weights=%.9g routes=%d/%d\n",
           attn_error, ffn_error, router_error, weight_error, route_hits, TOP_K);
    CHECK(attn_error < 3e-4f && ffn_error < 3e-4f && router_error < 3e-4f);
    CHECK(weight_error < 3e-5f && route_hits == TOP_K);
    puts("v4 real layer0 tests: ok");
    return 0;
}
