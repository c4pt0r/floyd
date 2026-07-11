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
    if (argc != 4) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark> <layer0-oracle> <layers-0-2-oracle>\n", argv[0]);
        return 2;
    }
    enum { D = 4096, HC = 4, EXPERTS = 256, TOP_K = 6 };
    shards model, oracle, oracle_0_2;
    st_init(&model, argv[1]);
    st_init(&oracle, argv[2]);
    st_init(&oracle_0_2, argv[3]);
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
#define ACCUM_ERROR(group, member, field, count) do { \
    float *expected = load_f32(&oracle, field, count); CHECK(expected); \
    float error = max_error(capture.member, expected, count); \
    if (error > group##_error) group##_error = error; free(expected); \
} while (0)
    ACCUM_ERROR(attn, attn_post, "attn.post", HC);
    ACCUM_ERROR(attn, attn_comb, "attn.comb", HC * HC);
    ACCUM_ERROR(attn, attn_collapsed, "attn.collapsed", D);
    ACCUM_ERROR(attn, attn_norm, "attn.norm", D);
    ACCUM_ERROR(attn, attn_output, "attn.output", D);
    ACCUM_ERROR(attn, after_attn, "after_attn", HC * D);
    ACCUM_ERROR(ffn, ffn_post, "ffn.post", HC);
    ACCUM_ERROR(ffn, ffn_comb, "ffn.comb", HC * HC);
    ACCUM_ERROR(ffn, ffn_collapsed, "ffn.collapsed", D);
    ACCUM_ERROR(ffn, ffn_norm, "ffn.norm", D);
    ACCUM_ERROR(ffn, moe_output, "moe.output", D);
    ACCUM_ERROR(ffn, output, "output", HC * D);
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

    float *layer2_input = load_f32(&oracle_0_2, "layer.2.input", 4 * HC * D);
    float csa_kv[512], index_scores[4], block_bias[4];
    int64_t index_ids[4];
    V4RealLayer2CSACapture csa = {
        .compressed_kv = csa_kv, .index_scores = index_scores,
        .index_ids = index_ids, .block_bias = block_bias,
    };
    CHECK(layer2_input && v4_real_layer2_csa_forward(&model, layer2_input, &csa));
    float *expected_kv = load_f32(&oracle_0_2, "layer.2.compressor.kv", 512);
    float *expected_index_scores = load_f32(&oracle_0_2, "layer.2.indexer.scores", 4);
    int64_t expected_index_ids[4];
    st_read_raw(&oracle_0_2, "layer.2.indexer.indices", expected_index_ids, 0);
    float csa_error = max_error(csa_kv, expected_kv, 512);
    float index_error = max_error(index_scores, expected_index_scores, 4);
    int index_hits = 0, bias_hits = 0;
    for (int i = 0; i < 4; i++) {
        if (index_ids[i] == expected_index_ids[i]) index_hits++;
        if ((index_ids[i] >= 0 && block_bias[i] == 0.0f) ||
            (index_ids[i] < 0 && isinf(block_bias[i]) && block_bias[i] < 0.0f)) bias_hits++;
    }
    printf("v4 real layer2 CSA: compressor=%.9g indexer=%.9g indices=%d/4 bias=%d/4\n",
           csa_error, index_error, index_hits, bias_hits);
    CHECK(csa_error < 3e-4f && index_error < 3e-4f);
    CHECK(index_hits == 4 && bias_hits == 4);

    int token_ids[4] = {3, 14, 15, 9};
    float *streams = load_f32(&oracle_0_2, "layer.0.input", 4 * HC * D);
    float *layer_outputs = malloc((size_t)3 * 4 * HC * D * sizeof(float));
    float runner_kv[512], runner_scores[4], runner_bias[4];
    int64_t runner_ids[4];
    V4RealLayer2CSACapture runner_csa = {
        .compressed_kv = runner_kv, .index_scores = runner_scores,
        .index_ids = runner_ids, .block_bias = runner_bias,
    };
    CHECK(streams && layer_outputs);
    CHECK(v4_real_layers_forward(&model, token_ids, 4, 0, 3, streams,
                                 layer_outputs, &runner_csa));
    float layer_errors[3];
    for (int layer = 0; layer < 3; layer++) {
        char name[64];
        snprintf(name, sizeof(name), "layer.%d.output", layer);
        float *expected = load_f32(&oracle_0_2, name, 4 * HC * D);
        CHECK(expected);
        layer_errors[layer] = max_error(
            layer_outputs + (int64_t)layer * 4 * HC * D,
            expected, 4 * HC * D);
        free(expected);
    }
    printf("v4 real layers 0-2: layer0=%.9g layer1=%.9g layer2=%.9g\n",
           layer_errors[0], layer_errors[1], layer_errors[2]);
    CHECK(layer_errors[0] < 3e-4f);
    CHECK(layer_errors[1] < 6e-4f);
    CHECK(layer_errors[2] < 9e-4f);
    free(streams);
    free(layer_outputs);
    puts("v4 real layer0 tests: ok");
    return 0;
}
