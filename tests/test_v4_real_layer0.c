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
    if (argc != 7) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark> <layer0-oracle> <layers-0-2-oracle> <layer3-hca-oracle> <layers-3-4-oracle> <base-forward-oracle>\n", argv[0]);
        return 2;
    }
    enum { D = 4096, HC = 4, EXPERTS = 256, TOP_K = 6 };
    shards model, oracle, oracle_0_2, oracle_3, oracle_3_4, oracle_base;
    st_init(&model, argv[1]);
    st_init(&oracle, argv[2]);
    st_init(&oracle_0_2, argv[3]);
    st_init(&oracle_3, argv[4]);
    st_init(&oracle_3_4, argv[5]);
    st_init(&oracle_base, argv[6]);
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

    enum { HCA_TOKENS = 128 };
    int hca_token_ids[HCA_TOKENS];
    for (int i = 0; i < HCA_TOKENS; i++) hca_token_ids[i] = 3;
    float *hca_streams = load_f32(&oracle_3, "layer.3.input",
                                  HCA_TOKENS * HC * D);
    float *hca_projected_kv = malloc((size_t)HCA_TOKENS * 512 * sizeof(float));
    float *hca_projected_gate = malloc((size_t)HCA_TOKENS * 512 * sizeof(float));
    float *hca_router_scores = malloc((size_t)HCA_TOKENS * EXPERTS * sizeof(float));
    float *hca_router_weights = malloc((size_t)HCA_TOKENS * TOP_K * sizeof(float));
    int64_t *hca_router_indices = malloc((size_t)HCA_TOKENS * TOP_K * sizeof(int64_t));
    CHECK(hca_streams && hca_projected_kv && hca_projected_gate &&
          hca_router_scores && hca_router_weights && hca_router_indices);
    V4RealLayerState hca_state;
    V4RealLayerCapture hca_capture = {
        .projected_kv = hca_projected_kv,
        .projected_gate = hca_projected_gate,
        .router_scores = hca_router_scores,
        .router_weights = hca_router_weights,
        .router_indices = hca_router_indices,
    };
    CHECK(v4_real_layer_state_init(&hca_state, 3, HCA_TOKENS));
    CHECK(v4_real_layer_forward_state(&model, 3, hca_token_ids, HCA_TOKENS,
                                      hca_streams, &hca_state, &hca_capture));
    float *expected_hca_kv = load_f32(&oracle_3, "layer.3.hca.kv", HCA_TOKENS * 512);
    float *expected_hca_gate = load_f32(&oracle_3, "layer.3.hca.gate", HCA_TOKENS * 512);
    float *expected_hca_output = load_f32(&oracle_3, "layer.3.hca.output", 512);
    float *expected_hca_scores = load_f32(&oracle_3, "layer.3.router.scores",
                                          HCA_TOKENS * EXPERTS);
    float *expected_hca_weights = load_f32(&oracle_3, "layer.3.router.weights",
                                           HCA_TOKENS * TOP_K);
    int64_t *expected_hca_indices = malloc((size_t)HCA_TOKENS * TOP_K * sizeof(int64_t));
    float *expected_hca_layer = load_f32(&oracle_3, "layer.3.output",
                                         HCA_TOKENS * HC * D);
    CHECK(expected_hca_kv && expected_hca_gate && expected_hca_output &&
          expected_hca_scores && expected_hca_weights && expected_hca_indices &&
          expected_hca_layer);
    st_read_raw(&oracle_3, "layer.3.router.indices", expected_hca_indices, 0);
    float hca_kv_error = max_error(hca_projected_kv, expected_hca_kv,
                                   HCA_TOKENS * 512);
    float hca_gate_error = max_error(hca_projected_gate, expected_hca_gate,
                                     HCA_TOKENS * 512);
    float hca_compress_error = max_error(v4_kv_cache_key(&hca_state.compressed, 0),
                                         expected_hca_output, 512);
    float hca_score_error = max_error(hca_router_scores, expected_hca_scores,
                                      HCA_TOKENS * EXPERTS);
    float hca_weight_error = max_error(hca_router_weights, expected_hca_weights,
                                       HCA_TOKENS * TOP_K);
    float hca_layer_error = max_error(hca_streams, expected_hca_layer,
                                      HCA_TOKENS * HC * D);
    int hca_route_hits = 0;
    for (int i = 0; i < HCA_TOKENS * TOP_K; i++)
        if (hca_router_indices[i] == expected_hca_indices[i]) hca_route_hits++;
    printf("v4 real layer3 HCA: kv=%.9g gate=%.9g compress=%.9g router=%.9g weights=%.9g routes=%d/%d output=%.9g\n",
           hca_kv_error, hca_gate_error, hca_compress_error, hca_score_error,
           hca_weight_error, hca_route_hits, HCA_TOKENS * TOP_K, hca_layer_error);
    CHECK(hca_kv_error < 3e-4f && hca_gate_error < 3e-4f);
    CHECK(hca_compress_error < 3e-4f && hca_score_error < 3e-4f);
    CHECK(hca_weight_error < 3e-5f && hca_route_hits == HCA_TOKENS * TOP_K);
    CHECK(hca_layer_error < 3e-4f);
    v4_real_layer_state_free(&hca_state);
    free(hca_streams); free(hca_projected_kv); free(hca_projected_gate);
    free(hca_router_scores); free(hca_router_weights); free(hca_router_indices);
    free(expected_hca_kv); free(expected_hca_gate); free(expected_hca_output);
    free(expected_hca_scores); free(expected_hca_weights); free(expected_hca_indices);
    free(expected_hca_layer);

    float *transition_streams = load_f32(&oracle_3_4, "layer.3.input", 4 * HC * D);
    float transition_scores[4], transition_bias[4];
    int64_t transition_ids[4], transition_routes[4 * TOP_K];
    V4RealLayerCapture transition_capture = {
        .index_scores = transition_scores,
        .index_ids = transition_ids,
        .block_bias = transition_bias,
        .router_indices = transition_routes,
    };
    V4RealLayerState transition_hca, transition_csa;
    CHECK(transition_streams);
    CHECK(v4_real_layer_state_init(&transition_hca, 3, 4));
    CHECK(v4_real_layer_forward_state(&model, 3, token_ids, 4,
                                      transition_streams, &transition_hca, NULL));
    float *expected_layer3 = load_f32(&oracle_3_4, "layer.3.output", 4 * HC * D);
    CHECK(expected_layer3);
    float transition_layer3_error = max_error(transition_streams, expected_layer3,
                                              4 * HC * D);
    CHECK(v4_kv_cache_count(&transition_hca.compressed) == 0);
    CHECK(transition_hca.next_position == 4);
    CHECK(v4_real_layer_state_init(&transition_csa, 4, 4));
    CHECK(v4_real_layer_forward_state(&model, 4, token_ids, 4,
                                      transition_streams, &transition_csa,
                                      &transition_capture));
    float *expected_layer4 = load_f32(&oracle_3_4, "layer.4.output", 4 * HC * D);
    float *expected_layer4_kv = load_f32(&oracle_3_4, "layer.4.compressor.kv", 512);
    float *expected_layer4_scores = load_f32(&oracle_3_4,
                                             "layer.4.indexer.scores", 4);
    int64_t expected_layer4_ids[4], expected_layer4_routes[4 * TOP_K];
    st_read_raw(&oracle_3_4, "layer.4.indexer.indices", expected_layer4_ids, 0);
    st_read_raw(&oracle_3_4, "layer.4.router.indices", expected_layer4_routes, 0);
    CHECK(expected_layer4 && expected_layer4_kv && expected_layer4_scores);
    float transition_layer4_error = max_error(transition_streams, expected_layer4,
                                              4 * HC * D);
    float transition_compress_error = max_error(
        v4_kv_cache_key(&transition_csa.compressed, 0), expected_layer4_kv, 512);
    float transition_index_error = max_error(transition_scores,
                                             expected_layer4_scores, 4);
    int transition_index_hits = 0, transition_route_hits = 0, transition_bias_hits = 0;
    for (int i = 0; i < 4; i++) {
        if (transition_ids[i] == expected_layer4_ids[i]) transition_index_hits++;
        if ((transition_ids[i] >= 0 && transition_bias[i] == 0.0f) ||
            (transition_ids[i] < 0 && isinf(transition_bias[i]) &&
             transition_bias[i] < 0.0f)) transition_bias_hits++;
    }
    for (int i = 0; i < 4 * TOP_K; i++)
        if (transition_routes[i] == expected_layer4_routes[i]) transition_route_hits++;
    printf("v4 real layers 3-4: layer3=%.9g layer4=%.9g compress=%.9g indexer=%.9g indices=%d/4 bias=%d/4 routes=%d/24\n",
           transition_layer3_error, transition_layer4_error,
           transition_compress_error, transition_index_error,
           transition_index_hits, transition_bias_hits, transition_route_hits);
    CHECK(transition_layer3_error < 3e-4f && transition_layer4_error < 3e-4f);
    CHECK(transition_compress_error < 3e-4f && transition_index_error < 3e-4f);
    CHECK(transition_index_hits == 4 && transition_bias_hits == 4);
    CHECK(transition_route_hits == 4 * TOP_K);
    v4_real_layer_state_free(&transition_hca);
    v4_real_layer_state_free(&transition_csa);
    free(transition_streams); free(expected_layer3); free(expected_layer4);
    free(expected_layer4_kv); free(expected_layer4_scores);

    int checkpoint_layers[43];
    for (int layer = 0; layer < 43; layer++) checkpoint_layers[layer] = layer;
    float *base_streams = malloc((size_t)4 * HC * D * sizeof(float));
    float *base_checkpoints = malloc((size_t)43 * 4 * HC * D * sizeof(float));
    float *base_targets = malloc((size_t)3 * 4 * D * sizeof(float));
    int64_t *base_routes = malloc((size_t)43 * 4 * TOP_K * sizeof(int64_t));
    CHECK(base_streams && base_checkpoints && base_targets && base_routes);
    V4RealModelState base_state;
    V4RealBaseCapture base_capture = {
        .layer_ids = checkpoint_layers,
        .n_layer_ids = 43,
        .layer_outputs = base_checkpoints,
        .target_means = base_targets,
        .router_indices = base_routes,
    };
    CHECK(v4_real_model_state_init(&base_state, 4));
    CHECK(v4_real_base_layers_forward(&model, token_ids, 4, &base_state,
                                      base_streams, &base_capture));
    float base_errors[43];
    int first_route_miss = -1, first_route_set_miss = -1;
    for (int i = 0; i < 43; i++) {
        char name[64];
        snprintf(name, sizeof(name), "layer.%d.output", checkpoint_layers[i]);
        float *expected = load_f32(&oracle_base, name, 4 * HC * D);
        CHECK(expected);
        base_errors[i] = max_error(base_checkpoints + (int64_t)i * 4 * HC * D,
                                   expected, 4 * HC * D);
        free(expected);
        if (i >= 3) {
            snprintf(name, sizeof(name), "layer.%d.router.indices", i);
            int64_t expected_routes[4 * TOP_K];
            st_read_raw(&oracle_base, name, expected_routes, 0);
            for (int route = 0; route < 4 * TOP_K; route++)
                if (base_routes[((int64_t)i * 4 * TOP_K) + route]
                        != expected_routes[route] && first_route_miss < 0)
                    first_route_miss = i;
            for (int token = 0; token < 4; token++)
                for (int route = 0; route < TOP_K; route++) {
                    int found = 0;
                    int64_t actual = base_routes[
                        ((int64_t)i * 4 + token) * TOP_K + route];
                    for (int expected_rank = 0; expected_rank < TOP_K; expected_rank++)
                        if (actual == expected_routes[token * TOP_K + expected_rank])
                            found = 1;
                    if (!found && first_route_set_miss < 0)
                        first_route_set_miss = i;
                }
        }
    }
    float target_error = 0.0f;
    for (int i = 0; i < 3; i++) {
        char name[64];
        snprintf(name, sizeof(name), "layer.%d.mean", 40 + i);
        float *expected = load_f32(&oracle_base, name, 4 * D);
        CHECK(expected);
        float error = max_error(base_targets + (int64_t)i * 4 * D,
                                expected, 4 * D);
        if (error > target_error) target_error = error;
        free(expected);
    }
    printf("v4 real base layers: l4=%.9g l20=%.9g l40=%.9g l41=%.9g l42=%.9g targets=%.9g order_miss=%d set_miss=%d\n",
           base_errors[4], base_errors[20], base_errors[40], base_errors[41],
           base_errors[42], target_error, first_route_miss, first_route_set_miss);
    CHECK(base_errors[4] < 3e-4f);
    for (int i = 0; i < 43; i++) CHECK(isfinite(base_errors[i]));
    CHECK(isfinite(target_error));

    enum { VOCAB = 129280 };
    float *head_input = load_f32(&oracle_base, "layer.42.output", 4 * HC * D);
    float *head_hidden = malloc((size_t)4 * D * sizeof(float));
    float *head_norm = malloc((size_t)4 * D * sizeof(float));
    float *head_logits = malloc((size_t)VOCAB * sizeof(float));
    CHECK(head_input && head_hidden && head_norm && head_logits);
    V4RealHeadCapture head_capture = {
        .hidden = head_hidden, .normalized = head_norm, .logits = head_logits,
    };
    CHECK(v4_real_base_head_forward(&model, head_input, 4, &head_capture));
    float *expected_head_hidden = load_f32(&oracle_base, "final.hidden", 4 * D);
    float *expected_head_norm = load_f32(&oracle_base, "final.norm", 4 * D);
    float *expected_head_logits = load_f32(&oracle_base, "final.logits", VOCAB);
    int64_t expected_argmax;
    st_read_raw(&oracle_base, "final.argmax", &expected_argmax, 0);
    CHECK(expected_head_hidden && expected_head_norm && expected_head_logits);
    float head_hidden_error = max_error(head_hidden, expected_head_hidden, 4 * D);
    float head_norm_error = max_error(head_norm, expected_head_norm, 4 * D);
    float head_logits_error = max_error(head_logits, expected_head_logits, VOCAB);
    int isolated_argmax = 0;
    for (int token = 1; token < VOCAB; token++)
        if (head_logits[token] > head_logits[isolated_argmax]) isolated_argmax = token;
    CHECK(v4_real_base_head_forward(&model, base_streams, 4, &head_capture));
    int continuous_argmax = 0;
    for (int token = 1; token < VOCAB; token++)
        if (head_logits[token] > head_logits[continuous_argmax]) continuous_argmax = token;
    printf("v4 real base head: hidden=%.9g norm=%.9g logits=%.9g isolated=%d expected=%lld continuous=%d\n",
           head_hidden_error, head_norm_error, head_logits_error,
           isolated_argmax, (long long)expected_argmax, continuous_argmax);
    CHECK(head_hidden_error < 3e-4f && head_norm_error < 3e-4f);
    CHECK(head_logits_error < 3e-3f);
    CHECK(isolated_argmax == expected_argmax);
    CHECK(continuous_argmax == expected_argmax);
    free(head_input); free(head_hidden); free(head_norm); free(head_logits);
    free(expected_head_hidden); free(expected_head_norm); free(expected_head_logits);
    v4_real_model_state_free(&base_state);
    free(base_streams); free(base_checkpoints); free(base_targets); free(base_routes);
    puts("v4 real layer0 tests: ok");
    return 0;
}
