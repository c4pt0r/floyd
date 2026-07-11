#ifndef DEEPSEEK_V4_FORWARD_LAYER0_H
#define DEEPSEEK_V4_FORWARD_LAYER0_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moe_route.h"
#include "st.h"
#include "deepseek_v4_compress.h"
#include "deepseek_v4_hc.h"
#include "deepseek_v4_indexer.h"
#include "deepseek_v4_kv_cache.h"
#include "deepseek_v4_quant.h"

enum {
    DEEPSEEK_V4_FORWARD_D = 4096,
    DEEPSEEK_V4_FORWARD_HC = 4,
    DEEPSEEK_V4_FORWARD_HEADS = 64,
    DEEPSEEK_V4_FORWARD_HD = 512,
    DEEPSEEK_V4_FORWARD_QR = 1024,
    DEEPSEEK_V4_FORWARD_INTER = 2048,
    DEEPSEEK_V4_FORWARD_EXPERTS = 256,
    DEEPSEEK_V4_FORWARD_TOPK = 6,
};

typedef struct {
    float *input;
    float *attn_post;
    float *attn_comb;
    float *attn_collapsed;
    float *attn_norm;
    float *attn_output;
    float *after_attn;
    float *ffn_post;
    float *ffn_comb;
    float *ffn_collapsed;
    float *ffn_norm;
    float *router_scores;
    float *router_weights;
    int64_t *router_indices;
    float *moe_output;
    float *output;
} DeepSeekV4ForwardLayer0Capture;

typedef struct {
    float *compressed_kv;
    float *index_scores;
    int64_t *index_ids;
    float *block_bias;
} DeepSeekV4ForwardLayer2CSACapture;

typedef struct {
    float *projected_kv;
    float *projected_gate;
    float *index_scores;
    int64_t *index_ids;
    float *block_bias;
    float *router_scores;
    float *router_weights;
    int64_t *router_indices;
} DeepSeekV4ForwardLayerCapture;

typedef struct {
    int layer;
    int ratio;
    int max_context;
    int next_position;
    DeepSeekV4KVCacheF32 window;
    DeepSeekV4KVCacheF32 compressed;
    DeepSeekV4KVCacheF32 index;
    float *window_keys;
    int64_t *window_positions;
    float *compressed_keys;
    int64_t *compressed_positions;
    float *index_keys;
    int64_t *index_positions;
    float *compressor_kv_state;
    float *compressor_gate_state;
    float *index_kv_state;
    float *index_gate_state;
} DeepSeekV4ForwardLayerState;

typedef struct {
    int max_context;
    DeepSeekV4ForwardLayerState layers[43];
} DeepSeekV4ForwardModelState;

typedef struct {
    const int *layer_ids;
    int n_layer_ids;
    float *layer_outputs;
    float *target_means;
    int64_t *router_indices;
} DeepSeekV4ForwardBaseCapture;

typedef struct {
    float *hidden;
    float *normalized;
    float *logits;
} DeepSeekV4ForwardHeadCapture;

typedef struct {
    float *main_kv;
    int next_position;
} DeepSeekV4ForwardDSparkState;

typedef struct {
    float *prefill_kv;
    float *stage_outputs;
    float *hidden;
    int64_t *output_ids;
    float *confidence;
} DeepSeekV4ForwardDSparkCapture;

static inline int deepseek_v4_forward_dspark_state_init(DeepSeekV4ForwardDSparkState *state) {
    if (!state) return 0;
    memset(state, 0, sizeof(*state));
    state->main_kv = calloc((size_t)3 * 128 * DEEPSEEK_V4_FORWARD_HD, sizeof(float));
    return state->main_kv != NULL;
}

static inline void deepseek_v4_forward_dspark_state_free(DeepSeekV4ForwardDSparkState *state) {
    if (!state) return;
    free(state->main_kv);
    memset(state, 0, sizeof(*state));
}

static inline int deepseek_v4_forward_layer_ratio(int layer) {
    if (layer < 0 || layer > 42) return -1;
    if (layer < 2) return 0;
    return (layer & 1) ? 128 : 4;
}

static inline void deepseek_v4_forward_layer_state_free(DeepSeekV4ForwardLayerState *state) {
    if (!state) return;
    free(state->window_keys); free(state->window_positions);
    free(state->compressed_keys); free(state->compressed_positions);
    free(state->index_keys); free(state->index_positions);
    free(state->compressor_kv_state); free(state->compressor_gate_state);
    free(state->index_kv_state); free(state->index_gate_state);
    memset(state, 0, sizeof(*state));
}

static inline int deepseek_v4_forward_layer_state_init(DeepSeekV4ForwardLayerState *state, int layer,
                                            int max_context) {
    if (!state || max_context <= 0) return 0;
    int ratio = deepseek_v4_forward_layer_ratio(layer);
    if (ratio < 0) return 0;
    memset(state, 0, sizeof(*state));
    state->layer = layer;
    state->ratio = ratio;
    state->max_context = max_context;
    int compressed_capacity = ratio ? (max_context + ratio - 1) / ratio : 1;
    state->window_keys = malloc((size_t)128 * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    state->window_positions = malloc(128 * sizeof(int64_t));
    state->compressed_keys = malloc((size_t)compressed_capacity * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    state->compressed_positions = malloc((size_t)compressed_capacity * sizeof(int64_t));
    if (!state->window_keys || !state->window_positions ||
        !state->compressed_keys || !state->compressed_positions) goto fail;
    if (!deepseek_v4_kv_cache_init(&state->window, DEEPSEEK_V4_KV_CACHE_RING, 128, DEEPSEEK_V4_FORWARD_HD,
                          state->window_keys, state->window_positions) ||
        !deepseek_v4_kv_cache_init(&state->compressed, DEEPSEEK_V4_KV_CACHE_APPEND,
                          compressed_capacity, DEEPSEEK_V4_FORWARD_HD,
                          state->compressed_keys, state->compressed_positions)) goto fail;
    if (ratio) {
        int overlap = ratio == 4;
        int rows = (1 + overlap) * ratio;
        int width = (1 + overlap) * DEEPSEEK_V4_FORWARD_HD;
        state->compressor_kv_state = calloc((size_t)rows * width, sizeof(float));
        state->compressor_gate_state = malloc((size_t)rows * width * sizeof(float));
        if (!state->compressor_kv_state || !state->compressor_gate_state) goto fail;
        for (int64_t i = 0; i < (int64_t)rows * width; i++)
            state->compressor_gate_state[i] = -INFINITY;
    }
    if (ratio == 4) {
        state->index_keys = malloc((size_t)compressed_capacity * 128 * sizeof(float));
        state->index_positions = malloc((size_t)compressed_capacity * sizeof(int64_t));
        state->index_kv_state = calloc((size_t)8 * 256, sizeof(float));
        state->index_gate_state = malloc((size_t)8 * 256 * sizeof(float));
        if (!state->index_keys || !state->index_positions ||
            !state->index_kv_state || !state->index_gate_state) goto fail;
        for (int i = 0; i < 8 * 256; i++) state->index_gate_state[i] = -INFINITY;
        if (!deepseek_v4_kv_cache_init(&state->index, DEEPSEEK_V4_KV_CACHE_APPEND,
                              compressed_capacity, 128,
                              state->index_keys, state->index_positions)) goto fail;
    }
    return 1;
fail:
    deepseek_v4_forward_layer_state_free(state);
    return 0;
}

static inline void deepseek_v4_forward_model_state_free(DeepSeekV4ForwardModelState *state) {
    if (!state) return;
    for (int layer = 0; layer < 43; layer++)
        deepseek_v4_forward_layer_state_free(&state->layers[layer]);
    memset(state, 0, sizeof(*state));
}

static inline int deepseek_v4_forward_model_state_init(DeepSeekV4ForwardModelState *state,
                                            int max_context) {
    if (!state || max_context <= 0) return 0;
    memset(state, 0, sizeof(*state));
    state->max_context = max_context;
    for (int layer = 0; layer < 43; layer++)
        if (!deepseek_v4_forward_layer_state_init(&state->layers[layer], layer,
                                      max_context)) {
            deepseek_v4_forward_model_state_free(state);
            return 0;
        }
    return 1;
}

static inline float *deepseek_v4_forward_read_f32(shards *source, const char *name) {
    int64_t count = st_numel(source, name);
    if (count <= 0 || (uint64_t)count > SIZE_MAX / sizeof(float)) return NULL;
    float *data = malloc((size_t)count * sizeof(float));
    if (!data) return NULL;
    st_read_f32(source, name, data, 0);
    return data;
}

static inline uint8_t *deepseek_v4_forward_read_raw(shards *source, const char *name) {
    int64_t count = st_nbytes(source, name);
    if (count <= 0 || (uint64_t)count > SIZE_MAX) return NULL;
    uint8_t *data = malloc((size_t)count);
    if (!data) return NULL;
    st_read_raw(source, name, data, 0);
    return data;
}

static inline void deepseek_v4_forward_matvec(float *output, const float *weight,
                                  const float *input, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        const float *w = weight + (int64_t)row * columns;
        for (int column = 0; column < columns; column++) sum += w[column] * input[column];
        output[row] = sum;
    }
}

static inline int deepseek_v4_forward_fp8_linear(shards *source, const char *prefix,
                                     const float *input, int rows, int columns,
                                     float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = deepseek_v4_forward_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = deepseek_v4_forward_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = deepseek_v4_fp8_matmul_f32(output, input, 1, weight, scale, rows, columns, 128);
    free(weight); free(scale);
    return ok;
}

static inline int deepseek_v4_forward_fp4_linear(shards *source, const char *prefix,
                                     const float *input, int rows, int columns,
                                     float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = deepseek_v4_forward_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = deepseek_v4_forward_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = deepseek_v4_fp4_matmul_f32(output, input, 1, weight, scale, rows, columns);
    free(weight); free(scale);
    return ok;
}

static inline int deepseek_v4_forward_norm(shards *source, const char *name,
                               const float *input, int count, float *output) {
    float *weight = deepseek_v4_forward_read_f32(source, name);
    if (!weight) return 0;
    float mean_square = 0.0f;
    for (int i = 0; i < count; i++) mean_square += input[i] * input[i];
    float scale = 1.0f / sqrtf(mean_square / count + 1e-6f);
    for (int i = 0; i < count; i++) output[i] = input[i] * scale * weight[i];
    free(weight);
    return 1;
}

static inline int deepseek_v4_forward_hc_layer(shards *source, int layer, const char *site,
                                   const float *input, int n_tokens,
                                   float *post, float *comb, float *collapsed) {
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.hc_%s_fn", layer, site);
    float *fn = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.hc_%s_base", layer, site);
    float *base = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.hc_%s_scale", layer, site);
    float *scale = deepseek_v4_forward_read_f32(source, name);
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    DeepSeekV4HyperConnection model = {
        .streams = DEEPSEEK_V4_FORWARD_HC, .hidden = DEEPSEEK_V4_FORWARD_D, .sinkhorn_iters = 20,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f, .fn = fn, .base = base, .scale = scale,
    };
    int ok = deepseek_v4_hc_forward(&model, input, n_tokens, post, comb, collapsed);
    free(fn); free(base); free(scale);
    return ok;
}

static inline int deepseek_v4_forward_hc(shards *source, const char *site, const float *input,
                             float *post, float *comb, float *collapsed) {
    return deepseek_v4_forward_hc_layer(source, 0, site, input, 1, post, comb, collapsed);
}

static inline void deepseek_v4_forward_hc_post(float *output, const float *block,
                                   const float *residual, const float *post,
                                   const float *comb) {
    for (int row = 0; row < DEEPSEEK_V4_FORWARD_HC; row++) {
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) {
            float sum = post[row] * block[d];
            for (int column = 0; column < DEEPSEEK_V4_FORWARD_HC; column++)
                sum += comb[row * DEEPSEEK_V4_FORWARD_HC + column]
                     * residual[column * DEEPSEEK_V4_FORWARD_D + d];
            output[row * DEEPSEEK_V4_FORWARD_D + d] = sum;
        }
    }
}

static inline int deepseek_v4_forward_attention(shards *source, const float *input, float *output) {
    float *q_a = malloc(DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q = malloc((size_t)DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *kv = malloc(DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *context = malloc((size_t)DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *grouped = malloc(8192 * sizeof(float));
    if (!q_a || !q || !kv || !context || !grouped) goto fail;
    if (!deepseek_v4_forward_fp8_linear(source, "layers.0.attn.wq_a", input,
                            DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    float *q_norm = malloc(DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    if (!q_norm || !deepseek_v4_forward_norm(source, "layers.0.attn.q_norm.weight",
                                  q_a, DEEPSEEK_V4_FORWARD_QR, q_norm)) { free(q_norm); goto fail; }
    if (!deepseek_v4_forward_fp8_linear(source, "layers.0.attn.wq_b", q_norm,
                            DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_QR, q)) {
        free(q_norm); goto fail;
    }
    free(q_norm);
    for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
        float *row = q + (int64_t)head * DEEPSEEK_V4_FORWARD_HD;
        float mean_square = 0.0f;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) mean_square += row[d] * row[d];
        float scale = 1.0f / sqrtf(mean_square / DEEPSEEK_V4_FORWARD_HD + 1e-6f);
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) row[d] *= scale;
    }
    if (!deepseek_v4_forward_fp8_linear(source, "layers.0.attn.wkv", input,
                            DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_D, kv)) goto fail;
    float *kv_norm = malloc(DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    if (!kv_norm || !deepseek_v4_forward_norm(source, "layers.0.attn.kv_norm.weight",
                                   kv, DEEPSEEK_V4_FORWARD_HD, kv_norm)) { free(kv_norm); goto fail; }
    memcpy(kv, kv_norm, DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    free(kv_norm);
    float *sinks = deepseek_v4_forward_read_f32(source, "layers.0.attn.attn_sink");
    if (!sinks) goto fail;
    float attention_scale = 1.0f / sqrtf((float)DEEPSEEK_V4_FORWARD_HD);
    for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
        float score = 0.0f;
        const float *query = q + (int64_t)head * DEEPSEEK_V4_FORWARD_HD;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) score += query[d] * kv[d];
        score *= attention_scale;
        float maximum = score > sinks[head] ? score : sinks[head];
        float probability = expf(score - maximum)
            / (expf(score - maximum) + expf(sinks[head] - maximum));
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
            context[(int64_t)head * DEEPSEEK_V4_FORWARD_HD + d] = probability * kv[d];
    }
    free(sinks);

    uint8_t *wo_a = deepseek_v4_forward_read_raw(source, "layers.0.attn.wo_a.weight");
    uint8_t *wo_a_scale = deepseek_v4_forward_read_raw(source, "layers.0.attn.wo_a.scale");
    if (!wo_a || !wo_a_scale) { free(wo_a); free(wo_a_scale); goto fail; }
    for (int group = 0; group < 8; group++) {
        if (!deepseek_v4_fp8_matmul_f32(grouped + group * 1024, context + group * 4096, 1,
                wo_a + (int64_t)group * 1024 * 4096,
                wo_a_scale + (int64_t)group * 8 * 32, 1024, 4096, 128)) {
            free(wo_a); free(wo_a_scale); goto fail;
        }
    }
    free(wo_a); free(wo_a_scale);
    if (!deepseek_v4_forward_fp8_linear(source, "layers.0.attn.wo_b", grouped,
                            DEEPSEEK_V4_FORWARD_D, 8192, output)) goto fail;
    free(q_a); free(q); free(kv); free(context); free(grouped);
    return 1;
fail:
    free(q_a); free(q); free(kv); free(context); free(grouped);
    return 0;
}

static inline int deepseek_v4_forward_expert(shards *source, const char *prefix,
                                 const float *input, int fp4, float coefficient,
                                 float *output) {
    float *gate = malloc(DEEPSEEK_V4_FORWARD_INTER * sizeof(float));
    float *up = malloc(DEEPSEEK_V4_FORWARD_INTER * sizeof(float));
    float *activated = malloc(DEEPSEEK_V4_FORWARD_INTER * sizeof(float));
    if (!gate || !up || !activated) { free(gate); free(up); free(activated); return 0; }
    char name[320];
    snprintf(name, sizeof(name), "%s.w1", prefix);
    int ok = fp4 ? deepseek_v4_forward_fp4_linear(source, name, input, DEEPSEEK_V4_FORWARD_INTER, DEEPSEEK_V4_FORWARD_D, gate)
                 : deepseek_v4_forward_fp8_linear(source, name, input, DEEPSEEK_V4_FORWARD_INTER, DEEPSEEK_V4_FORWARD_D, gate);
    snprintf(name, sizeof(name), "%s.w3", prefix);
    ok = ok && (fp4 ? deepseek_v4_forward_fp4_linear(source, name, input, DEEPSEEK_V4_FORWARD_INTER, DEEPSEEK_V4_FORWARD_D, up)
                    : deepseek_v4_forward_fp8_linear(source, name, input, DEEPSEEK_V4_FORWARD_INTER, DEEPSEEK_V4_FORWARD_D, up));
    if (!ok) { free(gate); free(up); free(activated); return 0; }
    for (int i = 0; i < DEEPSEEK_V4_FORWARD_INTER; i++) {
        if (gate[i] > 10.0f) gate[i] = 10.0f;
        if (up[i] > 10.0f) up[i] = 10.0f;
        if (up[i] < -10.0f) up[i] = -10.0f;
        activated[i] = coefficient * (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    }
    snprintf(name, sizeof(name), "%s.w2", prefix);
    ok = fp4 ? deepseek_v4_forward_fp4_linear(source, name, activated, DEEPSEEK_V4_FORWARD_D, DEEPSEEK_V4_FORWARD_INTER, output)
             : deepseek_v4_forward_fp8_linear(source, name, activated, DEEPSEEK_V4_FORWARD_D, DEEPSEEK_V4_FORWARD_INTER, output);
    free(gate); free(up); free(activated);
    return ok;
}

static inline int deepseek_v4_forward_moe(shards *source, const float *input, int token_id,
                              DeepSeekV4ForwardLayer0Capture *capture) {
    float *router = deepseek_v4_forward_read_f32(source, "layers.0.ffn.gate.weight");
    float logits[DEEPSEEK_V4_FORWARD_EXPERTS];
    int indices[DEEPSEEK_V4_FORWARD_TOPK];
    if (!router) return 0;
    deepseek_v4_forward_matvec(logits, router, input, DEEPSEEK_V4_FORWARD_EXPERTS, DEEPSEEK_V4_FORWARD_D);
    free(router);
    st_read_slice_raw(source, "layers.0.ffn.gate.tid2eid",
                      (int64_t)token_id * DEEPSEEK_V4_FORWARD_TOPK * sizeof(int64_t),
                      DEEPSEEK_V4_FORWARD_TOPK * sizeof(int64_t), capture->router_indices, 0);
    for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++) indices[k] = (int)capture->router_indices[k];
    if (moe_route_fixed(logits, DEEPSEEK_V4_FORWARD_EXPERTS, indices, DEEPSEEK_V4_FORWARD_TOPK,
                        MOE_SCORE_SQRT_SOFTPLUS, capture->router_weights,
                        capture->router_scores) != DEEPSEEK_V4_FORWARD_TOPK) return 0;
    moe_route_finalize(capture->router_weights, DEEPSEEK_V4_FORWARD_TOPK, 1, 1.5f);
    memset(capture->moe_output, 0, DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *expert_output = malloc(DEEPSEEK_V4_FORWARD_D * sizeof(float));
    if (!expert_output) return 0;
    char prefix[320];
    for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++) {
        snprintf(prefix, sizeof(prefix), "layers.0.ffn.experts.%d", indices[k]);
        if (!deepseek_v4_forward_expert(source, prefix, input, 1,
                            capture->router_weights[k], expert_output)) {
            free(expert_output); return 0;
        }
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) capture->moe_output[d] += expert_output[d];
    }
    if (!deepseek_v4_forward_expert(source, "layers.0.ffn.shared_experts", input, 0, 1.0f,
                        expert_output)) { free(expert_output); return 0; }
    for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) capture->moe_output[d] += expert_output[d];
    free(expert_output);
    return 1;
}

static inline int deepseek_v4_forward_fp8_batch(shards *source, const char *prefix,
                                    const float *input, int n_inputs,
                                    int rows, int columns, float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = deepseek_v4_forward_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = deepseek_v4_forward_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = deepseek_v4_fp8_matmul_f32(output, input, n_inputs, weight, scale,
                               rows, columns, 128);
    free(weight); free(scale);
    return ok;
}

static inline int deepseek_v4_forward_dense_batch(shards *source, const char *name,
                                      const float *input, int n_inputs,
                                      int rows, int columns, float *output) {
    float *weight = deepseek_v4_forward_read_f32(source, name);
    if (!weight) return 0;
    for (int token = 0; token < n_inputs; token++)
        deepseek_v4_forward_matvec(output + (int64_t)token * rows, weight,
                       input + (int64_t)token * columns, rows, columns);
    free(weight);
    return 1;
}

static inline int deepseek_v4_forward_norm_batch(shards *source, const char *name,
                                     const float *input, int n_inputs,
                                     int count, float *output) {
    float *weight = deepseek_v4_forward_read_f32(source, name);
    if (!weight) return 0;
    for (int token = 0; token < n_inputs; token++) {
        const float *x = input + (int64_t)token * count;
        float *y = output + (int64_t)token * count;
        float mean_square = 0.0f;
        for (int i = 0; i < count; i++) mean_square += x[i] * x[i];
        float scale = 1.0f / sqrtf(mean_square / count + 1e-6f);
        for (int i = 0; i < count; i++) y[i] = x[i] * scale * weight[i];
    }
    free(weight);
    return 1;
}

static inline float deepseek_v4_forward_compress_frequency(int pair) {
    const int dim = 64;
    const float base = 160000.0f;
    float frequency = powf(base, -(2.0f * pair) / dim);
    float low_value = dim * logf(65536.0f / (32.0f * 2.0f * (float)M_PI))
                    / (2.0f * logf(base));
    float high_value = dim * logf(65536.0f / (1.0f * 2.0f * (float)M_PI))
                     / (2.0f * logf(base));
    int low = (int)floorf(low_value); if (low < 0) low = 0;
    int high = (int)ceilf(high_value); if (high > dim - 1) high = dim - 1;
    float ramp = ((float)pair - low) / (high - low);
    if (ramp < 0.0f) ramp = 0.0f; if (ramp > 1.0f) ramp = 1.0f;
    float smooth = 1.0f - ramp;
    return frequency / 16.0f * (1.0f - smooth) + frequency * smooth;
}

static inline void deepseek_v4_forward_compress_rope(float *vector, int width,
                                         int64_t position) {
    int start = width - 64;
    for (int pair = 0; pair < 32; pair++) {
        float angle = (float)position * deepseek_v4_forward_compress_frequency(pair);
        float cosine = cosf(angle), sine = sinf(angle);
        int index = start + pair * 2;
        float first = vector[index], second = vector[index + 1];
        vector[index] = first * cosine - second * sine;
        vector[index + 1] = second * cosine + first * sine;
    }
}

static inline int deepseek_v4_forward_layer2_csa_forward(shards *source, const float *streams,
                                              DeepSeekV4ForwardLayer2CSACapture *capture) {
    if (!source || !streams || !capture || !capture->compressed_kv ||
        !capture->index_scores || !capture->index_ids || !capture->block_bias) return 0;
    enum { TOKENS = 4 };
    float *post = malloc(TOKENS * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *comb = malloc(TOKENS * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *collapsed = malloc(TOKENS * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *hidden = malloc(TOKENS * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *q_a = malloc(TOKENS * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q_norm = malloc(TOKENS * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !q_a || !q_norm) goto fail;
    if (!deepseek_v4_forward_hc_layer(source, 2, "attn", streams, TOKENS,
                          post, comb, collapsed)) goto fail;
    if (!deepseek_v4_forward_norm_batch(source, "layers.2.attn_norm.weight", collapsed,
                            TOKENS, DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (!deepseek_v4_forward_fp8_batch(source, "layers.2.attn.wq_a", hidden, TOKENS,
                           DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    if (!deepseek_v4_forward_norm_batch(source, "layers.2.attn.q_norm.weight", q_a,
                            TOKENS, DEEPSEEK_V4_FORWARD_QR, q_norm)) goto fail;

    float *outer_kv = malloc(TOKENS * 1024 * sizeof(float));
    float *outer_gate = malloc(TOKENS * 1024 * sizeof(float));
    float *inner_kv = malloc(TOKENS * 256 * sizeof(float));
    float *inner_gate = malloc(TOKENS * 256 * sizeof(float));
    if (!outer_kv || !outer_gate || !inner_kv || !inner_gate) goto fail2;
    if (!deepseek_v4_forward_dense_batch(source, "layers.2.attn.compressor.wkv.weight",
                             hidden, TOKENS, 1024, DEEPSEEK_V4_FORWARD_D, outer_kv) ||
        !deepseek_v4_forward_dense_batch(source, "layers.2.attn.compressor.wgate.weight",
                             hidden, TOKENS, 1024, DEEPSEEK_V4_FORWARD_D, outer_gate) ||
        !deepseek_v4_forward_dense_batch(source, "layers.2.attn.indexer.compressor.wkv.weight",
                             hidden, TOKENS, 256, DEEPSEEK_V4_FORWARD_D, inner_kv) ||
        !deepseek_v4_forward_dense_batch(source, "layers.2.attn.indexer.compressor.wgate.weight",
                             hidden, TOKENS, 256, DEEPSEEK_V4_FORWARD_D, inner_gate)) goto fail2;
    float *outer_ape = deepseek_v4_forward_read_f32(source, "layers.2.attn.compressor.ape");
    float *outer_norm = deepseek_v4_forward_read_f32(source, "layers.2.attn.compressor.norm.weight");
    float *inner_ape = deepseek_v4_forward_read_f32(source, "layers.2.attn.indexer.compressor.ape");
    float *inner_norm = deepseek_v4_forward_read_f32(source, "layers.2.attn.indexer.compressor.norm.weight");
    float index_kv[128];
    if (!outer_ape || !outer_norm || !inner_ape || !inner_norm) goto fail3;
    DeepSeekV4PrefillCompressorF32 outer = {
        .ratio = 4, .head_dim = 512, .rope_dim = 64, .overlap = 1,
        .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = outer_ape, .norm_weight = outer_norm,
    };
    DeepSeekV4PrefillCompressorF32 inner = {
        .ratio = 4, .head_dim = 128, .rope_dim = 64, .overlap = 1,
        .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = inner_ape, .norm_weight = inner_norm,
    };
    if (!deepseek_v4_compress_prefill_f32(&outer, outer_kv, outer_gate, TOKENS,
                                  capture->compressed_kv) ||
        !deepseek_v4_compress_prefill_f32(&inner, inner_kv, inner_gate, TOKENS, index_kv)) goto fail3;

    float *queries = malloc(TOKENS * 64 * 128 * sizeof(float));
    float *head_weights = malloc(TOKENS * 64 * sizeof(float));
    if (!queries || !head_weights) { free(queries); free(head_weights); goto fail3; }
    if (!deepseek_v4_forward_fp8_batch(source, "layers.2.attn.indexer.wq_b", q_norm,
                           TOKENS, 64 * 128, DEEPSEEK_V4_FORWARD_QR, queries) ||
        !deepseek_v4_forward_dense_batch(source, "layers.2.attn.indexer.weights_proj.weight",
                             hidden, TOKENS, 64, DEEPSEEK_V4_FORWARD_D, head_weights)) {
        free(queries); free(head_weights); goto fail3;
    }
    for (int token = 0; token < TOKENS; token++)
        for (int head = 0; head < 64; head++)
            deepseek_v4_forward_compress_rope(queries + ((int64_t)token * 64 + head) * 128,
                                  128, token);
    int64_t positions[4] = {0, 1, 2, 3};
    DeepSeekV4LightningIndexerF32 indexer = {
        .n_heads = 64, .head_dim = 128, .ratio = 4, .top_k = 1,
    };
    int ok = deepseek_v4_indexer_forward_f32(&indexer, queries, index_kv, head_weights,
                                    positions, TOKENS, 1, capture->index_scores,
                                    capture->index_ids);
    for (int token = 0; token < TOKENS; token++)
        capture->block_bias[token] = capture->index_ids[token] >= 0 ? 0.0f : -INFINITY;
    free(queries); free(head_weights);
    free(outer_ape); free(outer_norm); free(inner_ape); free(inner_norm);
    free(outer_kv); free(outer_gate); free(inner_kv); free(inner_gate);
    free(post); free(comb); free(collapsed); free(hidden); free(q_a); free(q_norm);
    return ok;
fail3:
    free(outer_ape); free(outer_norm); free(inner_ape); free(inner_norm);
fail2:
    free(outer_kv); free(outer_gate); free(inner_kv); free(inner_gate);
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(q_a); free(q_norm);
    return 0;
}

static inline int deepseek_v4_forward_compressor_update(
        shards *source, const char *prefix, const float *hidden, int n_tokens,
        int start_position, int ratio, int head_dim,
        float *kv_state, float *gate_state, DeepSeekV4KVCacheF32 *cache,
        float *projected_kv_capture, float *projected_gate_capture) {
    if (!source || !prefix || !hidden || n_tokens <= 0 || start_position < 0 ||
        ratio <= 0 || head_dim <= 0 || !kv_state || !gate_state || !cache)
        return 0;
    int overlap = ratio == 4;
    int width = (1 + overlap) * head_dim;
    float *kv = malloc((size_t)n_tokens * width * sizeof(float));
    float *gate = malloc((size_t)n_tokens * width * sizeof(float));
    float *pooled = malloc((size_t)head_dim * sizeof(float));
    char name[320];
    if (!kv || !gate || !pooled) goto fail;
    snprintf(name, sizeof(name), "%s.wkv.weight", prefix);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             width, DEEPSEEK_V4_FORWARD_D, kv)) goto fail;
    snprintf(name, sizeof(name), "%s.wgate.weight", prefix);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             width, DEEPSEEK_V4_FORWARD_D, gate)) goto fail;
    if (projected_kv_capture)
        memcpy(projected_kv_capture, kv,
               (size_t)n_tokens * width * sizeof(float));
    if (projected_gate_capture)
        memcpy(projected_gate_capture, gate,
               (size_t)n_tokens * width * sizeof(float));
    snprintf(name, sizeof(name), "%s.ape", prefix);
    float *ape = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.norm.weight", prefix);
    float *norm = deepseek_v4_forward_read_f32(source, name);
    if (!ape || !norm) { free(ape); free(norm); goto fail; }

    for (int token = 0; token < n_tokens; token++) {
        int position = start_position + token;
        int slot = position % ratio;
        int row = overlap ? ratio + slot : slot;
        memcpy(kv_state + (int64_t)row * width,
               kv + (int64_t)token * width, (size_t)width * sizeof(float));
        for (int d = 0; d < width; d++)
            gate_state[(int64_t)row * width + d]
                = gate[(int64_t)token * width + d]
                + ape[(int64_t)slot * width + d];
        if ((position + 1) % ratio) continue;

        int slots = overlap ? 2 * ratio : ratio;
        for (int d = 0; d < head_dim; d++) {
            float maximum = -INFINITY;
            for (int pool_slot = 0; pool_slot < slots; pool_slot++) {
                int source_d = overlap && pool_slot >= ratio
                             ? d + head_dim : d;
                float score = gate_state[(int64_t)pool_slot * width + source_d];
                if (score > maximum) maximum = score;
            }
            float denominator = 0.0f, weighted = 0.0f;
            for (int pool_slot = 0; pool_slot < slots; pool_slot++) {
                int source_d = overlap && pool_slot >= ratio
                             ? d + head_dim : d;
                float score = gate_state[(int64_t)pool_slot * width + source_d];
                float weight = expf(score - maximum);
                denominator += weight;
                weighted += weight
                          * kv_state[(int64_t)pool_slot * width + source_d];
            }
            pooled[d] = weighted / denominator;
        }
        float mean_square = 0.0f;
        for (int d = 0; d < head_dim; d++)
            mean_square += pooled[d] * pooled[d];
        float scale = 1.0f / sqrtf(mean_square / head_dim + 1e-6f);
        for (int d = 0; d < head_dim; d++) pooled[d] *= scale * norm[d];
        int64_t block_position = position + 1 - ratio;
        deepseek_v4_forward_compress_rope(pooled, head_dim, block_position);
        if (!deepseek_v4_kv_cache_append(cache, pooled, block_position)) {
            free(ape); free(norm); goto fail;
        }
        if (overlap) {
            memcpy(kv_state, kv_state + (int64_t)ratio * width,
                   (size_t)ratio * width * sizeof(float));
            memcpy(gate_state, gate_state + (int64_t)ratio * width,
                   (size_t)ratio * width * sizeof(float));
        }
    }
    free(ape); free(norm); free(kv); free(gate); free(pooled);
    return 1;
fail:
    free(kv); free(gate); free(pooled);
    return 0;
}

static inline int deepseek_v4_forward_index_layer_update(
        shards *source, int layer, const float *hidden, int n_tokens,
        DeepSeekV4ForwardLayerState *state, DeepSeekV4ForwardLayerCapture *capture) {
    if (!source || !hidden || !state || state->layer != layer ||
        state->ratio != 4 || n_tokens <= 0) return 0;
    char prefix[320], name[320];
    snprintf(prefix, sizeof(prefix),
             "layers.%d.attn.indexer.compressor", layer);
    if (!deepseek_v4_forward_compressor_update(
            source, prefix, hidden, n_tokens, state->next_position,
            4, 128, state->index_kv_state, state->index_gate_state,
            &state->index, NULL, NULL)) return 0;
    int n_entries = state->index.count;
    if (n_entries == 0) return 1;
    int top_k = n_entries < 512 ? n_entries : 512;
    if (!capture || !capture->index_scores || !capture->index_ids ||
        !capture->block_bias) return 0;

    float *q_a = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q_norm = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *queries = malloc((size_t)n_tokens * 64 * 128 * sizeof(float));
    float *head_weights = malloc((size_t)n_tokens * 64 * sizeof(float));
    int64_t *positions = malloc((size_t)n_tokens * sizeof(int64_t));
    if (!q_a || !q_norm || !queries || !head_weights || !positions) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.wq_a", layer);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, n_tokens,
                           DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.q_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, q_a, n_tokens,
                            DEEPSEEK_V4_FORWARD_QR, q_norm)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.indexer.wq_b", layer);
    if (!deepseek_v4_forward_fp8_batch(source, name, q_norm, n_tokens,
                           64 * 128, DEEPSEEK_V4_FORWARD_QR, queries)) goto fail;
    snprintf(name, sizeof(name),
             "layers.%d.attn.indexer.weights_proj.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             64, DEEPSEEK_V4_FORWARD_D, head_weights)) goto fail;
    for (int token = 0; token < n_tokens; token++) {
        positions[token] = state->next_position + token;
        for (int head = 0; head < 64; head++)
            deepseek_v4_forward_compress_rope(
                queries + ((int64_t)token * 64 + head) * 128,
                128, positions[token]);
    }
    DeepSeekV4LightningIndexerF32 indexer = {
        .n_heads = 64, .head_dim = 128, .ratio = 4, .top_k = top_k,
    };
    int ok = deepseek_v4_indexer_forward_f32(
        &indexer, queries, state->index_keys, head_weights, positions,
        n_tokens, n_entries, capture->index_scores, capture->index_ids);
    if (ok)
        for (int token = 0; token < n_tokens; token++)
            for (int rank = 0; rank < top_k; rank++) {
                int64_t id = capture->index_ids[(int64_t)token * top_k + rank];
                capture->block_bias[(int64_t)token * top_k + rank]
                    = id >= 0 ? 0.0f : -INFINITY;
            }
    free(q_a); free(q_norm); free(queries); free(head_weights); free(positions);
    return ok;
fail:
    free(q_a); free(q_norm); free(queries); free(head_weights); free(positions);
    return 0;
}

static inline int deepseek_v4_forward_compress_layer_prefill(
        shards *source, int layer, const float *hidden, int n_tokens,
        DeepSeekV4ForwardLayerState *state, DeepSeekV4ForwardLayerCapture *capture) {
    if (!source || !hidden || !state || state->layer != layer ||
        state->ratio <= 0 || state->next_position != 0 || n_tokens <= 0)
        return 0;
    int overlap = state->ratio == 4;
    int width = (1 + overlap) * DEEPSEEK_V4_FORWARD_HD;
    int n_entries = n_tokens / state->ratio;
    float *kv = malloc((size_t)n_tokens * width * sizeof(float));
    float *gate = malloc((size_t)n_tokens * width * sizeof(float));
    float *compressed = malloc((size_t)(n_entries ? n_entries : 1)
                               * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    char name[320];
    if (!kv || !gate || !compressed) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.wkv.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             width, DEEPSEEK_V4_FORWARD_D, kv)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.wgate.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             width, DEEPSEEK_V4_FORWARD_D, gate)) goto fail;
    if (capture && capture->projected_kv)
        memcpy(capture->projected_kv, kv,
               (size_t)n_tokens * width * sizeof(float));
    if (capture && capture->projected_gate)
        memcpy(capture->projected_gate, gate,
               (size_t)n_tokens * width * sizeof(float));
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.ape", layer);
    float *ape = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.norm.weight", layer);
    float *norm = deepseek_v4_forward_read_f32(source, name);
    if (!ape || !norm) { free(ape); free(norm); goto fail; }
    DeepSeekV4PrefillCompressorF32 model = {
        .ratio = state->ratio, .head_dim = DEEPSEEK_V4_FORWARD_HD, .rope_dim = 64,
        .overlap = overlap, .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = ape, .norm_weight = norm,
    };
    int cutoff = n_entries * state->ratio;
    int remainder = n_tokens - cutoff;
    if (overlap && cutoff >= state->ratio) {
        memcpy(state->compressor_kv_state,
               kv + (int64_t)(cutoff - state->ratio) * width,
               (size_t)state->ratio * width * sizeof(float));
        for (int row = 0; row < state->ratio; row++)
            for (int d = 0; d < width; d++)
                state->compressor_gate_state[(int64_t)row * width + d]
                    = gate[((int64_t)cutoff - state->ratio + row) * width + d]
                    + ape[(int64_t)row * width + d];
    }
    int remainder_offset = overlap ? state->ratio : 0;
    for (int row = 0; row < remainder; row++) {
        memcpy(state->compressor_kv_state
                   + (int64_t)(remainder_offset + row) * width,
               kv + (int64_t)(cutoff + row) * width,
               (size_t)width * sizeof(float));
        for (int d = 0; d < width; d++)
            state->compressor_gate_state[
                (int64_t)(remainder_offset + row) * width + d]
                = gate[(int64_t)(cutoff + row) * width + d]
                + ape[(int64_t)row * width + d];
    }
    int ok = n_entries == 0 || deepseek_v4_compress_prefill_f32(
        &model, kv, gate, cutoff, compressed);
    free(ape); free(norm);
    if (!ok) goto fail;
    for (int entry = 0; entry < n_entries; entry++) {
        int64_t position = (int64_t)entry * state->ratio;
        if (!deepseek_v4_kv_cache_append(&state->compressed,
                                compressed + (int64_t)entry * DEEPSEEK_V4_FORWARD_HD,
                                position)) goto fail;
    }
    free(kv); free(gate); free(compressed);
    return 1;
fail:
    free(kv); free(gate); free(compressed);
    return 0;
}

static inline int deepseek_v4_forward_index_layer_prefill(
        shards *source, int layer, const float *hidden, int n_tokens,
        DeepSeekV4ForwardLayerState *state, DeepSeekV4ForwardLayerCapture *capture) {
    if (!source || !hidden || !state || !capture || state->layer != layer ||
        state->ratio != 4 || state->next_position != 0 || n_tokens <= 0 ||
        n_tokens % 4 != 0 || !capture->index_scores || !capture->index_ids ||
        !capture->block_bias) return 0;
    int n_entries = n_tokens / 4;
    float *kv = malloc((size_t)n_tokens * 256 * sizeof(float));
    float *gate = malloc((size_t)n_tokens * 256 * sizeof(float));
    float *compressed = malloc((size_t)n_entries * 128 * sizeof(float));
    float *q_a = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q_norm = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *queries = malloc((size_t)n_tokens * 64 * 128 * sizeof(float));
    float *head_weights = malloc((size_t)n_tokens * 64 * sizeof(float));
    int64_t *positions = malloc((size_t)n_tokens * sizeof(int64_t));
    char name[320];
    if (!kv || !gate || !compressed || !q_a || !q_norm || !queries ||
        !head_weights || !positions) goto fail;
    snprintf(name, sizeof(name),
             "layers.%d.attn.indexer.compressor.wkv.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens, 256, DEEPSEEK_V4_FORWARD_D, kv))
        goto fail;
    snprintf(name, sizeof(name),
             "layers.%d.attn.indexer.compressor.wgate.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens, 256, DEEPSEEK_V4_FORWARD_D, gate))
        goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.indexer.compressor.ape", layer);
    float *ape = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name),
             "layers.%d.attn.indexer.compressor.norm.weight", layer);
    float *norm = deepseek_v4_forward_read_f32(source, name);
    if (!ape || !norm) { free(ape); free(norm); goto fail; }
    DeepSeekV4PrefillCompressorF32 compressor = {
        .ratio = 4, .head_dim = 128, .rope_dim = 64, .overlap = 1,
        .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = ape, .norm_weight = norm,
    };
    int ok = deepseek_v4_compress_prefill_f32(&compressor, kv, gate, n_tokens,
                                      compressed);
    free(ape); free(norm);
    if (!ok) goto fail;
    for (int entry = 0; entry < n_entries; entry++)
        if (!deepseek_v4_kv_cache_append(&state->index,
                                compressed + (int64_t)entry * 128,
                                (int64_t)entry * 4)) goto fail;

    snprintf(name, sizeof(name), "layers.%d.attn.wq_a", layer);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, n_tokens,
                           DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.q_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, q_a, n_tokens,
                            DEEPSEEK_V4_FORWARD_QR, q_norm)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.indexer.wq_b", layer);
    if (!deepseek_v4_forward_fp8_batch(source, name, q_norm, n_tokens,
                           64 * 128, DEEPSEEK_V4_FORWARD_QR, queries)) goto fail;
    snprintf(name, sizeof(name),
             "layers.%d.attn.indexer.weights_proj.weight", layer);
    if (!deepseek_v4_forward_dense_batch(source, name, hidden, n_tokens,
                             64, DEEPSEEK_V4_FORWARD_D, head_weights)) goto fail;
    for (int token = 0; token < n_tokens; token++) {
        positions[token] = token;
        for (int head = 0; head < 64; head++)
            deepseek_v4_forward_compress_rope(
                queries + ((int64_t)token * 64 + head) * 128, 128, token);
    }
    DeepSeekV4LightningIndexerF32 indexer = {
        .n_heads = 64, .head_dim = 128, .ratio = 4,
        .top_k = n_entries < 512 ? n_entries : 512,
    };
    ok = deepseek_v4_indexer_forward_f32(&indexer, queries, compressed, head_weights,
                                positions, n_tokens, n_entries,
                                capture->index_scores, capture->index_ids);
    if (ok)
        for (int token = 0; token < n_tokens; token++)
            for (int rank = 0; rank < indexer.top_k; rank++) {
                int64_t id = capture->index_ids[(int64_t)token * indexer.top_k + rank];
                capture->block_bias[(int64_t)token * indexer.top_k + rank]
                    = id >= 0 ? 0.0f : -INFINITY;
            }
    free(kv); free(gate); free(compressed); free(q_a); free(q_norm);
    free(queries); free(head_weights); free(positions);
    return ok;
fail:
    free(kv); free(gate); free(compressed); free(q_a); free(q_norm);
    free(queries); free(head_weights); free(positions);
    return 0;
}

static inline float deepseek_v4_forward_rope_frequency(int pair, int compressed) {
    if (compressed) return deepseek_v4_forward_compress_frequency(pair);
    return powf(10000.0f, -(2.0f * pair) / 64.0f);
}

static inline void deepseek_v4_forward_rope(float *vector, int width, int64_t position,
                                int compressed, int inverse) {
    int start = width - 64;
    for (int pair = 0; pair < 32; pair++) {
        float angle = (float)position * deepseek_v4_forward_rope_frequency(pair, compressed);
        float cosine = cosf(angle), sine = sinf(angle);
        if (inverse) sine = -sine;
        int index = start + pair * 2;
        float first = vector[index], second = vector[index + 1];
        vector[index] = first * cosine - second * sine;
        vector[index + 1] = second * cosine + first * sine;
    }
}

static inline void deepseek_v4_forward_hc_post_batch(float *output, const float *block,
                                         const float *residual, const float *post,
                                         const float *comb, int n_tokens) {
    for (int token = 0; token < n_tokens; token++) {
        const float *token_block = block + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        const float *token_residual = residual
            + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        float *token_output = output + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        for (int row = 0; row < DEEPSEEK_V4_FORWARD_HC; row++) {
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) {
                float sum = post[(int64_t)token * DEEPSEEK_V4_FORWARD_HC + row]
                          * token_block[d];
                for (int column = 0; column < DEEPSEEK_V4_FORWARD_HC; column++)
                    sum += comb[((int64_t)token * DEEPSEEK_V4_FORWARD_HC + row)
                                * DEEPSEEK_V4_FORWARD_HC + column]
                         * token_residual[column * DEEPSEEK_V4_FORWARD_D + d];
                token_output[row * DEEPSEEK_V4_FORWARD_D + d] = sum;
            }
        }
    }
}

static inline int deepseek_v4_forward_attention_batch(shards *source, int layer,
                                           const float *hidden, int n_tokens,
                                           const DeepSeekV4ForwardLayer2CSACapture *csa,
                                           DeepSeekV4ForwardLayerState *state,
                                           const DeepSeekV4ForwardLayerCapture *capture,
                                           float *output) {
    char prefix[320], name[320];
    snprintf(prefix, sizeof(prefix), "layers.%d.attn", layer);
    size_t q_count = (size_t)n_tokens * DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD;
    float *q_a = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q_norm = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q = malloc(q_count * sizeof(float));
    float *kv = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *context = malloc(q_count * sizeof(float));
    float *group_input = malloc((size_t)n_tokens * 4096 * sizeof(float));
    float *group_output = malloc((size_t)n_tokens * 1024 * sizeof(float));
    float *grouped = malloc((size_t)n_tokens * 8192 * sizeof(float));
    if (!q_a || !q_norm || !q || !kv || !context || !group_input ||
        !group_output || !grouped) goto fail;
    snprintf(name, sizeof(name), "%s.wq_a", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, n_tokens,
                           DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    snprintf(name, sizeof(name), "%s.q_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, q_a, n_tokens,
                            DEEPSEEK_V4_FORWARD_QR, q_norm)) goto fail;
    snprintf(name, sizeof(name), "%s.wq_b", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, q_norm, n_tokens,
                           DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_QR, q)) goto fail;
    snprintf(name, sizeof(name), "%s.wkv", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, n_tokens,
                           DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_D, kv)) goto fail;
    snprintf(name, sizeof(name), "%s.kv_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, kv, n_tokens,
                            DEEPSEEK_V4_FORWARD_HD, kv)) goto fail;

    int compressed = layer >= 2;
    int start_position = state ? state->next_position : 0;
    for (int token = 0; token < n_tokens; token++) {
        int64_t position = start_position + token;
        deepseek_v4_forward_rope(kv + (int64_t)token * DEEPSEEK_V4_FORWARD_HD,
                     DEEPSEEK_V4_FORWARD_HD, position, compressed, 0);
        for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
            float *query = q + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head)
                             * DEEPSEEK_V4_FORWARD_HD;
            float mean_square = 0.0f;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                mean_square += query[d] * query[d];
            float scale = 1.0f / sqrtf(mean_square / DEEPSEEK_V4_FORWARD_HD + 1e-6f);
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) query[d] *= scale;
            deepseek_v4_forward_rope(query, DEEPSEEK_V4_FORWARD_HD, position, compressed, 0);
        }
    }

    snprintf(name, sizeof(name), "%s.attn_sink", prefix);
    float *sinks = deepseek_v4_forward_read_f32(source, name);
    if (!sinks) goto fail;
    float attention_scale = 1.0f / sqrtf((float)DEEPSEEK_V4_FORWARD_HD);
    for (int token = 0; token < n_tokens; token++) {
        int64_t position = start_position + token;
        int local_start = token >= 127 ? token - 127 : 0;
        int use_compressed = layer == 2 && csa && csa->index_ids[token] >= 0;
        for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
            const float *query = q + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head)
                                   * DEEPSEEK_V4_FORWARD_HD;
            float maximum = sinks[head];
            if (state) {
                for (int key = 0; key < state->window.count; key++) {
                    int64_t key_position = deepseek_v4_kv_cache_position(&state->window, key);
                    if (key_position < position - 127) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->window, key);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    score *= attention_scale;
                    if (score > maximum) maximum = score;
                }
            }
            for (int key = local_start; key <= token; key++) {
                const float *key_vector = kv + (int64_t)key * DEEPSEEK_V4_FORWARD_HD;
                float score = 0.0f;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    score += query[d] * key_vector[d];
                score *= attention_scale;
                if (score > maximum) maximum = score;
            }
            float compressed_score = -INFINITY;
            if (use_compressed) {
                compressed_score = 0.0f;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    compressed_score += query[d] * csa->compressed_kv[d];
                compressed_score *= attention_scale;
                if (compressed_score > maximum) maximum = compressed_score;
            }
            if (state && state->ratio == 128) {
                for (int entry = 0; entry < state->compressed.count; entry++) {
                    if (deepseek_v4_kv_cache_position(&state->compressed, entry)
                            + state->ratio > position + 1) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->compressed, entry);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    score *= attention_scale;
                    if (score > maximum) maximum = score;
                }
            }
            if (state && state->ratio == 4 && capture) {
                int top_k = state->index.count < 512 ? state->index.count : 512;
                for (int rank = 0; rank < top_k; rank++) {
                    int64_t id = capture->index_ids[(int64_t)token * top_k + rank];
                    if (id < 0) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->compressed,
                                                              (int)id);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    score *= attention_scale;
                    if (score > maximum) maximum = score;
                }
            }
            float denominator = expf(sinks[head] - maximum);
            float *result = context
                + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head) * DEEPSEEK_V4_FORWARD_HD;
            memset(result, 0, DEEPSEEK_V4_FORWARD_HD * sizeof(float));
            if (state) {
                for (int key = 0; key < state->window.count; key++) {
                    int64_t key_position = deepseek_v4_kv_cache_position(&state->window, key);
                    if (key_position < position - 127) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->window, key);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    float probability = expf(score * attention_scale - maximum);
                    denominator += probability;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        result[d] += probability * key_vector[d];
                }
            }
            for (int key = local_start; key <= token; key++) {
                const float *key_vector = kv + (int64_t)key * DEEPSEEK_V4_FORWARD_HD;
                float score = 0.0f;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    score += query[d] * key_vector[d];
                float probability = expf(score * attention_scale - maximum);
                denominator += probability;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    result[d] += probability * key_vector[d];
            }
            if (use_compressed) {
                float probability = expf(compressed_score - maximum);
                denominator += probability;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    result[d] += probability * csa->compressed_kv[d];
            }
            if (state && state->ratio == 128) {
                for (int entry = 0; entry < state->compressed.count; entry++) {
                    if (deepseek_v4_kv_cache_position(&state->compressed, entry)
                            + state->ratio > position + 1) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->compressed, entry);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    float probability = expf(score * attention_scale - maximum);
                    denominator += probability;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        result[d] += probability * key_vector[d];
                }
            }
            if (state && state->ratio == 4 && capture) {
                int top_k = state->index.count < 512 ? state->index.count : 512;
                for (int rank = 0; rank < top_k; rank++) {
                    int64_t id = capture->index_ids[(int64_t)token * top_k + rank];
                    if (id < 0) continue;
                    const float *key_vector = deepseek_v4_kv_cache_key(&state->compressed,
                                                              (int)id);
                    float score = 0.0f;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        score += query[d] * key_vector[d];
                    float probability = expf(score * attention_scale - maximum);
                    denominator += probability;
                    for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                        result[d] += probability * key_vector[d];
                }
            }
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) result[d] /= denominator;
            deepseek_v4_forward_rope(result, DEEPSEEK_V4_FORWARD_HD, position, compressed, 1);
        }
    }
    free(sinks);
    if (state)
        for (int token = 0; token < n_tokens; token++)
            if (!deepseek_v4_kv_cache_append(&state->window,
                                    kv + (int64_t)token * DEEPSEEK_V4_FORWARD_HD,
                                    start_position + token)) goto fail;

    snprintf(name, sizeof(name), "%s.wo_a.weight", prefix);
    uint8_t *wo_a = deepseek_v4_forward_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.wo_a.scale", prefix);
    uint8_t *wo_a_scale = deepseek_v4_forward_read_raw(source, name);
    if (!wo_a || !wo_a_scale) { free(wo_a); free(wo_a_scale); goto fail; }
    for (int group = 0; group < 8; group++) {
        for (int token = 0; token < n_tokens; token++)
            memcpy(group_input + (int64_t)token * 4096,
                   context + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + group * 8)
                             * DEEPSEEK_V4_FORWARD_HD,
                   4096 * sizeof(float));
        if (!deepseek_v4_fp8_matmul_f32(group_output, group_input, n_tokens,
                wo_a + (int64_t)group * 1024 * 4096,
                wo_a_scale + (int64_t)group * 8 * 32,
                1024, 4096, 128)) {
            free(wo_a); free(wo_a_scale); goto fail;
        }
        for (int token = 0; token < n_tokens; token++)
            memcpy(grouped + (int64_t)token * 8192 + group * 1024,
                   group_output + (int64_t)token * 1024,
                   1024 * sizeof(float));
    }
    free(wo_a); free(wo_a_scale);
    snprintf(name, sizeof(name), "%s.wo_b", prefix);
    int ok = deepseek_v4_forward_fp8_batch(source, name, grouped, n_tokens,
                               DEEPSEEK_V4_FORWARD_D, 8192, output);
    free(q_a); free(q_norm); free(q); free(kv); free(context);
    free(group_input); free(group_output); free(grouped);
    return ok;
fail:
    free(q_a); free(q_norm); free(q); free(kv); free(context);
    free(group_input); free(group_output); free(grouped);
    return 0;
}

static inline int deepseek_v4_forward_moe_batch(shards *source, int layer,
                                     const float *hidden, const int *token_ids,
                                     int n_tokens, DeepSeekV4ForwardLayerCapture *capture,
                                     float *output) {
    char name[320], prefix[320];
    snprintf(name, sizeof(name), "layers.%d.ffn.gate.weight", layer);
    float *router = deepseek_v4_forward_read_f32(source, name);
    float *bias = NULL;
    if (layer >= 3) {
        snprintf(name, sizeof(name), "layers.%d.ffn.gate.bias", layer);
        bias = deepseek_v4_forward_read_f32(source, name);
    }
    float *expert_output = malloc(DEEPSEEK_V4_FORWARD_D * sizeof(float));
    if (!router || (layer >= 3 && !bias) || !expert_output) {
        free(router); free(bias); free(expert_output); return 0;
    }
    for (int token = 0; token < n_tokens; token++) {
        const float *input = hidden + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        float logits[DEEPSEEK_V4_FORWARD_EXPERTS], scores[DEEPSEEK_V4_FORWARD_EXPERTS];
        float weights[DEEPSEEK_V4_FORWARD_TOPK];
        int indices[DEEPSEEK_V4_FORWARD_TOPK];
        int64_t stored_indices[DEEPSEEK_V4_FORWARD_TOPK];
        deepseek_v4_forward_matvec(logits, router, input, DEEPSEEK_V4_FORWARD_EXPERTS, DEEPSEEK_V4_FORWARD_D);
        if (layer < 3) {
            snprintf(name, sizeof(name), "layers.%d.ffn.gate.tid2eid", layer);
            st_read_slice_raw(source, name,
                              (int64_t)token_ids[token] * DEEPSEEK_V4_FORWARD_TOPK * sizeof(int64_t),
                              DEEPSEEK_V4_FORWARD_TOPK * sizeof(int64_t), stored_indices, 0);
            for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++) indices[k] = (int)stored_indices[k];
            if (moe_route_fixed(logits, DEEPSEEK_V4_FORWARD_EXPERTS, indices, DEEPSEEK_V4_FORWARD_TOPK,
                                MOE_SCORE_SQRT_SOFTPLUS, weights, scores)
                != DEEPSEEK_V4_FORWARD_TOPK) goto fail;
        } else {
            float selection_scores[DEEPSEEK_V4_FORWARD_EXPERTS];
            if (moe_route_select(logits, bias, DEEPSEEK_V4_FORWARD_EXPERTS, DEEPSEEK_V4_FORWARD_TOPK,
                                 MOE_SCORE_SQRT_SOFTPLUS, indices, weights,
                                 scores, selection_scores) != DEEPSEEK_V4_FORWARD_TOPK) goto fail;
        }
        moe_route_finalize(weights, DEEPSEEK_V4_FORWARD_TOPK, 1, 1.5f);
        if (capture) {
            if (capture->router_scores)
                memcpy(capture->router_scores + (int64_t)token * DEEPSEEK_V4_FORWARD_EXPERTS,
                       scores, DEEPSEEK_V4_FORWARD_EXPERTS * sizeof(float));
            if (capture->router_weights)
                memcpy(capture->router_weights + (int64_t)token * DEEPSEEK_V4_FORWARD_TOPK,
                       weights, DEEPSEEK_V4_FORWARD_TOPK * sizeof(float));
            if (capture->router_indices)
                for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++)
                    capture->router_indices[(int64_t)token * DEEPSEEK_V4_FORWARD_TOPK + k]
                        = indices[k];
        }
        float *token_output = output + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        memset(token_output, 0, DEEPSEEK_V4_FORWARD_D * sizeof(float));
        for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++) {
            snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d",
                     layer, indices[k]);
            if (!deepseek_v4_forward_expert(source, prefix, input, 1,
                                weights[k], expert_output)) goto fail;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) token_output[d] += expert_output[d];
        }
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.shared_experts", layer);
        if (!deepseek_v4_forward_expert(source, prefix, input, 0, 1.0f,
                            expert_output)) goto fail;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) token_output[d] += expert_output[d];
    }
    free(router); free(bias); free(expert_output);
    return 1;
fail:
    free(router); free(bias); free(expert_output);
    return 0;
}

static inline int deepseek_v4_forward_layer_forward(shards *source, int layer,
                                         const int *token_ids, int n_tokens,
                                         float *streams,
                                         DeepSeekV4ForwardLayer2CSACapture *csa) {
    if (!source || !token_ids || !streams || layer < 0 || layer > 2 ||
        n_tokens <= 0 || (layer == 2 && (n_tokens != 4 || !csa))) return 0;
    size_t streams_count = (size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
    float *post = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *comb = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *collapsed = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *hidden = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *block = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *residual = malloc(streams_count * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !block || !residual) goto fail;
    if (!deepseek_v4_forward_hc_layer(source, layer, "attn", streams, n_tokens,
                          post, comb, collapsed)) goto fail;
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, n_tokens,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (layer == 2 && !deepseek_v4_forward_layer2_csa_forward(source, streams, csa)) goto fail;
    if (!deepseek_v4_forward_attention_batch(source, layer, hidden, n_tokens,
                                 csa, NULL, NULL, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(residual, block, streams, post, comb, n_tokens);

    if (!deepseek_v4_forward_hc_layer(source, layer, "ffn", residual, n_tokens,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, n_tokens,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (!deepseek_v4_forward_moe_batch(source, layer, hidden, token_ids, n_tokens,
                           NULL, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(streams, block, residual, post, comb, n_tokens);
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 1;
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 0;
}

static inline int deepseek_v4_forward_layer_forward_state(
        shards *source, int layer, const int *token_ids, int n_tokens,
        float *streams, DeepSeekV4ForwardLayerState *state, DeepSeekV4ForwardLayerCapture *capture) {
    if (!source || !token_ids || !streams || !state || state->layer != layer ||
        layer < 0 || layer > 42 || state->ratio < 0 || n_tokens <= 0 ||
        state->next_position + n_tokens > state->max_context) return 0;
    size_t streams_count = (size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
    float *post = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *comb = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *collapsed = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *hidden = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *block = malloc((size_t)n_tokens * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *residual = malloc(streams_count * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !block || !residual) goto fail;
    if (!deepseek_v4_forward_hc_layer(source, layer, "attn", streams, n_tokens,
                          post, comb, collapsed)) goto fail;
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, n_tokens,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (state->ratio > 0) {
        snprintf(name, sizeof(name), "layers.%d.attn.compressor", layer);
        if (!deepseek_v4_forward_compressor_update(
                source, name, hidden, n_tokens, state->next_position,
                state->ratio, DEEPSEEK_V4_FORWARD_HD,
                state->compressor_kv_state, state->compressor_gate_state,
                &state->compressed,
                capture ? capture->projected_kv : NULL,
                capture ? capture->projected_gate : NULL)) goto fail;
    }
    if (state->ratio == 4 &&
        !deepseek_v4_forward_index_layer_update(source, layer, hidden, n_tokens,
                                    state, capture)) goto fail;
    if (!deepseek_v4_forward_attention_batch(source, layer, hidden, n_tokens,
                                 NULL, state, capture, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(residual, block, streams, post, comb, n_tokens);

    if (!deepseek_v4_forward_hc_layer(source, layer, "ffn", residual, n_tokens,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, n_tokens,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (!deepseek_v4_forward_moe_batch(source, layer, hidden, token_ids, n_tokens,
                           capture, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(streams, block, residual, post, comb, n_tokens);
    state->next_position += n_tokens;
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 1;
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 0;
}

static inline void deepseek_v4_forward_base_capture_layer(
        const DeepSeekV4ForwardBaseCapture *capture, int layer, const float *streams,
        int n_tokens) {
    if (!capture || !capture->layer_ids || !capture->layer_outputs) return;
    size_t layer_size = (size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
    for (int index = 0; index < capture->n_layer_ids; index++)
        if (capture->layer_ids[index] == layer)
            memcpy(capture->layer_outputs + (size_t)index * layer_size,
                   streams, layer_size * sizeof(float));
}

static inline int deepseek_v4_forward_base_layers_forward(
        shards *source, const int *token_ids, int n_tokens,
        DeepSeekV4ForwardModelState *state, float *streams, DeepSeekV4ForwardBaseCapture *capture) {
    if (!source || !token_ids || !state || !streams || n_tokens <= 0 ||
        state->layers[0].next_position + n_tokens > state->max_context)
        return 0;
    for (int token = 0; token < n_tokens; token++) {
        float *embedding = streams
            + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        st_read_slice_f32(source, "embed.weight",
                          (int64_t)token_ids[token] * DEEPSEEK_V4_FORWARD_D,
                          DEEPSEEK_V4_FORWARD_D, embedding, 0);
        for (int stream = 1; stream < DEEPSEEK_V4_FORWARD_HC; stream++)
            memcpy(embedding + (int64_t)stream * DEEPSEEK_V4_FORWARD_D,
                   embedding, DEEPSEEK_V4_FORWARD_D * sizeof(float));
    }

    for (int layer = 0; layer < 43; layer++) {
        DeepSeekV4ForwardLayerCapture layer_capture = {0};
        float *index_scores = NULL, *block_bias = NULL;
        int64_t *index_ids = NULL;
        int entries = (state->layers[layer].next_position + n_tokens) / 4;
        int top_k = entries < 512 ? entries : 512;
        if (state->layers[layer].ratio == 4 && entries > 0) {
            index_scores = malloc((size_t)n_tokens * entries * sizeof(float));
            index_ids = malloc((size_t)n_tokens * top_k * sizeof(int64_t));
            block_bias = malloc((size_t)n_tokens * top_k * sizeof(float));
            if (!index_scores || !index_ids || !block_bias) {
                free(index_scores); free(index_ids); free(block_bias);
                return 0;
            }
            layer_capture.index_scores = index_scores;
            layer_capture.index_ids = index_ids;
            layer_capture.block_bias = block_bias;
        }
        if (capture && capture->router_indices)
            layer_capture.router_indices = capture->router_indices
                + (int64_t)layer * n_tokens * DEEPSEEK_V4_FORWARD_TOPK;
        int ok = deepseek_v4_forward_layer_forward_state(
            source, layer, token_ids, n_tokens, streams,
            &state->layers[layer],
            (state->layers[layer].ratio == 4 ||
             layer_capture.router_indices) ? &layer_capture : NULL);
        free(index_scores); free(index_ids); free(block_bias);
        if (!ok) return 0;
        deepseek_v4_forward_base_capture_layer(capture, layer, streams, n_tokens);
        if (capture && capture->target_means && layer >= 40) {
            float *target = capture->target_means
                + (int64_t)(layer - 40) * n_tokens * DEEPSEEK_V4_FORWARD_D;
            for (int token = 0; token < n_tokens; token++)
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) {
                    float sum = 0.0f;
                    for (int stream = 0; stream < DEEPSEEK_V4_FORWARD_HC; stream++)
                        sum += streams[((int64_t)token * DEEPSEEK_V4_FORWARD_HC + stream)
                                       * DEEPSEEK_V4_FORWARD_D + d];
                    target[(int64_t)token * DEEPSEEK_V4_FORWARD_D + d]
                        = sum / DEEPSEEK_V4_FORWARD_HC;
                }
        }
    }
    return 1;
}

static inline int deepseek_v4_forward_base_head_forward(shards *source,
                                             const float *streams,
                                             int n_tokens,
                                             DeepSeekV4ForwardHeadCapture *capture) {
    if (!source || !streams || n_tokens <= 0 || !capture || !capture->hidden ||
        !capture->normalized || !capture->logits ||
        st_numel(source, "head.weight") != (int64_t)129280 * DEEPSEEK_V4_FORWARD_D)
        return 0;
    float *fn = deepseek_v4_forward_read_f32(source, "hc_head_fn");
    float *base = deepseek_v4_forward_read_f32(source, "hc_head_base");
    float *scale = deepseek_v4_forward_read_f32(source, "hc_head_scale");
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    for (int token = 0; token < n_tokens; token++) {
        const float *input = streams
            + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        float mean_square = 0.0f;
        for (int i = 0; i < DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D; i++)
            mean_square += input[i] * input[i];
        float inv_rms = 1.0f / sqrtf(
            mean_square / (DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D) + 1e-6f);
        float pre[DEEPSEEK_V4_FORWARD_HC];
        for (int row = 0; row < DEEPSEEK_V4_FORWARD_HC; row++) {
            float mix = 0.0f;
            const float *weight = fn
                + (int64_t)row * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
            for (int i = 0; i < DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D; i++)
                mix += weight[i] * input[i];
            pre[row] = deepseek_v4_hc_sigmoid(mix * inv_rms * scale[0] + base[row])
                     + 1e-6f;
        }
        float *hidden = capture->hidden + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) {
            float sum = 0.0f;
            for (int stream = 0; stream < DEEPSEEK_V4_FORWARD_HC; stream++)
                sum += pre[stream] * input[(int64_t)stream * DEEPSEEK_V4_FORWARD_D + d];
            hidden[d] = sum;
        }
    }
    free(fn); free(base); free(scale);
    if (!deepseek_v4_forward_norm_batch(source, "norm.weight", capture->hidden,
                            n_tokens, DEEPSEEK_V4_FORWARD_D, capture->normalized)) return 0;

    st_tensor *head = st_find(source, "head.weight");
    if (!head || (head->dtype != ST_DTYPE_BF16 &&
                  head->dtype != ST_DTYPE_F16 && head->dtype != ST_DTYPE_F32))
        return 0;
    uint8_t *raw = deepseek_v4_forward_read_raw(source, "head.weight");
    if (!raw) return 0;
    const float *input = capture->normalized
        + (int64_t)(n_tokens - 1) * DEEPSEEK_V4_FORWARD_D;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int token = 0; token < 129280; token++) {
        float sum = 0.0f;
        int64_t offset = (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        if (head->dtype == ST_DTYPE_BF16) {
            const uint16_t *weight = (const uint16_t *)raw + offset;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++)
                sum += bf16_to_f32(weight[d]) * input[d];
        } else if (head->dtype == ST_DTYPE_F16) {
            const uint16_t *weight = (const uint16_t *)raw + offset;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++)
                sum += f16_to_f32(weight[d]) * input[d];
        } else {
            const float *weight = (const float *)raw + offset;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) sum += weight[d] * input[d];
        }
        capture->logits[token] = sum;
    }
    free(raw);
    return 1;
}

static inline int deepseek_v4_forward_hc_named(shards *source, const char *prefix,
                                    const char *site, const float *input,
                                    int n_tokens, float *post, float *comb,
                                    float *collapsed) {
    char name[320];
    snprintf(name, sizeof(name), "%s.hc_%s_fn", prefix, site);
    float *fn = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.hc_%s_base", prefix, site);
    float *base = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.hc_%s_scale", prefix, site);
    float *scale = deepseek_v4_forward_read_f32(source, name);
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    DeepSeekV4HyperConnection model = {
        .streams = DEEPSEEK_V4_FORWARD_HC, .hidden = DEEPSEEK_V4_FORWARD_D, .sinkhorn_iters = 20,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f, .fn = fn, .base = base, .scale = scale,
    };
    int ok = deepseek_v4_hc_forward(&model, input, n_tokens, post, comb, collapsed);
    free(fn); free(base); free(scale);
    return ok;
}

static inline int deepseek_v4_forward_moe_named(shards *source, const char *prefix,
                                     const float *hidden, int n_tokens,
                                     float *output) {
    char name[320], expert_prefix[320];
    snprintf(name, sizeof(name), "%s.gate.weight", prefix);
    float *router = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.gate.bias", prefix);
    float *bias = deepseek_v4_forward_read_f32(source, name);
    float *expert_output = malloc(DEEPSEEK_V4_FORWARD_D * sizeof(float));
    if (!router || !bias || !expert_output) {
        free(router); free(bias); free(expert_output); return 0;
    }
    for (int token = 0; token < n_tokens; token++) {
        const float *input = hidden + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        float logits[DEEPSEEK_V4_FORWARD_EXPERTS], scores[DEEPSEEK_V4_FORWARD_EXPERTS];
        float selection[DEEPSEEK_V4_FORWARD_EXPERTS], weights[DEEPSEEK_V4_FORWARD_TOPK];
        int indices[DEEPSEEK_V4_FORWARD_TOPK];
        deepseek_v4_forward_matvec(logits, router, input, DEEPSEEK_V4_FORWARD_EXPERTS, DEEPSEEK_V4_FORWARD_D);
        if (moe_route_select(logits, bias, DEEPSEEK_V4_FORWARD_EXPERTS, DEEPSEEK_V4_FORWARD_TOPK,
                             MOE_SCORE_SQRT_SOFTPLUS, indices, weights,
                             scores, selection) != DEEPSEEK_V4_FORWARD_TOPK) goto fail;
        moe_route_finalize(weights, DEEPSEEK_V4_FORWARD_TOPK, 1, 1.5f);
        float *token_output = output + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        memset(token_output, 0, DEEPSEEK_V4_FORWARD_D * sizeof(float));
        for (int k = 0; k < DEEPSEEK_V4_FORWARD_TOPK; k++) {
            snprintf(expert_prefix, sizeof(expert_prefix), "%s.experts.%d",
                     prefix, indices[k]);
            if (!deepseek_v4_forward_expert(source, expert_prefix, input, 1,
                                weights[k], expert_output)) goto fail;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) token_output[d] += expert_output[d];
        }
        snprintf(expert_prefix, sizeof(expert_prefix), "%s.shared_experts", prefix);
        if (!deepseek_v4_forward_expert(source, expert_prefix, input, 0, 1.0f,
                            expert_output)) goto fail;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) token_output[d] += expert_output[d];
    }
    free(router); free(bias); free(expert_output);
    return 1;
fail:
    free(router); free(bias); free(expert_output);
    return 0;
}

static inline int deepseek_v4_forward_dspark_attention(
        shards *source, int stage, const float *hidden, const float *main_x,
        int start_position, DeepSeekV4ForwardDSparkState *state, float *output) {
    enum { TOKENS = 5, MAX_KEYS = 133 };
    if (!source || stage < 0 || stage >= 3 || !hidden || !main_x ||
        start_position < 0 || !state || !state->main_kv || !output)
        return 0;
    char prefix[320], name[320];
    snprintf(prefix, sizeof(prefix), "mtp.%d.attn", stage);
    float *main_kv = state->main_kv + (int64_t)stage * 128 * DEEPSEEK_V4_FORWARD_HD;
    float decode_kv[DEEPSEEK_V4_FORWARD_HD];
    snprintf(name, sizeof(name), "%s.wkv", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, main_x, 1,
                           DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_D, decode_kv)) return 0;
    snprintf(name, sizeof(name), "%s.kv_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, decode_kv, 1,
                            DEEPSEEK_V4_FORWARD_HD, decode_kv)) return 0;
    deepseek_v4_forward_rope(decode_kv, DEEPSEEK_V4_FORWARD_HD, start_position, 0, 0);
    memcpy(main_kv + (int64_t)(start_position % 128) * DEEPSEEK_V4_FORWARD_HD,
           decode_kv, DEEPSEEK_V4_FORWARD_HD * sizeof(float));

    int history = start_position + 1;
    if (history > 128) history = 128;
    int n_keys = history + TOKENS;

    float *q_a = malloc(TOKENS * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q_norm = malloc(TOKENS * DEEPSEEK_V4_FORWARD_QR * sizeof(float));
    float *q = malloc((size_t)TOKENS * DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *kv = malloc(TOKENS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *keys = malloc((size_t)n_keys * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *context = malloc((size_t)TOKENS * DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    float *group_input = malloc(TOKENS * 4096 * sizeof(float));
    float *group_output = malloc(TOKENS * 1024 * sizeof(float));
    float *grouped = malloc(TOKENS * 8192 * sizeof(float));
    if (!q_a || !q_norm || !q || !kv || !keys || !context || !group_input ||
        !group_output || !grouped) goto fail;
    snprintf(name, sizeof(name), "%s.wq_a", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, TOKENS,
                           DEEPSEEK_V4_FORWARD_QR, DEEPSEEK_V4_FORWARD_D, q_a)) goto fail;
    snprintf(name, sizeof(name), "%s.q_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, q_a, TOKENS,
                            DEEPSEEK_V4_FORWARD_QR, q_norm)) goto fail;
    snprintf(name, sizeof(name), "%s.wq_b", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, q_norm, TOKENS,
                           DEEPSEEK_V4_FORWARD_HEADS * DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_QR, q)) goto fail;
    snprintf(name, sizeof(name), "%s.wkv", prefix);
    if (!deepseek_v4_forward_fp8_batch(source, name, hidden, TOKENS,
                           DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_D, kv)) goto fail;
    snprintf(name, sizeof(name), "%s.kv_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, kv, TOKENS,
                            DEEPSEEK_V4_FORWARD_HD, kv)) goto fail;
    for (int token = 0; token < TOKENS; token++) {
        int position = start_position + 1 + token;
        deepseek_v4_forward_rope(kv + (int64_t)token * DEEPSEEK_V4_FORWARD_HD,
                     DEEPSEEK_V4_FORWARD_HD, position, 0, 0);
        for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
            float *query = q + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head)
                             * DEEPSEEK_V4_FORWARD_HD;
            float mean_square = 0.0f;
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) mean_square += query[d] * query[d];
            float scale = 1.0f / sqrtf(mean_square / DEEPSEEK_V4_FORWARD_HD + 1e-6f);
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) query[d] *= scale;
            deepseek_v4_forward_rope(query, DEEPSEEK_V4_FORWARD_HD, position, 0, 0);
        }
    }
    memcpy(keys, main_kv, (size_t)history * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    memcpy(keys + (int64_t)history * DEEPSEEK_V4_FORWARD_HD, kv,
           TOKENS * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    snprintf(name, sizeof(name), "%s.attn_sink", prefix);
    float *sinks = deepseek_v4_forward_read_f32(source, name);
    if (!sinks) goto fail;
    float attention_scale = 1.0f / sqrtf((float)DEEPSEEK_V4_FORWARD_HD);
    for (int token = 0; token < TOKENS; token++)
        for (int head = 0; head < DEEPSEEK_V4_FORWARD_HEADS; head++) {
            const float *query = q + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head)
                                   * DEEPSEEK_V4_FORWARD_HD;
            float scores[MAX_KEYS], maximum = sinks[head];
            for (int key = 0; key < n_keys; key++) {
                float score = 0.0f;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    score += query[d] * keys[(int64_t)key * DEEPSEEK_V4_FORWARD_HD + d];
                scores[key] = score * attention_scale;
                if (scores[key] > maximum) maximum = scores[key];
            }
            float denominator = expf(sinks[head] - maximum);
            float *result = context
                + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + head) * DEEPSEEK_V4_FORWARD_HD;
            memset(result, 0, DEEPSEEK_V4_FORWARD_HD * sizeof(float));
            for (int key = 0; key < n_keys; key++) {
                float probability = expf(scores[key] - maximum);
                denominator += probability;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++)
                    result[d] += probability
                               * keys[(int64_t)key * DEEPSEEK_V4_FORWARD_HD + d];
            }
            for (int d = 0; d < DEEPSEEK_V4_FORWARD_HD; d++) result[d] /= denominator;
            deepseek_v4_forward_rope(result, DEEPSEEK_V4_FORWARD_HD, start_position + 1 + token, 0, 1);
        }
    free(sinks);
    snprintf(name, sizeof(name), "%s.wo_a.weight", prefix);
    uint8_t *wo_a = deepseek_v4_forward_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.wo_a.scale", prefix);
    uint8_t *wo_a_scale = deepseek_v4_forward_read_raw(source, name);
    if (!wo_a || !wo_a_scale) { free(wo_a); free(wo_a_scale); goto fail; }
    for (int group = 0; group < 8; group++) {
        for (int token = 0; token < TOKENS; token++)
            memcpy(group_input + (int64_t)token * 4096,
                   context + ((int64_t)token * DEEPSEEK_V4_FORWARD_HEADS + group * 8)
                             * DEEPSEEK_V4_FORWARD_HD, 4096 * sizeof(float));
        if (!deepseek_v4_fp8_matmul_f32(group_output, group_input, TOKENS,
                wo_a + (int64_t)group * 1024 * 4096,
                wo_a_scale + (int64_t)group * 8 * 32,
                1024, 4096, 128)) {
            free(wo_a); free(wo_a_scale); goto fail;
        }
        for (int token = 0; token < TOKENS; token++)
            memcpy(grouped + (int64_t)token * 8192 + group * 1024,
                   group_output + (int64_t)token * 1024, 1024 * sizeof(float));
    }
    free(wo_a); free(wo_a_scale);
    snprintf(name, sizeof(name), "%s.wo_b", prefix);
    int ok = deepseek_v4_forward_fp8_batch(source, name, grouped, TOKENS,
                               DEEPSEEK_V4_FORWARD_D, 8192, output);
    free(q_a); free(q_norm); free(q); free(kv); free(keys); free(context);
    free(group_input); free(group_output); free(grouped);
    return ok;
fail:
    free(q_a); free(q_norm); free(q); free(kv); free(keys); free(context);
    free(group_input); free(group_output); free(grouped);
    return 0;
}

static inline int deepseek_v4_forward_dspark_stage(
        shards *source, int stage, float *streams, const float *main_x,
        int start_position, DeepSeekV4ForwardDSparkState *state) {
    enum { TOKENS = 5 };
    char prefix[64], name[128];
    snprintf(prefix, sizeof(prefix), "mtp.%d", stage);
    float *post = malloc(TOKENS * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *comb = malloc(TOKENS * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_HC * sizeof(float));
    float *collapsed = malloc(TOKENS * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *hidden = malloc(TOKENS * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *block = malloc(TOKENS * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *residual = malloc((size_t)TOKENS * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !block || !residual) goto fail;
    if (!deepseek_v4_forward_hc_named(source, prefix, "attn", streams, TOKENS,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "%s.attn_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, TOKENS,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    if (!deepseek_v4_forward_dspark_attention(source, stage, hidden, main_x, start_position,
                                  state, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(residual, block, streams, post, comb, TOKENS);
    if (!deepseek_v4_forward_hc_named(source, prefix, "ffn", residual, TOKENS,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "%s.ffn_norm.weight", prefix);
    if (!deepseek_v4_forward_norm_batch(source, name, collapsed, TOKENS,
                            DEEPSEEK_V4_FORWARD_D, hidden)) goto fail;
    snprintf(name, sizeof(name), "%s.ffn", prefix);
    if (!deepseek_v4_forward_moe_named(source, name, hidden, TOKENS, block)) goto fail;
    deepseek_v4_forward_hc_post_batch(streams, block, residual, post, comb, TOKENS);
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 1;
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 0;
}

static inline int deepseek_v4_forward_hc_head_named(shards *source, const char *prefix,
                                         const float *streams, int n_tokens,
                                         float *hidden) {
    char name[128];
    snprintf(name, sizeof(name), "%s.hc_head_fn", prefix);
    float *fn = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.hc_head_base", prefix);
    float *base = deepseek_v4_forward_read_f32(source, name);
    snprintf(name, sizeof(name), "%s.hc_head_scale", prefix);
    float *scale = deepseek_v4_forward_read_f32(source, name);
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    for (int token = 0; token < n_tokens; token++) {
        const float *input = streams
            + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        float mean_square = 0.0f;
        for (int i = 0; i < DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D; i++)
            mean_square += input[i] * input[i];
        float inv_rms = 1.0f / sqrtf(
            mean_square / (DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D) + 1e-6f);
        float pre[DEEPSEEK_V4_FORWARD_HC];
        for (int row = 0; row < DEEPSEEK_V4_FORWARD_HC; row++) {
            float mix = 0.0f;
            const float *weight = fn
                + (int64_t)row * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
            for (int i = 0; i < DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D; i++)
                mix += weight[i] * input[i];
            pre[row] = deepseek_v4_hc_sigmoid(mix * inv_rms * scale[0] + base[row])
                     + 1e-6f;
        }
        float *out = hidden + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) {
            float sum = 0.0f;
            for (int stream = 0; stream < DEEPSEEK_V4_FORWARD_HC; stream++)
                sum += pre[stream] * input[(int64_t)stream * DEEPSEEK_V4_FORWARD_D + d];
            out[d] = sum;
        }
    }
    free(fn); free(base); free(scale);
    return 1;
}

static inline int deepseek_v4_forward_vocab_logits(shards *source, const float *input,
                                        int n_tokens, float *logits) {
    st_tensor *head = st_find(source, "head.weight");
    if (!head || !input || !logits || n_tokens <= 0) return 0;
    uint8_t *raw = deepseek_v4_forward_read_raw(source, "head.weight");
    if (!raw) return 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int vocab = 0; vocab < 129280; vocab++) {
        int64_t offset = (int64_t)vocab * DEEPSEEK_V4_FORWARD_D;
        for (int token = 0; token < n_tokens; token++) {
            float sum = 0.0f;
            const float *x = input + (int64_t)token * DEEPSEEK_V4_FORWARD_D;
            if (head->dtype == ST_DTYPE_BF16) {
                const uint16_t *weight = (const uint16_t *)raw + offset;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++)
                    sum += bf16_to_f32(weight[d]) * x[d];
            } else if (head->dtype == ST_DTYPE_F16) {
                const uint16_t *weight = (const uint16_t *)raw + offset;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++)
                    sum += f16_to_f32(weight[d]) * x[d];
            } else {
                const float *weight = (const float *)raw + offset;
                for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) sum += weight[d] * x[d];
            }
            logits[(int64_t)token * 129280 + vocab] = sum;
        }
    }
    free(raw);
    return 1;
}

static inline int deepseek_v4_forward_dspark_append_main(
        shards *source, const float *main_x, int n_main, int start_position,
        DeepSeekV4ForwardDSparkState *state, float *capture_kv) {
    enum { STAGES = 3 };
    if (!source || !main_x || n_main <= 0 || !state || !state->main_kv ||
        start_position < 0 || state->next_position != start_position)
        return 0;
    char name[128];
    float *projected = malloc((size_t)n_main * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    if (!projected) return 0;
    for (int stage = 0; stage < STAGES; stage++) {
        snprintf(name, sizeof(name), "mtp.%d.attn.wkv", stage);
        if (!deepseek_v4_forward_fp8_batch(source, name, main_x, n_main,
                               DEEPSEEK_V4_FORWARD_HD, DEEPSEEK_V4_FORWARD_D, projected)) goto fail;
        snprintf(name, sizeof(name), "mtp.%d.attn.kv_norm.weight", stage);
        if (!deepseek_v4_forward_norm_batch(source, name, projected, n_main,
                                DEEPSEEK_V4_FORWARD_HD, projected)) goto fail;
        for (int token = 0; token < n_main; token++) {
            float *row = projected + (int64_t)token * DEEPSEEK_V4_FORWARD_HD;
            int position = start_position + token;
            deepseek_v4_forward_rope(row, DEEPSEEK_V4_FORWARD_HD, position, 0, 0);
            memcpy(state->main_kv
                       + ((int64_t)stage * 128 + position % 128) * DEEPSEEK_V4_FORWARD_HD,
                   row, DEEPSEEK_V4_FORWARD_HD * sizeof(float));
        }
        if (capture_kv)
            memcpy(capture_kv + (int64_t)stage * n_main * DEEPSEEK_V4_FORWARD_HD,
                   projected, (size_t)n_main * DEEPSEEK_V4_FORWARD_HD * sizeof(float));
    }
    state->next_position = start_position + n_main;
    free(projected);
    return 1;
fail:
    free(projected);
    return 0;
}

static inline int deepseek_v4_forward_dspark_prefill(
        shards *source, const float *main_x, int n_main,
        DeepSeekV4ForwardDSparkState *state, DeepSeekV4ForwardDSparkCapture *capture) {
    if (!capture || !capture->prefill_kv || !state || state->next_position != 0)
        return 0;
    return deepseek_v4_forward_dspark_append_main(
        source, main_x, n_main, 0, state, capture->prefill_kv);
}

static inline int deepseek_v4_forward_dspark_propose(
        shards *source, const float *main_x, int start_position, int input_id,
        DeepSeekV4ForwardDSparkState *state, DeepSeekV4ForwardDSparkCapture *capture) {
    enum { STAGES = 3, DRAFT = 5, NOISE_ID = 128799, VOCAB = 129280 };
    if (!source || !main_x || start_position < 0 || input_id < 0 || !state ||
        !state->main_kv ||
        (state->next_position != start_position &&
         state->next_position != start_position + 1) || !capture ||
        !capture->stage_outputs || !capture->hidden || !capture->output_ids ||
        !capture->confidence) return 0;
    char name[128];
    int ids[DRAFT] = {input_id, NOISE_ID, NOISE_ID, NOISE_ID, NOISE_ID};
    float *streams = malloc((size_t)DRAFT * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *normalized = malloc(DRAFT * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    float *logits = malloc((size_t)DRAFT * VOCAB * sizeof(float));
    float *markov_w2 = NULL;
    if (!streams || !normalized || !logits) goto fail;
    for (int token = 0; token < DRAFT; token++) {
        float *embedding = streams
            + (int64_t)token * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
        st_read_slice_f32(source, "embed.weight", (int64_t)ids[token] * DEEPSEEK_V4_FORWARD_D,
                          DEEPSEEK_V4_FORWARD_D, embedding, 0);
        for (int stream = 1; stream < DEEPSEEK_V4_FORWARD_HC; stream++)
            memcpy(embedding + (int64_t)stream * DEEPSEEK_V4_FORWARD_D,
                   embedding, DEEPSEEK_V4_FORWARD_D * sizeof(float));
    }
    for (int stage = 0; stage < STAGES; stage++) {
        if (!deepseek_v4_forward_dspark_stage(source, stage, streams,
                                  main_x, start_position, state)) goto fail;
        memcpy(capture->stage_outputs
                   + (int64_t)stage * DRAFT * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D,
               streams, (size_t)DRAFT * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D * sizeof(float));
    }
    if (!deepseek_v4_forward_hc_head_named(source, "mtp.2", streams, DRAFT,
                               capture->hidden)) goto fail;
    if (!deepseek_v4_forward_norm_batch(source, "mtp.2.norm.weight", capture->hidden,
                            DRAFT, DEEPSEEK_V4_FORWARD_D, normalized)) goto fail;
    if (!deepseek_v4_forward_vocab_logits(source, normalized, DRAFT, logits)) goto fail;
    markov_w2 = deepseek_v4_forward_read_f32(source, "mtp.2.markov_head.markov_w2.weight");
    if (!markov_w2) goto fail;
    capture->output_ids[0] = input_id;
    float markov_embeds[DRAFT * 256];
    for (int position = 0; position < DRAFT; position++) {
        float *embed = markov_embeds + position * 256;
        st_read_slice_f32(source, "mtp.2.markov_head.markov_w1.weight",
                          capture->output_ids[position] * 256, 256, embed, 0);
        int best = 0;
        for (int vocab = 0; vocab < VOCAB; vocab++) {
            float bias = 0.0f;
            const float *weight = markov_w2 + (int64_t)vocab * 256;
            for (int d = 0; d < 256; d++) bias += weight[d] * embed[d];
            float value = (logits[(int64_t)position * VOCAB + vocab] += bias);
            if (vocab == 0 || value > logits[(int64_t)position * VOCAB + best])
                best = vocab;
        }
        capture->output_ids[position + 1] = best;
    }
    snprintf(name, sizeof(name), "mtp.2.confidence_head.proj.weight");
    float *confidence_weight = deepseek_v4_forward_read_f32(source, name);
    if (!confidence_weight) goto fail;
    for (int position = 0; position < DRAFT; position++) {
        float sum = 0.0f;
        const float *hidden = capture->hidden + (int64_t)position * DEEPSEEK_V4_FORWARD_D;
        const float *embed = markov_embeds + position * 256;
        for (int d = 0; d < DEEPSEEK_V4_FORWARD_D; d++) sum += confidence_weight[d] * hidden[d];
        for (int d = 0; d < 256; d++) sum += confidence_weight[DEEPSEEK_V4_FORWARD_D + d] * embed[d];
        capture->confidence[position] = sum;
    }
    if (state->next_position < start_position + 1)
        state->next_position = start_position + 1;
    free(confidence_weight); free(markov_w2);
    free(streams); free(normalized); free(logits);
    return 1;
fail:
    free(markov_w2); free(streams); free(normalized); free(logits);
    return 0;
}

static inline int deepseek_v4_forward_dspark_forward(
        shards *source, const float *main_x, int n_main, int input_id,
        DeepSeekV4ForwardDSparkState *state, DeepSeekV4ForwardDSparkCapture *capture) {
    if (!source || !main_x || n_main != 4 || !state || !capture) return 0;
    if (!deepseek_v4_forward_dspark_prefill(source, main_x, n_main, state, capture))
        return 0;
    return deepseek_v4_forward_dspark_propose(
        source, main_x + (int64_t)(n_main - 1) * DEEPSEEK_V4_FORWARD_D,
        n_main, input_id, state, capture);
}

static inline int deepseek_v4_forward_layers_forward(shards *source, const int *token_ids,
                                          int n_tokens, int first_layer,
                                          int layer_count, float *streams,
                                          float *layer_outputs,
                                          DeepSeekV4ForwardLayer2CSACapture *csa) {
    if (!source || !token_ids || !streams || !layer_outputs || n_tokens <= 0 ||
        first_layer < 0 || layer_count <= 0 || first_layer + layer_count > 3)
        return 0;
    size_t layer_size = (size_t)n_tokens * DEEPSEEK_V4_FORWARD_HC * DEEPSEEK_V4_FORWARD_D;
    for (int offset = 0; offset < layer_count; offset++) {
        int layer = first_layer + offset;
        if (!deepseek_v4_forward_layer_forward(source, layer, token_ids, n_tokens,
                                   streams, csa)) return 0;
        memcpy(layer_outputs + (size_t)offset * layer_size, streams,
               layer_size * sizeof(float));
    }
    return 1;
}

static inline int deepseek_v4_forward_forward(shards *source, int token_id,
                                         DeepSeekV4ForwardLayer0Capture *capture) {
    if (!source || !capture || token_id < 0 || !capture->input ||
        !capture->attn_post || !capture->attn_comb || !capture->attn_collapsed ||
        !capture->attn_norm || !capture->attn_output || !capture->after_attn ||
        !capture->ffn_post || !capture->ffn_comb || !capture->ffn_collapsed ||
        !capture->ffn_norm || !capture->router_scores || !capture->router_weights ||
        !capture->router_indices || !capture->moe_output || !capture->output) return 0;
    float embedding[DEEPSEEK_V4_FORWARD_D];
    st_read_slice_f32(source, "embed.weight", (int64_t)token_id * DEEPSEEK_V4_FORWARD_D,
                      DEEPSEEK_V4_FORWARD_D, embedding, 0);
    for (int h = 0; h < DEEPSEEK_V4_FORWARD_HC; h++)
        memcpy(capture->input + h * DEEPSEEK_V4_FORWARD_D, embedding, DEEPSEEK_V4_FORWARD_D * sizeof(float));
    if (!deepseek_v4_forward_hc(source, "attn", capture->input, capture->attn_post,
                    capture->attn_comb, capture->attn_collapsed)) return 0;
    if (!deepseek_v4_forward_norm(source, "layers.0.attn_norm.weight", capture->attn_collapsed,
                      DEEPSEEK_V4_FORWARD_D, capture->attn_norm)) return 0;
    if (!deepseek_v4_forward_attention(source, capture->attn_norm, capture->attn_output)) return 0;
    deepseek_v4_forward_hc_post(capture->after_attn, capture->attn_output, capture->input,
                    capture->attn_post, capture->attn_comb);
    if (!deepseek_v4_forward_hc(source, "ffn", capture->after_attn, capture->ffn_post,
                    capture->ffn_comb, capture->ffn_collapsed)) return 0;
    if (!deepseek_v4_forward_norm(source, "layers.0.ffn_norm.weight", capture->ffn_collapsed,
                      DEEPSEEK_V4_FORWARD_D, capture->ffn_norm)) return 0;
    if (!deepseek_v4_forward_moe(source, capture->ffn_norm, token_id, capture)) return 0;
    deepseek_v4_forward_hc_post(capture->output, capture->moe_output, capture->after_attn,
                    capture->ffn_post, capture->ffn_comb);
    return 1;
}

#endif
