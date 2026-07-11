#ifndef V4_REAL_LAYER0_H
#define V4_REAL_LAYER0_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moe_route.h"
#include "st.h"
#include "v4_compress.h"
#include "v4_hc.h"
#include "v4_indexer.h"
#include "v4_kv_cache.h"
#include "v4_quant.h"

enum {
    V4_REAL_D = 4096,
    V4_REAL_HC = 4,
    V4_REAL_HEADS = 64,
    V4_REAL_HD = 512,
    V4_REAL_QR = 1024,
    V4_REAL_INTER = 2048,
    V4_REAL_EXPERTS = 256,
    V4_REAL_TOPK = 6,
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
} V4RealLayer0Capture;

typedef struct {
    float *compressed_kv;
    float *index_scores;
    int64_t *index_ids;
    float *block_bias;
} V4RealLayer2CSACapture;

typedef struct {
    float *projected_kv;
    float *projected_gate;
    float *router_scores;
    float *router_weights;
    int64_t *router_indices;
} V4RealLayerCapture;

typedef struct {
    int layer;
    int ratio;
    int max_context;
    int next_position;
    V4KVCacheF32 window;
    V4KVCacheF32 compressed;
    V4KVCacheF32 index;
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
} V4RealLayerState;

static inline int v4_real_layer_ratio(int layer) {
    if (layer < 0 || layer > 42) return -1;
    if (layer < 2) return 0;
    return (layer & 1) ? 128 : 4;
}

static inline void v4_real_layer_state_free(V4RealLayerState *state) {
    if (!state) return;
    free(state->window_keys); free(state->window_positions);
    free(state->compressed_keys); free(state->compressed_positions);
    free(state->index_keys); free(state->index_positions);
    free(state->compressor_kv_state); free(state->compressor_gate_state);
    free(state->index_kv_state); free(state->index_gate_state);
    memset(state, 0, sizeof(*state));
}

static inline int v4_real_layer_state_init(V4RealLayerState *state, int layer,
                                            int max_context) {
    if (!state || max_context <= 0) return 0;
    int ratio = v4_real_layer_ratio(layer);
    if (ratio < 0) return 0;
    memset(state, 0, sizeof(*state));
    state->layer = layer;
    state->ratio = ratio;
    state->max_context = max_context;
    int compressed_capacity = ratio ? (max_context + ratio - 1) / ratio : 1;
    state->window_keys = malloc((size_t)128 * V4_REAL_HD * sizeof(float));
    state->window_positions = malloc(128 * sizeof(int64_t));
    state->compressed_keys = malloc((size_t)compressed_capacity * V4_REAL_HD * sizeof(float));
    state->compressed_positions = malloc((size_t)compressed_capacity * sizeof(int64_t));
    if (!state->window_keys || !state->window_positions ||
        !state->compressed_keys || !state->compressed_positions) goto fail;
    if (!v4_kv_cache_init(&state->window, V4_KV_CACHE_RING, 128, V4_REAL_HD,
                          state->window_keys, state->window_positions) ||
        !v4_kv_cache_init(&state->compressed, V4_KV_CACHE_APPEND,
                          compressed_capacity, V4_REAL_HD,
                          state->compressed_keys, state->compressed_positions)) goto fail;
    if (ratio) {
        int overlap = ratio == 4;
        int rows = (1 + overlap) * ratio;
        int width = (1 + overlap) * V4_REAL_HD;
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
        if (!v4_kv_cache_init(&state->index, V4_KV_CACHE_APPEND,
                              compressed_capacity, 128,
                              state->index_keys, state->index_positions)) goto fail;
    }
    return 1;
fail:
    v4_real_layer_state_free(state);
    return 0;
}

static inline float *v4_real_read_f32(shards *source, const char *name) {
    int64_t count = st_numel(source, name);
    if (count <= 0 || (uint64_t)count > SIZE_MAX / sizeof(float)) return NULL;
    float *data = malloc((size_t)count * sizeof(float));
    if (!data) return NULL;
    st_read_f32(source, name, data, 0);
    return data;
}

static inline uint8_t *v4_real_read_raw(shards *source, const char *name) {
    int64_t count = st_nbytes(source, name);
    if (count <= 0 || (uint64_t)count > SIZE_MAX) return NULL;
    uint8_t *data = malloc((size_t)count);
    if (!data) return NULL;
    st_read_raw(source, name, data, 0);
    return data;
}

static inline void v4_real_matvec(float *output, const float *weight,
                                  const float *input, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        const float *w = weight + (int64_t)row * columns;
        for (int column = 0; column < columns; column++) sum += w[column] * input[column];
        output[row] = sum;
    }
}

static inline int v4_real_fp8_linear(shards *source, const char *prefix,
                                     const float *input, int rows, int columns,
                                     float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = v4_real_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = v4_real_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = v4_fp8_matmul_f32(output, input, 1, weight, scale, rows, columns, 128);
    free(weight); free(scale);
    return ok;
}

static inline int v4_real_fp4_linear(shards *source, const char *prefix,
                                     const float *input, int rows, int columns,
                                     float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = v4_real_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = v4_real_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = v4_fp4_matmul_f32(output, input, 1, weight, scale, rows, columns);
    free(weight); free(scale);
    return ok;
}

static inline int v4_real_norm(shards *source, const char *name,
                               const float *input, int count, float *output) {
    float *weight = v4_real_read_f32(source, name);
    if (!weight) return 0;
    float mean_square = 0.0f;
    for (int i = 0; i < count; i++) mean_square += input[i] * input[i];
    float scale = 1.0f / sqrtf(mean_square / count + 1e-6f);
    for (int i = 0; i < count; i++) output[i] = input[i] * scale * weight[i];
    free(weight);
    return 1;
}

static inline int v4_real_hc_layer(shards *source, int layer, const char *site,
                                   const float *input, int n_tokens,
                                   float *post, float *comb, float *collapsed) {
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.hc_%s_fn", layer, site);
    float *fn = v4_real_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.hc_%s_base", layer, site);
    float *base = v4_real_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.hc_%s_scale", layer, site);
    float *scale = v4_real_read_f32(source, name);
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    V4HyperConnection model = {
        .streams = V4_REAL_HC, .hidden = V4_REAL_D, .sinkhorn_iters = 20,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f, .fn = fn, .base = base, .scale = scale,
    };
    int ok = v4_hc_forward(&model, input, n_tokens, post, comb, collapsed);
    free(fn); free(base); free(scale);
    return ok;
}

static inline int v4_real_hc(shards *source, const char *site, const float *input,
                             float *post, float *comb, float *collapsed) {
    return v4_real_hc_layer(source, 0, site, input, 1, post, comb, collapsed);
}

static inline void v4_real_hc_post(float *output, const float *block,
                                   const float *residual, const float *post,
                                   const float *comb) {
    for (int row = 0; row < V4_REAL_HC; row++) {
        for (int d = 0; d < V4_REAL_D; d++) {
            float sum = post[row] * block[d];
            for (int column = 0; column < V4_REAL_HC; column++)
                sum += comb[row * V4_REAL_HC + column]
                     * residual[column * V4_REAL_D + d];
            output[row * V4_REAL_D + d] = sum;
        }
    }
}

static inline int v4_real_attention(shards *source, const float *input, float *output) {
    float *q_a = malloc(V4_REAL_QR * sizeof(float));
    float *q = malloc((size_t)V4_REAL_HEADS * V4_REAL_HD * sizeof(float));
    float *kv = malloc(V4_REAL_HD * sizeof(float));
    float *context = malloc((size_t)V4_REAL_HEADS * V4_REAL_HD * sizeof(float));
    float *grouped = malloc(8192 * sizeof(float));
    if (!q_a || !q || !kv || !context || !grouped) goto fail;
    if (!v4_real_fp8_linear(source, "layers.0.attn.wq_a", input,
                            V4_REAL_QR, V4_REAL_D, q_a)) goto fail;
    float *q_norm = malloc(V4_REAL_QR * sizeof(float));
    if (!q_norm || !v4_real_norm(source, "layers.0.attn.q_norm.weight",
                                  q_a, V4_REAL_QR, q_norm)) { free(q_norm); goto fail; }
    if (!v4_real_fp8_linear(source, "layers.0.attn.wq_b", q_norm,
                            V4_REAL_HEADS * V4_REAL_HD, V4_REAL_QR, q)) {
        free(q_norm); goto fail;
    }
    free(q_norm);
    for (int head = 0; head < V4_REAL_HEADS; head++) {
        float *row = q + (int64_t)head * V4_REAL_HD;
        float mean_square = 0.0f;
        for (int d = 0; d < V4_REAL_HD; d++) mean_square += row[d] * row[d];
        float scale = 1.0f / sqrtf(mean_square / V4_REAL_HD + 1e-6f);
        for (int d = 0; d < V4_REAL_HD; d++) row[d] *= scale;
    }
    if (!v4_real_fp8_linear(source, "layers.0.attn.wkv", input,
                            V4_REAL_HD, V4_REAL_D, kv)) goto fail;
    float *kv_norm = malloc(V4_REAL_HD * sizeof(float));
    if (!kv_norm || !v4_real_norm(source, "layers.0.attn.kv_norm.weight",
                                   kv, V4_REAL_HD, kv_norm)) { free(kv_norm); goto fail; }
    memcpy(kv, kv_norm, V4_REAL_HD * sizeof(float));
    free(kv_norm);
    float *sinks = v4_real_read_f32(source, "layers.0.attn.attn_sink");
    if (!sinks) goto fail;
    float attention_scale = 1.0f / sqrtf((float)V4_REAL_HD);
    for (int head = 0; head < V4_REAL_HEADS; head++) {
        float score = 0.0f;
        const float *query = q + (int64_t)head * V4_REAL_HD;
        for (int d = 0; d < V4_REAL_HD; d++) score += query[d] * kv[d];
        score *= attention_scale;
        float maximum = score > sinks[head] ? score : sinks[head];
        float probability = expf(score - maximum)
            / (expf(score - maximum) + expf(sinks[head] - maximum));
        for (int d = 0; d < V4_REAL_HD; d++)
            context[(int64_t)head * V4_REAL_HD + d] = probability * kv[d];
    }
    free(sinks);

    uint8_t *wo_a = v4_real_read_raw(source, "layers.0.attn.wo_a.weight");
    uint8_t *wo_a_scale = v4_real_read_raw(source, "layers.0.attn.wo_a.scale");
    if (!wo_a || !wo_a_scale) { free(wo_a); free(wo_a_scale); goto fail; }
    for (int group = 0; group < 8; group++) {
        if (!v4_fp8_matmul_f32(grouped + group * 1024, context + group * 4096, 1,
                wo_a + (int64_t)group * 1024 * 4096,
                wo_a_scale + (int64_t)group * 8 * 32, 1024, 4096, 128)) {
            free(wo_a); free(wo_a_scale); goto fail;
        }
    }
    free(wo_a); free(wo_a_scale);
    if (!v4_real_fp8_linear(source, "layers.0.attn.wo_b", grouped,
                            V4_REAL_D, 8192, output)) goto fail;
    free(q_a); free(q); free(kv); free(context); free(grouped);
    return 1;
fail:
    free(q_a); free(q); free(kv); free(context); free(grouped);
    return 0;
}

static inline int v4_real_expert(shards *source, const char *prefix,
                                 const float *input, int fp4, float coefficient,
                                 float *output) {
    float *gate = malloc(V4_REAL_INTER * sizeof(float));
    float *up = malloc(V4_REAL_INTER * sizeof(float));
    float *activated = malloc(V4_REAL_INTER * sizeof(float));
    if (!gate || !up || !activated) { free(gate); free(up); free(activated); return 0; }
    char name[320];
    snprintf(name, sizeof(name), "%s.w1", prefix);
    int ok = fp4 ? v4_real_fp4_linear(source, name, input, V4_REAL_INTER, V4_REAL_D, gate)
                 : v4_real_fp8_linear(source, name, input, V4_REAL_INTER, V4_REAL_D, gate);
    snprintf(name, sizeof(name), "%s.w3", prefix);
    ok = ok && (fp4 ? v4_real_fp4_linear(source, name, input, V4_REAL_INTER, V4_REAL_D, up)
                    : v4_real_fp8_linear(source, name, input, V4_REAL_INTER, V4_REAL_D, up));
    if (!ok) { free(gate); free(up); free(activated); return 0; }
    for (int i = 0; i < V4_REAL_INTER; i++) {
        if (gate[i] > 10.0f) gate[i] = 10.0f;
        if (up[i] > 10.0f) up[i] = 10.0f;
        if (up[i] < -10.0f) up[i] = -10.0f;
        activated[i] = coefficient * (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    }
    snprintf(name, sizeof(name), "%s.w2", prefix);
    ok = fp4 ? v4_real_fp4_linear(source, name, activated, V4_REAL_D, V4_REAL_INTER, output)
             : v4_real_fp8_linear(source, name, activated, V4_REAL_D, V4_REAL_INTER, output);
    free(gate); free(up); free(activated);
    return ok;
}

static inline int v4_real_moe(shards *source, const float *input, int token_id,
                              V4RealLayer0Capture *capture) {
    float *router = v4_real_read_f32(source, "layers.0.ffn.gate.weight");
    float logits[V4_REAL_EXPERTS];
    int indices[V4_REAL_TOPK];
    if (!router) return 0;
    v4_real_matvec(logits, router, input, V4_REAL_EXPERTS, V4_REAL_D);
    free(router);
    st_read_slice_raw(source, "layers.0.ffn.gate.tid2eid",
                      (int64_t)token_id * V4_REAL_TOPK * sizeof(int64_t),
                      V4_REAL_TOPK * sizeof(int64_t), capture->router_indices, 0);
    for (int k = 0; k < V4_REAL_TOPK; k++) indices[k] = (int)capture->router_indices[k];
    if (moe_route_fixed(logits, V4_REAL_EXPERTS, indices, V4_REAL_TOPK,
                        MOE_SCORE_SQRT_SOFTPLUS, capture->router_weights,
                        capture->router_scores) != V4_REAL_TOPK) return 0;
    moe_route_finalize(capture->router_weights, V4_REAL_TOPK, 1, 1.5f);
    memset(capture->moe_output, 0, V4_REAL_D * sizeof(float));
    float *expert_output = malloc(V4_REAL_D * sizeof(float));
    if (!expert_output) return 0;
    char prefix[320];
    for (int k = 0; k < V4_REAL_TOPK; k++) {
        snprintf(prefix, sizeof(prefix), "layers.0.ffn.experts.%d", indices[k]);
        if (!v4_real_expert(source, prefix, input, 1,
                            capture->router_weights[k], expert_output)) {
            free(expert_output); return 0;
        }
        for (int d = 0; d < V4_REAL_D; d++) capture->moe_output[d] += expert_output[d];
    }
    if (!v4_real_expert(source, "layers.0.ffn.shared_experts", input, 0, 1.0f,
                        expert_output)) { free(expert_output); return 0; }
    for (int d = 0; d < V4_REAL_D; d++) capture->moe_output[d] += expert_output[d];
    free(expert_output);
    return 1;
}

static inline int v4_real_fp8_batch(shards *source, const char *prefix,
                                    const float *input, int n_inputs,
                                    int rows, int columns, float *output) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint8_t *weight = v4_real_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    uint8_t *scale = v4_real_read_raw(source, name);
    if (!weight || !scale) { free(weight); free(scale); return 0; }
    int ok = v4_fp8_matmul_f32(output, input, n_inputs, weight, scale,
                               rows, columns, 128);
    free(weight); free(scale);
    return ok;
}

static inline int v4_real_dense_batch(shards *source, const char *name,
                                      const float *input, int n_inputs,
                                      int rows, int columns, float *output) {
    float *weight = v4_real_read_f32(source, name);
    if (!weight) return 0;
    for (int token = 0; token < n_inputs; token++)
        v4_real_matvec(output + (int64_t)token * rows, weight,
                       input + (int64_t)token * columns, rows, columns);
    free(weight);
    return 1;
}

static inline int v4_real_norm_batch(shards *source, const char *name,
                                     const float *input, int n_inputs,
                                     int count, float *output) {
    float *weight = v4_real_read_f32(source, name);
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

static inline float v4_real_compress_frequency(int pair) {
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

static inline void v4_real_compress_rope(float *vector, int width,
                                         int64_t position) {
    int start = width - 64;
    for (int pair = 0; pair < 32; pair++) {
        float angle = (float)position * v4_real_compress_frequency(pair);
        float cosine = cosf(angle), sine = sinf(angle);
        int index = start + pair * 2;
        float first = vector[index], second = vector[index + 1];
        vector[index] = first * cosine - second * sine;
        vector[index + 1] = second * cosine + first * sine;
    }
}

static inline int v4_real_layer2_csa_forward(shards *source, const float *streams,
                                             V4RealLayer2CSACapture *capture) {
    if (!source || !streams || !capture || !capture->compressed_kv ||
        !capture->index_scores || !capture->index_ids || !capture->block_bias) return 0;
    enum { TOKENS = 4 };
    float *post = malloc(TOKENS * V4_REAL_HC * sizeof(float));
    float *comb = malloc(TOKENS * V4_REAL_HC * V4_REAL_HC * sizeof(float));
    float *collapsed = malloc(TOKENS * V4_REAL_D * sizeof(float));
    float *hidden = malloc(TOKENS * V4_REAL_D * sizeof(float));
    float *q_a = malloc(TOKENS * V4_REAL_QR * sizeof(float));
    float *q_norm = malloc(TOKENS * V4_REAL_QR * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !q_a || !q_norm) goto fail;
    if (!v4_real_hc_layer(source, 2, "attn", streams, TOKENS,
                          post, comb, collapsed)) goto fail;
    if (!v4_real_norm_batch(source, "layers.2.attn_norm.weight", collapsed,
                            TOKENS, V4_REAL_D, hidden)) goto fail;
    if (!v4_real_fp8_batch(source, "layers.2.attn.wq_a", hidden, TOKENS,
                           V4_REAL_QR, V4_REAL_D, q_a)) goto fail;
    if (!v4_real_norm_batch(source, "layers.2.attn.q_norm.weight", q_a,
                            TOKENS, V4_REAL_QR, q_norm)) goto fail;

    float *outer_kv = malloc(TOKENS * 1024 * sizeof(float));
    float *outer_gate = malloc(TOKENS * 1024 * sizeof(float));
    float *inner_kv = malloc(TOKENS * 256 * sizeof(float));
    float *inner_gate = malloc(TOKENS * 256 * sizeof(float));
    if (!outer_kv || !outer_gate || !inner_kv || !inner_gate) goto fail2;
    if (!v4_real_dense_batch(source, "layers.2.attn.compressor.wkv.weight",
                             hidden, TOKENS, 1024, V4_REAL_D, outer_kv) ||
        !v4_real_dense_batch(source, "layers.2.attn.compressor.wgate.weight",
                             hidden, TOKENS, 1024, V4_REAL_D, outer_gate) ||
        !v4_real_dense_batch(source, "layers.2.attn.indexer.compressor.wkv.weight",
                             hidden, TOKENS, 256, V4_REAL_D, inner_kv) ||
        !v4_real_dense_batch(source, "layers.2.attn.indexer.compressor.wgate.weight",
                             hidden, TOKENS, 256, V4_REAL_D, inner_gate)) goto fail2;
    float *outer_ape = v4_real_read_f32(source, "layers.2.attn.compressor.ape");
    float *outer_norm = v4_real_read_f32(source, "layers.2.attn.compressor.norm.weight");
    float *inner_ape = v4_real_read_f32(source, "layers.2.attn.indexer.compressor.ape");
    float *inner_norm = v4_real_read_f32(source, "layers.2.attn.indexer.compressor.norm.weight");
    float index_kv[128];
    if (!outer_ape || !outer_norm || !inner_ape || !inner_norm) goto fail3;
    V4PrefillCompressorF32 outer = {
        .ratio = 4, .head_dim = 512, .rope_dim = 64, .overlap = 1,
        .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = outer_ape, .norm_weight = outer_norm,
    };
    V4PrefillCompressorF32 inner = {
        .ratio = 4, .head_dim = 128, .rope_dim = 64, .overlap = 1,
        .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = inner_ape, .norm_weight = inner_norm,
    };
    if (!v4_compress_prefill_f32(&outer, outer_kv, outer_gate, TOKENS,
                                  capture->compressed_kv) ||
        !v4_compress_prefill_f32(&inner, inner_kv, inner_gate, TOKENS, index_kv)) goto fail3;

    float *queries = malloc(TOKENS * 64 * 128 * sizeof(float));
    float *head_weights = malloc(TOKENS * 64 * sizeof(float));
    if (!queries || !head_weights) { free(queries); free(head_weights); goto fail3; }
    if (!v4_real_fp8_batch(source, "layers.2.attn.indexer.wq_b", q_norm,
                           TOKENS, 64 * 128, V4_REAL_QR, queries) ||
        !v4_real_dense_batch(source, "layers.2.attn.indexer.weights_proj.weight",
                             hidden, TOKENS, 64, V4_REAL_D, head_weights)) {
        free(queries); free(head_weights); goto fail3;
    }
    for (int token = 0; token < TOKENS; token++)
        for (int head = 0; head < 64; head++)
            v4_real_compress_rope(queries + ((int64_t)token * 64 + head) * 128,
                                  128, token);
    int64_t positions[4] = {0, 1, 2, 3};
    V4LightningIndexerF32 indexer = {
        .n_heads = 64, .head_dim = 128, .ratio = 4, .top_k = 1,
    };
    int ok = v4_indexer_forward_f32(&indexer, queries, index_kv, head_weights,
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

static inline int v4_real_compress_layer_prefill(
        shards *source, int layer, const float *hidden, int n_tokens,
        V4RealLayerState *state, V4RealLayerCapture *capture) {
    if (!source || !hidden || !state || state->layer != layer ||
        state->ratio <= 0 || state->next_position != 0 || n_tokens <= 0 ||
        n_tokens % state->ratio != 0) return 0;
    int overlap = state->ratio == 4;
    int width = (1 + overlap) * V4_REAL_HD;
    int n_entries = n_tokens / state->ratio;
    float *kv = malloc((size_t)n_tokens * width * sizeof(float));
    float *gate = malloc((size_t)n_tokens * width * sizeof(float));
    float *compressed = malloc((size_t)n_entries * V4_REAL_HD * sizeof(float));
    char name[320];
    if (!kv || !gate || !compressed) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.wkv.weight", layer);
    if (!v4_real_dense_batch(source, name, hidden, n_tokens,
                             width, V4_REAL_D, kv)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.wgate.weight", layer);
    if (!v4_real_dense_batch(source, name, hidden, n_tokens,
                             width, V4_REAL_D, gate)) goto fail;
    if (capture && capture->projected_kv)
        memcpy(capture->projected_kv, kv,
               (size_t)n_tokens * width * sizeof(float));
    if (capture && capture->projected_gate)
        memcpy(capture->projected_gate, gate,
               (size_t)n_tokens * width * sizeof(float));
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.ape", layer);
    float *ape = v4_real_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.%d.attn.compressor.norm.weight", layer);
    float *norm = v4_real_read_f32(source, name);
    if (!ape || !norm) { free(ape); free(norm); goto fail; }
    V4PrefillCompressorF32 model = {
        .ratio = state->ratio, .head_dim = V4_REAL_HD, .rope_dim = 64,
        .overlap = overlap, .norm_eps = 1e-6f, .rope_theta = 160000.0f,
        .position_bias = ape, .norm_weight = norm,
    };
    int ok = v4_compress_prefill_f32(&model, kv, gate, n_tokens, compressed);
    free(ape); free(norm);
    if (!ok) goto fail;
    for (int entry = 0; entry < n_entries; entry++) {
        int64_t position = (int64_t)entry * state->ratio;
        if (!v4_kv_cache_append(&state->compressed,
                                compressed + (int64_t)entry * V4_REAL_HD,
                                position)) goto fail;
    }
    free(kv); free(gate); free(compressed);
    return 1;
fail:
    free(kv); free(gate); free(compressed);
    return 0;
}

static inline float v4_real_rope_frequency(int pair, int compressed) {
    if (compressed) return v4_real_compress_frequency(pair);
    return powf(10000.0f, -(2.0f * pair) / 64.0f);
}

static inline void v4_real_rope(float *vector, int width, int64_t position,
                                int compressed, int inverse) {
    int start = width - 64;
    for (int pair = 0; pair < 32; pair++) {
        float angle = (float)position * v4_real_rope_frequency(pair, compressed);
        float cosine = cosf(angle), sine = sinf(angle);
        if (inverse) sine = -sine;
        int index = start + pair * 2;
        float first = vector[index], second = vector[index + 1];
        vector[index] = first * cosine - second * sine;
        vector[index + 1] = second * cosine + first * sine;
    }
}

static inline void v4_real_hc_post_batch(float *output, const float *block,
                                         const float *residual, const float *post,
                                         const float *comb, int n_tokens) {
    for (int token = 0; token < n_tokens; token++) {
        const float *token_block = block + (int64_t)token * V4_REAL_D;
        const float *token_residual = residual
            + (int64_t)token * V4_REAL_HC * V4_REAL_D;
        float *token_output = output + (int64_t)token * V4_REAL_HC * V4_REAL_D;
        for (int row = 0; row < V4_REAL_HC; row++) {
            for (int d = 0; d < V4_REAL_D; d++) {
                float sum = post[(int64_t)token * V4_REAL_HC + row]
                          * token_block[d];
                for (int column = 0; column < V4_REAL_HC; column++)
                    sum += comb[((int64_t)token * V4_REAL_HC + row)
                                * V4_REAL_HC + column]
                         * token_residual[column * V4_REAL_D + d];
                token_output[row * V4_REAL_D + d] = sum;
            }
        }
    }
}

static inline int v4_real_attention_batch(shards *source, int layer,
                                           const float *hidden, int n_tokens,
                                           const V4RealLayer2CSACapture *csa,
                                           V4RealLayerState *state,
                                           float *output) {
    char prefix[320], name[320];
    snprintf(prefix, sizeof(prefix), "layers.%d.attn", layer);
    size_t q_count = (size_t)n_tokens * V4_REAL_HEADS * V4_REAL_HD;
    float *q_a = malloc((size_t)n_tokens * V4_REAL_QR * sizeof(float));
    float *q_norm = malloc((size_t)n_tokens * V4_REAL_QR * sizeof(float));
    float *q = malloc(q_count * sizeof(float));
    float *kv = malloc((size_t)n_tokens * V4_REAL_HD * sizeof(float));
    float *context = malloc(q_count * sizeof(float));
    float *group_input = malloc((size_t)n_tokens * 4096 * sizeof(float));
    float *group_output = malloc((size_t)n_tokens * 1024 * sizeof(float));
    float *grouped = malloc((size_t)n_tokens * 8192 * sizeof(float));
    if (!q_a || !q_norm || !q || !kv || !context || !group_input ||
        !group_output || !grouped) goto fail;
    snprintf(name, sizeof(name), "%s.wq_a", prefix);
    if (!v4_real_fp8_batch(source, name, hidden, n_tokens,
                           V4_REAL_QR, V4_REAL_D, q_a)) goto fail;
    snprintf(name, sizeof(name), "%s.q_norm.weight", prefix);
    if (!v4_real_norm_batch(source, name, q_a, n_tokens,
                            V4_REAL_QR, q_norm)) goto fail;
    snprintf(name, sizeof(name), "%s.wq_b", prefix);
    if (!v4_real_fp8_batch(source, name, q_norm, n_tokens,
                           V4_REAL_HEADS * V4_REAL_HD, V4_REAL_QR, q)) goto fail;
    snprintf(name, sizeof(name), "%s.wkv", prefix);
    if (!v4_real_fp8_batch(source, name, hidden, n_tokens,
                           V4_REAL_HD, V4_REAL_D, kv)) goto fail;
    snprintf(name, sizeof(name), "%s.kv_norm.weight", prefix);
    if (!v4_real_norm_batch(source, name, kv, n_tokens,
                            V4_REAL_HD, kv)) goto fail;

    int compressed = layer >= 2;
    int start_position = state ? state->next_position : 0;
    for (int token = 0; token < n_tokens; token++) {
        int64_t position = start_position + token;
        v4_real_rope(kv + (int64_t)token * V4_REAL_HD,
                     V4_REAL_HD, position, compressed, 0);
        for (int head = 0; head < V4_REAL_HEADS; head++) {
            float *query = q + ((int64_t)token * V4_REAL_HEADS + head)
                             * V4_REAL_HD;
            float mean_square = 0.0f;
            for (int d = 0; d < V4_REAL_HD; d++)
                mean_square += query[d] * query[d];
            float scale = 1.0f / sqrtf(mean_square / V4_REAL_HD + 1e-6f);
            for (int d = 0; d < V4_REAL_HD; d++) query[d] *= scale;
            v4_real_rope(query, V4_REAL_HD, position, compressed, 0);
        }
    }

    snprintf(name, sizeof(name), "%s.attn_sink", prefix);
    float *sinks = v4_real_read_f32(source, name);
    if (!sinks) goto fail;
    float attention_scale = 1.0f / sqrtf((float)V4_REAL_HD);
    for (int token = 0; token < n_tokens; token++) {
        int64_t position = start_position + token;
        int local_start = token >= 127 ? token - 127 : 0;
        int use_compressed = layer == 2 && csa && csa->index_ids[token] >= 0;
        for (int head = 0; head < V4_REAL_HEADS; head++) {
            const float *query = q + ((int64_t)token * V4_REAL_HEADS + head)
                                   * V4_REAL_HD;
            float maximum = sinks[head];
            for (int key = local_start; key <= token; key++) {
                const float *key_vector = kv + (int64_t)key * V4_REAL_HD;
                float score = 0.0f;
                for (int d = 0; d < V4_REAL_HD; d++)
                    score += query[d] * key_vector[d];
                score *= attention_scale;
                if (score > maximum) maximum = score;
            }
            float compressed_score = -INFINITY;
            if (use_compressed) {
                compressed_score = 0.0f;
                for (int d = 0; d < V4_REAL_HD; d++)
                    compressed_score += query[d] * csa->compressed_kv[d];
                compressed_score *= attention_scale;
                if (compressed_score > maximum) maximum = compressed_score;
            }
            if (state && state->ratio == 128) {
                for (int entry = 0; entry < state->compressed.count; entry++) {
                    if (v4_kv_cache_position(&state->compressed, entry)
                            + state->ratio > position + 1) continue;
                    const float *key_vector = v4_kv_cache_key(&state->compressed, entry);
                    float score = 0.0f;
                    for (int d = 0; d < V4_REAL_HD; d++)
                        score += query[d] * key_vector[d];
                    score *= attention_scale;
                    if (score > maximum) maximum = score;
                }
            }
            float denominator = expf(sinks[head] - maximum);
            float *result = context
                + ((int64_t)token * V4_REAL_HEADS + head) * V4_REAL_HD;
            memset(result, 0, V4_REAL_HD * sizeof(float));
            for (int key = local_start; key <= token; key++) {
                const float *key_vector = kv + (int64_t)key * V4_REAL_HD;
                float score = 0.0f;
                for (int d = 0; d < V4_REAL_HD; d++)
                    score += query[d] * key_vector[d];
                float probability = expf(score * attention_scale - maximum);
                denominator += probability;
                for (int d = 0; d < V4_REAL_HD; d++)
                    result[d] += probability * key_vector[d];
            }
            if (use_compressed) {
                float probability = expf(compressed_score - maximum);
                denominator += probability;
                for (int d = 0; d < V4_REAL_HD; d++)
                    result[d] += probability * csa->compressed_kv[d];
            }
            if (state && state->ratio == 128) {
                for (int entry = 0; entry < state->compressed.count; entry++) {
                    if (v4_kv_cache_position(&state->compressed, entry)
                            + state->ratio > position + 1) continue;
                    const float *key_vector = v4_kv_cache_key(&state->compressed, entry);
                    float score = 0.0f;
                    for (int d = 0; d < V4_REAL_HD; d++)
                        score += query[d] * key_vector[d];
                    float probability = expf(score * attention_scale - maximum);
                    denominator += probability;
                    for (int d = 0; d < V4_REAL_HD; d++)
                        result[d] += probability * key_vector[d];
                }
            }
            for (int d = 0; d < V4_REAL_HD; d++) result[d] /= denominator;
            v4_real_rope(result, V4_REAL_HD, position, compressed, 1);
        }
    }
    free(sinks);
    if (state)
        for (int token = 0; token < n_tokens; token++)
            if (!v4_kv_cache_append(&state->window,
                                    kv + (int64_t)token * V4_REAL_HD,
                                    start_position + token)) goto fail;

    snprintf(name, sizeof(name), "%s.wo_a.weight", prefix);
    uint8_t *wo_a = v4_real_read_raw(source, name);
    snprintf(name, sizeof(name), "%s.wo_a.scale", prefix);
    uint8_t *wo_a_scale = v4_real_read_raw(source, name);
    if (!wo_a || !wo_a_scale) { free(wo_a); free(wo_a_scale); goto fail; }
    for (int group = 0; group < 8; group++) {
        for (int token = 0; token < n_tokens; token++)
            memcpy(group_input + (int64_t)token * 4096,
                   context + ((int64_t)token * V4_REAL_HEADS + group * 8)
                             * V4_REAL_HD,
                   4096 * sizeof(float));
        if (!v4_fp8_matmul_f32(group_output, group_input, n_tokens,
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
    int ok = v4_real_fp8_batch(source, name, grouped, n_tokens,
                               V4_REAL_D, 8192, output);
    free(q_a); free(q_norm); free(q); free(kv); free(context);
    free(group_input); free(group_output); free(grouped);
    return ok;
fail:
    free(q_a); free(q_norm); free(q); free(kv); free(context);
    free(group_input); free(group_output); free(grouped);
    return 0;
}

static inline int v4_real_moe_batch(shards *source, int layer,
                                     const float *hidden, const int *token_ids,
                                     int n_tokens, V4RealLayerCapture *capture,
                                     float *output) {
    char name[320], prefix[320];
    snprintf(name, sizeof(name), "layers.%d.ffn.gate.weight", layer);
    float *router = v4_real_read_f32(source, name);
    float *bias = NULL;
    if (layer >= 3) {
        snprintf(name, sizeof(name), "layers.%d.ffn.gate.bias", layer);
        bias = v4_real_read_f32(source, name);
    }
    float *expert_output = malloc(V4_REAL_D * sizeof(float));
    if (!router || (layer >= 3 && !bias) || !expert_output) {
        free(router); free(bias); free(expert_output); return 0;
    }
    for (int token = 0; token < n_tokens; token++) {
        const float *input = hidden + (int64_t)token * V4_REAL_D;
        float logits[V4_REAL_EXPERTS], scores[V4_REAL_EXPERTS];
        float weights[V4_REAL_TOPK];
        int indices[V4_REAL_TOPK];
        int64_t stored_indices[V4_REAL_TOPK];
        v4_real_matvec(logits, router, input, V4_REAL_EXPERTS, V4_REAL_D);
        if (layer < 3) {
            snprintf(name, sizeof(name), "layers.%d.ffn.gate.tid2eid", layer);
            st_read_slice_raw(source, name,
                              (int64_t)token_ids[token] * V4_REAL_TOPK * sizeof(int64_t),
                              V4_REAL_TOPK * sizeof(int64_t), stored_indices, 0);
            for (int k = 0; k < V4_REAL_TOPK; k++) indices[k] = (int)stored_indices[k];
            if (moe_route_fixed(logits, V4_REAL_EXPERTS, indices, V4_REAL_TOPK,
                                MOE_SCORE_SQRT_SOFTPLUS, weights, scores)
                != V4_REAL_TOPK) goto fail;
        } else {
            float selection_scores[V4_REAL_EXPERTS];
            if (moe_route_select(logits, bias, V4_REAL_EXPERTS, V4_REAL_TOPK,
                                 MOE_SCORE_SQRT_SOFTPLUS, indices, weights,
                                 scores, selection_scores) != V4_REAL_TOPK) goto fail;
        }
        moe_route_finalize(weights, V4_REAL_TOPK, 1, 1.5f);
        if (capture) {
            if (capture->router_scores)
                memcpy(capture->router_scores + (int64_t)token * V4_REAL_EXPERTS,
                       scores, V4_REAL_EXPERTS * sizeof(float));
            if (capture->router_weights)
                memcpy(capture->router_weights + (int64_t)token * V4_REAL_TOPK,
                       weights, V4_REAL_TOPK * sizeof(float));
            if (capture->router_indices)
                for (int k = 0; k < V4_REAL_TOPK; k++)
                    capture->router_indices[(int64_t)token * V4_REAL_TOPK + k]
                        = indices[k];
        }
        float *token_output = output + (int64_t)token * V4_REAL_D;
        memset(token_output, 0, V4_REAL_D * sizeof(float));
        for (int k = 0; k < V4_REAL_TOPK; k++) {
            snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d",
                     layer, indices[k]);
            if (!v4_real_expert(source, prefix, input, 1,
                                weights[k], expert_output)) goto fail;
            for (int d = 0; d < V4_REAL_D; d++) token_output[d] += expert_output[d];
        }
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.shared_experts", layer);
        if (!v4_real_expert(source, prefix, input, 0, 1.0f,
                            expert_output)) goto fail;
        for (int d = 0; d < V4_REAL_D; d++) token_output[d] += expert_output[d];
    }
    free(router); free(bias); free(expert_output);
    return 1;
fail:
    free(router); free(bias); free(expert_output);
    return 0;
}

static inline int v4_real_layer_forward(shards *source, int layer,
                                         const int *token_ids, int n_tokens,
                                         float *streams,
                                         V4RealLayer2CSACapture *csa) {
    if (!source || !token_ids || !streams || layer < 0 || layer > 2 ||
        n_tokens <= 0 || (layer == 2 && (n_tokens != 4 || !csa))) return 0;
    size_t streams_count = (size_t)n_tokens * V4_REAL_HC * V4_REAL_D;
    float *post = malloc((size_t)n_tokens * V4_REAL_HC * sizeof(float));
    float *comb = malloc((size_t)n_tokens * V4_REAL_HC * V4_REAL_HC * sizeof(float));
    float *collapsed = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *hidden = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *block = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *residual = malloc(streams_count * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !block || !residual) goto fail;
    if (!v4_real_hc_layer(source, layer, "attn", streams, n_tokens,
                          post, comb, collapsed)) goto fail;
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", layer);
    if (!v4_real_norm_batch(source, name, collapsed, n_tokens,
                            V4_REAL_D, hidden)) goto fail;
    if (layer == 2 && !v4_real_layer2_csa_forward(source, streams, csa)) goto fail;
    if (!v4_real_attention_batch(source, layer, hidden, n_tokens,
                                 csa, NULL, block)) goto fail;
    v4_real_hc_post_batch(residual, block, streams, post, comb, n_tokens);

    if (!v4_real_hc_layer(source, layer, "ffn", residual, n_tokens,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer);
    if (!v4_real_norm_batch(source, name, collapsed, n_tokens,
                            V4_REAL_D, hidden)) goto fail;
    if (!v4_real_moe_batch(source, layer, hidden, token_ids, n_tokens,
                           NULL, block)) goto fail;
    v4_real_hc_post_batch(streams, block, residual, post, comb, n_tokens);
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 1;
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 0;
}

static inline int v4_real_layer_forward_state(
        shards *source, int layer, const int *token_ids, int n_tokens,
        float *streams, V4RealLayerState *state, V4RealLayerCapture *capture) {
    if (!source || !token_ids || !streams || !state || state->layer != layer ||
        layer < 3 || layer > 42 || state->ratio != 128 ||
        state->next_position != 0 || n_tokens <= 0 ||
        n_tokens % state->ratio != 0) return 0;
    size_t streams_count = (size_t)n_tokens * V4_REAL_HC * V4_REAL_D;
    float *post = malloc((size_t)n_tokens * V4_REAL_HC * sizeof(float));
    float *comb = malloc((size_t)n_tokens * V4_REAL_HC * V4_REAL_HC * sizeof(float));
    float *collapsed = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *hidden = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *block = malloc((size_t)n_tokens * V4_REAL_D * sizeof(float));
    float *residual = malloc(streams_count * sizeof(float));
    if (!post || !comb || !collapsed || !hidden || !block || !residual) goto fail;
    if (!v4_real_hc_layer(source, layer, "attn", streams, n_tokens,
                          post, comb, collapsed)) goto fail;
    char name[320];
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", layer);
    if (!v4_real_norm_batch(source, name, collapsed, n_tokens,
                            V4_REAL_D, hidden)) goto fail;
    if (!v4_real_compress_layer_prefill(source, layer, hidden, n_tokens,
                                        state, capture)) goto fail;
    if (!v4_real_attention_batch(source, layer, hidden, n_tokens,
                                 NULL, state, block)) goto fail;
    v4_real_hc_post_batch(residual, block, streams, post, comb, n_tokens);

    if (!v4_real_hc_layer(source, layer, "ffn", residual, n_tokens,
                          post, comb, collapsed)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer);
    if (!v4_real_norm_batch(source, name, collapsed, n_tokens,
                            V4_REAL_D, hidden)) goto fail;
    if (!v4_real_moe_batch(source, layer, hidden, token_ids, n_tokens,
                           capture, block)) goto fail;
    v4_real_hc_post_batch(streams, block, residual, post, comb, n_tokens);
    state->next_position += n_tokens;
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 1;
fail:
    free(post); free(comb); free(collapsed); free(hidden); free(block); free(residual);
    return 0;
}

static inline int v4_real_layers_forward(shards *source, const int *token_ids,
                                          int n_tokens, int first_layer,
                                          int layer_count, float *streams,
                                          float *layer_outputs,
                                          V4RealLayer2CSACapture *csa) {
    if (!source || !token_ids || !streams || !layer_outputs || n_tokens <= 0 ||
        first_layer < 0 || layer_count <= 0 || first_layer + layer_count > 3)
        return 0;
    size_t layer_size = (size_t)n_tokens * V4_REAL_HC * V4_REAL_D;
    for (int offset = 0; offset < layer_count; offset++) {
        int layer = first_layer + offset;
        if (!v4_real_layer_forward(source, layer, token_ids, n_tokens,
                                   streams, csa)) return 0;
        memcpy(layer_outputs + (size_t)offset * layer_size, streams,
               layer_size * sizeof(float));
    }
    return 1;
}

static inline int v4_real_layer0_forward(shards *source, int token_id,
                                         V4RealLayer0Capture *capture) {
    if (!source || !capture || token_id < 0 || !capture->input ||
        !capture->attn_post || !capture->attn_comb || !capture->attn_collapsed ||
        !capture->attn_norm || !capture->attn_output || !capture->after_attn ||
        !capture->ffn_post || !capture->ffn_comb || !capture->ffn_collapsed ||
        !capture->ffn_norm || !capture->router_scores || !capture->router_weights ||
        !capture->router_indices || !capture->moe_output || !capture->output) return 0;
    float embedding[V4_REAL_D];
    st_read_slice_f32(source, "embed.weight", (int64_t)token_id * V4_REAL_D,
                      V4_REAL_D, embedding, 0);
    for (int h = 0; h < V4_REAL_HC; h++)
        memcpy(capture->input + h * V4_REAL_D, embedding, V4_REAL_D * sizeof(float));
    if (!v4_real_hc(source, "attn", capture->input, capture->attn_post,
                    capture->attn_comb, capture->attn_collapsed)) return 0;
    if (!v4_real_norm(source, "layers.0.attn_norm.weight", capture->attn_collapsed,
                      V4_REAL_D, capture->attn_norm)) return 0;
    if (!v4_real_attention(source, capture->attn_norm, capture->attn_output)) return 0;
    v4_real_hc_post(capture->after_attn, capture->attn_output, capture->input,
                    capture->attn_post, capture->attn_comb);
    if (!v4_real_hc(source, "ffn", capture->after_attn, capture->ffn_post,
                    capture->ffn_comb, capture->ffn_collapsed)) return 0;
    if (!v4_real_norm(source, "layers.0.ffn_norm.weight", capture->ffn_collapsed,
                      V4_REAL_D, capture->ffn_norm)) return 0;
    if (!v4_real_moe(source, capture->ffn_norm, token_id, capture)) return 0;
    v4_real_hc_post(capture->output, capture->moe_output, capture->after_attn,
                    capture->ffn_post, capture->ffn_comb);
    return 1;
}

#endif
