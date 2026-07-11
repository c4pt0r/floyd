#ifndef V4_REAL_LAYER0_H
#define V4_REAL_LAYER0_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moe_route.h"
#include "st.h"
#include "v4_hc.h"
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

static inline int v4_real_hc(shards *source, const char *site, const float *input,
                             float *post, float *comb, float *collapsed) {
    char name[320];
    snprintf(name, sizeof(name), "layers.0.hc_%s_fn", site);
    float *fn = v4_real_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.0.hc_%s_base", site);
    float *base = v4_real_read_f32(source, name);
    snprintf(name, sizeof(name), "layers.0.hc_%s_scale", site);
    float *scale = v4_real_read_f32(source, name);
    if (!fn || !base || !scale) { free(fn); free(base); free(scale); return 0; }
    V4HyperConnection model = {
        .streams = V4_REAL_HC, .hidden = V4_REAL_D, .sinkhorn_iters = 20,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f, .fn = fn, .base = base, .scale = scale,
    };
    int ok = v4_hc_forward(&model, input, 1, post, comb, collapsed);
    free(fn); free(base); free(scale);
    return ok;
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
