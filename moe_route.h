#ifndef MOE_ROUTE_H
#define MOE_ROUTE_H

#include <math.h>

typedef enum {
    MOE_SCORE_SOFTMAX,
    MOE_SCORE_SIGMOID,
    MOE_SCORE_SQRT_SOFTPLUS
} MoeScoreFn;

static inline float moe_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float moe_sqrt_softplus(float x) {
    float softplus;
    if (x > 20.0f) softplus = x;
    else if (x < -20.0f) softplus = expf(x);
    else softplus = log1pf(expf(x));
    return sqrtf(softplus);
}

static inline void moe_route_scores(const float *logits, int n_experts,
                                    MoeScoreFn score_fn, float *scores) {
    if (score_fn == MOE_SCORE_SOFTMAX) {
        float max_logit = logits[0];
        for (int e = 1; e < n_experts; e++)
            if (logits[e] > max_logit) max_logit = logits[e];
        float sum = 0.0f;
        for (int e = 0; e < n_experts; e++) {
            scores[e] = expf(logits[e] - max_logit);
            sum += scores[e];
        }
        for (int e = 0; e < n_experts; e++) scores[e] /= sum;
        return;
    }

    for (int e = 0; e < n_experts; e++) {
        scores[e] = score_fn == MOE_SCORE_SQRT_SOFTPLUS
            ? moe_sqrt_softplus(logits[e]) : moe_sigmoid(logits[e]);
    }
}

/* Bias changes expert selection only. Returned weights always come from the
 * unbiased affinity scores and are finalized after caller-specific pruning. */
static inline int moe_route_select(const float *logits, const float *selection_bias,
                                   int n_experts, int top_k, MoeScoreFn score_fn,
                                   int *indices, float *weights,
                                   float *scores, float *selection_scores) {
    if (n_experts <= 0 || top_k <= 0) return 0;
    if (top_k > n_experts) top_k = n_experts;

    moe_route_scores(logits, n_experts, score_fn, scores);
    for (int e = 0; e < n_experts; e++)
        selection_scores[e] = scores[e] + (selection_bias ? selection_bias[e] : 0.0f);

    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_value = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            int selected = 0;
            for (int j = 0; j < k; j++)
                if (indices[j] == e) { selected = 1; break; }
            if (!selected && selection_scores[e] > best_value) {
                best = e;
                best_value = selection_scores[e];
            }
        }
        indices[k] = best;
        weights[k] = scores[best];
    }
    return top_k;
}

static inline void moe_route_finalize(float *weights, int count,
                                      int normalize, float routed_scale) {
    if (count <= 0) return;
    if (normalize) {
        float sum = 0.0f;
        for (int k = 0; k < count; k++) sum += weights[k];
        sum += 1e-20f;
        for (int k = 0; k < count; k++) weights[k] /= sum;
    }
    for (int k = 0; k < count; k++) weights[k] *= routed_scale;
}

#endif
