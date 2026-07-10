#ifndef MOE_EXEC_H
#define MOE_EXEC_H

#include <math.h>
#include <stdint.h>

#include "moe_route.h"

typedef struct {
    const float *w1;
    const float *w2;
    const float *w3;
} MoeExpertF32;

typedef struct {
    int hidden;
    int intermediate;
    int n_experts;
    int top_k;
    int vocab_size;
    MoeScoreFn score_fn;
    int normalize;
    float route_scale;
    float swiglu_limit;
    const float *router;
    const float *selection_bias;
    const int64_t *fixed_experts;
    const MoeExpertF32 *experts;
    const MoeExpertF32 *shared_experts;
    int n_shared_experts;
} MoeF32;

static inline void moe_f32_matvec(float *output, const float *weight,
                                  const float *input, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        const float *w = weight + (int64_t)row * columns;
        for (int column = 0; column < columns; column++) sum += w[column] * input[column];
        output[row] = sum;
    }
}

static inline float moe_f32_silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline void moe_f32_add_expert(const MoeF32 *model, const MoeExpertF32 *expert,
                                      const float *input, float coefficient, float *output) {
    int hidden = model->hidden, intermediate = model->intermediate;
    float gate[intermediate], up[intermediate], activated[intermediate], projected[hidden];
    moe_f32_matvec(gate, expert->w1, input, intermediate, hidden);
    moe_f32_matvec(up, expert->w3, input, intermediate, hidden);
    for (int i = 0; i < intermediate; i++) {
        if (model->swiglu_limit > 0.0f) {
            if (gate[i] > model->swiglu_limit) gate[i] = model->swiglu_limit;
            if (up[i] > model->swiglu_limit) up[i] = model->swiglu_limit;
            if (up[i] < -model->swiglu_limit) up[i] = -model->swiglu_limit;
        }
        activated[i] = moe_f32_silu(gate[i]) * up[i];
    }
    moe_f32_matvec(projected, expert->w2, activated, hidden, intermediate);
    for (int d = 0; d < hidden; d++) output[d] += coefficient * projected[d];
}

static inline int moe_f32_forward(const MoeF32 *model, const float *input,
                                  const int *token_ids, int n_tokens, float *output) {
    if (!model || !input || !output || n_tokens < 0 || model->hidden <= 0 ||
        model->intermediate <= 0 || model->n_experts <= 0 || model->top_k <= 0 ||
        model->top_k > model->n_experts || !model->router || !model->experts) return 0;

    int hidden = model->hidden, experts = model->n_experts, top_k = model->top_k;
    float logits[experts], scores[experts], choices[experts], weights[top_k];
    int indices[top_k], order[top_k];

    for (int token = 0; token < n_tokens; token++) {
        const float *x = input + (int64_t)token * hidden;
        float *y = output + (int64_t)token * hidden;
        for (int d = 0; d < hidden; d++) y[d] = 0.0f;
        moe_f32_matvec(logits, model->router, x, experts, hidden);

        int selected;
        if (model->fixed_experts) {
            if (!token_ids || token_ids[token] < 0 || token_ids[token] >= model->vocab_size) return 0;
            const int64_t *fixed = model->fixed_experts + (int64_t)token_ids[token] * top_k;
            for (int k = 0; k < top_k; k++) indices[k] = (int)fixed[k];
            selected = moe_route_fixed(logits, experts, indices, top_k,
                                       model->score_fn, weights, scores);
        } else {
            selected = moe_route_select(logits, model->selection_bias, experts, top_k,
                                        model->score_fn, indices, weights, scores, choices);
        }
        if (selected != top_k) return 0;
        moe_route_finalize(weights, selected, model->normalize, model->route_scale);

        for (int k = 0; k < selected; k++) order[k] = k;
        for (int a = 1; a < selected; a++) {
            int value = order[a], b = a - 1;
            while (b >= 0 && indices[order[b]] > indices[value]) {
                order[b + 1] = order[b];
                b--;
            }
            order[b + 1] = value;
        }
        for (int k = 0; k < selected; k++) {
            int route = order[k];
            moe_f32_add_expert(model, &model->experts[indices[route]], x, weights[route], y);
        }
        for (int shared = 0; shared < model->n_shared_experts; shared++)
            moe_f32_add_expert(model, &model->shared_experts[shared], x, 1.0f, y);
    }
    return 1;
}

#endif
