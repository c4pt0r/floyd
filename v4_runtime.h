#ifndef V4_RUNTIME_H
#define V4_RUNTIME_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "st.h"
#include "v4_real_layer0.h"

typedef struct {
    shards model;
    V4RealModelState state;
    V4RealDSparkState dspark_state;
    int max_context;
    int last_count;
    int last_start_position;
    int dspark_ready;
    float *streams;
    float *head_hidden;
    float *head_normalized;
    float *logits;
    float *target_means;
    float *main_input;
    float *main_x;
    float *dspark_prefill_kv;
    float *dspark_stage_outputs;
    float *dspark_hidden;
    float dspark_confidence[5];
    int64_t dspark_output_ids[6];
    size_t token_capacity;
} V4Runtime;

static inline int v4_runtime_resize(V4Runtime *runtime, int n_tokens) {
    if ((size_t)n_tokens <= runtime->token_capacity) return 1;
    size_t capacity = runtime->token_capacity ? runtime->token_capacity : 1;
    while (capacity < (size_t)n_tokens) capacity *= 2;
    float *streams = realloc(
        runtime->streams,
        capacity * V4_REAL_HC * V4_REAL_D * sizeof(float));
    if (!streams) return 0;
    runtime->streams = streams;
    float *hidden = realloc(runtime->head_hidden,
                            capacity * V4_REAL_D * sizeof(float));
    if (!hidden) return 0;
    runtime->head_hidden = hidden;
    float *normalized = realloc(runtime->head_normalized,
                                capacity * V4_REAL_D * sizeof(float));
    if (!normalized) return 0;
    runtime->head_normalized = normalized;
    float *targets = realloc(runtime->target_means,
                             capacity * 3 * V4_REAL_D * sizeof(float));
    if (!targets) return 0;
    runtime->target_means = targets;
    float *main_input = realloc(runtime->main_input,
                                capacity * 3 * V4_REAL_D * sizeof(float));
    if (!main_input) return 0;
    runtime->main_input = main_input;
    float *main_x = realloc(runtime->main_x,
                            capacity * V4_REAL_D * sizeof(float));
    if (!main_x) return 0;
    runtime->main_x = main_x;
    float *prefill_kv = realloc(runtime->dspark_prefill_kv,
                                capacity * 3 * V4_REAL_HD * sizeof(float));
    if (!prefill_kv) return 0;
    runtime->dspark_prefill_kv = prefill_kv;
    runtime->token_capacity = capacity;
    return 1;
}

static inline int v4_runtime_init(V4Runtime *runtime, const char *model_dir,
                                  int max_context) {
    if (!runtime || !model_dir || max_context <= 0) return 0;
    memset(runtime, 0, sizeof(*runtime));
    runtime->max_context = max_context;
    st_init(&runtime->model, model_dir);
    if (!v4_real_model_state_init(&runtime->state, max_context) ||
        !v4_real_dspark_state_init(&runtime->dspark_state)) return 0;
    runtime->logits = malloc((size_t)129280 * sizeof(float));
    runtime->dspark_stage_outputs = malloc(
        (size_t)3 * 5 * V4_REAL_HC * V4_REAL_D * sizeof(float));
    runtime->dspark_hidden = malloc((size_t)5 * V4_REAL_D * sizeof(float));
    if (!runtime->logits || !runtime->dspark_stage_outputs ||
        !runtime->dspark_hidden || !v4_runtime_resize(runtime, 1)) {
        v4_real_model_state_free(&runtime->state);
        v4_real_dspark_state_free(&runtime->dspark_state);
        free(runtime->streams); free(runtime->head_hidden);
        free(runtime->head_normalized); free(runtime->logits);
        free(runtime->target_means); free(runtime->main_input);
        free(runtime->main_x); free(runtime->dspark_prefill_kv);
        free(runtime->dspark_stage_outputs); free(runtime->dspark_hidden);
        memset(runtime, 0, sizeof(*runtime));
        return 0;
    }
    return 1;
}

static inline void v4_runtime_reset(V4Runtime *runtime) {
    if (!runtime) return;
    for (int layer = 0; layer < 43; layer++) {
        V4RealLayerState *state = &runtime->state.layers[layer];
        state->next_position = 0;
        v4_kv_cache_reset(&state->window);
        v4_kv_cache_reset(&state->compressed);
        v4_kv_cache_reset(&state->index);
        if (state->ratio > 0) {
            int overlap = state->ratio == 4;
            int rows = (1 + overlap) * state->ratio;
            int width = (1 + overlap) * V4_REAL_HD;
            memset(state->compressor_kv_state, 0,
                   (size_t)rows * width * sizeof(float));
            for (int64_t i = 0; i < (int64_t)rows * width; i++)
                state->compressor_gate_state[i] = -INFINITY;
        }
        if (state->ratio == 4) {
            memset(state->index_kv_state, 0, 8 * 256 * sizeof(float));
            for (int i = 0; i < 8 * 256; i++)
                state->index_gate_state[i] = -INFINITY;
        }
    }
    memset(runtime->dspark_state.main_kv, 0,
           (size_t)3 * 128 * V4_REAL_HD * sizeof(float));
    runtime->dspark_state.next_position = 0;
    runtime->dspark_ready = 0;
    runtime->last_start_position = 0;
    runtime->last_count = 0;
}

static inline int v4_runtime_project_main(V4Runtime *runtime, int n_tokens) {
    if (!runtime || n_tokens <= 0 || n_tokens > runtime->last_count) return 0;
    for (int token = 0; token < n_tokens; token++)
        for (int layer = 0; layer < 3; layer++)
            memcpy(runtime->main_input
                       + ((int64_t)token * 3 + layer) * V4_REAL_D,
                   runtime->target_means
                       + ((int64_t)layer * n_tokens + token) * V4_REAL_D,
                   V4_REAL_D * sizeof(float));
    if (!v4_real_fp8_batch(&runtime->model, "mtp.0.main_proj",
                           runtime->main_input, n_tokens,
                           V4_REAL_D, 3 * V4_REAL_D, runtime->main_x))
        return 0;
    return v4_real_norm_batch(&runtime->model, "mtp.0.main_norm.weight",
                              runtime->main_x, n_tokens,
                              V4_REAL_D, runtime->main_x);
}

static inline int v4_runtime_forward(V4Runtime *runtime, const int *token_ids,
                                     int n_tokens, const float **logits) {
    if (!runtime || !token_ids || n_tokens <= 0 ||
        runtime->state.layers[0].next_position + n_tokens
            > runtime->max_context ||
        !v4_runtime_resize(runtime, n_tokens)) return 0;
    int start_position = runtime->state.layers[0].next_position;
    V4RealBaseCapture base_capture = {
        .target_means = runtime->target_means,
    };
    if (!v4_real_base_layers_forward(
            &runtime->model, token_ids, n_tokens, &runtime->state,
            runtime->streams, &base_capture)) return 0;
    V4RealHeadCapture head_capture = {
        .hidden = runtime->head_hidden,
        .normalized = runtime->head_normalized,
        .logits = runtime->logits,
    };
    if (!v4_real_base_head_forward(&runtime->model, runtime->streams,
                                   n_tokens, &head_capture)) return 0;
    runtime->last_start_position = start_position;
    runtime->last_count = n_tokens;
    if (runtime->dspark_ready) {
        if (!v4_runtime_project_main(runtime, n_tokens) ||
            !v4_real_dspark_append_main(
                &runtime->model, runtime->main_x, n_tokens, start_position,
                &runtime->dspark_state, NULL)) return 0;
    }
    if (logits) *logits = runtime->logits;
    return 1;
}

static inline int v4_runtime_dspark_prefill(V4Runtime *runtime) {
    if (!runtime || runtime->dspark_ready || runtime->last_count <= 0 ||
        runtime->last_start_position != 0 ||
        !v4_runtime_project_main(runtime, runtime->last_count)) return 0;
    V4RealDSparkCapture capture = {
        .prefill_kv = runtime->dspark_prefill_kv,
        .stage_outputs = runtime->dspark_stage_outputs,
        .hidden = runtime->dspark_hidden,
        .output_ids = runtime->dspark_output_ids,
        .confidence = runtime->dspark_confidence,
    };
    if (!v4_real_dspark_prefill(
            &runtime->model, runtime->main_x, runtime->last_count,
            &runtime->dspark_state, &capture)) return 0;
    runtime->dspark_ready = 1;
    return 1;
}

static inline int v4_runtime_dspark_propose(
        V4Runtime *runtime, int input_id, int64_t output_ids[6],
        float confidence[5]) {
    if (!runtime || !runtime->dspark_ready || runtime->last_count <= 0 ||
        input_id < 0) return 0;
    int current_position = runtime->last_start_position
                         + runtime->last_count - 1;
    const float *current_main = runtime->main_x
                              + (int64_t)(runtime->last_count - 1) * V4_REAL_D;
    V4RealDSparkCapture capture = {
        .prefill_kv = runtime->dspark_prefill_kv,
        .stage_outputs = runtime->dspark_stage_outputs,
        .hidden = runtime->dspark_hidden,
        .output_ids = runtime->dspark_output_ids,
        .confidence = runtime->dspark_confidence,
    };
    if (!v4_real_dspark_propose(
            &runtime->model, current_main, current_position,
            input_id, &runtime->dspark_state, &capture)) return 0;
    if (output_ids)
        memcpy(output_ids, runtime->dspark_output_ids,
               6 * sizeof(int64_t));
    if (confidence)
        memcpy(confidence, runtime->dspark_confidence, 5 * sizeof(float));
    return 1;
}

static inline const float *v4_runtime_last_streams(const V4Runtime *runtime) {
    if (!runtime || runtime->last_count <= 0) return NULL;
    return runtime->streams
         + (int64_t)(runtime->last_count - 1) * V4_REAL_HC * V4_REAL_D;
}

static inline int v4_runtime_argmax(const float *logits) {
    if (!logits) return -1;
    int best = 0;
    for (int token = 1; token < 129280; token++)
        if (logits[token] > logits[best]) best = token;
    return best;
}

static inline int v4_runtime_verify_token(const float *logits,
                                          int64_t proposal, int *accepted) {
    int base = v4_runtime_argmax(logits);
    if (accepted) *accepted = base >= 0 && proposal == base;
    return base;
}

static inline void v4_runtime_free(V4Runtime *runtime) {
    if (!runtime) return;
    v4_real_model_state_free(&runtime->state);
    v4_real_dspark_state_free(&runtime->dspark_state);
    free(runtime->streams); free(runtime->head_hidden);
    free(runtime->head_normalized); free(runtime->logits);
    free(runtime->target_means); free(runtime->main_input);
    free(runtime->main_x); free(runtime->dspark_prefill_kv);
    free(runtime->dspark_stage_outputs); free(runtime->dspark_hidden);
    memset(runtime, 0, sizeof(*runtime));
}

#endif
