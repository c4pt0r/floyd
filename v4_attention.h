#ifndef V4_ATTENTION_H
#define V4_ATTENTION_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "v4_kv_cache.h"

typedef struct {
    int hidden;
    int heads;
    int head_dim;
    int q_rank;
    int rope_dim;
    int o_groups;
    int o_rank;
    int window;
    float norm_eps;
    float rope_theta;
    const float *q_a;
    const float *q_a_norm;
    const float *q_b;
    const float *kv;
    const float *kv_norm;
    const float *sinks;
    const float *o_a;
    const float *o_b;
} V4SlidingAttentionF32;

static inline void v4_attn_matvec(float *output, const float *weight,
                                  const float *input, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        const float *w = weight + (int64_t)row * columns;
        float sum = 0.0f;
        for (int column = 0; column < columns; column++) sum += w[column] * input[column];
        output[row] = sum;
    }
}

static inline void v4_attn_rmsnorm(float *output, const float *input,
                                   const float *weight, int count, float eps) {
    float mean_square = 0.0f;
    for (int i = 0; i < count; i++) mean_square += input[i] * input[i];
    mean_square /= count;
    float scale = 1.0f / sqrtf(mean_square + eps);
    for (int i = 0; i < count; i++)
        output[i] = input[i] * scale * (weight ? weight[i] : 1.0f);
}

static inline void v4_attn_rope(float *vector, int head_dim, int rope_dim,
                                int64_t position, float theta, int inverse) {
    int start = head_dim - rope_dim;
    for (int pair = 0; pair < rope_dim / 2; pair++) {
        float frequency = powf(theta, -(2.0f * pair) / rope_dim);
        float angle = position * frequency;
        float cosine = cosf(angle), sine = sinf(angle);
        if (inverse) sine = -sine;
        int i = start + pair * 2;
        float first = vector[i], second = vector[i + 1];
        vector[i] = first * cosine - second * sine;
        vector[i + 1] = second * cosine + first * sine;
    }
}

static inline int v4_sliding_attention_cached_f32(
        const V4SlidingAttentionF32 *model, V4KVCacheF32 *cache,
        const float *input, int n_tokens, int64_t start_position, float *output) {
    if (!model || !cache || !input || !output || n_tokens < 0 || start_position < 0 ||
        model->hidden <= 0 || model->heads <= 0 || model->head_dim <= 0 ||
        model->q_rank <= 0 || model->rope_dim <= 0 || (model->rope_dim & 1) ||
        model->rope_dim > model->head_dim || model->o_groups <= 0 ||
        model->heads % model->o_groups || model->o_rank <= 0 || model->window <= 0 ||
        !model->q_a || !model->q_a_norm || !model->q_b || !model->kv ||
        !model->kv_norm || !model->sinks || !model->o_a || !model->o_b ||
        cache->mode != V4_KV_CACHE_RING || cache->capacity != model->window ||
        cache->head_dim != model->head_dim)
        return 0;
    if (cache->count > 0 &&
        start_position != v4_kv_cache_position(cache, cache->count - 1) + 1)
        return 0;
    if (n_tokens == 0) return 1;

    int hidden = model->hidden, heads = model->heads, head_dim = model->head_dim;
    int q_width = heads * head_dim;
    int group_input = q_width / model->o_groups;
    int grouped_width = model->o_groups * model->o_rank;
    float *query = malloc((size_t)q_width * sizeof(float));
    float *q_a = malloc((size_t)model->q_rank * sizeof(float));
    float *q_norm = malloc((size_t)model->q_rank * sizeof(float));
    float *kv_raw = malloc((size_t)head_dim * sizeof(float));
    float *key = malloc((size_t)head_dim * sizeof(float));
    float *context = malloc((size_t)q_width * sizeof(float));
    float *grouped = malloc((size_t)grouped_width * sizeof(float));
    float *scores = malloc((size_t)model->window * sizeof(float));
    if (!query || !q_a || !q_norm || !kv_raw || !key || !context || !grouped || !scores) {
        free(query); free(q_a); free(q_norm); free(kv_raw); free(key);
        free(context); free(grouped); free(scores);
        return 0;
    }

    float attention_scale = 1.0f / sqrtf((float)head_dim);
    for (int token = 0; token < n_tokens; token++) {
        int64_t position = start_position + token;
        const float *x = input + (int64_t)token * hidden;
        v4_attn_matvec(q_a, model->q_a, x, model->q_rank, hidden);
        v4_attn_rmsnorm(q_norm, q_a, model->q_a_norm, model->q_rank, model->norm_eps);
        v4_attn_matvec(query, model->q_b, q_norm, q_width, model->q_rank);
        for (int head = 0; head < heads; head++) {
            float *row = query + head * head_dim;
            v4_attn_rmsnorm(row, row, NULL, head_dim, model->norm_eps);
            v4_attn_rope(row, head_dim, model->rope_dim, position, model->rope_theta, 0);
        }
        v4_attn_matvec(kv_raw, model->kv, x, head_dim, hidden);
        v4_attn_rmsnorm(key, kv_raw, model->kv_norm, head_dim, model->norm_eps);
        v4_attn_rope(key, head_dim, model->rope_dim, position, model->rope_theta, 0);
        if (!v4_kv_cache_append(cache, key, position)) {
            free(query); free(q_a); free(q_norm); free(kv_raw); free(key);
            free(context); free(grouped); free(scores);
            return 0;
        }

        int key_count = v4_kv_cache_count(cache);
        for (int head = 0; head < heads; head++) {
            const float *head_query = query + head * head_dim;
            float maximum = model->sinks[head];
            for (int index = 0; index < key_count; index++) {
                const float *cached_key = v4_kv_cache_key(cache, index);
                float dot = 0.0f;
                for (int dimension = 0; dimension < head_dim; dimension++)
                    dot += head_query[dimension] * cached_key[dimension];
                scores[index] = dot * attention_scale;
                if (scores[index] > maximum) maximum = scores[index];
            }
            float denominator = expf(model->sinks[head] - maximum);
            for (int index = 0; index < key_count; index++) {
                scores[index] = expf(scores[index] - maximum);
                denominator += scores[index];
            }
            float *head_output = context + head * head_dim;
            for (int dimension = 0; dimension < head_dim; dimension++) {
                float sum = 0.0f;
                for (int index = 0; index < key_count; index++)
                    sum += scores[index] * v4_kv_cache_key(cache, index)[dimension];
                head_output[dimension] = sum / denominator;
            }
            v4_attn_rope(head_output, head_dim, model->rope_dim,
                         position, model->rope_theta, 1);
        }

        for (int group = 0; group < model->o_groups; group++) {
            const float *group_x = context + group * group_input;
            for (int rank = 0; rank < model->o_rank; rank++) {
                int row = group * model->o_rank + rank;
                const float *weight = model->o_a + (int64_t)row * group_input;
                float sum = 0.0f;
                for (int i = 0; i < group_input; i++) sum += weight[i] * group_x[i];
                grouped[row] = sum;
            }
        }
        v4_attn_matvec(output + (int64_t)token * hidden, model->o_b,
                       grouped, hidden, grouped_width);
    }

    free(query); free(q_a); free(q_norm); free(kv_raw); free(key);
    free(context); free(grouped); free(scores);
    return 1;
}

static inline int v4_sliding_attention_forward(const V4SlidingAttentionF32 *model,
                                                const float *input, int n_tokens,
                                                float *output) {
    if (!model || !input || !output || n_tokens < 0 || model->hidden <= 0 ||
        model->heads <= 0 || model->head_dim <= 0 || model->q_rank <= 0 ||
        model->rope_dim <= 0 || (model->rope_dim & 1) || model->rope_dim > model->head_dim ||
        model->o_groups <= 0 || model->heads % model->o_groups || model->o_rank <= 0 ||
        model->window <= 0 || !model->q_a || !model->q_a_norm || !model->q_b ||
        !model->kv || !model->kv_norm || !model->sinks || !model->o_a || !model->o_b)
        return 0;
    if (n_tokens == 0) return 1;

    int hidden = model->hidden, heads = model->heads, head_dim = model->head_dim;
    int q_width = heads * head_dim;
    int group_input = q_width / model->o_groups;
    int grouped_width = model->o_groups * model->o_rank;
    float *queries = malloc((size_t)n_tokens * q_width * sizeof(float));
    float *keys = malloc((size_t)n_tokens * head_dim * sizeof(float));
    float *q_a = malloc((size_t)model->q_rank * sizeof(float));
    float *q_norm = malloc((size_t)model->q_rank * sizeof(float));
    float *kv_raw = malloc((size_t)head_dim * sizeof(float));
    float *context = malloc((size_t)q_width * sizeof(float));
    float *grouped = malloc((size_t)grouped_width * sizeof(float));
    float *scores = malloc((size_t)model->window * sizeof(float));
    if (!queries || !keys || !q_a || !q_norm || !kv_raw || !context || !grouped || !scores) {
        free(queries); free(keys); free(q_a); free(q_norm); free(kv_raw);
        free(context); free(grouped); free(scores);
        return 0;
    }

    for (int token = 0; token < n_tokens; token++) {
        const float *x = input + (int64_t)token * hidden;
        float *query = queries + (int64_t)token * q_width;
        float *key = keys + (int64_t)token * head_dim;
        v4_attn_matvec(q_a, model->q_a, x, model->q_rank, hidden);
        v4_attn_rmsnorm(q_norm, q_a, model->q_a_norm, model->q_rank, model->norm_eps);
        v4_attn_matvec(query, model->q_b, q_norm, q_width, model->q_rank);
        for (int head = 0; head < heads; head++) {
            float *row = query + head * head_dim;
            v4_attn_rmsnorm(row, row, NULL, head_dim, model->norm_eps);
            v4_attn_rope(row, head_dim, model->rope_dim, token, model->rope_theta, 0);
        }
        v4_attn_matvec(kv_raw, model->kv, x, head_dim, hidden);
        v4_attn_rmsnorm(key, kv_raw, model->kv_norm, head_dim, model->norm_eps);
        v4_attn_rope(key, head_dim, model->rope_dim, token, model->rope_theta, 0);
    }

    float attention_scale = 1.0f / sqrtf((float)head_dim);
    for (int token = 0; token < n_tokens; token++) {
        int first_key = token - model->window + 1;
        if (first_key < 0) first_key = 0;
        int key_count = token - first_key + 1;
        for (int head = 0; head < heads; head++) {
            const float *query = queries + ((int64_t)token * heads + head) * head_dim;
            float maximum = model->sinks[head];
            for (int k = 0; k < key_count; k++) {
                const float *key = keys + (int64_t)(first_key + k) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += query[d] * key[d];
                scores[k] = dot * attention_scale;
                if (scores[k] > maximum) maximum = scores[k];
            }
            float denominator = expf(model->sinks[head] - maximum);
            for (int k = 0; k < key_count; k++) {
                scores[k] = expf(scores[k] - maximum);
                denominator += scores[k];
            }
            float *head_output = context + head * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int k = 0; k < key_count; k++)
                    sum += scores[k] * keys[(int64_t)(first_key + k) * head_dim + d];
                head_output[d] = sum / denominator;
            }
            v4_attn_rope(head_output, head_dim, model->rope_dim,
                         token, model->rope_theta, 1);
        }

        for (int group = 0; group < model->o_groups; group++) {
            const float *group_x = context + group * group_input;
            for (int rank = 0; rank < model->o_rank; rank++) {
                int row = group * model->o_rank + rank;
                const float *weight = model->o_a + (int64_t)row * group_input;
                float sum = 0.0f;
                for (int i = 0; i < group_input; i++) sum += weight[i] * group_x[i];
                grouped[row] = sum;
            }
        }
        v4_attn_matvec(output + (int64_t)token * hidden, model->o_b,
                       grouped, hidden, grouped_width);
    }

    free(queries); free(keys); free(q_a); free(q_norm); free(kv_raw);
    free(context); free(grouped); free(scores);
    return 1;
}

#endif
