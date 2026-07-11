#ifndef DEEPSEEK_V4_INDEXER_H
#define DEEPSEEK_V4_INDEXER_H

#include <math.h>
#include <stdint.h>

typedef struct {
    int n_heads;
    int head_dim;
    int ratio;
    int top_k;
} DeepSeekV4LightningIndexerF32;

static inline int deepseek_v4_indexer_forward_f32(const DeepSeekV4LightningIndexerF32 *model,
                                         const float *queries, const float *keys,
                                         const float *head_weights,
                                         const int64_t *positions,
                                         int n_queries, int n_entries,
                                         float *scores, int64_t *indices) {
    if (!model || !queries || !keys || !head_weights || !positions ||
        !scores || !indices || model->n_heads <= 0 || model->head_dim <= 0 ||
        model->ratio <= 0 || model->top_k <= 0 || n_queries <= 0 ||
        n_entries <= 0 || model->top_k > n_entries)
        return 0;
    for (int query = 0; query < n_queries; query++)
        if (positions[query] < 0) return 0;

    int n_heads = model->n_heads, head_dim = model->head_dim;
    float score_scale = 1.0f / sqrtf((float)head_dim);
    float weight_scale = 1.0f / sqrtf((float)n_heads);
    for (int query = 0; query < n_queries; query++) {
        for (int entry = 0; entry < n_entries; entry++) {
            float sum = 0.0f;
            const float *key = keys + (int64_t)entry * head_dim;
            for (int head = 0; head < n_heads; head++) {
                const float *q = queries
                    + ((int64_t)query * n_heads + head) * head_dim;
                float dot = 0.0f;
                for (int dimension = 0; dimension < head_dim; dimension++)
                    dot += q[dimension] * key[dimension];
                float head_score = dot > 0.0f ? dot * score_scale : 0.0f;
                float weight = head_weights[(int64_t)query * n_heads + head]
                             * weight_scale;
                sum += head_score * weight;
            }
            scores[(int64_t)query * n_entries + entry] = sum;
        }
    }

    for (int query = 0; query < n_queries; query++) {
        int64_t available = (positions[query] + 1) / model->ratio;
        if (available > n_entries) available = n_entries;
        int64_t *row = indices + (int64_t)query * model->top_k;
        for (int rank = 0; rank < model->top_k; rank++) {
            int best = -1;
            float best_score = -INFINITY;
            for (int entry = 0; entry < available; entry++) {
                int selected = 0;
                for (int prior = 0; prior < rank; prior++)
                    if (row[prior] == entry) { selected = 1; break; }
                float value = scores[(int64_t)query * n_entries + entry];
                if (!selected && value > best_score) {
                    best = entry;
                    best_score = value;
                }
            }
            row[rank] = best;
        }
    }
    return 1;
}

#endif
