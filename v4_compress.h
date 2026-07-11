#ifndef V4_COMPRESS_H
#define V4_COMPRESS_H

#include <math.h>
#include <stdint.h>

typedef struct {
    int ratio;
    int head_dim;
    int rope_dim;
    int overlap;
    float norm_eps;
    float rope_theta;
    const float *position_bias;
    const float *norm_weight;
} V4PrefillCompressorF32;

static inline void v4_compress_rope(float *vector, int head_dim, int rope_dim,
                                    int64_t position, float theta) {
    int start = head_dim - rope_dim;
    for (int pair = 0; pair < rope_dim / 2; pair++) {
        float frequency = powf(theta, -(2.0f * pair) / rope_dim);
        float angle = (float)position * frequency;
        float cosine = cosf(angle), sine = sinf(angle);
        int index = start + pair * 2;
        float first = vector[index], second = vector[index + 1];
        vector[index] = first * cosine - second * sine;
        vector[index + 1] = second * cosine + first * sine;
    }
}

static inline float v4_compress_gate(const V4PrefillCompressorF32 *model,
                                     const float *gate, int window, int slot,
                                     int dimension) {
    int ratio = model->ratio, head_dim = model->head_dim;
    int input_width = model->overlap ? 2 * head_dim : head_dim;
    int source_window = window;
    int source_slot = slot;
    int source_dimension = dimension;

    if (model->overlap) {
        if (slot < ratio) {
            if (window == 0) return -INFINITY;
            source_window = window - 1;
        } else {
            source_slot -= ratio;
            source_dimension += head_dim;
        }
    }

    int64_t token = (int64_t)source_window * ratio + source_slot;
    return gate[token * input_width + source_dimension]
         + model->position_bias[source_slot * input_width + source_dimension];
}

static inline float v4_compress_value(const V4PrefillCompressorF32 *model,
                                      const float *kv, int window, int slot,
                                      int dimension) {
    int ratio = model->ratio, head_dim = model->head_dim;
    int input_width = model->overlap ? 2 * head_dim : head_dim;
    int source_window = window;
    int source_slot = slot;
    int source_dimension = dimension;

    if (model->overlap) {
        if (slot < ratio) source_window = window - 1;
        else {
            source_slot -= ratio;
            source_dimension += head_dim;
        }
    }
    if (source_window < 0) return 0.0f;

    int64_t token = (int64_t)source_window * ratio + source_slot;
    return kv[token * input_width + source_dimension];
}

static inline int v4_compress_prefill_f32(const V4PrefillCompressorF32 *model,
                                          const float *kv, const float *gate,
                                          int n_tokens, float *output) {
    if (!model || !kv || !gate || !output || !model->position_bias ||
        !model->norm_weight || model->ratio <= 0 || model->head_dim <= 0 ||
        model->rope_dim <= 0 || (model->rope_dim & 1) ||
        model->rope_dim > model->head_dim ||
        (model->overlap != 0 && model->overlap != 1) ||
        model->norm_eps < 0.0f || model->rope_theta <= 0.0f ||
        n_tokens <= 0 || n_tokens % model->ratio != 0)
        return 0;

    int ratio = model->ratio, head_dim = model->head_dim;
    int slots = model->overlap ? 2 * ratio : ratio;
    int n_windows = n_tokens / ratio;

    for (int window = 0; window < n_windows; window++) {
        float *row = output + (int64_t)window * head_dim;
        for (int dimension = 0; dimension < head_dim; dimension++) {
            float maximum = -INFINITY;
            for (int slot = 0; slot < slots; slot++) {
                float score = v4_compress_gate(model, gate, window, slot, dimension);
                if (score > maximum) maximum = score;
            }

            float denominator = 0.0f, weighted = 0.0f;
            for (int slot = 0; slot < slots; slot++) {
                float score = v4_compress_gate(model, gate, window, slot, dimension);
                float weight = expf(score - maximum);
                denominator += weight;
                weighted += weight * v4_compress_value(model, kv, window, slot, dimension);
            }
            row[dimension] = weighted / denominator;
        }

        float mean_square = 0.0f;
        for (int dimension = 0; dimension < head_dim; dimension++)
            mean_square += row[dimension] * row[dimension];
        mean_square /= head_dim;
        float scale = 1.0f / sqrtf(mean_square + model->norm_eps);
        for (int dimension = 0; dimension < head_dim; dimension++)
            row[dimension] *= scale * model->norm_weight[dimension];

        v4_compress_rope(row, head_dim, model->rope_dim,
                         (int64_t)window * ratio, model->rope_theta);
    }
    return 1;
}

#endif
