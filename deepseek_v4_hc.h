#ifndef DEEPSEEK_V4_HC_H
#define DEEPSEEK_V4_HC_H

#include <math.h>
#include <stdint.h>

typedef struct {
    int streams;
    int hidden;
    int sinkhorn_iters;
    float norm_eps;
    float hc_eps;
    const float *fn;
    const float *base;
    const float *scale;
} DeepSeekV4HyperConnection;

static inline float deepseek_v4_hc_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x);
    return z / (1.0f + z);
}

static inline int deepseek_v4_hc_forward(const DeepSeekV4HyperConnection *model, const float *input,
                                int n_tokens, float *post, float *comb,
                                float *collapsed) {
    if (!model || !input || !post || !comb || !collapsed || n_tokens < 0 ||
        model->streams <= 0 || model->hidden <= 0 || model->sinkhorn_iters <= 0 ||
        !model->fn || !model->base || !model->scale) return 0;

    int streams = model->streams, hidden = model->hidden;
    int flat_size = streams * hidden;
    int mix_size = (2 + streams) * streams;
    float normalized[flat_size], mixes[mix_size], pre[streams];
    float row_sums[streams], column_sums[streams];

    for (int token = 0; token < n_tokens; token++) {
        const float *x = input + (int64_t)token * flat_size;
        float mean_square = 0.0f;
        for (int i = 0; i < flat_size; i++) mean_square += x[i] * x[i];
        mean_square /= flat_size;
        float inv_rms = 1.0f / sqrtf(mean_square + model->norm_eps);
        for (int i = 0; i < flat_size; i++) normalized[i] = x[i] * inv_rms;

        for (int row = 0; row < mix_size; row++) {
            const float *weight = model->fn + (int64_t)row * flat_size;
            float sum = 0.0f;
            for (int i = 0; i < flat_size; i++) sum += weight[i] * normalized[i];
            mixes[row] = sum;
        }

        for (int h = 0; h < streams; h++) {
            pre[h] = deepseek_v4_hc_sigmoid(mixes[h] * model->scale[0] + model->base[h]) + model->hc_eps;
            post[(int64_t)token * streams + h] = 2.0f * deepseek_v4_hc_sigmoid(
                mixes[streams + h] * model->scale[1] + model->base[streams + h]);
        }

        float *matrix = comb + (int64_t)token * streams * streams;
        for (int row = 0; row < streams; row++) {
            float maximum = -INFINITY;
            for (int column = 0; column < streams; column++) {
                int index = row * streams + column;
                float value = mixes[2 * streams + index] * model->scale[2]
                            + model->base[2 * streams + index];
                matrix[index] = value;
                if (value > maximum) maximum = value;
            }
            float sum = 0.0f;
            for (int column = 0; column < streams; column++) {
                int index = row * streams + column;
                matrix[index] = expf(matrix[index] - maximum);
                sum += matrix[index];
            }
            for (int column = 0; column < streams; column++)
                matrix[row * streams + column] = matrix[row * streams + column] / sum + model->hc_eps;
        }

        for (int column = 0; column < streams; column++) {
            float sum = 0.0f;
            for (int row = 0; row < streams; row++) sum += matrix[row * streams + column];
            column_sums[column] = sum;
        }
        for (int row = 0; row < streams; row++)
            for (int column = 0; column < streams; column++)
                matrix[row * streams + column] /= column_sums[column] + model->hc_eps;

        for (int iteration = 1; iteration < model->sinkhorn_iters; iteration++) {
            for (int row = 0; row < streams; row++) {
                float sum = 0.0f;
                for (int column = 0; column < streams; column++) sum += matrix[row * streams + column];
                row_sums[row] = sum;
            }
            for (int row = 0; row < streams; row++)
                for (int column = 0; column < streams; column++)
                    matrix[row * streams + column] /= row_sums[row] + model->hc_eps;

            for (int column = 0; column < streams; column++) {
                float sum = 0.0f;
                for (int row = 0; row < streams; row++) sum += matrix[row * streams + column];
                column_sums[column] = sum;
            }
            for (int row = 0; row < streams; row++)
                for (int column = 0; column < streams; column++)
                    matrix[row * streams + column] /= column_sums[column] + model->hc_eps;
        }

        for (int d = 0; d < hidden; d++) {
            float sum = 0.0f;
            for (int h = 0; h < streams; h++) sum += pre[h] * x[h * hidden + d];
            collapsed[(int64_t)token * hidden + d] = sum;
        }
    }
    return 1;
}

#endif
