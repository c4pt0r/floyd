#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../moonlight_metal.h"
#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float *read_f32(shards *source, const char *name, int64_t *count) {
    *count = st_numel(source, name);
    if (*count <= 0) return NULL;
    float *values = malloc((size_t)*count * sizeof(*values));
    st_read_f32(source, name, values, 0);
    return values;
}

static float max_abs(const float *actual, const float *expected, int64_t count) {
    float result = 0;
    for (int64_t index = 0; index < count; ++index) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > result) result = difference;
    }
    return result;
}

static float *quantized_matmul_reference(shards *model, const char *name,
                                         const float *input, int rows,
                                         int input_width, int output_width) {
    char scale_name[512];
    int64_t weight_bytes;
    uint8_t *weight;
    float *scale;
    float *output;
    snprintf(scale_name, sizeof(scale_name), "%s.qs", name);
    if (!st_has(model, scale_name)) return NULL;
    weight_bytes = st_nbytes(model, name);
    weight = malloc((size_t)weight_bytes);
    scale = malloc((size_t)output_width * sizeof(*scale));
    output = calloc((size_t)rows * output_width, sizeof(*output));
    st_read_raw(model, name, weight, 0);
    st_read_f32(model, scale_name, scale, 0);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < output_width; ++column) {
            const float *x = input + (int64_t)row * input_width;
            float sum = 0;
            if (weight_bytes == (int64_t)output_width * input_width) {
                const int8_t *values = (const int8_t *)weight +
                                       (int64_t)column * input_width;
                for (int inner = 0; inner < input_width; ++inner)
                    sum += x[inner] * values[inner];
            } else {
                int row_bytes = (input_width + 1) / 2;
                const uint8_t *values = weight + (int64_t)column * row_bytes;
                for (int inner = 0; inner < input_width; ++inner) {
                    uint8_t packed = values[inner / 2];
                    int value = (inner & 1) ? (packed >> 4) - 8
                                            : (packed & 15) - 8;
                    sum += x[inner] * value;
                }
            }
            output[(int64_t)row * output_width + column] = sum * scale[column];
        }
    }
    free(weight);
    free(scale);
    return output;
}

int main(int argc, char **argv) {
    MoonlightModel *model = NULL;
    MoonlightSession *session = NULL;
    MoonlightOptions options = {.context_size = 64, .max_batch = 32};
    MoonlightStats stats;
    shards oracle;
    shards checkpoint;
    char error[512] = {0};
    int64_t id_count;
    int64_t embed_count;
    int64_t final_input_count;
    int64_t final_norm_count;
    int64_t input_norm_count;
    int64_t q_count;
    int64_t logits_count;
    int64_t raw_id_count;
    int64_t *raw_ids;
    int *ids;
    float *expected_embed;
    float *expected_final_input;
    float *expected_final_norm;
    float *expected_input_norm;
    float *expected_q;
    float *expected_logits;
    float *actual_embed;
    float *actual_final_norm;
    float *actual_q;
    float *actual_logits;
    float *quantized_q;
    float embed_error;
    float norm_error;
    float q_error;
    float logits_error;

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL ORACLE_DIR\n", argv[0]);
        return 2;
    }
    st_init(&oracle, argv[2]);
    st_init(&checkpoint, argv[1]);
    raw_id_count = st_numel(&oracle, "input_ids");
    CHECK(raw_id_count > 0 && raw_id_count <= 32);
    id_count = raw_id_count;
    raw_ids = malloc((size_t)id_count * sizeof(*raw_ids));
    ids = malloc((size_t)id_count * sizeof(*ids));
    st_read_raw(&oracle, "input_ids", raw_ids, 0);
    for (int64_t index = 0; index < id_count; ++index) ids[index] = (int)raw_ids[index];
    expected_embed = read_f32(&oracle, "embed", &embed_count);
    expected_final_input = read_f32(&oracle, "final_input", &final_input_count);
    expected_final_norm = read_f32(&oracle, "final_norm", &final_norm_count);
    expected_input_norm = read_f32(&oracle, "layer.0.input_norm", &input_norm_count);
    expected_q = read_f32(&oracle, "layer.0.q", &q_count);
    expected_logits = read_f32(&oracle, "logits", &logits_count);
    CHECK(expected_embed && expected_final_input && expected_final_norm &&
          expected_input_norm && expected_q && expected_logits);
    CHECK(embed_count == id_count * 256);
    CHECK(final_input_count == id_count * 256);
    CHECK(final_norm_count == id_count * 256);
    CHECK(input_norm_count == id_count * 256);
    CHECK(q_count == id_count * 192);
    CHECK(logits_count == id_count * 512);

    actual_embed = calloc((size_t)embed_count, sizeof(*actual_embed));
    actual_final_norm = calloc((size_t)final_norm_count, sizeof(*actual_final_norm));
    actual_q = calloc((size_t)q_count, sizeof(*actual_q));
    actual_logits = calloc((size_t)logits_count, sizeof(*actual_logits));
    if (!moonlight_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "model open failed: %s\n", error);
        return 1;
    }
    if (!moonlight_session_create(&session, model, &options,
                                  error, sizeof(error))) {
        fprintf(stderr, "session create failed: %s\n", error);
        return 1;
    }

    CHECK(moonlight_test_embed(session, ids, (int)id_count, actual_embed) == 1);
    CHECK(moonlight_test_rmsnorm(session, expected_final_input,
          "model.norm.weight", (int)id_count, 256, actual_final_norm) == 1);
    CHECK(moonlight_test_matmul(session, "model.layers.0.self_attn.q_proj.weight",
          expected_input_norm, (int)id_count, 256, 192, actual_q) == 1);
    CHECK(moonlight_test_matmul(session, "lm_head.weight", expected_final_norm,
          (int)id_count, 256, 512, actual_logits) == 1);

    quantized_q = quantized_matmul_reference(
        &checkpoint, "model.layers.0.self_attn.q_proj.weight",
        expected_input_norm, (int)id_count, 256, 192);
    embed_error = max_abs(actual_embed, expected_embed, embed_count);
    norm_error = max_abs(actual_final_norm, expected_final_norm, final_norm_count);
    q_error = max_abs(actual_q, quantized_q ? quantized_q : expected_q, q_count);
    logits_error = max_abs(actual_logits, expected_logits, logits_count);
    CHECK(embed_error < 3e-6f);
    CHECK(norm_error < 3e-5f);
    CHECK(q_error < 3e-5f);
    CHECK(logits_error < 3e-5f);
    stats = moonlight_session_stats(session);
    CHECK(stats.command_buffers == 4);
    CHECK(stats.cpu_fallbacks == 0);

    printf("Moonlight Metal primitives: embed=%.9g norm=%.9g q=%.9g logits=%.9g\n",
           embed_error, norm_error, q_error, logits_error);
    moonlight_session_destroy(session);
    moonlight_model_close(model);
    free(ids);
    free(raw_ids);
    free(expected_embed);
    free(expected_final_input);
    free(expected_final_norm);
    free(expected_input_norm);
    free(expected_q);
    free(expected_logits);
    free(actual_embed);
    free(actual_final_norm);
    free(actual_q);
    free(actual_logits);
    free(quantized_q);
    return 0;
}
