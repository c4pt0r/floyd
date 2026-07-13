#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void silu_multiply(float *gate, const float *up, int64_t count) {
    for (int64_t index = 0; index < count; ++index)
        gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
}

static int quantized_moe_reference(shards *model, int layer,
                                   const MoonlightModelInfo *info,
                                   const float *input, int rows,
                                   const int *route_ids,
                                   const float *route_weights,
                                   float *routed, float *shared, float *output) {
    char name[512];
    int hidden = info->hidden_size;
    int intermediate = info->moe_intermediate_size;
    int shared_width = intermediate * info->shared_expert_count;
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.shared_experts.gate_proj.weight", layer);
    char scale_name[544];
    snprintf(scale_name, sizeof(scale_name), "%s.qs", name);
    if (!st_has(model, scale_name)) return 0;

    float *shared_gate = quantized_matmul_reference(
        model, name, input, rows, hidden, shared_width);
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.shared_experts.up_proj.weight", layer);
    float *shared_up = quantized_matmul_reference(
        model, name, input, rows, hidden, shared_width);
    silu_multiply(shared_gate, shared_up, (int64_t)rows * shared_width);
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.shared_experts.down_proj.weight", layer);
    float *shared_result = quantized_matmul_reference(
        model, name, shared_gate, rows, shared_width, hidden);
    memcpy(shared, shared_result, (size_t)rows * hidden * sizeof(*shared));
    memset(routed, 0, (size_t)rows * hidden * sizeof(*routed));

    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < info->experts_per_token; ++slot) {
            int index = row * info->experts_per_token + slot;
            int expert = route_ids[index];
            snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.gate_proj.weight",
                     layer, expert);
            float *gate = quantized_matmul_reference(
                model, name, input + (int64_t)row * hidden,
                1, hidden, intermediate);
            snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.up_proj.weight",
                     layer, expert);
            float *up = quantized_matmul_reference(
                model, name, input + (int64_t)row * hidden,
                1, hidden, intermediate);
            silu_multiply(gate, up, intermediate);
            snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.down_proj.weight",
                     layer, expert);
            float *down = quantized_matmul_reference(
                model, name, gate, 1, intermediate, hidden);
            for (int column = 0; column < hidden; ++column)
                routed[(int64_t)row * hidden + column] +=
                    route_weights[index] * down[column];
            free(gate);
            free(up);
            free(down);
        }
    }
    for (int64_t index = 0; index < (int64_t)rows * hidden; ++index)
        output[index] = routed[index] + shared[index];
    free(shared_gate);
    free(shared_up);
    free(shared_result);
    return 1;
}

int main(int argc, char **argv) {
    MoonlightModel *model = NULL;
    MoonlightSession *session = NULL;
    MoonlightOptions options = {.context_size = 64, .max_batch = 32};
    MoonlightModelInfo info;
    MoonlightStats stats;
    shards oracle;
    shards checkpoint;
    char error[512] = {0};
    int64_t id_count;
    int64_t input_count;
    int64_t score_count;
    int64_t id_value_count;
    int64_t weight_count;
    int64_t routed_count;
    int64_t shared_count;
    int64_t output_count;
    float *input;
    float *expected_scores;
    int64_t *expected_ids64;
    int *expected_ids;
    float *expected_weights;
    float *expected_routed;
    float *expected_shared;
    float *expected_output;
    int *actual_ids;
    float *actual_weights;
    float *actual_scores;
    float *actual_routed;
    float *actual_shared;
    float *actual_output;
    float score_error;
    float weight_error;
    float routed_error;
    float shared_error;
    float output_error;

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL ORACLE_DIR\n", argv[0]);
        return 2;
    }
    CHECK(moonlight_model_open(&model, argv[1], error, sizeof(error)) == 1);
    info = moonlight_model_info(model);
    st_init(&oracle, argv[2]);
    st_init(&checkpoint, argv[1]);
    id_count = st_numel(&oracle, "input_ids");
    input = read_f32(&oracle, "layer.1.post_norm", &input_count);
    expected_scores = read_f32(&oracle, "layer.1.router_scores", &score_count);
    id_value_count = st_numel(&oracle, "layer.1.router_ids");
    expected_ids64 = malloc((size_t)id_value_count * sizeof(*expected_ids64));
    st_read_raw(&oracle, "layer.1.router_ids", expected_ids64, 0);
    expected_weights = read_f32(&oracle, "layer.1.router_weights", &weight_count);
    expected_routed = read_f32(&oracle, "layer.1.routed_mlp", &routed_count);
    expected_shared = read_f32(&oracle, "layer.1.shared_mlp", &shared_count);
    expected_output = read_f32(&oracle, "layer.1.mlp", &output_count);
    CHECK(id_count > 0 && id_count <= options.max_batch);
    CHECK(input && expected_scores && expected_weights && expected_routed &&
          expected_shared && expected_output);
    CHECK(input_count == id_count * info.hidden_size);
    CHECK(score_count == id_count * info.expert_count);
    CHECK(id_value_count == id_count * info.experts_per_token);
    CHECK(weight_count == id_value_count);
    CHECK(routed_count == input_count && shared_count == input_count &&
          output_count == input_count);

    expected_ids = malloc((size_t)id_value_count * sizeof(*expected_ids));
    for (int64_t index = 0; index < id_value_count; ++index)
        expected_ids[index] = (int)expected_ids64[index];
    actual_ids = calloc((size_t)id_value_count, sizeof(*actual_ids));
    actual_weights = calloc((size_t)weight_count, sizeof(*actual_weights));
    actual_scores = calloc((size_t)score_count, sizeof(*actual_scores));
    actual_routed = calloc((size_t)routed_count, sizeof(*actual_routed));
    actual_shared = calloc((size_t)shared_count, sizeof(*actual_shared));
    actual_output = calloc((size_t)output_count, sizeof(*actual_output));
    CHECK(moonlight_session_create(&session, model, &options,
                                   error, sizeof(error)) == 1);
    CHECK(moonlight_test_moe(session, 1, input, (int)id_count,
                             actual_ids, actual_weights, actual_scores,
                             actual_routed, actual_shared, actual_output) == 1);
    quantized_moe_reference(&checkpoint, 1, &info, input, (int)id_count,
                            actual_ids, actual_weights, expected_routed,
                            expected_shared, expected_output);
    weight_error = 0;
    for (int row = 0; row < id_count; ++row) {
        for (int slot = 0; slot < info.experts_per_token; ++slot) {
            int index = row * info.experts_per_token + slot;
            int expected_slot = -1;
            for (int candidate = 0; candidate < info.experts_per_token; ++candidate)
                if (expected_ids[row * info.experts_per_token + candidate] ==
                    actual_ids[index]) expected_slot = candidate;
            CHECK(expected_slot >= 0);
            float difference = fabsf(
                actual_weights[index] -
                expected_weights[row * info.experts_per_token + expected_slot]);
            if (difference > weight_error) weight_error = difference;
        }
    }
    score_error = max_abs(actual_scores, expected_scores, score_count);
    routed_error = max_abs(actual_routed, expected_routed, routed_count);
    shared_error = max_abs(actual_shared, expected_shared, shared_count);
    output_error = max_abs(actual_output, expected_output, output_count);
    printf("Moonlight Metal MoE: scores=%.9g weights=%.9g routed=%.9g shared=%.9g output=%.9g\n",
           score_error, weight_error, routed_error, shared_error, output_error);
    CHECK(score_error < 3e-5f);
    CHECK(weight_error < 3e-6f);
    CHECK(routed_error < 3e-4f);
    CHECK(shared_error < 3e-4f);
    CHECK(output_error < 5e-4f);
    stats = moonlight_session_stats(session);
    CHECK(stats.command_buffers == 2);
    CHECK(stats.cpu_fallbacks == 0);

    moonlight_session_destroy(session);
    moonlight_model_close(model);
    free(input);
    free(expected_scores);
    free(expected_ids64);
    free(expected_ids);
    free(expected_weights);
    free(expected_routed);
    free(expected_shared);
    free(expected_output);
    free(actual_ids);
    free(actual_weights);
    free(actual_scores);
    free(actual_routed);
    free(actual_shared);
    free(actual_output);
    return 0;
}
