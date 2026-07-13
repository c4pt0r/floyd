#include <math.h>
#include <stdint.h>
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

static void rmsnorm_reference(const float *input, const float *weight,
                              float *output, int rows, int width,
                              float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const float *source = input + (int64_t)row * width;
        float sum = 0;
        for (int column = 0; column < width; ++column)
            sum += source[column] * source[column];
        float scale = 1.0f / sqrtf(sum / width + epsilon);
        for (int column = 0; column < width; ++column)
            output[(int64_t)row * width + column] =
                source[column] * scale * weight[column];
    }
}

static void rope_reference(const float *input, float *output, int rows,
                           int width, float theta) {
    int half = width / 2;
    for (int row = 0; row < rows; ++row) {
        const float *source = input + (int64_t)row * width;
        float *target = output + (int64_t)row * width;
        for (int pair = 0; pair < half; ++pair) {
            float angle = row * powf(theta, -2.0f * pair / width);
            float cosine = cosf(angle);
            float sine = sinf(angle);
            float first = source[2 * pair];
            float second = source[2 * pair + 1];
            target[pair] = first * cosine - second * sine;
            target[half + pair] = second * cosine + first * sine;
        }
    }
}

int main(int argc, char **argv) {
    MoonlightModel *model = NULL;
    MoonlightSession *prefill_session = NULL;
    MoonlightSession *split_session = NULL;
    MoonlightOptions options = {.context_size = 64, .max_batch = 32};
    MoonlightStats stats;
    MoonlightModelInfo info;
    shards oracle;
    shards checkpoint;
    char error[512] = {0};
    int64_t input_count;
    int64_t attention_count;
    int64_t kv_a_count;
    int64_t kv_norm_count;
    int64_t id_count;
    float *input;
    float *expected_attention;
    float *kv_a;
    float *expected_latent;
    float *quantized_kv_a;
    float *expected_rope;
    float *prefill;
    float *prefix;
    float *decode;
    float *actual_latent;
    float *actual_rope;
    float attention_error;
    float split_error;
    float latent_error;
    float rope_error;

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL ORACLE_DIR\n", argv[0]);
        return 2;
    }
    CHECK(moonlight_model_open(&model, argv[1], error, sizeof(error)) == 1);
    info = moonlight_model_info(model);
    st_init(&oracle, argv[2]);
    st_init(&checkpoint, argv[1]);
    id_count = st_numel(&oracle, "input_ids");
    input = read_f32(&oracle, "layer.0.input_norm", &input_count);
    expected_attention = read_f32(&oracle, "layer.0.attn", &attention_count);
    kv_a = read_f32(&oracle, "layer.0.kv_a", &kv_a_count);
    expected_latent = read_f32(&oracle, "layer.0.kv_norm", &kv_norm_count);
    CHECK(id_count > 1 && id_count <= options.max_batch);
    CHECK(input && expected_attention && kv_a && expected_latent);
    CHECK(input_count == id_count * info.hidden_size);
    CHECK(attention_count == id_count * info.hidden_size);
    CHECK(kv_a_count == id_count * (info.kv_lora_rank + info.qk_rope_dim));
    CHECK(kv_norm_count == id_count * info.kv_lora_rank);

    quantized_kv_a = quantized_matmul_reference(
        &checkpoint, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        input, (int)id_count, info.hidden_size,
        info.kv_lora_rank + info.qk_rope_dim);
    if (quantized_kv_a) {
        int64_t norm_weight_count;
        float *norm_weight = read_f32(
            &checkpoint, "model.layers.0.self_attn.kv_a_layernorm.weight",
            &norm_weight_count);
        CHECK(norm_weight && norm_weight_count == info.kv_lora_rank);
        free(kv_a);
        kv_a = quantized_kv_a;
        float *latent_source = malloc((size_t)id_count * info.kv_lora_rank *
                                      sizeof(*latent_source));
        for (int row = 0; row < id_count; ++row)
            for (int column = 0; column < info.kv_lora_rank; ++column)
                latent_source[(int64_t)row * info.kv_lora_rank + column] =
                    kv_a[(int64_t)row *
                         (info.kv_lora_rank + info.qk_rope_dim) + column];
        rmsnorm_reference(latent_source, norm_weight, expected_latent,
                          (int)id_count, info.kv_lora_rank,
                          info.rms_norm_epsilon);
        free(latent_source);
        free(norm_weight);
    }

    expected_rope = malloc((size_t)id_count * info.qk_rope_dim *
                           sizeof(*expected_rope));
    float *rope_source = malloc((size_t)id_count * info.qk_rope_dim *
                                sizeof(*rope_source));
    for (int row = 0; row < id_count; ++row)
        for (int column = 0; column < info.qk_rope_dim; ++column)
            rope_source[(int64_t)row * info.qk_rope_dim + column] =
                kv_a[(int64_t)row *
                     (info.kv_lora_rank + info.qk_rope_dim) +
                     info.kv_lora_rank + column];
    rope_reference(rope_source, expected_rope, (int)id_count,
                   info.qk_rope_dim, info.rope_theta);
    free(rope_source);

    prefill = calloc((size_t)attention_count, sizeof(*prefill));
    prefix = calloc((size_t)(id_count - 1) * info.hidden_size, sizeof(*prefix));
    decode = calloc((size_t)info.hidden_size, sizeof(*decode));
    actual_latent = calloc((size_t)id_count * info.kv_lora_rank,
                           sizeof(*actual_latent));
    actual_rope = calloc((size_t)id_count * info.qk_rope_dim,
                         sizeof(*actual_rope));
    CHECK(moonlight_session_create(&prefill_session, model, &options,
                                   error, sizeof(error)) == 1);
    CHECK(moonlight_session_create(&split_session, model, &options,
                                   error, sizeof(error)) == 1);

    CHECK(moonlight_test_attention(prefill_session, 0, input, (int)id_count,
                                   0, prefill) == 1);
    CHECK(moonlight_test_kv_length(prefill_session, 0) == id_count);
    CHECK(moonlight_session_position(prefill_session) == id_count);
    CHECK(moonlight_test_copy_kv(prefill_session, 0, actual_latent, actual_rope,
                                 (int)id_count) == 1);
    CHECK(moonlight_test_attention(split_session, 0, input,
                                   (int)id_count - 1, 0, prefix) == 1);
    CHECK(moonlight_test_attention(split_session, 0,
                                   input + (id_count - 1) * info.hidden_size,
                                   1, (int)id_count - 1, decode) == 1);
    CHECK(moonlight_test_kv_length(split_session, 0) == id_count);
    CHECK(moonlight_test_attention(split_session, 0, input, 1,
                                   (int)id_count + 1, decode) == 0);

    attention_error = max_abs(prefill, expected_attention, attention_count);
    split_error = max_abs(decode,
                          prefill + (id_count - 1) * info.hidden_size,
                          info.hidden_size);
    latent_error = max_abs(actual_latent, expected_latent, kv_norm_count);
    rope_error = max_abs(actual_rope, expected_rope,
                         id_count * info.qk_rope_dim);
    printf("Moonlight Metal MLA: attention=%.9g split=%.9g latent=%.9g rope=%.9g\n",
           attention_error, split_error, latent_error, rope_error);
    CHECK(attention_error < 3e-4f);
    CHECK(split_error < 3e-4f);
    CHECK(latent_error < 3e-5f);
    CHECK(rope_error < 3e-5f);
    stats = moonlight_session_stats(prefill_session);
    CHECK(stats.cpu_fallbacks == 0);
    moonlight_session_reset(split_session);
    CHECK(moonlight_test_kv_length(split_session, 0) == 0);
    CHECK(moonlight_session_position(split_session) == 0);

    moonlight_session_destroy(split_session);
    moonlight_session_destroy(prefill_session);
    moonlight_model_close(model);
    free(input);
    free(expected_attention);
    free(kv_a);
    free(expected_latent);
    free(expected_rope);
    free(prefill);
    free(prefix);
    free(decode);
    free(actual_latent);
    free(actual_rope);
    return 0;
}
