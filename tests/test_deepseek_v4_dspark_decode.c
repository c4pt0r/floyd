#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../deepseek_v4_forward.h"

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static float *load_f32(shards *source, const char *name, int64_t count) {
    if (st_numel(source, name) != count) return NULL;
    float *data = malloc((size_t)count * sizeof(float));
    if (data) st_read_f32(source, name, data, 0);
    return data;
}

static float max_abs(const float *actual, const float *expected, int64_t count) {
    float maximum = 0.0f;
    for (int64_t i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
    }
    return maximum;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    enum { D = 4096, HC = 4 };
    shards model, oracle;
    st_init(&model, argv[1]);
    st_init(&oracle, argv[2]);
    float *prompt_main = load_f32(&oracle, "dspark.prompt_main_x", 5 * D);
    float *current_main = load_f32(&oracle, "dspark.current_main_x", D);
    int64_t input_id = -1, expected_ids[6];
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected_ids, 0);
    float *prefill = malloc((size_t)3 * 5 * 512 * sizeof(float));
    float *stages = malloc((size_t)3 * 5 * HC * D * sizeof(float));
    float *hidden = malloc((size_t)5 * D * sizeof(float));
    float confidence[5];
    int64_t output_ids[6];
    CHECK(prompt_main && current_main && prefill && stages && hidden);
    DeepSeekV4ForwardDSparkState state;
    DeepSeekV4ForwardDSparkCapture capture = {
        .prefill_kv = prefill, .stage_outputs = stages, .hidden = hidden,
        .output_ids = output_ids, .confidence = confidence,
    };
    CHECK(deepseek_v4_forward_dspark_state_init(&state));
    CHECK(deepseek_v4_forward_dspark_prefill(&model, prompt_main, 5, &state, &capture));
    CHECK(deepseek_v4_forward_dspark_propose(&model, current_main, 5, (int)input_id,
                                 &state, &capture));
    float prefill_error = 0.0f, stage_error = 0.0f;
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.prefill_kv.%d", stage);
        float *expected_prefill = load_f32(&oracle, name, 5 * 512);
        snprintf(name, sizeof(name), "dspark.stage.%d.output", stage);
        float *expected_stage = load_f32(&oracle, name, 5 * HC * D);
        CHECK(expected_prefill && expected_stage);
        float error = max_abs(prefill + (int64_t)stage * 5 * 512,
                              expected_prefill, 5 * 512);
        if (error > prefill_error) prefill_error = error;
        error = max_abs(stages + (int64_t)stage * 5 * HC * D,
                        expected_stage, 5 * HC * D);
        if (error > stage_error) stage_error = error;
        free(expected_prefill); free(expected_stage);
    }
    int hits = 0;
    for (int i = 0; i < 6; i++) if (output_ids[i] == expected_ids[i]) hits++;
    printf("DeepSeek V4 DSpark decode: prefill=%.9g stages=%.9g ids=%d/6\n",
           prefill_error, stage_error, hits);
    CHECK(prefill_error < 3e-4f && stage_error < 1e-2f && hits == 6);
    deepseek_v4_forward_dspark_state_free(&state);
    free(prompt_main); free(current_main); free(prefill); free(stages); free(hidden);
    return 0;
}
