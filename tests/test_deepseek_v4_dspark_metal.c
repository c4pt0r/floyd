#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "../st.h"

#define D 4096
#define HC 4
#define DRAFT 5
#define KV 512

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float max_abs(const float *actual, const float *expected, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) return INFINITY;
        float error = fabsf(actual[i] - expected[i]);
        if (error > result) result = error;
    }
    return result;
}

static void similarity(const float *actual, const float *expected, size_t count,
                       float *rmse, float *cosine) {
    double error2 = 0.0, aa = 0.0, ee = 0.0, ae = 0.0;
    for (size_t i = 0; i < count; i++) {
        double delta = (double)actual[i] - expected[i];
        error2 += delta * delta;
        aa += (double)actual[i] * actual[i];
        ee += (double)expected[i] * expected[i];
        ae += (double)actual[i] * expected[i];
    }
    *rmse = (float)sqrt(error2 / count);
    *cosine = (float)(ae / sqrt(aa * ee));
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base.gguf> <dspark.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }
    enum { VOCAB = 129280 };
    float *argmax_logits = malloc((size_t)VOCAB * sizeof(*argmax_logits));
    CHECK(argmax_logits);
    for (int i = 0; i < VOCAB; i++) argmax_logits[i] = -10.0f;
    argmax_logits[17] = 3.0f;
    argmax_logits[42] = 3.0f;
    CHECK(ds4_gpu_init());
    ds4_gpu_tensor *argmax_input =
        ds4_gpu_tensor_alloc((uint64_t)VOCAB * sizeof(*argmax_logits));
    ds4_gpu_tensor *argmax_output = ds4_gpu_tensor_alloc(sizeof(int32_t));
    CHECK(argmax_input && argmax_output);
    CHECK(ds4_gpu_tensor_write(argmax_input, 0, argmax_logits,
                               (uint64_t)VOCAB * sizeof(*argmax_logits)));
    CHECK(ds4_gpu_argmax_tensor(argmax_output, argmax_input, VOCAB));
    int32_t argmax_id = -1;
    CHECK(ds4_gpu_tensor_read(argmax_output, 0, &argmax_id, sizeof(argmax_id)));
    CHECK(argmax_id == 17);
    ds4_gpu_kernel_stats argmax_stats_before;
    ds4_gpu_get_kernel_stats(&argmax_stats_before);
    CHECK(argmax_stats_before.fast_argmax_calls > 0);
    ds4_gpu_tensor_free(argmax_output);
    ds4_gpu_tensor_free(argmax_input);
    free(argmax_logits);

    shards oracle;
    st_init(&oracle, argv[3]);
    float *main_x = malloc(D * sizeof(float));
    float *prefill = malloc(3u * DRAFT * KV * sizeof(float));
    float *actual = malloc(3u * DRAFT * HC * D * sizeof(float));
    float *expected = malloc(DRAFT * HC * D * sizeof(float));
    int64_t input_id = -1;
    int64_t expected_ids[6];
    int output_ids[6];
    CHECK(main_x && prefill && actual && expected);
    st_read_raw(&oracle, "dspark.current_main_x", main_x, 0);
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected_ids, 0);
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.prefill_kv.%d", stage);
        st_read_raw(&oracle, name, prefill + stage * DRAFT * KV, 0);
    }

    CHECK(ds4_dspark_probe_stages(argv[1], argv[2], prefill, DRAFT,
                                  main_x, (int)input_id, actual, output_ids) == 0);
    ds4_gpu_kernel_stats argmax_stats_after;
    ds4_gpu_get_kernel_stats(&argmax_stats_after);
    CHECK(argmax_stats_after.fast_argmax_calls >=
          argmax_stats_before.fast_argmax_calls + DRAFT);
    float errors[3], rmses[3], cosines[3];
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.stage.%d.output", stage);
        st_read_raw(&oracle, name, expected, 0);
        errors[stage] = max_abs(actual + (size_t)stage * DRAFT * HC * D,
                                expected, DRAFT * HC * D);
        similarity(actual + (size_t)stage * DRAFT * HC * D,
                   expected, DRAFT * HC * D, &rmses[stage], &cosines[stage]);
        CHECK(isfinite(errors[stage]));
    }
    printf("DeepSeek V4 DSpark Metal stages: max_abs=%.9g/%.9g/%.9g\n",
           errors[0], errors[1], errors[2]);
    printf("DeepSeek V4 DSpark Metal stages: rmse=%.9g/%.9g/%.9g cosine=%.9g/%.9g/%.9g\n",
           rmses[0], rmses[1], rmses[2], cosines[0], cosines[1], cosines[2]);
    int id_hits = 0;
    for (int i = 0; i < 6; i++) if (output_ids[i] == expected_ids[i]) id_hits++;
    printf("DeepSeek V4 DSpark Metal proposals: ids=%d/6\n", id_hits);
    printf("DeepSeek V4 DSpark fast argmax: calls=%llu tie_id=%d\n",
           (unsigned long long)argmax_stats_after.fast_argmax_calls,
           argmax_id);
    printf("  actual=%d,%d,%d,%d,%d,%d expected=%lld,%lld,%lld,%lld,%lld,%lld\n",
           output_ids[0], output_ids[1], output_ids[2], output_ids[3],
           output_ids[4], output_ids[5],
           (long long)expected_ids[0], (long long)expected_ids[1],
           (long long)expected_ids[2], (long long)expected_ids[3],
           (long long)expected_ids[4], (long long)expected_ids[5]);
    CHECK(rmses[0] < 0.02f);
    CHECK(rmses[1] < 0.05f);
    CHECK(rmses[2] < 7.0f);
    CHECK(cosines[0] > 0.999f);
    CHECK(cosines[1] > 0.999f);
    CHECK(cosines[2] > 0.70f);
    CHECK(id_hits == 6);
    free(expected);
    free(actual);
    free(prefill);
    free(main_x);
    return 0;
}
