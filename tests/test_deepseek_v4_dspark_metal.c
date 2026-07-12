#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ds4.h"
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

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base.gguf> <dspark.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }
    shards oracle;
    st_init(&oracle, argv[3]);
    float *main_x = malloc(D * sizeof(float));
    float *prefill = malloc(3u * DRAFT * KV * sizeof(float));
    float *actual = malloc(3u * DRAFT * HC * D * sizeof(float));
    float *expected = malloc(DRAFT * HC * D * sizeof(float));
    int64_t input_id = -1;
    CHECK(main_x && prefill && actual && expected);
    st_read_raw(&oracle, "dspark.current_main_x", main_x, 0);
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.prefill_kv.%d", stage);
        st_read_raw(&oracle, name, prefill + stage * DRAFT * KV, 0);
    }

    CHECK(ds4_dspark_probe_stages(argv[1], argv[2], prefill, DRAFT,
                                  main_x, (int)input_id, actual) == 0);
    float errors[3];
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.stage.%d.output", stage);
        st_read_raw(&oracle, name, expected, 0);
        errors[stage] = max_abs(actual + (size_t)stage * DRAFT * HC * D,
                                expected, DRAFT * HC * D);
        CHECK(isfinite(errors[stage]));
    }
    printf("DeepSeek V4 DSpark Metal stages: max_abs=%.9g/%.9g/%.9g\n",
           errors[0], errors[1], errors[2]);
    CHECK(errors[0] < 0.25f && errors[1] < 0.25f && errors[2] < 0.25f);
    free(expected);
    free(actual);
    free(prefill);
    free(main_x);
    return 0;
}
