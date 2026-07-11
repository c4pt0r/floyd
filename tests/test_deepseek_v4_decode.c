#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../deepseek_v4_runtime.h"

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
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model-dir> <oracle-dir>\n", argv[0]);
        return 2;
    }
    enum { D = 4096, HC = 4, VOCAB = 129280 };
    shards oracle;
    st_init(&oracle, argv[2]);

    int prefill_ids[4] = {3, 14, 15, 9};
    int decode_id = 5;
    DeepSeekV4Runtime runtime;
    const float *logits = NULL;
    CHECK(deepseek_v4_runtime_init(&runtime, argv[1], 5));
    CHECK(deepseek_v4_runtime_forward(&runtime, prefill_ids, 4, &logits));
    CHECK(deepseek_v4_runtime_forward(&runtime, &decode_id, 1, &logits));
    const float *decode = deepseek_v4_runtime_last_streams(&runtime);

    float *expected_layers = load_f32(&oracle, "layer.42.output",
                                      (int64_t)5 * HC * D);
    CHECK(expected_layers && decode && logits);
    float layer_error = max_abs(decode, expected_layers + (int64_t)4 * HC * D,
                                (int64_t)HC * D);
    float *expected_logits = load_f32(&oracle, "final.logits", VOCAB);
    CHECK(expected_logits);
    float logits_error = max_abs(logits, expected_logits, VOCAB);
    int actual_top = 0;
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > logits[actual_top]) actual_top = i;
    int64_t expected_top = -1;
    st_read_raw(&oracle, "final.argmax", &expected_top, 0);
    printf("DeepSeek V4 real decode: layer42=%.9g logits=%.9g top1=%d/%lld position=%d\n",
           layer_error, logits_error, actual_top, (long long)expected_top,
           runtime.state.layers[42].next_position);
    CHECK(layer_error < 1e-3f && logits_error < 5e-3f);
    CHECK(actual_top == expected_top && runtime.state.layers[42].next_position == 5);
    deepseek_v4_runtime_reset(&runtime);
    CHECK(runtime.state.layers[42].next_position == 0);

    deepseek_v4_runtime_free(&runtime);
    free(expected_layers); free(expected_logits);
    return 0;
}
