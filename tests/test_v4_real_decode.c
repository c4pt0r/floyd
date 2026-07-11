#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_real_layer0.h"

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
    shards model, oracle;
    st_init(&model, argv[1]);
    st_init(&oracle, argv[2]);

    int prefill_ids[4] = {3, 14, 15, 9};
    int decode_id = 5;
    float *prefill = malloc((size_t)4 * HC * D * sizeof(float));
    float *decode = malloc((size_t)HC * D * sizeof(float));
    V4RealModelState state;
    CHECK(prefill && decode && v4_real_model_state_init(&state, 5));
    CHECK(v4_real_base_layers_forward(&model, prefill_ids, 4,
                                      &state, prefill, NULL));
    CHECK(v4_real_base_layers_forward(&model, &decode_id, 1,
                                      &state, decode, NULL));

    float *expected_layers = load_f32(&oracle, "layer.42.output",
                                      (int64_t)5 * HC * D);
    float *hidden = malloc(D * sizeof(float));
    float *normalized = malloc(D * sizeof(float));
    float *logits = malloc((size_t)VOCAB * sizeof(float));
    CHECK(expected_layers && hidden && normalized && logits);
    float layer_error = max_abs(decode, expected_layers + (int64_t)4 * HC * D,
                                (int64_t)HC * D);
    V4RealHeadCapture capture = {
        .hidden = hidden, .normalized = normalized, .logits = logits,
    };
    CHECK(v4_real_base_head_forward(&model, decode, 1, &capture));
    float *expected_logits = load_f32(&oracle, "final.logits", VOCAB);
    CHECK(expected_logits);
    float logits_error = max_abs(logits, expected_logits, VOCAB);
    int actual_top = 0;
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > logits[actual_top]) actual_top = i;
    int64_t expected_top = -1;
    st_read_raw(&oracle, "final.argmax", &expected_top, 0);
    printf("v4 real decode: layer42=%.9g logits=%.9g top1=%d/%lld position=%d\n",
           layer_error, logits_error, actual_top, (long long)expected_top,
           state.layers[42].next_position);
    CHECK(layer_error < 1e-3f && logits_error < 5e-3f);
    CHECK(actual_top == expected_top && state.layers[42].next_position == 5);

    v4_real_model_state_free(&state);
    free(prefill); free(decode); free(expected_layers);
    free(hidden); free(normalized); free(logits); free(expected_logits);
    return 0;
}
