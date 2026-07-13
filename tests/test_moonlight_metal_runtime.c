#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../moonlight_metal.h"
#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float max_abs(const float *actual, const float *expected, int count) {
    float result = 0;
    for (int index = 0; index < count; ++index) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > result) result = difference;
    }
    return result;
}

static int argmax(const float *values, int count) {
    int best = 0;
    for (int index = 1; index < count; ++index)
        if (values[index] > values[best]) best = index;
    return best;
}

static int read_greedy_tokens(const char *directory, int *tokens, int count) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/ref.json", directory) >=
        (int)sizeof(path)) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = malloc((size_t)length + 1);
    if (!source || fread(source, 1, (size_t)length, file) != (size_t)length) {
        free(source);
        fclose(file);
        return 0;
    }
    fclose(file);
    source[length] = 0;
    jval *root = json_parse_full(source, NULL, 0);
    jval *values = json_get(root, "greedy_ids");
    int ok = values && values->t == J_ARR && values->len >= count;
    for (int index = 0; ok && index < count; ++index) {
        ok = values->kids[index]->t == J_NUM;
        if (ok) tokens[index] = (int)values->kids[index]->num;
    }
    json_free(root);
    free(source);
    return ok;
}

int main(int argc, char **argv) {
    MoonlightModel *model = NULL;
    MoonlightSession *session = NULL;
    MoonlightOptions options = {.context_size = 64, .max_batch = 32};
    MoonlightStats before;
    MoonlightStats after;
    MoonlightModelInfo info;
    char error[512] = {0};
    char device[256];
    shards oracle;
    int64_t id_count;
    int64_t logit_count;
    int64_t *ids64;
    int *ids;
    float *expected_logits;
    float *logits;
    float *layer_outputs;
    float *dense_input;
    float *expected_dense;
    float *actual_dense;
    float logit_error;
    int greedy[2];

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL ORACLE_DIR\n", argv[0]);
        return 2;
    }

    CHECK(moonlight_model_open(&model, argv[1], error, sizeof(error)) == 1);
    CHECK(model != NULL);
    CHECK(strstr(moonlight_device_name(model), "Apple") != NULL);
    snprintf(device, sizeof(device), "%s", moonlight_device_name(model));
    info = moonlight_model_info(model);
    CHECK(info.hidden_size > 0);
    CHECK(info.layer_count > 0);
    CHECK(info.vocab_size > 0);
    CHECK(info.resident_bytes > 0);

    CHECK(moonlight_session_create(&session, model, &options,
                                   error, sizeof(error)) == 1);
    CHECK(session != NULL);
    before = moonlight_session_stats(session);
    CHECK(before.buffer_allocations >= 5);
    CHECK(before.resident_bytes > info.resident_bytes);
    CHECK(before.cpu_fallbacks == 0);
    CHECK(moonlight_session_position(session) == 0);

    moonlight_session_reset(session);
    after = moonlight_session_stats(session);
    CHECK(after.buffer_allocations == before.buffer_allocations);
    CHECK(after.resident_bytes == before.resident_bytes);
    CHECK(after.cpu_fallbacks == 0);
    CHECK(moonlight_session_position(session) == 0);

    st_init(&oracle, argv[2]);
    CHECK(read_greedy_tokens(argv[2], greedy, 2));
    id_count = st_numel(&oracle, "input_ids");
    logit_count = st_numel(&oracle, "logits");
    CHECK(id_count > 0 && id_count <= options.max_batch);
    CHECK(logit_count == id_count * info.vocab_size);
    ids64 = malloc((size_t)id_count * sizeof(*ids64));
    ids = malloc((size_t)id_count * sizeof(*ids));
    expected_logits = malloc((size_t)logit_count * sizeof(*expected_logits));
    logits = malloc((size_t)info.vocab_size * sizeof(*logits));
    layer_outputs = malloc((size_t)info.layer_count * id_count *
                           info.hidden_size * sizeof(*layer_outputs));
    st_read_raw(&oracle, "input_ids", ids64, 0);
    st_read_f32(&oracle, "logits", expected_logits, 0);
    for (int64_t index = 0; index < id_count; ++index) ids[index] = (int)ids64[index];

    int64_t dense_input_count = st_numel(&oracle, "layer.0.post_norm");
    int64_t dense_output_count = st_numel(&oracle, "layer.0.mlp");
    CHECK(dense_input_count == id_count * info.hidden_size);
    CHECK(dense_output_count == dense_input_count);
    dense_input = malloc((size_t)dense_input_count * sizeof(*dense_input));
    expected_dense = malloc((size_t)dense_output_count * sizeof(*expected_dense));
    actual_dense = malloc((size_t)dense_output_count * sizeof(*actual_dense));
    st_read_f32(&oracle, "layer.0.post_norm", dense_input, 0);
    st_read_f32(&oracle, "layer.0.mlp", expected_dense, 0);
    CHECK(moonlight_test_dense_mlp(session, 0, dense_input, (int)id_count,
                                   actual_dense) == 1);
    float dense_error = max_abs(actual_dense, expected_dense,
                                (int)dense_output_count);
    printf("Moonlight Metal dense MLP: max_abs=%.9g\n", dense_error);
    CHECK(dense_error < 3e-4f);

    CHECK(moonlight_test_prefill_layers(session, ids, (int)id_count,
                                        layer_outputs, logits,
                                        error, sizeof(error)) == 1);
    for (int layer = 0; layer < info.layer_count; ++layer) {
        char name[64];
        snprintf(name, sizeof(name), "layer.%d.output", layer);
        int64_t expected_count = st_numel(&oracle, name);
        CHECK(expected_count == id_count * info.hidden_size);
        float *expected_layer = malloc((size_t)expected_count *
                                       sizeof(*expected_layer));
        st_read_f32(&oracle, name, expected_layer, 0);
        float layer_error = max_abs(
            layer_outputs + (int64_t)layer * expected_count,
            expected_layer, (int)expected_count);
        printf("Moonlight Metal layer %d: max_abs=%.9g\n", layer, layer_error);
        /* Real q8 peaks at 1.343e-3 in layer 26 while all route sets match
         * and final logits remain below 5e-5. Keep both stricter gates below. */
        CHECK(layer_error < 2e-3f);
        free(expected_layer);
    }
    int route_mismatch_rows = 0;
    for (int layer = info.first_dense_layer_count;
         layer < info.layer_count; ++layer) {
        char name[64];
        snprintf(name, sizeof(name), "layer.%d.router_ids", layer);
        int64_t expected_count = st_numel(&oracle, name);
        CHECK(expected_count == id_count * info.experts_per_token);
        int64_t *expected_routes = malloc((size_t)expected_count *
                                          sizeof(*expected_routes));
        int *actual_routes = malloc((size_t)expected_count *
                                    sizeof(*actual_routes));
        st_read_raw(&oracle, name, expected_routes, 0);
        CHECK(moonlight_test_copy_routes(session, layer, actual_routes,
                                         (int)id_count) == 1);
        int layer_mismatches = 0;
        for (int row = 0; row < id_count; ++row) {
            int row_matches = 1;
            for (int slot = 0; slot < info.experts_per_token; ++slot) {
                int actual = actual_routes[row * info.experts_per_token + slot];
                int found = 0;
                for (int expected_slot = 0;
                     expected_slot < info.experts_per_token; ++expected_slot)
                    found = found || actual ==
                        expected_routes[row * info.experts_per_token +
                                        expected_slot];
                row_matches = row_matches && found;
            }
            if (!row_matches) layer_mismatches++;
        }
        printf("Moonlight Metal layer %d routes: mismatch_rows=%d/%lld\n",
               layer, layer_mismatches, (long long)id_count);
        route_mismatch_rows += layer_mismatches;
        free(expected_routes);
        free(actual_routes);
    }
    printf("Moonlight Metal route mismatches: %d rows\n", route_mismatch_rows);
    CHECK(route_mismatch_rows == 0);
    logit_error = max_abs(logits,
                          expected_logits + (id_count - 1) * info.vocab_size,
                          info.vocab_size);
    printf("Moonlight Metal prefill: logits=%.9g token=%d\n",
           logit_error, argmax(logits, info.vocab_size));
    CHECK(logit_error < 3e-4f);
    CHECK(argmax(logits, info.vocab_size) == greedy[0]);
    CHECK(moonlight_session_position(session) == id_count);
    CHECK(moonlight_session_decode(session, greedy[0], logits,
                                   error, sizeof(error)) == 1);
    CHECK(argmax(logits, info.vocab_size) == greedy[1]);
    CHECK(moonlight_session_position(session) == id_count + 1);
    after = moonlight_session_stats(session);
    CHECK(after.prefill_tokens == (uint64_t)id_count);
    CHECK(after.decode_tokens == 1);
    CHECK(after.cpu_fallbacks == 0);
    printf("Moonlight Metal runtime: logits=%.9g greedy=%d,%d commands=%llu\n",
           logit_error, greedy[0], greedy[1],
           (unsigned long long)after.command_buffers);
    printf("Moonlight Metal timing: prefill=%.3f ms (%.2f tok/s) decode=%.3f ms (%.2f tok/s)\n",
           after.prefill_ms, id_count * 1000.0 / after.prefill_ms,
           after.decode_ms, 1000.0 / after.decode_ms);

    moonlight_session_reset(session);
    CHECK(moonlight_session_prefill(session, ids, (int)id_count, logits,
                                    error, sizeof(error)) == 1);
    logit_error = max_abs(logits,
                          expected_logits + (id_count - 1) * info.vocab_size,
                          info.vocab_size);
    CHECK(logit_error < 3e-4f);
    CHECK(argmax(logits, info.vocab_size) == greedy[0]);
    CHECK(moonlight_session_decode(session, greedy[0], logits,
                                   error, sizeof(error)) == 1);
    CHECK(argmax(logits, info.vocab_size) == greedy[1]);
    after = moonlight_session_stats(session);
    printf("Moonlight Metal fast path: logits=%.9g commands=%llu prefill=%.3f ms decode=%.3f ms\n",
           logit_error, (unsigned long long)after.command_buffers,
           after.prefill_ms, after.decode_ms);
    CHECK(after.command_buffers <= (uint64_t)(2 * info.layer_count + 2));
    CHECK(after.cpu_fallbacks == 0);
    const char *max_prefill_ms = getenv("MOONLIGHT_MAX_PREFILL_MS");
    const char *max_decode_ms = getenv("MOONLIGHT_MAX_DECODE_MS");
    if (max_prefill_ms) CHECK(after.prefill_ms <= strtod(max_prefill_ms, NULL));
    if (max_decode_ms) CHECK(after.decode_ms <= strtod(max_decode_ms, NULL));

    free(ids64);
    free(ids);
    free(expected_logits);
    free(logits);
    free(layer_outputs);
    free(dense_input);
    free(expected_dense);
    free(actual_dense);

    moonlight_session_destroy(session);
    moonlight_model_close(model);
    printf("Moonlight Metal lifecycle: device=%s buffers=%llu resident=%.2f MiB\n",
           device,
           (unsigned long long)after.buffer_allocations,
           after.resident_bytes / 1048576.0);
    return 0;
}
