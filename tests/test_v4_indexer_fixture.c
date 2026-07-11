#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_indexer.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float *load_f32(shards *source, const char *name, int64_t expected) {
    int64_t count = st_numel(source, name);
    if (count != expected) {
        fprintf(stderr, "%s: elements=%lld expected=%lld\n", name,
                (long long)count, (long long)expected);
        exit(1);
    }
    float *data = malloc((size_t)count * sizeof(float));
    st_read_f32(source, name, data, 0);
    return data;
}

static int64_t *load_i64(shards *source, const char *name, int64_t expected) {
    int64_t count = st_numel(source, name);
    if (count != expected) {
        fprintf(stderr, "%s: elements=%lld expected=%lld\n", name,
                (long long)count, (long long)expected);
        exit(1);
    }
    int64_t *data = malloc((size_t)count * sizeof(int64_t));
    st_read_raw(source, name, data, 0);
    return data;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_v4>\n", argv[0]);
        return 2;
    }

    enum { TOKENS = 12, HEADS = 4, HEAD_DIM = 8, ENTRIES = 3, TOP_K = 2 };
    shards source;
    st_init(&source, argv[1]);
    float *queries = load_f32(&source, "indexer.q", TOKENS * HEADS * HEAD_DIM);
    float *keys = load_f32(&source, "indexer.kv", ENTRIES * HEAD_DIM);
    float *weights = load_f32(&source, "indexer.weights", TOKENS * HEADS);
    float *expected_scores = load_f32(&source, "indexer.scores", TOKENS * ENTRIES);
    int64_t *positions = load_i64(&source, "indexer.positions", TOKENS);
    int64_t *expected_indices = load_i64(&source, "indexer.indices", TOKENS * TOP_K);
    float actual_scores[TOKENS * ENTRIES];
    int64_t actual_indices[TOKENS * TOP_K];
    V4LightningIndexerF32 model = {
        .n_heads = HEADS, .head_dim = HEAD_DIM, .ratio = 4, .top_k = TOP_K,
    };

    CHECK(v4_indexer_forward_f32(&model, queries, keys, weights, positions,
                                 TOKENS, ENTRIES, actual_scores, actual_indices));
    int64_t negative_position = -1;
    CHECK(!v4_indexer_forward_f32(&model, queries, keys, weights, &negative_position,
                                  1, ENTRIES, actual_scores, actual_indices));
    V4LightningIndexerF32 invalid = model;
    invalid.top_k = ENTRIES + 1;
    CHECK(!v4_indexer_forward_f32(&invalid, queries, keys, weights, positions,
                                  TOKENS, ENTRIES, actual_scores, actual_indices));

    float max_abs = 0.0f;
    for (int i = 0; i < TOKENS * ENTRIES; i++) {
        float error = fabsf(actual_scores[i] - expected_scores[i]);
        if (error > max_abs) max_abs = error;
    }
    int matches = 0;
    for (int i = 0; i < TOKENS * TOP_K; i++)
        if (actual_indices[i] == expected_indices[i]) matches++;

    printf("v4 lightning indexer: max_abs=%.9g topk=%d/%d\n",
           max_abs, matches, TOKENS * TOP_K);
    CHECK(max_abs < 3e-5f);
    CHECK(matches == TOKENS * TOP_K);
    puts("v4 lightning indexer fixture tests: ok");
    return 0;
}
