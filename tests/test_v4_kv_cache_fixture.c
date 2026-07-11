#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_attention.h"
#include "../v4_indexer.h"
#include "../v4_kv_cache.h"

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

static float max_error(const float *actual, const float *expected, int64_t count) {
    float maximum = 0.0f;
    for (int64_t i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
    }
    return maximum;
}

static int test_sliding_cache(shards *source) {
    enum {
        D = 64, H = 4, HD = 16, QR = 32, GROUPS = 2, OR = 16,
        WINDOW = 8, TOKENS = 16,
    };
    V4SlidingAttentionF32 model = {
        .hidden = D, .heads = H, .head_dim = HD, .q_rank = QR,
        .rope_dim = 4, .o_groups = GROUPS, .o_rank = OR,
        .window = WINDOW, .norm_eps = 1e-6f, .rope_theta = 10000.0f,
        .q_a = load_f32(source, "model.layers.0.self_attn.q_a_proj.weight", QR * D),
        .q_a_norm = load_f32(source, "model.layers.0.self_attn.q_a_norm.weight", QR),
        .q_b = load_f32(source, "model.layers.0.self_attn.q_b_proj.weight", H * HD * QR),
        .kv = load_f32(source, "model.layers.0.self_attn.kv_proj.weight", HD * D),
        .kv_norm = load_f32(source, "model.layers.0.self_attn.kv_norm.weight", HD),
        .sinks = load_f32(source, "model.layers.0.self_attn.sinks", H),
        .o_a = load_f32(source, "model.layers.0.self_attn.o_a_proj.weight", GROUPS * OR * (H * HD / GROUPS)),
        .o_b = load_f32(source, "model.layers.0.self_attn.o_b_proj.weight", D * GROUPS * OR),
    };
    float *input = load_f32(source, "attn.0.input", TOKENS * D);
    float *expected_output = load_f32(source, "attn.0.output", TOKENS * D);
    float *expected_keys = load_f32(source, "attn.0.keys", TOKENS * HD);
    float prefill_output[TOKENS * D], decode_output[TOKENS * D];
    float prefill_storage[WINDOW * HD], decode_storage[WINDOW * HD];
    int64_t prefill_positions[WINDOW], decode_positions[WINDOW];
    V4KVCacheF32 prefill_cache, decode_cache;

    CHECK(v4_kv_cache_init(&prefill_cache, V4_KV_CACHE_RING,
                           WINDOW, HD, prefill_storage, prefill_positions));
    CHECK(v4_sliding_attention_cached_f32(
        &model, &prefill_cache, input, TOKENS, 0, prefill_output));
    CHECK(v4_kv_cache_count(&prefill_cache) == WINDOW);
    CHECK(!v4_sliding_attention_cached_f32(
        &model, &prefill_cache, input, 1, TOKENS + 1, prefill_output));

    float retained_keys[WINDOW * HD];
    int64_t retained_positions[WINDOW];
    CHECK(v4_kv_cache_copy(&prefill_cache, retained_keys, retained_positions));
    for (int i = 0; i < WINDOW; i++) CHECK(retained_positions[i] == TOKENS - WINDOW + i);
    float key_error = max_error(retained_keys, expected_keys + (TOKENS - WINDOW) * HD,
                                WINDOW * HD);
    float prefill_error = max_error(prefill_output, expected_output, TOKENS * D);

    CHECK(v4_kv_cache_init(&decode_cache, V4_KV_CACHE_RING,
                           WINDOW, HD, decode_storage, decode_positions));
    for (int token = 0; token < TOKENS; token++) {
        CHECK(v4_sliding_attention_cached_f32(
            &model, &decode_cache, input + token * D, 1, token,
            decode_output + token * D));
    }
    float decode_error = max_error(decode_output, expected_output, TOKENS * D);
    float consistency_error = max_error(decode_output, prefill_output, TOKENS * D);

    printf("v4 KV sliding: key=%.9g prefill=%.9g decode=%.9g consistency=%.9g\n",
           key_error, prefill_error, decode_error, consistency_error);
    CHECK(key_error < 3e-5f);
    CHECK(prefill_error < 3e-5f);
    CHECK(decode_error < 3e-5f);
    CHECK(consistency_error < 1e-7f);
    return 0;
}

static int test_indexer_append_cache(shards *source) {
    enum { TOKENS = 12, HEADS = 4, HD = 8, ENTRIES = 3, TOP_K = 2 };
    float *queries = load_f32(source, "indexer.q", TOKENS * HEADS * HD);
    float *oracle_keys = load_f32(source, "indexer.kv", ENTRIES * HD);
    float *weights = load_f32(source, "indexer.weights", TOKENS * HEADS);
    float *expected_scores = load_f32(source, "indexer.scores", TOKENS * ENTRIES);
    int64_t *query_positions = load_i64(source, "indexer.positions", TOKENS);
    int64_t *expected_indices = load_i64(source, "indexer.indices", TOKENS * TOP_K);
    float storage[ENTRIES * HD], keys[ENTRIES * HD], scores[TOKENS * ENTRIES];
    int64_t storage_positions[ENTRIES], key_positions[ENTRIES], indices[TOKENS * TOP_K];
    V4KVCacheF32 cache;

    CHECK(v4_kv_cache_init(&cache, V4_KV_CACHE_APPEND,
                           ENTRIES, HD, storage, storage_positions));
    CHECK(v4_kv_cache_append(&cache, oracle_keys, 0));
    CHECK(!v4_kv_cache_append(&cache, oracle_keys, 0));
    for (int entry = 1; entry < ENTRIES; entry++)
        CHECK(v4_kv_cache_append(&cache, oracle_keys + entry * HD, entry * 4));
    CHECK(!v4_kv_cache_append(&cache, oracle_keys, ENTRIES * 4));
    CHECK(v4_kv_cache_copy(&cache, keys, key_positions));
    for (int entry = 0; entry < ENTRIES; entry++) CHECK(key_positions[entry] == entry * 4);

    V4LightningIndexerF32 model = {
        .n_heads = HEADS, .head_dim = HD, .ratio = 4, .top_k = TOP_K,
    };
    CHECK(v4_indexer_forward_f32(&model, queries, keys, weights, query_positions,
                                 TOKENS, ENTRIES, scores, indices));
    float score_error = max_error(scores, expected_scores, TOKENS * ENTRIES);
    int matches = 0;
    for (int i = 0; i < TOKENS * TOP_K; i++)
        if (indices[i] == expected_indices[i]) matches++;
    printf("v4 KV indexer: score=%.9g topk=%d/%d\n",
           score_error, matches, TOKENS * TOP_K);
    CHECK(score_error < 3e-5f);
    CHECK(matches == TOKENS * TOP_K);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_v4>\n", argv[0]);
        return 2;
    }
    shards source;
    st_init(&source, argv[1]);
    CHECK(test_sliding_cache(&source) == 0);
    CHECK(test_indexer_append_cache(&source) == 0);
    puts("v4 KV cache fixture tests: ok");
    return 0;
}
