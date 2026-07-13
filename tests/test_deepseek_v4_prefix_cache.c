#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_prefix_cache.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int put_entry(DeepSeekV4PrefixCache *cache,
                     const char *fingerprint, uint64_t config_key,
                     const int *tokens, size_t token_count,
                     size_t snapshot_bytes, uint64_t available_bytes) {
    uint64_t total = deepseek_v4_prefix_cache_entry_bytes(
        fingerprint, token_count, snapshot_bytes);
    if (!deepseek_v4_prefix_cache_prepare_insert(
            cache, total, available_bytes)) return 0;
    unsigned char *snapshot = malloc(snapshot_bytes);
    if (!snapshot) return 0;
    memset(snapshot, tokens[0], snapshot_bytes);
    if (!deepseek_v4_prefix_cache_put_take(
            cache, fingerprint, config_key, tokens, token_count,
            snapshot, snapshot_bytes)) {
        free(snapshot);
        return 0;
    }
    return 1;
}

static int test_longest_prefix_and_key_isolation(void) {
    const int short_prefix[] = {1, 2};
    const int long_prefix[] = {1, 2, 3};
    const int request[] = {1, 2, 3, 4};
    DeepSeekV4PrefixCache cache;
    deepseek_v4_prefix_cache_init(&cache, 4096);
    CHECK(put_entry(&cache, "model-a", 7, short_prefix, 2, 16, UINT64_MAX));
    CHECK(put_entry(&cache, "model-a", 7, long_prefix, 3, 24, UINT64_MAX));

    DeepSeekV4PrefixCacheHit hit = {0};
    CHECK(deepseek_v4_prefix_cache_find_longest(
        &cache, "model-a", 7, request, 4, &hit));
    CHECK(hit.prefix_tokens == 3);
    CHECK(hit.snapshot_bytes == 24);
    CHECK(((const unsigned char *)hit.snapshot)[0] == 1);
    CHECK(!deepseek_v4_prefix_cache_find_longest(
        &cache, "model-b", 7, request, 4, &hit));
    CHECK(!deepseek_v4_prefix_cache_find_longest(
        &cache, "model-a", 8, request, 4, &hit));
    DeepSeekV4PrefixCacheStats stats = deepseek_v4_prefix_cache_stats(&cache);
    CHECK(stats.hits == 1);
    CHECK(stats.misses == 2);
    CHECK(stats.entries == 2);
    CHECK(stats.bytes <= stats.budget_bytes);
    deepseek_v4_prefix_cache_free(&cache);
    return 0;
}

static int test_lru_eviction(void) {
    const int a[] = {1};
    const int b[] = {2};
    const int c[] = {3};
    uint64_t each = deepseek_v4_prefix_cache_entry_bytes("m", 1, 32);
    DeepSeekV4PrefixCache cache;
    deepseek_v4_prefix_cache_init(&cache, each * 2);
    CHECK(put_entry(&cache, "m", 1, a, 1, 32, UINT64_MAX));
    CHECK(put_entry(&cache, "m", 1, b, 1, 32, UINT64_MAX));
    DeepSeekV4PrefixCacheHit hit;
    CHECK(deepseek_v4_prefix_cache_find_longest(&cache, "m", 1, a, 1, &hit));
    CHECK(put_entry(&cache, "m", 1, c, 1, 32, UINT64_MAX));
    CHECK(deepseek_v4_prefix_cache_find_longest(&cache, "m", 1, a, 1, &hit));
    CHECK(!deepseek_v4_prefix_cache_find_longest(&cache, "m", 1, b, 1, &hit));
    CHECK(deepseek_v4_prefix_cache_find_longest(&cache, "m", 1, c, 1, &hit));
    DeepSeekV4PrefixCacheStats stats = deepseek_v4_prefix_cache_stats(&cache);
    CHECK(stats.entries == 2);
    CHECK(stats.evictions == 1);
    deepseek_v4_prefix_cache_free(&cache);
    return 0;
}

static int test_budget_and_memory_pressure(void) {
    const int prefix[] = {9, 8, 7};
    uint64_t total = deepseek_v4_prefix_cache_entry_bytes("model", 3, 64);
    DeepSeekV4PrefixCache cache;

    deepseek_v4_prefix_cache_init(&cache, 0);
    CHECK(!deepseek_v4_prefix_cache_prepare_insert(&cache, total, UINT64_MAX));
    CHECK(deepseek_v4_prefix_cache_stats(&cache).disabled_skips == 1);
    deepseek_v4_prefix_cache_free(&cache);

    deepseek_v4_prefix_cache_init(&cache, total - 1);
    CHECK(!deepseek_v4_prefix_cache_prepare_insert(&cache, total, UINT64_MAX));
    CHECK(deepseek_v4_prefix_cache_stats(&cache).oversize_skips == 1);
    deepseek_v4_prefix_cache_free(&cache);

    deepseek_v4_prefix_cache_init(&cache, total * 2);
    CHECK(put_entry(&cache, "model", 1, prefix, 3, 64, UINT64_MAX));
    uint64_t available = DEEPSEEK_V4_PREFIX_CACHE_MEMORY_RESERVE_BYTES + total - 1;
    CHECK(deepseek_v4_prefix_cache_prepare_insert(&cache, total, available));
    CHECK(deepseek_v4_prefix_cache_stats(&cache).evictions == 1);
    CHECK(deepseek_v4_prefix_cache_stats(&cache).entries == 0);
    CHECK(!deepseek_v4_prefix_cache_prepare_insert(
        &cache, total,
        DEEPSEEK_V4_PREFIX_CACHE_MEMORY_RESERVE_BYTES + total - 1));
    CHECK(deepseek_v4_prefix_cache_stats(&cache).memory_skips == 1);
    deepseek_v4_prefix_cache_free(&cache);
    return 0;
}

int main(void) {
    CHECK(test_longest_prefix_and_key_isolation() == 0);
    CHECK(test_lru_eviction() == 0);
    CHECK(test_budget_and_memory_pressure() == 0);
    puts("DeepSeek V4 prefix cache tests: ok");
    return 0;
}
