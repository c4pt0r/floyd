#include "deepseek_v4_prefix_cache.h"

#include <stdlib.h>
#include <string.h>

struct DeepSeekV4PrefixCacheEntry {
    DeepSeekV4PrefixCacheEntry *prev;
    DeepSeekV4PrefixCacheEntry *next;
    char *fingerprint;
    uint64_t config_key;
    int *tokens;
    size_t token_count;
    void *snapshot;
    uint64_t snapshot_bytes;
    uint64_t accounted_bytes;
};

static uint64_t cache_add_saturating(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static char *cache_strdup(const char *value) {
    size_t size = strlen(value) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, value, size);
    return copy;
}

static void cache_unlink(
    DeepSeekV4PrefixCache *cache, DeepSeekV4PrefixCacheEntry *entry) {
    if (entry->prev) entry->prev->next = entry->next;
    else cache->head = entry->next;
    if (entry->next) entry->next->prev = entry->prev;
    else cache->tail = entry->prev;
    entry->prev = NULL;
    entry->next = NULL;
}

static void cache_link_head(
    DeepSeekV4PrefixCache *cache, DeepSeekV4PrefixCacheEntry *entry) {
    entry->prev = NULL;
    entry->next = cache->head;
    if (cache->head) cache->head->prev = entry;
    else cache->tail = entry;
    cache->head = entry;
}

static void cache_entry_free(DeepSeekV4PrefixCacheEntry *entry) {
    if (!entry) return;
    free(entry->fingerprint);
    free(entry->tokens);
    free(entry->snapshot);
    free(entry);
}

static uint64_t cache_remove(
    DeepSeekV4PrefixCache *cache, DeepSeekV4PrefixCacheEntry *entry,
    int eviction) {
    uint64_t bytes = entry->accounted_bytes;
    cache_unlink(cache, entry);
    cache->bytes -= bytes;
    cache->entries--;
    if (eviction) cache->evictions++;
    cache_entry_free(entry);
    return bytes;
}

void deepseek_v4_prefix_cache_init(
    DeepSeekV4PrefixCache *cache, uint64_t budget_bytes) {
    memset(cache, 0, sizeof(*cache));
    cache->budget_bytes = budget_bytes;
}

void deepseek_v4_prefix_cache_free(DeepSeekV4PrefixCache *cache) {
    if (!cache) return;
    while (cache->tail) cache_remove(cache, cache->tail, 0);
    memset(cache, 0, sizeof(*cache));
}

uint64_t deepseek_v4_prefix_cache_entry_bytes(
    const char *fingerprint, size_t token_count, uint64_t snapshot_bytes) {
    if (!fingerprint || !*fingerprint || token_count == 0 ||
        snapshot_bytes == 0 || token_count > SIZE_MAX / sizeof(int))
        return UINT64_MAX;
    uint64_t total = sizeof(DeepSeekV4PrefixCacheEntry);
    total = cache_add_saturating(total, strlen(fingerprint) + 1);
    total = cache_add_saturating(total, (uint64_t)token_count * sizeof(int));
    return cache_add_saturating(total, snapshot_bytes);
}

static int cache_memory_fits(
    uint64_t available, uint64_t freed, uint64_t entry_bytes) {
    if (available == UINT64_MAX) return 1;
    uint64_t after_eviction = cache_add_saturating(available, freed);
    uint64_t required = cache_add_saturating(
        DEEPSEEK_V4_PREFIX_CACHE_MEMORY_RESERVE_BYTES, entry_bytes);
    return after_eviction >= required;
}

int deepseek_v4_prefix_cache_prepare_insert(
    DeepSeekV4PrefixCache *cache, uint64_t entry_bytes,
    uint64_t available_memory_bytes) {
    if (!cache || cache->budget_bytes == 0) {
        if (cache) cache->disabled_skips++;
        return 0;
    }
    if (entry_bytes == UINT64_MAX || entry_bytes > cache->budget_bytes) {
        cache->oversize_skips++;
        return 0;
    }
    uint64_t freed = 0;
    while (cache->tail &&
           (entry_bytes > cache->budget_bytes - cache->bytes ||
            !cache_memory_fits(
                available_memory_bytes, freed, entry_bytes))) {
        freed = cache_add_saturating(
            freed, cache_remove(cache, cache->tail, 1));
    }
    if (entry_bytes > cache->budget_bytes - cache->bytes) {
        cache->oversize_skips++;
        return 0;
    }
    if (!cache_memory_fits(available_memory_bytes, freed, entry_bytes)) {
        cache->memory_skips++;
        return 0;
    }
    return 1;
}

static int cache_key_equal(
    const DeepSeekV4PrefixCacheEntry *entry, const char *fingerprint,
    uint64_t config_key, const int *tokens, size_t token_count) {
    return entry->config_key == config_key &&
           entry->token_count == token_count &&
           strcmp(entry->fingerprint, fingerprint) == 0 &&
           memcmp(entry->tokens, tokens, token_count * sizeof(int)) == 0;
}

int deepseek_v4_prefix_cache_put_take(
    DeepSeekV4PrefixCache *cache, const char *fingerprint,
    uint64_t config_key, const int *tokens, size_t token_count,
    void *snapshot, uint64_t snapshot_bytes) {
    if (!cache || !fingerprint || !tokens || !snapshot) return 0;
    uint64_t total = deepseek_v4_prefix_cache_entry_bytes(
        fingerprint, token_count, snapshot_bytes);
    if (total == UINT64_MAX || total > cache->budget_bytes) return 0;

    DeepSeekV4PrefixCacheEntry *entry = calloc(1, sizeof(*entry));
    char *fingerprint_copy = cache_strdup(fingerprint);
    int *token_copy = malloc(token_count * sizeof(*token_copy));
    if (!entry || !fingerprint_copy || !token_copy) {
        free(entry);
        free(fingerprint_copy);
        free(token_copy);
        return 0;
    }
    memcpy(token_copy, tokens, token_count * sizeof(*token_copy));

    DeepSeekV4PrefixCacheEntry *old = cache->head;
    while (old && !cache_key_equal(
            old, fingerprint, config_key, tokens, token_count)) old = old->next;
    uint64_t old_bytes = old ? old->accounted_bytes : 0;
    if (cache->bytes - old_bytes > cache->budget_bytes - total) {
        free(entry);
        free(fingerprint_copy);
        free(token_copy);
        return 0;
    }
    if (old) cache_remove(cache, old, 0);

    entry->fingerprint = fingerprint_copy;
    entry->config_key = config_key;
    entry->tokens = token_copy;
    entry->token_count = token_count;
    entry->snapshot = snapshot;
    entry->snapshot_bytes = snapshot_bytes;
    entry->accounted_bytes = total;
    cache_link_head(cache, entry);
    cache->bytes += total;
    cache->entries++;
    return 1;
}

int deepseek_v4_prefix_cache_find_longest(
    DeepSeekV4PrefixCache *cache, const char *fingerprint,
    uint64_t config_key, const int *tokens, size_t token_count,
    DeepSeekV4PrefixCacheHit *hit) {
    if (hit) memset(hit, 0, sizeof(*hit));
    if (!cache || !fingerprint || !tokens || !hit) return 0;
    DeepSeekV4PrefixCacheEntry *best = NULL;
    for (DeepSeekV4PrefixCacheEntry *entry = cache->head;
         entry; entry = entry->next) {
        if (entry->config_key != config_key ||
            entry->token_count > token_count ||
            strcmp(entry->fingerprint, fingerprint) != 0) continue;
        if (best && entry->token_count <= best->token_count) continue;
        if (memcmp(entry->tokens, tokens,
                   entry->token_count * sizeof(int)) == 0) best = entry;
    }
    if (!best) {
        cache->misses++;
        return 0;
    }
    if (best != cache->head) {
        cache_unlink(cache, best);
        cache_link_head(cache, best);
    }
    cache->hits++;
    hit->snapshot = best->snapshot;
    hit->snapshot_bytes = best->snapshot_bytes;
    hit->prefix_tokens = (int)best->token_count;
    return 1;
}

DeepSeekV4PrefixCacheStats deepseek_v4_prefix_cache_stats(
    const DeepSeekV4PrefixCache *cache) {
    DeepSeekV4PrefixCacheStats stats = {0};
    if (!cache) return stats;
    stats.budget_bytes = cache->budget_bytes;
    stats.bytes = cache->bytes;
    stats.entries = cache->entries;
    stats.hits = cache->hits;
    stats.misses = cache->misses;
    stats.evictions = cache->evictions;
    stats.disabled_skips = cache->disabled_skips;
    stats.oversize_skips = cache->oversize_skips;
    stats.memory_skips = cache->memory_skips;
    return stats;
}
