#ifndef DEEPSEEK_V4_PREFIX_CACHE_H
#define DEEPSEEK_V4_PREFIX_CACHE_H

#include <stddef.h>
#include <stdint.h>

#define DEEPSEEK_V4_PREFIX_CACHE_MEMORY_RESERVE_BYTES (UINT64_C(1) << 30)

typedef struct DeepSeekV4PrefixCacheEntry DeepSeekV4PrefixCacheEntry;

typedef struct {
    DeepSeekV4PrefixCacheEntry *head;
    DeepSeekV4PrefixCacheEntry *tail;
    uint64_t budget_bytes;
    uint64_t bytes;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t disabled_skips;
    uint64_t oversize_skips;
    uint64_t memory_skips;
} DeepSeekV4PrefixCache;

typedef struct {
    const void *snapshot;
    uint64_t snapshot_bytes;
    int prefix_tokens;
} DeepSeekV4PrefixCacheHit;

typedef struct {
    uint64_t budget_bytes;
    uint64_t bytes;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t disabled_skips;
    uint64_t oversize_skips;
    uint64_t memory_skips;
} DeepSeekV4PrefixCacheStats;

void deepseek_v4_prefix_cache_init(
    DeepSeekV4PrefixCache *cache, uint64_t budget_bytes);
void deepseek_v4_prefix_cache_free(DeepSeekV4PrefixCache *cache);
uint64_t deepseek_v4_prefix_cache_entry_bytes(
    const char *fingerprint, size_t token_count, uint64_t snapshot_bytes);
int deepseek_v4_prefix_cache_prepare_insert(
    DeepSeekV4PrefixCache *cache, uint64_t entry_bytes,
    uint64_t available_memory_bytes);
int deepseek_v4_prefix_cache_put_take(
    DeepSeekV4PrefixCache *cache, const char *fingerprint,
    uint64_t config_key, const int *tokens, size_t token_count,
    void *snapshot, uint64_t snapshot_bytes);
int deepseek_v4_prefix_cache_find_longest(
    DeepSeekV4PrefixCache *cache, const char *fingerprint,
    uint64_t config_key, const int *tokens, size_t token_count,
    DeepSeekV4PrefixCacheHit *hit);
DeepSeekV4PrefixCacheStats deepseek_v4_prefix_cache_stats(
    const DeepSeekV4PrefixCache *cache);

#endif
