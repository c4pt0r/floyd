#ifndef DEEPSEEK_V4_KV_CACHE_H
#define DEEPSEEK_V4_KV_CACHE_H

#include <stdint.h>
#include <string.h>

typedef enum {
    DEEPSEEK_V4_KV_CACHE_RING,
    DEEPSEEK_V4_KV_CACHE_APPEND
} DeepSeekV4KVCacheMode;

typedef struct {
    DeepSeekV4KVCacheMode mode;
    int capacity;
    int head_dim;
    int count;
    int write;
    float *keys;
    int64_t *positions;
} DeepSeekV4KVCacheF32;

static inline int deepseek_v4_kv_cache_init(DeepSeekV4KVCacheF32 *cache, DeepSeekV4KVCacheMode mode,
                                   int capacity, int head_dim, float *keys,
                                   int64_t *positions) {
    if (!cache || !keys || !positions || capacity <= 0 || head_dim <= 0 ||
        (mode != DEEPSEEK_V4_KV_CACHE_RING && mode != DEEPSEEK_V4_KV_CACHE_APPEND))
        return 0;
    cache->mode = mode;
    cache->capacity = capacity;
    cache->head_dim = head_dim;
    cache->count = 0;
    cache->write = 0;
    cache->keys = keys;
    cache->positions = positions;
    return 1;
}

static inline void deepseek_v4_kv_cache_reset(DeepSeekV4KVCacheF32 *cache) {
    if (!cache) return;
    cache->count = 0;
    cache->write = 0;
}

static inline int deepseek_v4_kv_cache_count(const DeepSeekV4KVCacheF32 *cache) {
    return cache ? cache->count : 0;
}

static inline int deepseek_v4_kv_cache_slot(const DeepSeekV4KVCacheF32 *cache, int index) {
    if (!cache || index < 0 || index >= cache->count) return -1;
    int oldest = cache->count == cache->capacity ? cache->write : 0;
    return (oldest + index) % cache->capacity;
}

static inline const float *deepseek_v4_kv_cache_key(const DeepSeekV4KVCacheF32 *cache, int index) {
    int slot = deepseek_v4_kv_cache_slot(cache, index);
    return slot < 0 ? NULL : cache->keys + (int64_t)slot * cache->head_dim;
}

static inline int64_t deepseek_v4_kv_cache_position(const DeepSeekV4KVCacheF32 *cache, int index) {
    int slot = deepseek_v4_kv_cache_slot(cache, index);
    return slot < 0 ? -1 : cache->positions[slot];
}

static inline int deepseek_v4_kv_cache_append(DeepSeekV4KVCacheF32 *cache, const float *key,
                                     int64_t position) {
    if (!cache || !key || position < 0 || !cache->keys || !cache->positions ||
        cache->capacity <= 0 || cache->head_dim <= 0)
        return 0;
    if (cache->count > 0 &&
        position <= deepseek_v4_kv_cache_position(cache, cache->count - 1))
        return 0;
    if (cache->mode == DEEPSEEK_V4_KV_CACHE_APPEND && cache->count == cache->capacity)
        return 0;

    int slot = cache->write;
    memcpy(cache->keys + (int64_t)slot * cache->head_dim, key,
           (size_t)cache->head_dim * sizeof(float));
    cache->positions[slot] = position;
    cache->write = (cache->write + 1) % cache->capacity;
    if (cache->count < cache->capacity) cache->count++;
    return 1;
}

static inline int deepseek_v4_kv_cache_copy(const DeepSeekV4KVCacheF32 *cache, float *keys,
                                   int64_t *positions) {
    if (!cache || !keys || !positions) return 0;
    for (int index = 0; index < cache->count; index++) {
        const float *key = deepseek_v4_kv_cache_key(cache, index);
        if (!key) return 0;
        memcpy(keys + (int64_t)index * cache->head_dim, key,
               (size_t)cache->head_dim * sizeof(float));
        positions[index] = deepseek_v4_kv_cache_position(cache, index);
    }
    return 1;
}

#endif
