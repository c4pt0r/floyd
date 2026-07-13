#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_prefix_cache.h"
#include "../deepseek_v4_serve.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    DeepSeekV4PrefixCache cache;
    char text[128];
} FakeSession;

static int fake_cached_handler(
    void *user_data, const DeepSeekV4ServeRequest *request,
    DeepSeekV4ServeResponse *response, char *error, size_t error_size) {
    FakeSession *session = user_data;
    if (!request->system || !request->prompt) {
        snprintf(error, error_size, "fake expects system and prompt");
        return 0;
    }
    size_t system_tokens = strlen(request->system);
    size_t prompt_tokens = strlen(request->prompt);
    size_t total_tokens = system_tokens + prompt_tokens;
    int *tokens = malloc(total_tokens * sizeof(*tokens));
    if (!tokens) return 0;
    for (size_t i = 0; i < system_tokens; i++) tokens[i] = request->system[i];
    for (size_t i = 0; i < prompt_tokens; i++)
        tokens[system_tokens + i] = request->prompt[i];

    DeepSeekV4PrefixCacheHit hit = {0};
    int cache_hit = deepseek_v4_prefix_cache_find_longest(
        &session->cache, "fake-model", 1, tokens, total_tokens, &hit);
    if (!cache_hit) {
        uint64_t bytes = deepseek_v4_prefix_cache_entry_bytes(
            "fake-model", system_tokens, 8);
        if (deepseek_v4_prefix_cache_prepare_insert(
                &session->cache, bytes, UINT64_MAX)) {
            void *snapshot = malloc(8);
            if (snapshot && !deepseek_v4_prefix_cache_put_take(
                    &session->cache, "fake-model", 1, tokens,
                    system_tokens, snapshot, 8)) free(snapshot);
        }
    }
    snprintf(session->text, sizeof(session->text), "reply:%s", request->prompt);
    response->text = session->text;
    response->prompt_tokens = (int)total_tokens;
    response->cached_tokens = cache_hit ? hit.prefix_tokens : 0;
    response->completion_tokens = 1;
    response->cache_hit = cache_hit;
    response->cache_prefix_tokens = response->cached_tokens;
    DeepSeekV4PrefixCacheStats stats =
        deepseek_v4_prefix_cache_stats(&session->cache);
    response->cache_entries = stats.entries;
    response->cache_bytes = stats.bytes;
    free(tokens);
    return 1;
}

int main(void) {
    FILE *input = tmpfile();
    FILE *output = tmpfile();
    CHECK(input && output);
    fputs("{\"id\":1,\"system\":\"shared-prefix\",\"prompt\":\"one\"}\n", input);
    fputs("{\"id\":2,\"system\":\"shared-prefix\",\"prompt\":\"two\"}\n", input);
    rewind(input);

    FakeSession session = {0};
    deepseek_v4_prefix_cache_init(&session.cache, 4096);
    CHECK(deepseek_v4_serve_stdio(
        input, output, fake_cached_handler, &session) == 0);
    rewind(output);
    char line[4096];
    CHECK(fgets(line, sizeof(line), output));
    CHECK(strstr(line, "\"id\":1"));
    CHECK(strstr(line, "\"cache_hit\":false"));
    CHECK(fgets(line, sizeof(line), output));
    CHECK(strstr(line, "\"id\":2"));
    CHECK(strstr(line, "\"text\":\"reply:two\""));
    CHECK(strstr(line, "\"cache_hit\":true"));
    CHECK(strstr(line, "\"cache_prefix_tokens\":13"));
    CHECK(!fgets(line, sizeof(line), output));

    DeepSeekV4PrefixCacheStats stats =
        deepseek_v4_prefix_cache_stats(&session.cache);
    CHECK(stats.hits == 1);
    CHECK(stats.misses == 1);
    deepseek_v4_prefix_cache_free(&session.cache);
    fclose(input);
    fclose(output);
    puts("DeepSeek V4 stdio cached-session smoke: ok");
    return 0;
}
