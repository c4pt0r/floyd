#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_ds4.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    char text[4096];
    size_t size;
} Output;

static int append_piece(int token, const char *piece, size_t size, void *data) {
    Output *output = data;
    (void)token;
    if (size > sizeof(output->text) - output->size - 1) return 0;
    memcpy(output->text + output->size, piece, size);
    output->size += size;
    output->text[output->size] = 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark checkpoint>\n", argv[0]);
        return 2;
    }
    char model[4096], error[4096] = {0};
    CHECK(deepseek_v4_ds4_find_model(
        argv[1], model, sizeof(model), error, sizeof(error)));
    CHECK(setenv("DRAFT", "3", 1) == 0);

    DeepSeekV4Ds4Session *session = deepseek_v4_ds4_open_cached(
        model, 512, 1, UINT64_C(2048) * 1024 * 1024,
        error, sizeof(error));
    CHECK(session != NULL);

    const DeepSeekV4Ds4Message first[] = {
        {"system", "Answer with one short word."},
        {"user", "Say alpha."},
    };
    const DeepSeekV4Ds4Message second[] = {
        {"system", "Answer with one short word."},
        {"user", "Say beta."},
    };
    DeepSeekV4Ds4RequestConfig config = {
        .max_tokens = 1,
        .temperature = 0.0f,
        .top_p = 1.0f,
        .draft = 2,
    };
    Output first_output = {0}, second_output = {0};
    DeepSeekV4Ds4Stats first_stats, second_stats;
    CHECK(deepseek_v4_ds4_generate_messages(
        session, first, 2, &config, append_piece, &first_output,
        &first_stats, error, sizeof(error)) == 1);
    CHECK(!first_stats.cache_hit);
    CHECK(first_stats.cache_entries > 0);
    CHECK(deepseek_v4_ds4_generate_messages(
        session, second, 2, &config, append_piece, &second_output,
        &second_stats, error, sizeof(error)) == 1);
    CHECK(second_stats.cache_hit);
    CHECK(second_stats.cache_prefix_tokens > 0);
    CHECK(second_stats.cached_tokens == second_stats.cache_prefix_tokens);
    CHECK(second_stats.cache_entries > 0);
    CHECK(second_stats.cache_bytes > 0);

    printf("DeepSeek V4 DS4 prefix cache: first=%d second=%d "
           "prefix=%d entries=%llu bytes=%llu outputs=%s/%s\n",
           first_stats.cache_hit, second_stats.cache_hit,
           second_stats.cache_prefix_tokens,
           second_stats.cache_entries, second_stats.cache_bytes,
           first_output.text, second_output.text);
    deepseek_v4_ds4_close(session);
    return 0;
}
