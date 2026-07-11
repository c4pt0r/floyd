#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_ggml.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    int first_token;
    char text[128];
    size_t used;
} Capture;

static int capture_token(int token, const char *piece, size_t piece_size,
                         void *user_data) {
    Capture *capture = (Capture *)user_data;
    if (capture->first_token < 0) capture->first_token = token;
    if (piece_size >= sizeof(capture->text) - capture->used) return 0;
    memcpy(capture->text + capture->used, piece, piece_size);
    capture->used += piece_size;
    capture->text[capture->used] = 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark>\n", argv[0]);
        return 2;
    }

    char error[2048];
    DeepSeekV4GgmlSession *session = deepseek_v4_ggml_open(
        argv[1], 512, error, sizeof(error));
    CHECK(session != NULL);

    Capture capture = {.first_token = -1};
    DeepSeekV4GgmlStats stats = {0};
    int generated = deepseek_v4_ggml_generate_user(
        session, "hello", 1, 1, capture_token, &capture,
        &stats, error, sizeof(error));
    CHECK(generated == 1);
    CHECK(capture.first_token == 19923);
    CHECK(strcmp(capture.text, "Hello") == 0);
    CHECK(stats.generated_tokens == 1);
    CHECK(stats.prompt_tokens > 0);
    CHECK(stats.load_ms > 0.0);

    printf("DeepSeek V4 ggml official: token=%d text=%s load_ms=%.3f prompt_ms=%.3f\n",
           capture.first_token, capture.text, stats.load_ms, stats.prompt_ms);
    deepseek_v4_ggml_close(session);
    return 0;
}
