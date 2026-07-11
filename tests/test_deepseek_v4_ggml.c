#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../deepseek_v4_ggml.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int touch_file(const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    fputs("GGUF", file);
    return fclose(file) == 0;
}

int main(void) {
    char root[] = "/tmp/floyd-dsv4-ggml.XXXXXX";
    CHECK(mkdtemp(root) != NULL);

    char checkpoint[1024], prepared[1024], shard[1024], override[1024];
    snprintf(checkpoint, sizeof(checkpoint), "%s/checkpoint", root);
    snprintf(prepared, sizeof(prepared), "%s-GGUF", checkpoint);
    snprintf(shard, sizeof(shard), "%s/model-mxfp4_moe-00001-of-00004.gguf", prepared);
    snprintf(override, sizeof(override), "%s/override.gguf", root);
    CHECK(mkdir(checkpoint, 0700) == 0);
    CHECK(mkdir(prepared, 0700) == 0);
    CHECK(touch_file(shard));
    CHECK(touch_file(override));

    char found[2048], error[2048];
    unsetenv("FLOYD_DEEPSEEK_V4_GGUF");
    CHECK(deepseek_v4_ggml_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strcmp(found, shard) == 0);
    CHECK(strcmp(deepseek_v4_ggml_backend_name(), "metal-ggml") == 0);

    CHECK(setenv("FLOYD_DEEPSEEK_V4_GGUF", override, 1) == 0);
    CHECK(deepseek_v4_ggml_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strcmp(found, override) == 0);

    CHECK(unlink(override) == 0);
    CHECK(!deepseek_v4_ggml_find_model(checkpoint, found, sizeof(found),
                                       error, sizeof(error)));
    CHECK(strstr(error, "FLOYD_DEEPSEEK_V4_GGUF") != NULL);

    unsetenv("FLOYD_DEEPSEEK_V4_GGUF");
    CHECK(unlink(shard) == 0);
    CHECK(!deepseek_v4_ggml_find_model(checkpoint, found, sizeof(found),
                                       error, sizeof(error)));
    CHECK(strstr(error, "make prepare-deepseek-v4-gguf") != NULL);

    CHECK(rmdir(prepared) == 0);
    CHECK(rmdir(checkpoint) == 0);
    CHECK(rmdir(root) == 0);
    puts("DeepSeek V4 ggml bridge contract: ok");
    return 0;
}
