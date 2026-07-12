#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../deepseek_v4_ds4.h"

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
    char root[] = "/tmp/floyd-dsv4-ds4.XXXXXX";
    CHECK(mkdtemp(root) != NULL);

    char checkpoint[1024], prepared[1024], model[2048], sidecar[2064];
    snprintf(checkpoint, sizeof(checkpoint), "%s/checkpoint", root);
    snprintf(prepared, sizeof(prepared), "%s-DS4", checkpoint);
    snprintf(model, sizeof(model),
             "%s/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
             prepared);
    snprintf(sidecar, sizeof(sidecar), "%s.aria2", model);
    CHECK(mkdir(checkpoint, 0700) == 0);
    CHECK(mkdir(prepared, 0700) == 0);
    CHECK(touch_file(model));
    CHECK(touch_file(sidecar));

    char found[4096], error[4096];
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    CHECK(!deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strstr(error, "incomplete") != NULL);
    CHECK(unlink(sidecar) == 0);
    CHECK(deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                     error, sizeof(error)));
    CHECK(strcmp(found, model) == 0);
    CHECK(strcmp(deepseek_v4_ds4_backend_name(), "metal-ds4") == 0);

    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_GGUF", model, 1) == 0);
    CHECK(deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                     error, sizeof(error)));
    CHECK(strcmp(found, model) == 0);

    CHECK(unlink(model) == 0);
    CHECK(!deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strstr(error, "FLOYD_DEEPSEEK_V4_DS4_GGUF") != NULL);

    unsetenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    CHECK(rmdir(prepared) == 0);
    CHECK(rmdir(checkpoint) == 0);
    CHECK(rmdir(root) == 0);
    puts("DeepSeek V4 DS4 bridge contract: ok");
    return 0;
}
