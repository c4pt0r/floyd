#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ds4.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <base.gguf> <three-stage-dspark.gguf>\n", argv[0]);
        return 2;
    }
    int stages = 0;
    CHECK(ds4_dspark_support_validate(argv[2], &stages) == 0);
    CHECK(stages == 3);

    char lock_path[] = "/tmp/floyd-dspark-support.XXXXXX";
    int lock_fd = mkstemp(lock_path);
    CHECK(lock_fd >= 0);
    CHECK(close(lock_fd) == 0);
    CHECK(setenv("DS4_LOCK_FILE", lock_path, 1) == 0);
    ds4_engine_options options = {
        .model_path = argv[1],
        .dspark_path = argv[2],
        .backend = DS4_BACKEND_METAL,
        .mtp_draft_tokens = 5,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(engine != NULL);
    CHECK(ds4_engine_has_dspark(engine));
    CHECK(ds4_engine_has_mtp(engine));
    CHECK(ds4_engine_mtp_draft_tokens(engine) == 5);
    ds4_engine_close(engine);
    CHECK(unlink(lock_path) == 0);
    unsetenv("DS4_LOCK_FILE");
    printf("DeepSeek V4 DSpark support: stages=%d resident=1 draft=5\n", stages);
    return 0;
}
