#include <stdio.h>

#include "ds4.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <three-stage-dspark.gguf>\n", argv[0]);
        return 2;
    }
    int stages = 0;
    CHECK(ds4_dspark_support_validate(argv[1], &stages) == 0);
    CHECK(stages == 3);
    printf("DeepSeek V4 DSpark support: stages=%d\n", stages);
    return 0;
}
