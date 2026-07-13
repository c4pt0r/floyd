#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../moonlight_oracle.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    const float values[] = {1.25f, -2.5f, 7.0f};
    char directory[] = "/tmp/floyd-moonlight-oracle-XXXXXX";
    char path[512];
    float actual[3] = {0};

    unsetenv("FLOYD_MOONLIGHT_ORACLE_DIR");
    CHECK(moonlight_oracle_enabled() == 0);
    CHECK(moonlight_oracle_write_f32("disabled", values, 3) == 1);

    CHECK(mkdtemp(directory) != NULL);
    CHECK(setenv("FLOYD_MOONLIGHT_ORACLE_DIR", directory, 1) == 0);
    CHECK(moonlight_oracle_enabled() == 1);
    CHECK(moonlight_oracle_write_f32("layer.0.attn", values, 3) == 1);
    CHECK(moonlight_oracle_write_f32("../escape", values, 3) == 0);
    CHECK(moonlight_oracle_write_f32("too_large", values,
          SIZE_MAX / sizeof(float) + 1) == 0);

    snprintf(path, sizeof(path), "%s/layer.0.attn.f32", directory);
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fread(actual, sizeof(float), 3, file) == 3);
    CHECK(fgetc(file) == EOF);
    CHECK(fclose(file) == 0);
    CHECK(memcmp(actual, values, sizeof(values)) == 0);

    CHECK(unlink(path) == 0);
    CHECK(rmdir(directory) == 0);
    puts("Moonlight CPU oracle writer: ok");
    return 0;
}
