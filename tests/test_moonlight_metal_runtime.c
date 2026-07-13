#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../moonlight_metal.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    MoonlightModel *model = NULL;
    MoonlightSession *session = NULL;
    MoonlightOptions options = {.context_size = 64, .max_batch = 32};
    MoonlightStats before;
    MoonlightStats after;
    MoonlightModelInfo info;
    char error[512] = {0};
    char device[256];

    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }

    CHECK(moonlight_model_open(&model, argv[1], error, sizeof(error)) == 1);
    CHECK(model != NULL);
    CHECK(strstr(moonlight_device_name(model), "Apple") != NULL);
    snprintf(device, sizeof(device), "%s", moonlight_device_name(model));
    info = moonlight_model_info(model);
    CHECK(info.hidden_size == 256);
    CHECK(info.layer_count == 4);
    CHECK(info.vocab_size == 512);
    CHECK(info.resident_bytes > 40 * 1024 * 1024ULL);

    CHECK(moonlight_session_create(&session, model, &options,
                                   error, sizeof(error)) == 1);
    CHECK(session != NULL);
    before = moonlight_session_stats(session);
    CHECK(before.buffer_allocations >= 5);
    CHECK(before.resident_bytes > info.resident_bytes);
    CHECK(before.cpu_fallbacks == 0);
    CHECK(moonlight_session_position(session) == 0);

    moonlight_session_reset(session);
    after = moonlight_session_stats(session);
    CHECK(after.buffer_allocations == before.buffer_allocations);
    CHECK(after.resident_bytes == before.resident_bytes);
    CHECK(after.cpu_fallbacks == 0);
    CHECK(moonlight_session_position(session) == 0);

    moonlight_session_destroy(session);
    moonlight_model_close(model);
    printf("Moonlight Metal lifecycle: device=%s buffers=%llu resident=%.2f MiB\n",
           device,
           (unsigned long long)after.buffer_allocations,
           after.resident_bytes / 1048576.0);
    return 0;
}
