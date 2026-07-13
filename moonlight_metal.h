#ifndef MOONLIGHT_METAL_H
#define MOONLIGHT_METAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct MoonlightModel MoonlightModel;
typedef struct MoonlightSession MoonlightSession;

typedef struct {
    int context_size;
    int max_batch;
} MoonlightOptions;

typedef struct {
    int hidden_size;
    int layer_count;
    int vocab_size;
    uint64_t resident_bytes;
} MoonlightModelInfo;

typedef struct {
    uint64_t command_buffers;
    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    uint64_t cpu_fallbacks;
    uint64_t buffer_allocations;
    uint64_t resident_bytes;
    double prefill_ms;
    double decode_ms;
} MoonlightStats;

int moonlight_model_open(MoonlightModel **out, const char *path,
                         char *error, size_t error_size);
void moonlight_model_close(MoonlightModel *model);
const char *moonlight_device_name(const MoonlightModel *model);
MoonlightModelInfo moonlight_model_info(const MoonlightModel *model);

int moonlight_session_create(MoonlightSession **out, MoonlightModel *model,
                             const MoonlightOptions *options,
                             char *error, size_t error_size);
void moonlight_session_destroy(MoonlightSession *session);
void moonlight_session_reset(MoonlightSession *session);
int moonlight_session_position(const MoonlightSession *session);
MoonlightStats moonlight_session_stats(const MoonlightSession *session);

int moonlight_test_embed(MoonlightSession *session, const int *ids,
                         int count, float *output);
int moonlight_test_rmsnorm(MoonlightSession *session, const float *input,
                           const char *weight_name, int rows, int width,
                           float *output);
int moonlight_test_matmul(MoonlightSession *session, const char *weight_name,
                          const float *input, int rows, int input_width,
                          int output_width, float *output);

#endif
