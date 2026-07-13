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
    int head_count;
    int kv_lora_rank;
    int qk_nope_dim;
    int qk_rope_dim;
    int value_dim;
    int expert_count;
    int experts_per_token;
    int moe_intermediate_size;
    int shared_expert_count;
    int first_dense_layer_count;
    float rms_norm_epsilon;
    float rope_theta;
    float routed_scale;
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
int moonlight_session_prefill(MoonlightSession *session, const int *ids,
                              int count, float *last_logits,
                              char *error, size_t error_size);
int moonlight_session_append(MoonlightSession *session, const int *ids,
                             int count, float *last_logits,
                             char *error, size_t error_size);
int moonlight_session_decode(MoonlightSession *session, int token,
                             float *logits, char *error, size_t error_size);
int moonlight_test_prefill_layers(MoonlightSession *session, const int *ids,
                                  int count, float *layer_outputs,
                                  float *last_logits,
                                  char *error, size_t error_size);

int moonlight_test_embed(MoonlightSession *session, const int *ids,
                         int count, float *output);
int moonlight_test_rmsnorm(MoonlightSession *session, const float *input,
                           const char *weight_name, int rows, int width,
                           float *output);
int moonlight_test_matmul(MoonlightSession *session, const char *weight_name,
                          const float *input, int rows, int input_width,
                          int output_width, float *output);
int moonlight_test_attention(MoonlightSession *session, int layer,
                             const float *input, int rows, int position,
                             float *output);
int moonlight_test_kv_length(const MoonlightSession *session, int layer);
int moonlight_test_copy_kv(const MoonlightSession *session, int layer,
                           float *latent, float *rope, int capacity);
int moonlight_test_moe(MoonlightSession *session, int layer,
                       const float *input, int rows, int *route_ids,
                       float *route_weights, float *router_scores,
                       float *routed_output, float *shared_output,
                       float *output);
int moonlight_test_dense_mlp(MoonlightSession *session, int layer,
                             const float *input, int rows, float *output);
int moonlight_test_copy_routes(const MoonlightSession *session, int layer,
                               int *route_ids, int capacity);

#endif
