#ifndef MOONLIGHT_MODEL_H
#define MOONLIGHT_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct MoonlightModelData MoonlightModelData;

typedef struct {
    int hidden_size;
    int layer_count;
    int head_count;
    int expert_count;
    int experts_per_token;
    int dense_intermediate_size;
    int moe_intermediate_size;
    int first_dense_layer_count;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_nope_dim;
    int qk_rope_dim;
    int value_dim;
    int shared_expert_count;
    int vocab_size;
    float rms_norm_epsilon;
    float rope_theta;
    float routed_scale;
} MoonlightConfig;

typedef struct {
    const char *name;
    int dtype;
    int64_t element_count;
    int64_t byte_count;
} MoonlightTensorInfo;

int moonlight_model_data_open(MoonlightModelData **out, const char *path,
                              char *error, size_t error_size);
void moonlight_model_data_close(MoonlightModelData *data);
const MoonlightConfig *moonlight_model_data_config(const MoonlightModelData *data);
int moonlight_model_data_tensor_count(const MoonlightModelData *data);
MoonlightTensorInfo moonlight_model_data_tensor(const MoonlightModelData *data,
                                                int index);
int moonlight_model_data_read_raw(const MoonlightModelData *data, int index,
                                  void *output, size_t output_size);
uint64_t moonlight_model_data_bytes(const MoonlightModelData *data);

#endif
