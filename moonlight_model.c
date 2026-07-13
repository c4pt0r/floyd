#include "moonlight_model.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st.h"

struct MoonlightModelData {
    MoonlightConfig config;
    shards tensors;
    uint64_t bytes;
};

static int set_error(char *error, size_t error_size, const char *format, ...) {
    va_list arguments;
    if (error && error_size) {
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int json_int(jval *root, const char *name) {
    jval *value = json_get(root, name);
    return value && value->t == J_NUM ? (int)value->num : 0;
}

static float json_float(jval *root, const char *name, float fallback) {
    jval *value = json_get(root, name);
    return value && value->t == J_NUM ? (float)value->num : fallback;
}

static int read_config(MoonlightConfig *config, const char *path,
                       char *error, size_t error_size) {
    char filename[4096];
    char *source;
    char *arena = NULL;
    jval *root;
    FILE *file;
    long length;

    if (snprintf(filename, sizeof(filename), "%s/config.json", path) >=
        (int)sizeof(filename))
        return set_error(error, error_size, "model path is too long");
    file = fopen(filename, "rb");
    if (!file) return set_error(error, error_size, "cannot open %s", filename);
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return set_error(error, error_size, "cannot read %s", filename);
    }
    source = malloc((size_t)length + 1);
    if (!source) {
        fclose(file);
        return set_error(error, error_size, "out of memory reading config");
    }
    if (fread(source, 1, (size_t)length, file) != (size_t)length) {
        free(source);
        fclose(file);
        return set_error(error, error_size, "short read from %s", filename);
    }
    fclose(file);
    source[length] = 0;
    root = json_parse(source, &arena);
    free(source);
    if (!root) return set_error(error, error_size, "invalid JSON in %s", filename);

    memset(config, 0, sizeof(*config));
    config->hidden_size = json_int(root, "hidden_size");
    config->layer_count = json_int(root, "num_hidden_layers");
    config->head_count = json_int(root, "num_attention_heads");
    config->expert_count = json_int(root, "n_routed_experts");
    config->experts_per_token = json_int(root, "num_experts_per_tok");
    config->dense_intermediate_size = json_int(root, "intermediate_size");
    config->moe_intermediate_size = json_int(root, "moe_intermediate_size");
    config->first_dense_layer_count = json_int(root, "first_k_dense_replace");
    config->q_lora_rank = json_int(root, "q_lora_rank");
    config->kv_lora_rank = json_int(root, "kv_lora_rank");
    config->qk_nope_dim = json_int(root, "qk_nope_head_dim");
    config->qk_rope_dim = json_int(root, "qk_rope_head_dim");
    config->value_dim = json_int(root, "v_head_dim");
    config->shared_expert_count = json_int(root, "n_shared_experts");
    config->vocab_size = json_int(root, "vocab_size");
    config->rms_norm_epsilon = json_float(root, "rms_norm_eps", 1e-5f);
    config->rope_theta = json_float(root, "rope_theta", 10000.0f);
    config->routed_scale = json_float(root, "routed_scaling_factor", 1.0f);
    if (config->rope_theta == 10000.0f) {
        jval *parameters = json_get(root, "rope_parameters");
        if (parameters)
            config->rope_theta = json_float(parameters, "rope_theta", 10000.0f);
    }
    free(arena);

    if (config->hidden_size <= 0 || config->layer_count <= 0 ||
        config->head_count <= 0 || config->expert_count <= 0 ||
        config->experts_per_token <= 0 || config->kv_lora_rank <= 0 ||
        config->qk_rope_dim <= 0 || config->vocab_size <= 0)
        return set_error(error, error_size, "unsupported Moonlight config in %s",
                         filename);
    return 1;
}

int moonlight_model_data_open(MoonlightModelData **out, const char *path,
                              char *error, size_t error_size) {
    MoonlightModelData *data;
    if (!out || !path || !path[0])
        return set_error(error, error_size, "model path is required");
    *out = NULL;
    data = calloc(1, sizeof(*data));
    if (!data) return set_error(error, error_size, "out of memory opening model");
    if (!read_config(&data->config, path, error, error_size)) {
        free(data);
        return 0;
    }
    st_init(&data->tensors, path);
    if (!data->tensors.n) {
        moonlight_model_data_close(data);
        return set_error(error, error_size, "no safetensors found in %s", path);
    }
    for (int index = 0; index < data->tensors.n; ++index)
        data->bytes += (uint64_t)data->tensors.t[index].nbytes;
    *out = data;
    return 1;
}

void moonlight_model_data_close(MoonlightModelData *data) {
    if (!data) return;
    for (int index = 0; index < data->tensors.nfd; ++index) {
        if (data->tensors.fds[index] >= 0) close(data->tensors.fds[index]);
        if (data->tensors.dfds[index] >= 0 &&
            data->tensors.dfds[index] != data->tensors.fds[index])
            close(data->tensors.dfds[index]);
        free(data->tensors.paths[index]);
    }
    for (int index = 0; index < data->tensors.n; ++index)
        free(data->tensors.t[index].name);
    free(data->tensors.t);
    free(data->tensors.hidx);
    free(data);
}

const MoonlightConfig *moonlight_model_data_config(const MoonlightModelData *data) {
    return data ? &data->config : NULL;
}

int moonlight_model_data_tensor_count(const MoonlightModelData *data) {
    return data ? data->tensors.n : 0;
}

int moonlight_model_data_find(const MoonlightModelData *data, const char *name) {
    st_tensor *tensor;
    if (!data || !name) return -1;
    tensor = st_find((shards *)&data->tensors, name);
    return tensor ? (int)(tensor - data->tensors.t) : -1;
}

MoonlightTensorInfo moonlight_model_data_tensor(const MoonlightModelData *data,
                                                int index) {
    MoonlightTensorInfo result = {0};
    if (!data || index < 0 || index >= data->tensors.n) return result;
    st_tensor *tensor = &data->tensors.t[index];
    result.name = tensor->name;
    result.dtype = tensor->dtype;
    result.element_count = tensor->numel;
    result.byte_count = tensor->nbytes;
    return result;
}

int moonlight_model_data_read_raw(const MoonlightModelData *data, int index,
                                  void *output, size_t output_size) {
    const st_tensor *tensor;
    if (!data || index < 0 || index >= data->tensors.n || !output) return 0;
    tensor = &data->tensors.t[index];
    if (tensor->nbytes < 0 || (uint64_t)tensor->nbytes > output_size) return 0;
    return pread(tensor->fd, output, (size_t)tensor->nbytes, tensor->off) ==
           tensor->nbytes;
}

uint64_t moonlight_model_data_bytes(const MoonlightModelData *data) {
    return data ? data->bytes : 0;
}
