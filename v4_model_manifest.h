#ifndef V4_MODEL_MANIFEST_H
#define V4_MODEL_MANIFEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st.h"

enum { V4_MANIFEST_MAX_LAYERS = 128 };

typedef struct {
    int vocab_size;
    int dim;
    int moe_inter_dim;
    int n_layers;
    int n_hash_layers;
    int n_mtp_layers;
    int n_heads;
    int head_dim;
    int n_experts;
    int n_shared_experts;
    int compress_ratios[V4_MANIFEST_MAX_LAYERS];
    int loaded_layers;
    int sliding_layers;
    int csa_layers;
    int hca_layers;
    int resolved_tensors;
} V4ModelManifest;

static inline int v4_manifest_number(jval *root, const char *name, int *output) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM) {
        fprintf(stderr, "v4 config: missing numeric %s\n", name);
        return 0;
    }
    *output = (int)value->num;
    return 1;
}

static inline jval *v4_manifest_config(const char *model_dir, char **arena) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/inference/config.json", model_dir);
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); return NULL; }
    if (fseek(file, 0, SEEK_END) || ftell(file) < 0) { fclose(file); return NULL; }
    long size = ftell(file);
    if (fseek(file, 0, SEEK_SET)) { fclose(file); return NULL; }
    char *buffer = malloc((size_t)size + 1);
    if (!buffer) { fclose(file); return NULL; }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer); fclose(file); return NULL;
    }
    fclose(file);
    buffer[size] = 0;
    jval *root = json_parse(buffer, arena);
    free(buffer);
    return root;
}

static inline int v4_manifest_require(V4ModelManifest *model, shards *source,
                                      const char *name, int dtype) {
    st_tensor *tensor = st_find(source, name);
    if (!tensor) {
        fprintf(stderr, "v4 manifest: missing %s\n", name);
        return 0;
    }
    if (dtype >= 0 && tensor->dtype != dtype) {
        fprintf(stderr, "v4 manifest: %s dtype=%d expected=%d\n",
                name, tensor->dtype, dtype);
        return 0;
    }
    model->resolved_tensors++;
    return 1;
}

static inline int v4_manifest_quant_pair(V4ModelManifest *model, shards *source,
                                         const char *prefix, int weight_dtype) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (!v4_manifest_require(model, source, name, weight_dtype)) return 0;
    snprintf(name, sizeof(name), "%s.scale", prefix);
    return v4_manifest_require(model, source, name, ST_DTYPE_F8_E8M0);
}

static inline int v4_manifest_layer(V4ModelManifest *model, shards *source,
                                    int layer) {
    static const char *dense_pairs[] = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"};
    static const char *expert_weights[] = {"w1", "w2", "w3"};
    char name[320], prefix[320];

    for (int i = 0; i < 5; i++) {
        snprintf(prefix, sizeof(prefix), "layers.%d.attn.%s", layer, dense_pairs[i]);
        if (!v4_manifest_quant_pair(model, source, prefix, ST_DTYPE_F8_E4M3)) return 0;
    }
    static const char *plain[] = {
        "attn.attn_sink", "attn.q_norm.weight", "attn.kv_norm.weight",
        "attn_norm.weight", "ffn_norm.weight",
        "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
        "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
        "ffn.gate.weight",
    };
    for (int i = 0; i < 12; i++) {
        snprintf(name, sizeof(name), "layers.%d.%s", layer, plain[i]);
        if (!v4_manifest_require(model, source, name, -1)) return 0;
    }
    snprintf(name, sizeof(name), "layers.%d.ffn.gate.%s", layer,
             layer < model->n_hash_layers ? "tid2eid" : "bias");
    if (!v4_manifest_require(model, source, name, -1)) return 0;

    for (int weight = 0; weight < 3; weight++) {
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.shared_experts.%s",
                 layer, expert_weights[weight]);
        if (!v4_manifest_quant_pair(model, source, prefix, ST_DTYPE_F8_E4M3)) return 0;
    }
    for (int expert = 0; expert < model->n_experts; expert++) {
        for (int weight = 0; weight < 3; weight++) {
            snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d.%s",
                     layer, expert, expert_weights[weight]);
            if (!v4_manifest_quant_pair(model, source, prefix, ST_DTYPE_U8)) return 0;
        }
    }

    int ratio = model->compress_ratios[layer];
    if (ratio == 0) {
        model->sliding_layers++;
    } else {
        static const char *compressor[] = {"ape", "norm.weight", "wgate.weight", "wkv.weight"};
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "layers.%d.attn.compressor.%s", layer, compressor[i]);
            if (!v4_manifest_require(model, source, name, -1)) return 0;
        }
        if (ratio == 4) {
            model->csa_layers++;
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "layers.%d.attn.indexer.compressor.%s",
                         layer, compressor[i]);
                if (!v4_manifest_require(model, source, name, -1)) return 0;
            }
            snprintf(name, sizeof(name), "layers.%d.attn.indexer.weights_proj.weight", layer);
            if (!v4_manifest_require(model, source, name, -1)) return 0;
            snprintf(prefix, sizeof(prefix), "layers.%d.attn.indexer.wq_b", layer);
            if (!v4_manifest_quant_pair(model, source, prefix, ST_DTYPE_F8_E4M3)) return 0;
        } else if (ratio == 128) {
            model->hca_layers++;
        } else {
            fprintf(stderr, "v4 manifest: layer %d unsupported compress ratio %d\n", layer, ratio);
            return 0;
        }
    }
    model->loaded_layers++;
    return 1;
}

static inline int v4_model_manifest_load(V4ModelManifest *model, shards *source,
                                         const char *model_dir) {
    if (!model || !source || !model_dir) return 0;
    memset(model, 0, sizeof(*model));
    char *arena = NULL;
    jval *root = v4_manifest_config(model_dir, &arena);
    if (!root) return 0;
    int valid =
        v4_manifest_number(root, "vocab_size", &model->vocab_size) &&
        v4_manifest_number(root, "dim", &model->dim) &&
        v4_manifest_number(root, "moe_inter_dim", &model->moe_inter_dim) &&
        v4_manifest_number(root, "n_layers", &model->n_layers) &&
        v4_manifest_number(root, "n_hash_layers", &model->n_hash_layers) &&
        v4_manifest_number(root, "n_mtp_layers", &model->n_mtp_layers) &&
        v4_manifest_number(root, "n_heads", &model->n_heads) &&
        v4_manifest_number(root, "head_dim", &model->head_dim) &&
        v4_manifest_number(root, "n_routed_experts", &model->n_experts) &&
        v4_manifest_number(root, "n_shared_experts", &model->n_shared_experts);
    jval *ratios = json_get(root, "compress_ratios");
    if (!valid || model->n_layers <= 0 || model->n_layers > V4_MANIFEST_MAX_LAYERS ||
        model->n_hash_layers < 0 || model->n_hash_layers > model->n_layers ||
        model->n_experts <= 0 || model->n_shared_experts != 1 ||
        !ratios || ratios->t != J_ARR || ratios->len < model->n_layers) {
        fprintf(stderr, "v4 config: unsupported architecture dimensions\n");
        free(arena);
        return 0;
    }
    for (int layer = 0; layer < model->n_layers; layer++)
        model->compress_ratios[layer] = (int)ratios->kids[layer]->num;
    free(arena);

    st_init(source, model_dir);
    static const char *global[] = {
        "embed.weight", "head.weight", "norm.weight",
        "hc_head_base", "hc_head_fn", "hc_head_scale",
    };
    for (int i = 0; i < 6; i++)
        if (!v4_manifest_require(model, source, global[i], -1)) return 0;
    for (int layer = 0; layer < model->n_layers; layer++)
        if (!v4_manifest_layer(model, source, layer)) return 0;
    return 1;
}

#endif
