#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "moonlight_kernels_metal.h"
#include "moonlight_metal.h"
#include "moonlight_model.h"

typedef struct {
    __strong id<MTLBuffer> buffer;
} MoonlightWeight;

struct MoonlightModel {
    MoonlightModelData *data;
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLLibrary> library;
    __strong id<MTLComputePipelineState> noop;
    __strong id<MTLComputePipelineState> embed_f32;
    __strong id<MTLComputePipelineState> rmsnorm;
    __strong id<MTLComputePipelineState> matmul_f32;
    __strong id<MTLComputePipelineState> matmul_q8;
    __strong id<MTLComputePipelineState> matmul_q8_reduction;
    __strong id<MTLComputePipelineState> matmul_q4;
    __strong id<MTLComputePipelineState> cache_append;
    __strong id<MTLComputePipelineState> rope_query;
    __strong id<MTLComputePipelineState> attention_absorbed;
    __strong id<MTLComputePipelineState> attention_absorb_query;
    __strong id<MTLComputePipelineState> attention_scores;
    __strong id<MTLComputePipelineState> attention_latent_context;
    __strong id<MTLComputePipelineState> attention_decode_value;
    __strong id<MTLComputePipelineState> route_sigmoid_topk;
    __strong id<MTLComputePipelineState> clear_f32;
    __strong id<MTLComputePipelineState> gather_rows;
    __strong id<MTLComputePipelineState> silu_multiply;
    __strong id<MTLComputePipelineState> scatter_expert;
    __strong id<MTLComputePipelineState> add_f32;
    MoonlightWeight *weights;
    int weight_count;
    uint64_t resident_bytes;
};

struct MoonlightSession {
    MoonlightModel *model;
    MoonlightOptions options;
    __strong id<MTLBuffer> activation_a;
    __strong id<MTLBuffer> activation_b;
    __strong id<MTLBuffer> latent_kv;
    __strong id<MTLBuffer> rope_kv;
    __strong id<MTLBuffer> token_ids;
    __strong id<MTLBuffer> logits;
    __strong id<MTLBuffer> query;
    __strong id<MTLBuffer> compressed;
    __strong id<MTLBuffer> context;
    __strong id<MTLBuffer> absorbed_query;
    __strong id<MTLBuffer> attention_scores;
    __strong id<MTLBuffer> latent_context;
    __strong id<MTLBuffer> router_logits;
    __strong id<MTLBuffer> route_ids;
    __strong id<MTLBuffer> route_weights;
    __strong id<MTLBuffer> expert_rows;
    __strong id<MTLBuffer> expert_slots;
    __strong id<MTLBuffer> expert_input;
    __strong id<MTLBuffer> expert_gate;
    __strong id<MTLBuffer> expert_up;
    __strong id<MTLBuffer> expert_hidden;
    __strong id<MTLBuffer> expert_output;
    __strong id<MTLBuffer> routed_output;
    __strong id<MTLBuffer> shared_output;
    MoonlightStats stats;
    int *kv_lengths;
    int *host_unique_experts;
    int *host_expert_row_counts;
    float *host_activation;
    float *host_norm;
    float *host_temporary;
    int *host_route_trace;
    int last_forward_rows;
    int position;
};

typedef struct {
    int weight_index;
    int scale_index;
    int format;
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
} MoonlightMatrix;

static double monotonic_seconds(void);

static int set_error(char *error, size_t error_size, const char *format, ...) {
    va_list arguments;
    if (error && error_size) {
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int multiply_size(size_t left, size_t right, size_t *result) {
    if (right && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static id<MTLComputePipelineState> make_pipeline(MoonlightModel *model,
                                                 NSString *name,
                                                 NSError **error) {
    id<MTLFunction> function = [model->library newFunctionWithName:name];
    if (!function) return nil;
    return [model->device newComputePipelineStateWithFunction:function error:error];
}

static int weight_index(const MoonlightModel *model, const char *name,
                        MoonlightTensorInfo *info) {
    int index = moonlight_model_data_find(model->data, name);
    if (index < 0) return -1;
    if (info) *info = moonlight_model_data_tensor(model->data, index);
    return index;
}

static int resolve_matrix(MoonlightModel *model, const char *name,
                          int input_width, int output_width,
                          MoonlightMatrix *matrix) {
    MoonlightTensorInfo info;
    MoonlightTensorInfo scale_info;
    char scale_name[512];

    memset(matrix, 0, sizeof(*matrix));
    matrix->scale_index = -1;
    matrix->weight_index = weight_index(model, name, &info);
    if (matrix->weight_index < 0) return 0;
    if (info.dtype == MOONLIGHT_TENSOR_F32) {
        if (info.element_count != (int64_t)input_width * output_width) return 0;
        matrix->pipeline = model->matmul_f32;
        return 1;
    }
    if (info.dtype != MOONLIGHT_TENSOR_U8 ||
        snprintf(scale_name, sizeof(scale_name), "%s.qs", name) >=
        (int)sizeof(scale_name)) return 0;
    matrix->scale_index = weight_index(model, scale_name, &scale_info);
    if (matrix->scale_index < 0 || scale_info.dtype != MOONLIGHT_TENSOR_F32 ||
        scale_info.element_count != output_width) return 0;
    if (info.byte_count == (int64_t)output_width * input_width) {
        matrix->format = 1;
        matrix->pipeline = model->matmul_q8;
        return 1;
    }
    if (info.byte_count == (int64_t)output_width * ((input_width + 1) / 2)) {
        matrix->format = 2;
        matrix->pipeline = model->matmul_q4;
        return 1;
    }
    return 0;
}

static void encode_matmul_at(MoonlightModel *model,
                             id<MTLComputeCommandEncoder> encoder,
                             const MoonlightMatrix *matrix,
                             id<MTLBuffer> input, NSUInteger input_offset,
                             id<MTLBuffer> output, NSUInteger output_offset,
                             int rows, int input_width, int output_width) {
    uint32_t shape[3] = {(uint32_t)output_width, (uint32_t)input_width,
                         (uint32_t)rows};
    id<MTLComputePipelineState> pipeline = matrix->format == 1
        ? model->matmul_q8_reduction : matrix->pipeline;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:model->weights[matrix->weight_index].buffer offset:0 atIndex:0];
    if (matrix->scale_index >= 0) {
        [encoder setBuffer:model->weights[matrix->scale_index].buffer offset:0 atIndex:1];
        [encoder setBuffer:input offset:input_offset atIndex:2];
        [encoder setBuffer:output offset:output_offset atIndex:3];
        [encoder setBytes:shape length:sizeof(shape) atIndex:4];
    } else {
        [encoder setBuffer:input offset:input_offset atIndex:1];
        [encoder setBuffer:output offset:output_offset atIndex:2];
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    }
    if (matrix->format == 1) {
        NSUInteger threads = MIN((NSUInteger)32,
                                 pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake(output_width, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        NSUInteger columns = MIN((NSUInteger)32, (NSUInteger)output_width);
        NSUInteger row_threads = MIN((NSUInteger)8, (NSUInteger)rows);
        [encoder dispatchThreads:MTLSizeMake(output_width, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(columns, row_threads, 1)];
    }
}

static void encode_matmul(MoonlightModel *model,
                          id<MTLComputeCommandEncoder> encoder,
                          const MoonlightMatrix *matrix,
                          id<MTLBuffer> input, id<MTLBuffer> output,
                          int rows, int input_width, int output_width) {
    encode_matmul_at(model, encoder, matrix, input, 0, output, 0,
                     rows, input_width, output_width);
}

static void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                        id<MTLComputePipelineState> pipeline,
                        NSUInteger count) {
    NSUInteger threads = MIN((NSUInteger)256,
                             pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MAX(threads, (NSUInteger)1), 1, 1)];
}

static int encode_embedding_buffer(MoonlightSession *session,
                                   id<MTLComputeCommandEncoder> encoder,
                                   const int *ids, int count,
                                   id<MTLBuffer> output) {
    MoonlightTensorInfo info;
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    int index = weight_index(session->model, "model.embed_tokens.weight", &info);
    if (index < 0 || info.dtype != MOONLIGHT_TENSOR_F32 ||
        info.element_count != (int64_t)config->vocab_size * config->hidden_size)
        return 0;
    memcpy(session->token_ids.contents, ids, (size_t)count * sizeof(int32_t));
    uint32_t shape[2] = {(uint32_t)count, (uint32_t)config->hidden_size};
    [encoder setComputePipelineState:session->model->embed_f32];
    [encoder setBuffer:session->model->weights[index].buffer offset:0 atIndex:0];
    [encoder setBuffer:session->token_ids offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    dispatch_1d(encoder, session->model->embed_f32,
                (NSUInteger)count * config->hidden_size);
    return 1;
}

static int encode_rmsnorm_buffer(MoonlightSession *session,
                                 id<MTLComputeCommandEncoder> encoder,
                                 id<MTLBuffer> input, NSUInteger input_offset,
                                 const char *weight_name, int rows, int width,
                                 id<MTLBuffer> output,
                                 NSUInteger output_offset) {
    MoonlightTensorInfo info;
    int index = weight_index(session->model, weight_name, &info);
    if (index < 0 || info.dtype != MOONLIGHT_TENSOR_F32 ||
        info.element_count != width) return 0;
    uint32_t shape[2] = {(uint32_t)rows, (uint32_t)width};
    float epsilon =
        moonlight_model_data_config(session->model->data)->rms_norm_epsilon;
    [encoder setComputePipelineState:session->model->rmsnorm];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:session->model->weights[index].buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
    dispatch_1d(encoder, session->model->rmsnorm, (NSUInteger)rows);
    return 1;
}

static void encode_add_buffer(MoonlightSession *session,
                              id<MTLComputeCommandEncoder> encoder,
                              id<MTLBuffer> left, id<MTLBuffer> right,
                              id<MTLBuffer> output, uint32_t count) {
    [encoder setComputePipelineState:session->model->add_f32];
    [encoder setBuffer:left offset:0 atIndex:0];
    [encoder setBuffer:right offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    dispatch_1d(encoder, session->model->add_f32, count);
}

static void encode_silu_buffer(MoonlightSession *session,
                               id<MTLComputeCommandEncoder> encoder,
                               id<MTLBuffer> gate, id<MTLBuffer> up,
                               id<MTLBuffer> output, uint32_t count) {
    [encoder setComputePipelineState:session->model->silu_multiply];
    [encoder setBuffer:gate offset:0 atIndex:0];
    [encoder setBuffer:up offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    dispatch_1d(encoder, session->model->silu_multiply, count);
}

static int encode_attention_buffer(MoonlightSession *session,
                                   id<MTLComputeCommandEncoder> encoder,
                                   int layer, id<MTLBuffer> input,
                                   int rows, int position,
                                   id<MTLBuffer> output) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix query_matrix;
    MoonlightMatrix kv_a_matrix;
    MoonlightMatrix kv_b_matrix;
    MoonlightMatrix output_matrix;
    MoonlightTensorInfo norm_info;
    char name[512];
    int norm_index;
    int query_width = config->head_count *
                      (config->qk_nope_dim + config->qk_rope_dim);
    int compressed_width = config->kv_lora_rank + config->qk_rope_dim;
    int context_width = config->head_count * config->value_dim;
    if (position != session->kv_lengths[layer] || config->q_lora_rank)
        return 0;
    snprintf(name, sizeof(name),
             "model.layers.%d.self_attn.q_proj.weight", layer);
    if (!resolve_matrix(session->model, name, config->hidden_size,
                        query_width, &query_matrix)) return 0;
    snprintf(name, sizeof(name),
             "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", layer);
    if (!resolve_matrix(session->model, name, config->hidden_size,
                        compressed_width, &kv_a_matrix)) return 0;
    snprintf(name, sizeof(name),
             "model.layers.%d.self_attn.kv_b_proj.weight", layer);
    if (!resolve_matrix(session->model, name, config->kv_lora_rank,
                        config->head_count *
                        (config->qk_nope_dim + config->value_dim),
                        &kv_b_matrix)) return 0;
    snprintf(name, sizeof(name),
             "model.layers.%d.self_attn.o_proj.weight", layer);
    if (!resolve_matrix(session->model, name, context_width,
                        config->hidden_size, &output_matrix)) return 0;
    snprintf(name, sizeof(name),
             "model.layers.%d.self_attn.kv_a_layernorm.weight", layer);
    norm_index = weight_index(session->model, name, &norm_info);
    if (norm_index < 0 || norm_info.dtype != MOONLIGHT_TENSOR_F32 ||
        norm_info.element_count != config->kv_lora_rank) return 0;

    encode_matmul(session->model, encoder, &query_matrix, input,
                  session->query, rows, config->hidden_size, query_width);
    encode_matmul(session->model, encoder, &kv_a_matrix, input,
                  session->compressed, rows, config->hidden_size,
                  compressed_width);
    uint32_t cache_shape[6] = {
        (uint32_t)rows, (uint32_t)config->kv_lora_rank,
        (uint32_t)config->qk_rope_dim,
        (uint32_t)session->options.context_size,
        (uint32_t)layer, (uint32_t)position,
    };
    [encoder setComputePipelineState:session->model->cache_append];
    [encoder setBuffer:session->compressed offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[norm_index].buffer offset:0 atIndex:1];
    [encoder setBuffer:session->latent_kv offset:0 atIndex:2];
    [encoder setBuffer:session->rope_kv offset:0 atIndex:3];
    [encoder setBytes:cache_shape length:sizeof(cache_shape) atIndex:4];
    [encoder setBytes:&config->rms_norm_epsilon
                length:sizeof(config->rms_norm_epsilon) atIndex:5];
    [encoder setBytes:&config->rope_theta length:sizeof(config->rope_theta) atIndex:6];
    dispatch_1d(encoder, session->model->cache_append, (NSUInteger)rows);

    uint32_t rope_shape[5] = {
        (uint32_t)rows, (uint32_t)config->head_count,
        (uint32_t)config->qk_nope_dim, (uint32_t)config->qk_rope_dim,
        (uint32_t)position,
    };
    [encoder setComputePipelineState:session->model->rope_query];
    [encoder setBuffer:session->query offset:0 atIndex:0];
    [encoder setBytes:rope_shape length:sizeof(rope_shape) atIndex:1];
    [encoder setBytes:&config->rope_theta length:sizeof(config->rope_theta) atIndex:2];
    NSUInteger query_count = (NSUInteger)rows * config->head_count;
    dispatch_1d(encoder, session->model->rope_query, query_count);

    uint32_t absorb_shape[7] = {
        (uint32_t)rows, (uint32_t)config->head_count,
        (uint32_t)config->qk_nope_dim, (uint32_t)config->qk_rope_dim,
        (uint32_t)config->value_dim, (uint32_t)config->kv_lora_rank,
        (uint32_t)kv_b_matrix.format,
    };
    float attention_scale = 1.0f /
        sqrtf((float)(config->qk_nope_dim + config->qk_rope_dim));
    id<MTLBuffer> kv_b_scale = kv_b_matrix.scale_index >= 0
        ? session->model->weights[kv_b_matrix.scale_index].buffer
        : session->model->weights[kv_b_matrix.weight_index].buffer;
    [encoder setComputePipelineState:session->model->attention_absorb_query];
    [encoder setBuffer:session->query offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[kv_b_matrix.weight_index].buffer
             offset:0 atIndex:1];
    [encoder setBuffer:kv_b_scale offset:0 atIndex:2];
    [encoder setBuffer:session->absorbed_query offset:0 atIndex:3];
    [encoder setBytes:absorb_shape length:sizeof(absorb_shape) atIndex:4];
    dispatch_1d(encoder, session->model->attention_absorb_query,
                query_count * config->kv_lora_rank);

    uint32_t score_shape[8] = {
        (uint32_t)rows, (uint32_t)config->head_count,
        (uint32_t)config->qk_nope_dim, (uint32_t)config->qk_rope_dim,
        (uint32_t)config->kv_lora_rank,
        (uint32_t)session->options.context_size,
        (uint32_t)layer, (uint32_t)position,
    };
    [encoder setComputePipelineState:session->model->attention_scores];
    [encoder setBuffer:session->absorbed_query offset:0 atIndex:0];
    [encoder setBuffer:session->query offset:0 atIndex:1];
    [encoder setBuffer:session->latent_kv offset:0 atIndex:2];
    [encoder setBuffer:session->rope_kv offset:0 atIndex:3];
    [encoder setBuffer:session->attention_scores offset:0 atIndex:4];
    [encoder setBytes:score_shape length:sizeof(score_shape) atIndex:5];
    [encoder setBytes:&attention_scale length:sizeof(attention_scale) atIndex:6];
    NSUInteger key_count = (NSUInteger)position + rows;
    [encoder dispatchThreads:MTLSizeMake(key_count, query_count, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)32, key_count),
                                          MIN((NSUInteger)8, query_count), 1)];

    uint32_t latent_shape[6] = {
        (uint32_t)rows, (uint32_t)config->head_count,
        (uint32_t)config->kv_lora_rank,
        (uint32_t)session->options.context_size,
        (uint32_t)layer, (uint32_t)position,
    };
    [encoder setComputePipelineState:session->model->attention_latent_context];
    [encoder setBuffer:session->attention_scores offset:0 atIndex:0];
    [encoder setBuffer:session->latent_kv offset:0 atIndex:1];
    [encoder setBuffer:session->latent_context offset:0 atIndex:2];
    [encoder setBytes:latent_shape length:sizeof(latent_shape) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(query_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    uint32_t value_shape[6] = {
        (uint32_t)rows, (uint32_t)config->head_count,
        (uint32_t)config->qk_nope_dim, (uint32_t)config->value_dim,
        (uint32_t)config->kv_lora_rank, (uint32_t)kv_b_matrix.format,
    };
    [encoder setComputePipelineState:session->model->attention_decode_value];
    [encoder setBuffer:session->latent_context offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[kv_b_matrix.weight_index].buffer
             offset:0 atIndex:1];
    [encoder setBuffer:kv_b_scale offset:0 atIndex:2];
    [encoder setBuffer:session->context offset:0 atIndex:3];
    [encoder setBytes:value_shape length:sizeof(value_shape) atIndex:4];
    [encoder dispatchThreadgroups:
        MTLSizeMake(query_count * config->value_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    encode_matmul(session->model, encoder, &output_matrix,
                  session->context, output, rows, context_width,
                  config->hidden_size);
    session->kv_lengths[layer] = position + rows;
    return 1;
}

static int encode_dense_residual_buffer(
        MoonlightSession *session, id<MTLComputeCommandEncoder> encoder,
        int layer, int rows) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix gate_matrix;
    MoonlightMatrix up_matrix;
    MoonlightMatrix down_matrix;
    char name[512];
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->dense_intermediate_size, &gate_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.up_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->dense_intermediate_size, &up_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.down_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->dense_intermediate_size,
                        config->hidden_size, &down_matrix)) return 0;
    encode_matmul(session->model, encoder, &gate_matrix,
                  session->activation_b, session->expert_gate, rows,
                  config->hidden_size, config->dense_intermediate_size);
    encode_matmul(session->model, encoder, &up_matrix,
                  session->activation_b, session->expert_up, rows,
                  config->hidden_size, config->dense_intermediate_size);
    encode_silu_buffer(session, encoder, session->expert_gate,
                       session->expert_up, session->expert_hidden,
                       (uint32_t)((int64_t)rows *
                                  config->dense_intermediate_size));
    encode_matmul(session->model, encoder, &down_matrix,
                  session->expert_hidden, session->expert_output, rows,
                  config->dense_intermediate_size, config->hidden_size);
    encode_add_buffer(session, encoder, session->routed_output,
                      session->expert_output, session->activation_a,
                      (uint32_t)((int64_t)rows * config->hidden_size));
    return 1;
}

static int encode_router_buffer(MoonlightSession *session,
                                id<MTLComputeCommandEncoder> encoder,
                                int layer, int rows) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix router_matrix;
    MoonlightTensorInfo bias_info;
    char name[512];
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->expert_count, &router_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate.e_score_correction_bias", layer) >=
            (int)sizeof(name)) return 0;
    int bias_index = weight_index(session->model, name, &bias_info);
    if (bias_index < 0 || bias_info.dtype != MOONLIGHT_TENSOR_F32 ||
        bias_info.element_count != config->expert_count) return 0;
    encode_matmul(session->model, encoder, &router_matrix,
                  session->activation_b, session->router_logits, rows,
                  config->hidden_size, config->expert_count);
    uint32_t shape[4] = {
        (uint32_t)rows, (uint32_t)config->expert_count,
        (uint32_t)config->experts_per_token,
        (uint32_t)config->normalize_topk,
    };
    [encoder setComputePipelineState:session->model->route_sigmoid_topk];
    [encoder setBuffer:session->router_logits offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[bias_index].buffer
             offset:0 atIndex:1];
    [encoder setBuffer:session->route_ids offset:0 atIndex:2];
    [encoder setBuffer:session->route_weights offset:0 atIndex:3];
    [encoder setBytes:shape length:sizeof(shape) atIndex:4];
    [encoder setBytes:&config->routed_scale length:sizeof(config->routed_scale)
              atIndex:5];
    dispatch_1d(encoder, session->model->route_sigmoid_topk, (NSUInteger)rows);
    return 1;
}

static int prepare_route_tables(MoonlightSession *session, int layer,
                                int rows, int *unique_count) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    int *selected = session->route_ids.contents;
    int *all_expert_rows = session->expert_rows.contents;
    int *all_expert_slots = session->expert_slots.contents;
    unsigned char seen[256] = {0};
    *unique_count = 0;
    memcpy(session->host_route_trace +
               (int64_t)layer * session->options.max_batch *
                   config->experts_per_token,
           selected,
           (size_t)rows * config->experts_per_token * sizeof(*selected));
    memset(session->host_expert_row_counts, 0,
           (size_t)config->expert_count *
               sizeof(*session->host_expert_row_counts));
    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < config->experts_per_token; ++slot) {
            int expert = selected[(int64_t)row * config->experts_per_token +
                                  slot];
            if (expert < 0 || expert >= config->expert_count) return 0;
            if (!seen[expert]) {
                seen[expert] = 1;
                session->host_unique_experts[(*unique_count)++] = expert;
            }
            int expert_row = session->host_expert_row_counts[expert]++;
            int64_t offset = (int64_t)expert * session->options.max_batch +
                             expert_row;
            all_expert_rows[offset] = row;
            all_expert_slots[offset] = slot;
        }
    }
    return 1;
}

static int encode_moe_residual_buffer(
        MoonlightSession *session, id<MTLComputeCommandEncoder> encoder,
        int layer, int rows, int unique_count) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix gate_matrix;
    MoonlightMatrix up_matrix;
    MoonlightMatrix down_matrix;
    char name[512];
    uint32_t hidden_count =
        (uint32_t)((int64_t)rows * config->hidden_size);
    [encoder setComputePipelineState:session->model->clear_f32];
    [encoder setBuffer:session->activation_a offset:0 atIndex:0];
    [encoder setBytes:&hidden_count length:sizeof(hidden_count) atIndex:1];
    dispatch_1d(encoder, session->model->clear_f32, hidden_count);

    for (int unique = 0; unique < unique_count; ++unique) {
        int expert = session->host_unique_experts[unique];
        int expert_row_count = session->host_expert_row_counts[expert];
        NSUInteger index_offset =
            (NSUInteger)expert * session->options.max_batch * sizeof(int32_t);
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.gate_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name, config->hidden_size,
                            config->moe_intermediate_size,
                            &gate_matrix)) return 0;
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.up_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name, config->hidden_size,
                            config->moe_intermediate_size,
                            &up_matrix)) return 0;
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.down_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name,
                            config->moe_intermediate_size,
                            config->hidden_size, &down_matrix)) return 0;
        uint32_t gather_shape[2] = {(uint32_t)expert_row_count,
                                    (uint32_t)config->hidden_size};
        [encoder setComputePipelineState:session->model->gather_rows];
        [encoder setBuffer:session->activation_b offset:0 atIndex:0];
        [encoder setBuffer:session->expert_rows offset:index_offset atIndex:1];
        [encoder setBuffer:session->expert_input offset:0 atIndex:2];
        [encoder setBytes:gather_shape length:sizeof(gather_shape) atIndex:3];
        dispatch_1d(encoder, session->model->gather_rows,
                    (NSUInteger)expert_row_count * config->hidden_size);
        encode_matmul(session->model, encoder, &gate_matrix,
                      session->expert_input, session->expert_gate,
                      expert_row_count, config->hidden_size,
                      config->moe_intermediate_size);
        encode_matmul(session->model, encoder, &up_matrix,
                      session->expert_input, session->expert_up,
                      expert_row_count, config->hidden_size,
                      config->moe_intermediate_size);
        encode_silu_buffer(session, encoder, session->expert_gate,
                           session->expert_up, session->expert_hidden,
                           (uint32_t)((int64_t)expert_row_count *
                                      config->moe_intermediate_size));
        encode_matmul(session->model, encoder, &down_matrix,
                      session->expert_hidden, session->expert_output,
                      expert_row_count, config->moe_intermediate_size,
                      config->hidden_size);
        uint32_t scatter_shape[3] = {
            (uint32_t)expert_row_count, (uint32_t)config->hidden_size,
            (uint32_t)config->experts_per_token,
        };
        [encoder setComputePipelineState:session->model->scatter_expert];
        [encoder setBuffer:session->expert_output offset:0 atIndex:0];
        [encoder setBuffer:session->expert_rows offset:index_offset atIndex:1];
        [encoder setBuffer:session->expert_slots offset:index_offset atIndex:2];
        [encoder setBuffer:session->route_weights offset:0 atIndex:3];
        [encoder setBuffer:session->activation_a offset:0 atIndex:4];
        [encoder setBytes:scatter_shape length:sizeof(scatter_shape) atIndex:5];
        dispatch_1d(encoder, session->model->scatter_expert,
                    (NSUInteger)expert_row_count * config->hidden_size);
    }

    int shared_width =
        config->moe_intermediate_size * config->shared_expert_count;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.gate_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        shared_width, &gate_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.up_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        shared_width, &up_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.down_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, shared_width,
                        config->hidden_size, &down_matrix)) return 0;
    encode_matmul(session->model, encoder, &gate_matrix,
                  session->activation_b, session->expert_gate, rows,
                  config->hidden_size, shared_width);
    encode_matmul(session->model, encoder, &up_matrix,
                  session->activation_b, session->expert_up, rows,
                  config->hidden_size, shared_width);
    encode_silu_buffer(session, encoder, session->expert_gate,
                       session->expert_up, session->expert_hidden,
                       (uint32_t)((int64_t)rows * shared_width));
    encode_matmul(session->model, encoder, &down_matrix,
                  session->expert_hidden, session->shared_output, rows,
                  shared_width, config->hidden_size);
    encode_add_buffer(session, encoder, session->activation_a,
                      session->shared_output, session->expert_output,
                      hidden_count);
    encode_add_buffer(session, encoder, session->routed_output,
                      session->expert_output, session->activation_a,
                      hidden_count);
    return 1;
}

static int encode_attention_block(MoonlightSession *session,
                                  id<MTLComputeCommandEncoder> encoder,
                                  int layer, int rows, int position,
                                  int encode_router) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    char name[512];
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.input_layernorm.weight", layer) >=
            (int)sizeof(name) ||
        !encode_rmsnorm_buffer(session, encoder, session->activation_a, 0,
                               name, rows, config->hidden_size,
                               session->activation_b, 0)) return 0;
    if (!encode_attention_buffer(session, encoder, layer,
                                 session->activation_b, rows, position,
                                 session->expert_output)) return 0;
    uint32_t hidden_count =
        (uint32_t)((int64_t)rows * config->hidden_size);
    encode_add_buffer(session, encoder, session->activation_a,
                      session->expert_output, session->routed_output,
                      hidden_count);
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.post_attention_layernorm.weight", layer) >=
            (int)sizeof(name) ||
        !encode_rmsnorm_buffer(session, encoder, session->routed_output, 0,
                               name, rows, config->hidden_size,
                               session->activation_b, 0)) return 0;
    return !encode_router || encode_router_buffer(session, encoder, layer, rows);
}

static int encode_final_logits(MoonlightSession *session,
                               id<MTLComputeCommandEncoder> encoder,
                               int rows) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix matrix;
    NSUInteger input_offset =
        (NSUInteger)(rows - 1) * config->hidden_size * sizeof(float);
    if (!encode_rmsnorm_buffer(session, encoder, session->activation_a,
                               input_offset, "model.norm.weight", 1,
                               config->hidden_size, session->activation_b, 0) ||
        !resolve_matrix(session->model, "lm_head.weight", config->hidden_size,
                        config->vocab_size, &matrix)) return 0;
    encode_matmul_at(session->model, encoder, &matrix, session->activation_b, 0,
                     session->logits, 0, 1, config->hidden_size,
                     config->vocab_size);
    return 1;
}

static int finish_command(MoonlightSession *session, id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    session->stats.command_buffers++;
    return command.status == MTLCommandBufferStatusCompleted;
}

static id<MTLBuffer> allocate_buffer(MoonlightSession *session, size_t length,
                                     char *error, size_t error_size) {
    id<MTLBuffer> buffer;
    if (!length) {
        set_error(error, error_size, "cannot allocate an empty Metal buffer");
        return nil;
    }
    if (length > session->model->device.maxBufferLength) {
        set_error(error, error_size, "Metal buffer exceeds device limit: %zu bytes",
                  length);
        return nil;
    }
    buffer = [session->model->device newBufferWithLength:length
                                                  options:MTLResourceStorageModeShared];
    if (!buffer) {
        set_error(error, error_size, "Metal allocation failed: %zu bytes", length);
        return nil;
    }
    session->stats.buffer_allocations++;
    session->stats.resident_bytes += length;
    return buffer;
}

int moonlight_model_open(MoonlightModel **out, const char *path,
                         char *error, size_t error_size) {
    MoonlightModel *model;
    NSError *metal_error = nil;
    NSString *source;
    uint64_t model_bytes;

    if (!out) return set_error(error, error_size, "model output is required");
    *out = NULL;
    model = calloc(1, sizeof(*model));
    if (!model) return set_error(error, error_size, "out of memory opening model");
    if (!moonlight_model_data_open(&model->data, path, error, error_size)) {
        free(model);
        return 0;
    }
    model->device = MTLCreateSystemDefaultDevice();
    if (!model->device) {
        moonlight_model_close(model);
        return set_error(error, error_size, "Metal device is unavailable");
    }
    model->queue = [model->device newCommandQueue];
    source = [[NSString alloc] initWithBytes:moonlight_kernels_metal
                                      length:moonlight_kernels_metal_len
                                    encoding:NSUTF8StringEncoding];
    model->library = [model->device newLibraryWithSource:source
                                                 options:nil error:&metal_error];
    if (!model->queue || !model->library) {
        NSString *message = metal_error.localizedDescription ?: @"unknown error";
        set_error(error, error_size, "Moonlight Metal initialization failed: %s",
                  message.UTF8String);
        moonlight_model_close(model);
        return 0;
    }
    model->noop = make_pipeline(model, @"moonlight_noop", &metal_error);
    model->embed_f32 = make_pipeline(model, @"moonlight_embed_f32", &metal_error);
    model->rmsnorm = make_pipeline(model, @"moonlight_rmsnorm", &metal_error);
    model->matmul_f32 = make_pipeline(model, @"moonlight_matmul_f32", &metal_error);
    model->matmul_q8 = make_pipeline(model, @"moonlight_matmul_q8", &metal_error);
    model->matmul_q8_reduction =
        make_pipeline(model, @"moonlight_matmul_q8_reduction", &metal_error);
    model->matmul_q4 = make_pipeline(model, @"moonlight_matmul_q4", &metal_error);
    model->cache_append = make_pipeline(model, @"moonlight_cache_append", &metal_error);
    model->rope_query = make_pipeline(model, @"moonlight_rope_query", &metal_error);
    model->attention_absorbed =
        make_pipeline(model, @"moonlight_attention_absorbed", &metal_error);
    model->attention_absorb_query =
        make_pipeline(model, @"moonlight_attention_absorb_query", &metal_error);
    model->attention_scores =
        make_pipeline(model, @"moonlight_attention_scores", &metal_error);
    model->attention_latent_context = make_pipeline(
        model, @"moonlight_attention_latent_context", &metal_error);
    model->attention_decode_value = make_pipeline(
        model, @"moonlight_attention_decode_value", &metal_error);
    model->route_sigmoid_topk =
        make_pipeline(model, @"moonlight_route_sigmoid_topk", &metal_error);
    model->clear_f32 = make_pipeline(model, @"moonlight_clear_f32", &metal_error);
    model->gather_rows = make_pipeline(model, @"moonlight_gather_rows", &metal_error);
    model->silu_multiply =
        make_pipeline(model, @"moonlight_silu_multiply", &metal_error);
    model->scatter_expert =
        make_pipeline(model, @"moonlight_scatter_expert", &metal_error);
    model->add_f32 = make_pipeline(model, @"moonlight_add_f32", &metal_error);
    if (!model->noop || !model->embed_f32 || !model->rmsnorm ||
        !model->matmul_f32 || !model->matmul_q8 ||
        !model->matmul_q8_reduction || !model->matmul_q4 ||
        !model->cache_append || !model->rope_query || !model->attention_absorbed ||
        !model->attention_absorb_query || !model->attention_scores ||
        !model->attention_latent_context || !model->attention_decode_value ||
        !model->route_sigmoid_topk || !model->clear_f32 || !model->gather_rows ||
        !model->silu_multiply || !model->scatter_expert || !model->add_f32) {
        NSString *message = metal_error.localizedDescription ?: @"unknown error";
        set_error(error, error_size, "Moonlight pipeline creation failed: %s",
                  message.UTF8String);
        moonlight_model_close(model);
        return 0;
    }

    model_bytes = moonlight_model_data_bytes(model->data);
    if (model->device.recommendedMaxWorkingSetSize &&
        model_bytes > model->device.recommendedMaxWorkingSetSize * 9 / 10) {
        set_error(error, error_size,
                  "model requires %.2f GiB, above the Metal working-set budget %.2f GiB",
                  model_bytes / 1073741824.0,
                  model->device.recommendedMaxWorkingSetSize / 1073741824.0);
        moonlight_model_close(model);
        return 0;
    }
    model->weight_count = moonlight_model_data_tensor_count(model->data);
    model->weights = calloc((size_t)model->weight_count, sizeof(*model->weights));
    if (!model->weights) {
        moonlight_model_close(model);
        return set_error(error, error_size, "out of memory indexing Metal weights");
    }
    for (int index = 0; index < model->weight_count; ++index) {
        MoonlightTensorInfo tensor = moonlight_model_data_tensor(model->data, index);
        if (tensor.byte_count <= 0 ||
            (uint64_t)tensor.byte_count > model->device.maxBufferLength) {
            set_error(error, error_size, "tensor %s exceeds the Metal buffer limit",
                      tensor.name ? tensor.name : "(unnamed)");
            moonlight_model_close(model);
            return 0;
        }
        model->weights[index].buffer =
            [model->device newBufferWithLength:(NSUInteger)tensor.byte_count
                                        options:MTLResourceStorageModeShared];
        if (!model->weights[index].buffer ||
            !moonlight_model_data_read_raw(model->data, index,
                model->weights[index].buffer.contents,
                (size_t)tensor.byte_count)) {
            set_error(error, error_size, "cannot upload tensor %s", tensor.name);
            moonlight_model_close(model);
            return 0;
        }
        model->resident_bytes += (uint64_t)tensor.byte_count;
    }
    *out = model;
    return 1;
}

void moonlight_model_close(MoonlightModel *model) {
    if (!model) return;
    if (model->weights) {
        for (int index = 0; index < model->weight_count; ++index)
            model->weights[index].buffer = nil;
    }
    free(model->weights);
    model->noop = nil;
    model->embed_f32 = nil;
    model->rmsnorm = nil;
    model->matmul_f32 = nil;
    model->matmul_q8 = nil;
    model->matmul_q8_reduction = nil;
    model->matmul_q4 = nil;
    model->cache_append = nil;
    model->rope_query = nil;
    model->attention_absorbed = nil;
    model->attention_absorb_query = nil;
    model->attention_scores = nil;
    model->attention_latent_context = nil;
    model->attention_decode_value = nil;
    model->route_sigmoid_topk = nil;
    model->clear_f32 = nil;
    model->gather_rows = nil;
    model->silu_multiply = nil;
    model->scatter_expert = nil;
    model->add_f32 = nil;
    model->library = nil;
    model->queue = nil;
    model->device = nil;
    moonlight_model_data_close(model->data);
    free(model);
}

const char *moonlight_device_name(const MoonlightModel *model) {
    return model && model->device ? model->device.name.UTF8String : "(none)";
}

MoonlightModelInfo moonlight_model_info(const MoonlightModel *model) {
    MoonlightModelInfo result = {0};
    if (!model) return result;
    const MoonlightConfig *config = moonlight_model_data_config(model->data);
    result.hidden_size = config->hidden_size;
    result.layer_count = config->layer_count;
    result.vocab_size = config->vocab_size;
    result.head_count = config->head_count;
    result.kv_lora_rank = config->kv_lora_rank;
    result.qk_nope_dim = config->qk_nope_dim;
    result.qk_rope_dim = config->qk_rope_dim;
    result.value_dim = config->value_dim;
    result.expert_count = config->expert_count;
    result.experts_per_token = config->experts_per_token;
    result.moe_intermediate_size = config->moe_intermediate_size;
    result.shared_expert_count = config->shared_expert_count;
    result.first_dense_layer_count = config->first_dense_layer_count;
    result.rms_norm_epsilon = config->rms_norm_epsilon;
    result.rope_theta = config->rope_theta;
    result.routed_scale = config->routed_scale;
    result.resident_bytes = model->resident_bytes;
    return result;
}

int moonlight_session_create(MoonlightSession **out, MoonlightModel *model,
                             const MoonlightOptions *options,
                             char *error, size_t error_size) {
    MoonlightSession *session;
    const MoonlightConfig *config;
    size_t activation_size;
    size_t latent_size;
    size_t rope_size;
    size_t token_size;
    size_t logit_size;
    size_t query_size;
    size_t compressed_size;
    size_t context_size;
    size_t absorbed_query_size;
    size_t attention_score_size;
    size_t latent_context_size;
    size_t router_size;
    size_t route_size;
    size_t row_index_size;
    size_t expert_input_size;
    size_t expert_intermediate_size;
    size_t expert_output_size;
    int shared_width;

    if (!out || !model || !options)
        return set_error(error, error_size, "model and session options are required");
    *out = NULL;
    if (options->context_size <= 0 || options->max_batch <= 0 ||
        options->max_batch > options->context_size)
        return set_error(error, error_size, "invalid context or batch size");
    config = moonlight_model_data_config(model->data);
    if (!multiply_size((size_t)options->max_batch, config->hidden_size,
                       &activation_size) ||
        !multiply_size(activation_size, sizeof(float), &activation_size) ||
        !multiply_size((size_t)config->layer_count, options->context_size,
                       &latent_size) ||
        !multiply_size(latent_size, config->kv_lora_rank, &latent_size) ||
        !multiply_size(latent_size, sizeof(float), &latent_size) ||
        !multiply_size((size_t)config->layer_count, options->context_size,
                       &rope_size) ||
        !multiply_size(rope_size, config->qk_rope_dim, &rope_size) ||
        !multiply_size(rope_size, sizeof(float), &rope_size) ||
        !multiply_size((size_t)options->max_batch, sizeof(int32_t), &token_size) ||
        !multiply_size((size_t)options->max_batch, config->vocab_size,
                       &logit_size) ||
        !multiply_size(logit_size, sizeof(float), &logit_size) ||
        !multiply_size((size_t)options->max_batch, config->head_count,
                       &query_size) ||
        !multiply_size(query_size, config->qk_nope_dim + config->qk_rope_dim,
                       &query_size) ||
        !multiply_size(query_size, sizeof(float), &query_size) ||
        !multiply_size((size_t)options->max_batch,
                       config->kv_lora_rank + config->qk_rope_dim,
                       &compressed_size) ||
        !multiply_size(compressed_size, sizeof(float), &compressed_size) ||
        !multiply_size((size_t)options->max_batch, config->head_count,
                       &context_size) ||
        !multiply_size(context_size, config->value_dim, &context_size) ||
        !multiply_size(context_size, sizeof(float), &context_size) ||
        !multiply_size((size_t)options->max_batch, config->head_count,
                       &absorbed_query_size) ||
        !multiply_size(absorbed_query_size, config->kv_lora_rank,
                       &absorbed_query_size) ||
        !multiply_size(absorbed_query_size, sizeof(float),
                       &absorbed_query_size) ||
        !multiply_size((size_t)options->max_batch, config->head_count,
                       &attention_score_size) ||
        !multiply_size(attention_score_size, options->context_size,
                       &attention_score_size) ||
        !multiply_size(attention_score_size, sizeof(float),
                       &attention_score_size) ||
        !multiply_size((size_t)options->max_batch, config->head_count,
                       &latent_context_size) ||
        !multiply_size(latent_context_size, config->kv_lora_rank,
                       &latent_context_size) ||
        !multiply_size(latent_context_size, sizeof(float),
                       &latent_context_size) ||
        !multiply_size((size_t)options->max_batch, config->expert_count,
                       &router_size) ||
        !multiply_size(router_size, sizeof(float), &router_size) ||
        !multiply_size((size_t)options->max_batch, config->experts_per_token,
                       &route_size) ||
        !multiply_size(route_size, sizeof(float), &route_size) ||
        !multiply_size((size_t)options->max_batch, config->expert_count,
                       &row_index_size) ||
        !multiply_size(row_index_size, sizeof(int32_t), &row_index_size) ||
        !multiply_size((size_t)options->max_batch, config->hidden_size,
                       &expert_input_size) ||
        !multiply_size(expert_input_size, sizeof(float), &expert_input_size))
        return set_error(error, error_size, "Moonlight session size overflow");
    shared_width = config->moe_intermediate_size * config->shared_expert_count;
    int maximum_intermediate = MAX(config->moe_intermediate_size, shared_width);
    maximum_intermediate = MAX(maximum_intermediate,
                               config->dense_intermediate_size);
    if (!multiply_size((size_t)options->max_batch,
                       (size_t)maximum_intermediate,
                       &expert_intermediate_size) ||
        !multiply_size(expert_intermediate_size, sizeof(float),
                       &expert_intermediate_size) ||
        !multiply_size((size_t)options->max_batch, config->hidden_size,
                       &expert_output_size) ||
        !multiply_size(expert_output_size, sizeof(float), &expert_output_size))
        return set_error(error, error_size, "Moonlight MoE session size overflow");
    if (config->kv_lora_rank > 512 || config->qk_rope_dim > 256 ||
        (config->qk_rope_dim & 1) || config->expert_count > 256 ||
        config->experts_per_token > config->expert_count)
        return set_error(error, error_size,
                         "unsupported Moonlight MLA dimensions");

    session = calloc(1, sizeof(*session));
    if (!session) return set_error(error, error_size, "out of memory creating session");
    session->model = model;
    session->options = *options;
    session->stats.resident_bytes = model->resident_bytes;
    session->activation_a = allocate_buffer(session, activation_size, error, error_size);
    session->activation_b = allocate_buffer(session, activation_size, error, error_size);
    session->latent_kv = allocate_buffer(session, latent_size, error, error_size);
    session->rope_kv = allocate_buffer(session, rope_size, error, error_size);
    session->token_ids = allocate_buffer(session, token_size, error, error_size);
    session->logits = allocate_buffer(session, logit_size, error, error_size);
    session->query = allocate_buffer(session, query_size, error, error_size);
    session->compressed = allocate_buffer(session, compressed_size, error, error_size);
    session->context = allocate_buffer(session, context_size, error, error_size);
    session->absorbed_query =
        allocate_buffer(session, absorbed_query_size, error, error_size);
    session->attention_scores =
        allocate_buffer(session, attention_score_size, error, error_size);
    session->latent_context =
        allocate_buffer(session, latent_context_size, error, error_size);
    session->router_logits = allocate_buffer(session, router_size, error, error_size);
    session->route_ids = allocate_buffer(session, route_size, error, error_size);
    session->route_weights = allocate_buffer(session, route_size, error, error_size);
    session->expert_rows = allocate_buffer(session, row_index_size, error, error_size);
    session->expert_slots = allocate_buffer(session, row_index_size, error, error_size);
    session->expert_input =
        allocate_buffer(session, expert_input_size, error, error_size);
    session->expert_gate =
        allocate_buffer(session, expert_intermediate_size, error, error_size);
    session->expert_up =
        allocate_buffer(session, expert_intermediate_size, error, error_size);
    session->expert_hidden =
        allocate_buffer(session, expert_intermediate_size, error, error_size);
    session->expert_output =
        allocate_buffer(session, expert_output_size, error, error_size);
    session->routed_output =
        allocate_buffer(session, expert_output_size, error, error_size);
    session->shared_output =
        allocate_buffer(session, expert_output_size, error, error_size);
    session->kv_lengths = calloc((size_t)config->layer_count,
                                 sizeof(*session->kv_lengths));
    session->host_unique_experts = calloc((size_t)config->expert_count,
                                          sizeof(*session->host_unique_experts));
    session->host_expert_row_counts =
        calloc((size_t)config->expert_count,
               sizeof(*session->host_expert_row_counts));
    session->host_activation = calloc(
        (size_t)options->max_batch * config->hidden_size,
        sizeof(*session->host_activation));
    session->host_norm = calloc(
        (size_t)options->max_batch * config->hidden_size,
        sizeof(*session->host_norm));
    session->host_temporary = calloc(
        (size_t)options->max_batch * config->hidden_size,
        sizeof(*session->host_temporary));
    session->host_route_trace = calloc(
        (size_t)config->layer_count * options->max_batch *
        config->experts_per_token, sizeof(*session->host_route_trace));
    if (!session->activation_a || !session->activation_b || !session->latent_kv ||
        !session->rope_kv || !session->token_ids || !session->logits ||
        !session->query || !session->compressed || !session->context ||
        !session->absorbed_query || !session->attention_scores ||
        !session->latent_context ||
        !session->router_logits || !session->route_ids ||
        !session->route_weights || !session->expert_rows ||
        !session->expert_slots || !session->expert_input ||
        !session->expert_gate || !session->expert_up ||
        !session->expert_hidden || !session->expert_output ||
        !session->routed_output || !session->shared_output ||
        !session->kv_lengths || !session->host_unique_experts ||
        !session->host_expert_row_counts || !session->host_activation ||
        !session->host_norm || !session->host_temporary ||
        !session->host_route_trace) {
        moonlight_session_destroy(session);
        return 0;
    }
    *out = session;
    return 1;
}

void moonlight_session_destroy(MoonlightSession *session) {
    if (!session) return;
    session->activation_a = nil;
    session->activation_b = nil;
    session->latent_kv = nil;
    session->rope_kv = nil;
    session->token_ids = nil;
    session->logits = nil;
    session->query = nil;
    session->compressed = nil;
    session->context = nil;
    session->absorbed_query = nil;
    session->attention_scores = nil;
    session->latent_context = nil;
    session->router_logits = nil;
    session->route_ids = nil;
    session->route_weights = nil;
    session->expert_rows = nil;
    session->expert_slots = nil;
    session->expert_input = nil;
    session->expert_gate = nil;
    session->expert_up = nil;
    session->expert_hidden = nil;
    session->expert_output = nil;
    session->routed_output = nil;
    session->shared_output = nil;
    free(session->kv_lengths);
    free(session->host_unique_experts);
    free(session->host_expert_row_counts);
    free(session->host_activation);
    free(session->host_norm);
    free(session->host_temporary);
    free(session->host_route_trace);
    free(session);
}

void moonlight_session_reset(MoonlightSession *session) {
    if (!session) return;
    session->position = 0;
    session->last_forward_rows = 0;
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    memset(session->kv_lengths, 0,
           (size_t)config->layer_count * sizeof(*session->kv_lengths));
    session->stats.command_buffers = 0;
    session->stats.prefill_tokens = 0;
    session->stats.decode_tokens = 0;
    session->stats.cpu_fallbacks = 0;
    session->stats.prefill_ms = 0;
    session->stats.decode_ms = 0;
}

int moonlight_session_position(const MoonlightSession *session) {
    return session ? session->position : 0;
}

MoonlightStats moonlight_session_stats(const MoonlightSession *session) {
    MoonlightStats result = {0};
    return session ? session->stats : result;
}

int moonlight_test_embed(MoonlightSession *session, const int *ids,
                         int count, float *output) {
    MoonlightTensorInfo info;
    int index;
    uint32_t shape[2];
    size_t output_size;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;

    if (!session || !ids || !output || count <= 0 ||
        count > session->options.max_batch) return 0;
    const MoonlightConfig *config = moonlight_model_data_config(session->model->data);
    index = weight_index(session->model, "model.embed_tokens.weight", &info);
    if (index < 0 || info.dtype != MOONLIGHT_TENSOR_F32 ||
        info.element_count != (int64_t)config->vocab_size * config->hidden_size)
        return 0;
    output_size = (size_t)count * config->hidden_size * sizeof(float);
    if ((size_t)count * sizeof(int32_t) > session->token_ids.length ||
        output_size > session->activation_a.length) return 0;
    memcpy(session->token_ids.contents, ids, (size_t)count * sizeof(int32_t));
    shape[0] = (uint32_t)count;
    shape[1] = (uint32_t)config->hidden_size;
    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:session->model->embed_f32];
    [encoder setBuffer:session->model->weights[index].buffer offset:0 atIndex:0];
    [encoder setBuffer:session->token_ids offset:0 atIndex:1];
    [encoder setBuffer:session->activation_a offset:0 atIndex:2];
    [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    NSUInteger threads = MIN((NSUInteger)256,
                             session->model->embed_f32.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count * config->hidden_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->activation_a.contents, output_size);
    return 1;
}

int moonlight_test_rmsnorm(MoonlightSession *session, const float *input,
                           const char *weight_name, int rows, int width,
                           float *output) {
    MoonlightTensorInfo info;
    int index;
    uint32_t shape[2];
    size_t byte_count;
    float epsilon;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;

    if (!session || !input || !weight_name || !output || rows <= 0 || width <= 0 ||
        rows > session->options.max_batch) return 0;
    index = weight_index(session->model, weight_name, &info);
    if (index < 0 || info.dtype != MOONLIGHT_TENSOR_F32 || info.element_count != width)
        return 0;
    byte_count = (size_t)rows * width * sizeof(float);
    if (byte_count > session->activation_a.length ||
        byte_count > session->activation_b.length) return 0;
    memcpy(session->activation_a.contents, input, byte_count);
    shape[0] = (uint32_t)rows;
    shape[1] = (uint32_t)width;
    epsilon = moonlight_model_data_config(session->model->data)->rms_norm_epsilon;
    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:session->model->rmsnorm];
    [encoder setBuffer:session->activation_a offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[index].buffer offset:0 atIndex:1];
    [encoder setBuffer:session->activation_b offset:0 atIndex:2];
    [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
    NSUInteger threads = MIN((NSUInteger)rows,
                             session->model->rmsnorm.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MAX(threads, (NSUInteger)1), 1, 1)];
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->activation_b.contents, byte_count);
    return 1;
}

int moonlight_test_matmul(MoonlightSession *session, const char *weight_name,
                          const float *input, int rows, int input_width,
                          int output_width, float *output) {
    MoonlightTensorInfo info;
    MoonlightTensorInfo scale_info;
    int index;
    int scale_index = -1;
    char scale_name[512];
    uint32_t shape[3];
    size_t input_size;
    size_t output_size;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pipeline;

    if (!session || !weight_name || !input || !output || rows <= 0 ||
        input_width <= 0 || output_width <= 0 ||
        rows > session->options.max_batch) return 0;
    index = weight_index(session->model, weight_name, &info);
    if (index < 0) return 0;
    if (info.dtype == MOONLIGHT_TENSOR_F32) {
        if (info.element_count != (int64_t)input_width * output_width) return 0;
        pipeline = session->model->matmul_f32;
    } else if (info.dtype == MOONLIGHT_TENSOR_U8) {
        if (snprintf(scale_name, sizeof(scale_name), "%s.qs", weight_name) >=
            (int)sizeof(scale_name)) return 0;
        scale_index = weight_index(session->model, scale_name, &scale_info);
        if (scale_index < 0 || scale_info.dtype != MOONLIGHT_TENSOR_F32 ||
            scale_info.element_count != output_width) return 0;
        if (info.byte_count == (int64_t)output_width * input_width)
            pipeline = session->model->matmul_q8;
        else if (info.byte_count ==
                 (int64_t)output_width * ((input_width + 1) / 2))
            pipeline = session->model->matmul_q4;
        else
            return 0;
    } else {
        return 0;
    }
    input_size = (size_t)rows * input_width * sizeof(float);
    output_size = (size_t)rows * output_width * sizeof(float);
    if (input_size > session->activation_a.length ||
        output_size > session->logits.length) return 0;
    memcpy(session->activation_a.contents, input, input_size);
    shape[0] = (uint32_t)output_width;
    shape[1] = (uint32_t)input_width;
    shape[2] = (uint32_t)rows;
    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:session->model->weights[index].buffer offset:0 atIndex:0];
    if (scale_index >= 0) {
        [encoder setBuffer:session->model->weights[scale_index].buffer offset:0 atIndex:1];
        [encoder setBuffer:session->activation_a offset:0 atIndex:2];
        [encoder setBuffer:session->logits offset:0 atIndex:3];
        [encoder setBytes:shape length:sizeof(shape) atIndex:4];
    } else {
        [encoder setBuffer:session->activation_a offset:0 atIndex:1];
        [encoder setBuffer:session->logits offset:0 atIndex:2];
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    }
    NSUInteger columns = MIN((NSUInteger)32, (NSUInteger)output_width);
    NSUInteger row_threads = MIN((NSUInteger)8, (NSUInteger)rows);
    [encoder dispatchThreads:MTLSizeMake(output_width, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(columns, row_threads, 1)];
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->logits.contents, output_size);
    return 1;
}

int moonlight_test_attention(MoonlightSession *session, int layer,
                             const float *input, int rows, int position,
                             float *output) {
    const MoonlightConfig *config;
    MoonlightMatrix query_matrix;
    MoonlightMatrix kv_a_matrix;
    MoonlightMatrix kv_b_matrix;
    MoonlightMatrix output_matrix;
    MoonlightTensorInfo norm_info;
    char name[512];
    int norm_index;
    int query_width;
    int compressed_width;
    int context_width;
    size_t input_size;
    size_t output_size;
    uint32_t cache_shape[6];
    uint32_t rope_shape[5];
    uint32_t attention_shape[10];
    float attention_scale;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> kv_b_scale;

    if (!session || !input || !output || rows <= 0 || position < 0 ||
        rows > session->options.max_batch) return 0;
    config = moonlight_model_data_config(session->model->data);
    if (layer < 0 || layer >= config->layer_count ||
        position != session->kv_lengths[layer] ||
        position + rows > session->options.context_size || config->q_lora_rank)
        return 0;
    query_width = config->head_count *
                  (config->qk_nope_dim + config->qk_rope_dim);
    compressed_width = config->kv_lora_rank + config->qk_rope_dim;
    context_width = config->head_count * config->value_dim;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.q_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        query_width, &query_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        compressed_width, &kv_a_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.kv_b_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->kv_lora_rank,
                        config->head_count *
                        (config->qk_nope_dim + config->value_dim),
                        &kv_b_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.o_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, context_width,
                        config->hidden_size, &output_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.kv_a_layernorm.weight", layer) >=
            (int)sizeof(name)) return 0;
    norm_index = weight_index(session->model, name, &norm_info);
    if (norm_index < 0 || norm_info.dtype != MOONLIGHT_TENSOR_F32 ||
        norm_info.element_count != config->kv_lora_rank) return 0;
    input_size = (size_t)rows * config->hidden_size * sizeof(float);
    output_size = input_size;
    if (input_size > session->activation_a.length ||
        output_size > session->activation_b.length) return 0;
    memcpy(session->activation_a.contents, input, input_size);

    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    encode_matmul(session->model, encoder, &query_matrix,
                  session->activation_a, session->query, rows,
                  config->hidden_size, query_width);
    encode_matmul(session->model, encoder, &kv_a_matrix,
                  session->activation_a, session->compressed, rows,
                  config->hidden_size, compressed_width);

    cache_shape[0] = (uint32_t)rows;
    cache_shape[1] = (uint32_t)config->kv_lora_rank;
    cache_shape[2] = (uint32_t)config->qk_rope_dim;
    cache_shape[3] = (uint32_t)session->options.context_size;
    cache_shape[4] = (uint32_t)layer;
    cache_shape[5] = (uint32_t)position;
    [encoder setComputePipelineState:session->model->cache_append];
    [encoder setBuffer:session->compressed offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[norm_index].buffer offset:0 atIndex:1];
    [encoder setBuffer:session->latent_kv offset:0 atIndex:2];
    [encoder setBuffer:session->rope_kv offset:0 atIndex:3];
    [encoder setBytes:cache_shape length:sizeof(cache_shape) atIndex:4];
    [encoder setBytes:&config->rms_norm_epsilon
                length:sizeof(config->rms_norm_epsilon) atIndex:5];
    [encoder setBytes:&config->rope_theta length:sizeof(config->rope_theta) atIndex:6];
    NSUInteger cache_threads = MIN((NSUInteger)rows,
        session->model->cache_append.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MAX(cache_threads, (NSUInteger)1), 1, 1)];

    rope_shape[0] = (uint32_t)rows;
    rope_shape[1] = (uint32_t)config->head_count;
    rope_shape[2] = (uint32_t)config->qk_nope_dim;
    rope_shape[3] = (uint32_t)config->qk_rope_dim;
    rope_shape[4] = (uint32_t)position;
    [encoder setComputePipelineState:session->model->rope_query];
    [encoder setBuffer:session->query offset:0 atIndex:0];
    [encoder setBytes:rope_shape length:sizeof(rope_shape) atIndex:1];
    [encoder setBytes:&config->rope_theta length:sizeof(config->rope_theta) atIndex:2];
    NSUInteger query_count = (NSUInteger)rows * config->head_count;
    NSUInteger rope_threads = MIN((NSUInteger)64,
        session->model->rope_query.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(query_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(rope_threads, 1, 1)];

    attention_shape[0] = (uint32_t)rows;
    attention_shape[1] = (uint32_t)config->head_count;
    attention_shape[2] = (uint32_t)config->qk_nope_dim;
    attention_shape[3] = (uint32_t)config->qk_rope_dim;
    attention_shape[4] = (uint32_t)config->value_dim;
    attention_shape[5] = (uint32_t)config->kv_lora_rank;
    attention_shape[6] = (uint32_t)session->options.context_size;
    attention_shape[7] = (uint32_t)layer;
    attention_shape[8] = (uint32_t)position;
    attention_shape[9] = (uint32_t)kv_b_matrix.format;
    attention_scale = 1.0f /
        sqrtf((float)(config->qk_nope_dim + config->qk_rope_dim));
    kv_b_scale = kv_b_matrix.scale_index >= 0
        ? session->model->weights[kv_b_matrix.scale_index].buffer
        : session->model->weights[kv_b_matrix.weight_index].buffer;
    [encoder setComputePipelineState:session->model->attention_absorbed];
    [encoder setBuffer:session->query offset:0 atIndex:0];
    [encoder setBuffer:session->latent_kv offset:0 atIndex:1];
    [encoder setBuffer:session->rope_kv offset:0 atIndex:2];
    [encoder setBuffer:session->model->weights[kv_b_matrix.weight_index].buffer
             offset:0 atIndex:3];
    [encoder setBuffer:kv_b_scale offset:0 atIndex:4];
    [encoder setBuffer:session->context offset:0 atIndex:5];
    [encoder setBytes:attention_shape length:sizeof(attention_shape) atIndex:6];
    [encoder setBytes:&attention_scale length:sizeof(attention_scale) atIndex:7];
    NSUInteger attention_threads = MIN((NSUInteger)32,
        session->model->attention_absorbed.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(query_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(attention_threads, 1, 1)];

    encode_matmul(session->model, encoder, &output_matrix,
                  session->context, session->activation_b, rows,
                  context_width, config->hidden_size);
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->activation_b.contents, output_size);
    session->kv_lengths[layer] = position + rows;
    if (session->position < position + rows) session->position = position + rows;
    return 1;
}

int moonlight_test_kv_length(const MoonlightSession *session, int layer) {
    if (!session) return 0;
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    if (layer < 0 || layer >= config->layer_count) return 0;
    return session->kv_lengths[layer];
}

int moonlight_test_copy_kv(const MoonlightSession *session, int layer,
                           float *latent, float *rope, int capacity) {
    if (!session || !latent || !rope || capacity < 0) return 0;
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    if (layer < 0 || layer >= config->layer_count ||
        capacity < session->kv_lengths[layer]) return 0;
    int length = session->kv_lengths[layer];
    size_t latent_offset = (size_t)layer * session->options.context_size *
                           config->kv_lora_rank;
    size_t rope_offset = (size_t)layer * session->options.context_size *
                         config->qk_rope_dim;
    memcpy(latent, (float *)session->latent_kv.contents + latent_offset,
           (size_t)length * config->kv_lora_rank * sizeof(float));
    memcpy(rope, (float *)session->rope_kv.contents + rope_offset,
           (size_t)length * config->qk_rope_dim * sizeof(float));
    return 1;
}

int moonlight_test_moe(MoonlightSession *session, int layer,
                       const float *input, int rows, int *route_ids,
                       float *route_weights, float *router_scores,
                       float *routed_output, float *shared_output,
                       float *output) {
    const MoonlightConfig *config;
    MoonlightMatrix router_matrix;
    MoonlightMatrix gate_matrix;
    MoonlightMatrix up_matrix;
    MoonlightMatrix down_matrix;
    MoonlightTensorInfo bias_info;
    char name[512];
    int bias_index;
    int shared_width;
    int unique_count = 0;
    size_t hidden_bytes;
    size_t route_bytes;
    size_t score_bytes;
    uint32_t route_shape[4];
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;

    if (!session || !input || !output ||
        rows <= 0 || rows > session->options.max_batch) return 0;
    config = moonlight_model_data_config(session->model->data);
    if (layer < config->first_dense_layer_count || layer >= config->layer_count ||
        config->expert_count <= 0 || config->experts_per_token <= 0 ||
        config->shared_expert_count <= 0) return 0;
    shared_width = config->moe_intermediate_size * config->shared_expert_count;
    hidden_bytes = (size_t)rows * config->hidden_size * sizeof(float);
    route_bytes = (size_t)rows * config->experts_per_token * sizeof(float);
    score_bytes = (size_t)rows * config->expert_count * sizeof(float);
    memcpy(session->activation_a.contents, input, hidden_bytes);

    if (snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->expert_count, &router_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate.e_score_correction_bias", layer) >=
            (int)sizeof(name)) return 0;
    bias_index = weight_index(session->model, name, &bias_info);
    if (bias_index < 0 || bias_info.dtype != MOONLIGHT_TENSOR_F32 ||
        bias_info.element_count != config->expert_count) return 0;

    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    encode_matmul(session->model, encoder, &router_matrix,
                  session->activation_a, session->router_logits, rows,
                  config->hidden_size, config->expert_count);
    route_shape[0] = (uint32_t)rows;
    route_shape[1] = (uint32_t)config->expert_count;
    route_shape[2] = (uint32_t)config->experts_per_token;
    route_shape[3] = (uint32_t)config->normalize_topk;
    [encoder setComputePipelineState:session->model->route_sigmoid_topk];
    [encoder setBuffer:session->router_logits offset:0 atIndex:0];
    [encoder setBuffer:session->model->weights[bias_index].buffer offset:0 atIndex:1];
    [encoder setBuffer:session->route_ids offset:0 atIndex:2];
    [encoder setBuffer:session->route_weights offset:0 atIndex:3];
    [encoder setBytes:route_shape length:sizeof(route_shape) atIndex:4];
    [encoder setBytes:&config->routed_scale length:sizeof(config->routed_scale)
                atIndex:5];
    dispatch_1d(encoder, session->model->route_sigmoid_topk, (NSUInteger)rows);
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;

    int *selected = session->route_ids.contents;
    int *all_expert_rows = session->expert_rows.contents;
    int *all_expert_slots = session->expert_slots.contents;
    unsigned char seen[256] = {0};
    memset(session->host_expert_row_counts, 0,
           (size_t)config->expert_count *
           sizeof(*session->host_expert_row_counts));
    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < config->experts_per_token; ++slot) {
            int expert = selected[(int64_t)row * config->experts_per_token + slot];
            if (expert < 0 || expert >= config->expert_count) return 0;
            if (!seen[expert]) {
                seen[expert] = 1;
                session->host_unique_experts[unique_count++] = expert;
            }
            int expert_row = session->host_expert_row_counts[expert]++;
            int64_t offset = (int64_t)expert * session->options.max_batch +
                             expert_row;
            all_expert_rows[offset] = row;
            all_expert_slots[offset] = slot;
        }
    }

    command = [session->model->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    uint32_t hidden_count = (uint32_t)((int64_t)rows * config->hidden_size);
    [encoder setComputePipelineState:session->model->clear_f32];
    [encoder setBuffer:session->routed_output offset:0 atIndex:0];
    [encoder setBytes:&hidden_count length:sizeof(hidden_count) atIndex:1];
    dispatch_1d(encoder, session->model->clear_f32, hidden_count);

    for (int unique = 0; unique < unique_count; ++unique) {
        int expert = session->host_unique_experts[unique];
        int expert_row_count = session->host_expert_row_counts[expert];
        NSUInteger index_offset =
            (NSUInteger)expert * session->options.max_batch * sizeof(int32_t);
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.gate_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name, config->hidden_size,
                            config->moe_intermediate_size, &gate_matrix)) return 0;
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.up_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name, config->hidden_size,
                            config->moe_intermediate_size, &up_matrix)) return 0;
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.down_proj.weight",
                     layer, expert) >= (int)sizeof(name) ||
            !resolve_matrix(session->model, name, config->moe_intermediate_size,
                            config->hidden_size, &down_matrix)) return 0;

        uint32_t gather_shape[2] = {(uint32_t)expert_row_count,
                                    (uint32_t)config->hidden_size};
        [encoder setComputePipelineState:session->model->gather_rows];
        [encoder setBuffer:session->activation_a offset:0 atIndex:0];
        [encoder setBuffer:session->expert_rows offset:index_offset atIndex:1];
        [encoder setBuffer:session->expert_input offset:0 atIndex:2];
        [encoder setBytes:gather_shape length:sizeof(gather_shape) atIndex:3];
        dispatch_1d(encoder, session->model->gather_rows,
                    (NSUInteger)expert_row_count * config->hidden_size);
        encode_matmul(session->model, encoder, &gate_matrix,
                      session->expert_input, session->expert_gate,
                      expert_row_count, config->hidden_size,
                      config->moe_intermediate_size);
        encode_matmul(session->model, encoder, &up_matrix,
                      session->expert_input, session->expert_up,
                      expert_row_count, config->hidden_size,
                      config->moe_intermediate_size);
        uint32_t intermediate_count =
            (uint32_t)((int64_t)expert_row_count * config->moe_intermediate_size);
        [encoder setComputePipelineState:session->model->silu_multiply];
        [encoder setBuffer:session->expert_gate offset:0 atIndex:0];
        [encoder setBuffer:session->expert_up offset:0 atIndex:1];
        [encoder setBuffer:session->expert_hidden offset:0 atIndex:2];
        [encoder setBytes:&intermediate_count length:sizeof(intermediate_count)
                  atIndex:3];
        dispatch_1d(encoder, session->model->silu_multiply, intermediate_count);
        encode_matmul(session->model, encoder, &down_matrix,
                      session->expert_hidden, session->expert_output,
                      expert_row_count, config->moe_intermediate_size,
                      config->hidden_size);
        uint32_t scatter_shape[3] = {(uint32_t)expert_row_count,
                                     (uint32_t)config->hidden_size,
                                     (uint32_t)config->experts_per_token};
        [encoder setComputePipelineState:session->model->scatter_expert];
        [encoder setBuffer:session->expert_output offset:0 atIndex:0];
        [encoder setBuffer:session->expert_rows offset:index_offset atIndex:1];
        [encoder setBuffer:session->expert_slots offset:index_offset atIndex:2];
        [encoder setBuffer:session->route_weights offset:0 atIndex:3];
        [encoder setBuffer:session->routed_output offset:0 atIndex:4];
        [encoder setBytes:scatter_shape length:sizeof(scatter_shape) atIndex:5];
        dispatch_1d(encoder, session->model->scatter_expert,
                    (NSUInteger)expert_row_count * config->hidden_size);
    }

    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.gate_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        shared_width, &gate_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.up_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        shared_width, &up_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.shared_experts.down_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, shared_width,
                        config->hidden_size, &down_matrix)) return 0;
    encode_matmul(session->model, encoder, &gate_matrix,
                  session->activation_a, session->expert_gate, rows,
                  config->hidden_size, shared_width);
    encode_matmul(session->model, encoder, &up_matrix,
                  session->activation_a, session->expert_up, rows,
                  config->hidden_size, shared_width);
    uint32_t shared_count = (uint32_t)((int64_t)rows * shared_width);
    [encoder setComputePipelineState:session->model->silu_multiply];
    [encoder setBuffer:session->expert_gate offset:0 atIndex:0];
    [encoder setBuffer:session->expert_up offset:0 atIndex:1];
    [encoder setBuffer:session->expert_hidden offset:0 atIndex:2];
    [encoder setBytes:&shared_count length:sizeof(shared_count) atIndex:3];
    dispatch_1d(encoder, session->model->silu_multiply, shared_count);
    encode_matmul(session->model, encoder, &down_matrix,
                  session->expert_hidden, session->shared_output, rows,
                  shared_width, config->hidden_size);
    [encoder setComputePipelineState:session->model->add_f32];
    [encoder setBuffer:session->routed_output offset:0 atIndex:0];
    [encoder setBuffer:session->shared_output offset:0 atIndex:1];
    [encoder setBuffer:session->activation_b offset:0 atIndex:2];
    [encoder setBytes:&hidden_count length:sizeof(hidden_count) atIndex:3];
    dispatch_1d(encoder, session->model->add_f32, hidden_count);
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;

    if (route_ids) memcpy(route_ids, session->route_ids.contents, route_bytes);
    if (route_weights)
        memcpy(route_weights, session->route_weights.contents, route_bytes);
    if (router_scores)
        memcpy(router_scores, session->router_logits.contents, score_bytes);
    if (routed_output)
        memcpy(routed_output, session->routed_output.contents, hidden_bytes);
    if (shared_output)
        memcpy(shared_output, session->shared_output.contents, hidden_bytes);
    memcpy(output, session->activation_b.contents, hidden_bytes);
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int run_residual_add(MoonlightSession *session, const float *left,
                            const float *right, int rows, float *output) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    uint32_t count = (uint32_t)((int64_t)rows * config->hidden_size);
    size_t byte_count = (size_t)count * sizeof(float);
    memcpy(session->activation_a.contents, left, byte_count);
    memcpy(session->activation_b.contents, right, byte_count);
    id<MTLCommandBuffer> command = [session->model->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:session->model->add_f32];
    [encoder setBuffer:session->activation_a offset:0 atIndex:0];
    [encoder setBuffer:session->activation_b offset:0 atIndex:1];
    [encoder setBuffer:session->expert_output offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    dispatch_1d(encoder, session->model->add_f32, count);
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->expert_output.contents, byte_count);
    return 1;
}

int moonlight_test_dense_mlp(MoonlightSession *session, int layer,
                             const float *input, int rows, float *output) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    MoonlightMatrix gate_matrix;
    MoonlightMatrix up_matrix;
    MoonlightMatrix down_matrix;
    char name[512];
    size_t byte_count = (size_t)rows * config->hidden_size * sizeof(float);
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->dense_intermediate_size, &gate_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.up_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->hidden_size,
                        config->dense_intermediate_size, &up_matrix)) return 0;
    if (snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.down_proj.weight", layer) >=
            (int)sizeof(name) ||
        !resolve_matrix(session->model, name, config->dense_intermediate_size,
                        config->hidden_size, &down_matrix)) return 0;
    memcpy(session->activation_a.contents, input, byte_count);
    id<MTLCommandBuffer> command = [session->model->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_matmul(session->model, encoder, &gate_matrix,
                  session->activation_a, session->expert_gate, rows,
                  config->hidden_size, config->dense_intermediate_size);
    encode_matmul(session->model, encoder, &up_matrix,
                  session->activation_a, session->expert_up, rows,
                  config->hidden_size, config->dense_intermediate_size);
    uint32_t intermediate_count =
        (uint32_t)((int64_t)rows * config->dense_intermediate_size);
    [encoder setComputePipelineState:session->model->silu_multiply];
    [encoder setBuffer:session->expert_gate offset:0 atIndex:0];
    [encoder setBuffer:session->expert_up offset:0 atIndex:1];
    [encoder setBuffer:session->expert_hidden offset:0 atIndex:2];
    [encoder setBytes:&intermediate_count length:sizeof(intermediate_count)
              atIndex:3];
    dispatch_1d(encoder, session->model->silu_multiply, intermediate_count);
    encode_matmul(session->model, encoder, &down_matrix,
                  session->expert_hidden, session->activation_b, rows,
                  config->dense_intermediate_size, config->hidden_size);
    [encoder endEncoding];
    if (!finish_command(session, command)) return 0;
    memcpy(output, session->activation_b.contents, byte_count);
    return 1;
}

static int run_tokens(MoonlightSession *session, const int *ids, int count,
                      int start_position, float *layer_outputs,
                      float *last_logits,
                      char *error, size_t error_size) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    char name[512];
    if (!moonlight_test_embed(session, ids, count, session->host_activation))
        return set_error(error, error_size, "Moonlight embedding failed");
    for (int layer = 0; layer < config->layer_count; ++layer) {
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.input_layernorm.weight", layer) >=
                (int)sizeof(name) ||
            !moonlight_test_rmsnorm(session, session->host_activation, name,
                                    count, config->hidden_size,
                                    session->host_norm))
            return set_error(error, error_size,
                             "Moonlight input norm failed at layer %d", layer);
        if (!moonlight_test_attention(session, layer, session->host_norm,
                                      count, start_position,
                                      session->host_temporary))
            return set_error(error, error_size,
                             "Moonlight attention failed at layer %d", layer);
        if (!run_residual_add(session, session->host_activation,
                              session->host_temporary, count,
                              session->host_activation))
            return set_error(error, error_size,
                             "Moonlight attention residual failed at layer %d",
                             layer);
        if (snprintf(name, sizeof(name),
                     "model.layers.%d.post_attention_layernorm.weight", layer) >=
                (int)sizeof(name) ||
            !moonlight_test_rmsnorm(session, session->host_activation, name,
                                    count, config->hidden_size,
                                    session->host_norm))
            return set_error(error, error_size,
                             "Moonlight post-attention norm failed at layer %d",
                             layer);
        if (layer < config->first_dense_layer_count) {
            if (!moonlight_test_dense_mlp(session, layer, session->host_norm,
                                          count, session->host_temporary))
                return set_error(error, error_size,
                                 "Moonlight dense MLP failed at layer %d", layer);
        } else if (!moonlight_test_moe(
                                       session, layer, session->host_norm,
                                       count,
                                       session->host_route_trace +
                                           (int64_t)layer *
                                           session->options.max_batch *
                                           config->experts_per_token,
                                       NULL, NULL, NULL, NULL,
                                       session->host_temporary)) {
            return set_error(error, error_size,
                             "Moonlight MoE failed at layer %d", layer);
        }
        if (!run_residual_add(session, session->host_activation,
                              session->host_temporary, count,
                              session->host_activation))
            return set_error(error, error_size,
                             "Moonlight MLP residual failed at layer %d", layer);
        if (layer_outputs)
            memcpy(layer_outputs + (int64_t)layer * count * config->hidden_size,
                   session->host_activation,
                   (size_t)count * config->hidden_size * sizeof(float));
    }
    const float *last_hidden = session->host_activation +
        (int64_t)(count - 1) * config->hidden_size;
    if (!moonlight_test_rmsnorm(session, last_hidden, "model.norm.weight",
                                1, config->hidden_size, session->host_norm))
        return set_error(error, error_size, "Moonlight final norm failed");
    if (!moonlight_test_matmul(session, "lm_head.weight", session->host_norm,
                               1, config->hidden_size, config->vocab_size,
                               last_logits))
        return set_error(error, error_size, "Moonlight lm-head failed");
    session->last_forward_rows = count;
    return 1;
}

static int run_tokens_fast(MoonlightSession *session, const int *ids, int count,
                           int start_position, float *last_logits,
                           char *error, size_t error_size) {
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    int first_sparse = MIN(config->first_dense_layer_count,
                           config->layer_count);
    int profile = getenv("MOONLIGHT_PROFILE") != NULL;
    double command_start = monotonic_seconds();
    id<MTLCommandBuffer> command = [session->model->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder ||
        !encode_embedding_buffer(session, encoder, ids, count,
                                 session->activation_a))
        return set_error(error, error_size,
                         "Moonlight fast embedding failed");

    for (int layer = 0; layer < first_sparse; ++layer) {
        if (!encode_attention_block(session, encoder, layer, count,
                                    start_position, 0) ||
            !encode_dense_residual_buffer(session, encoder, layer, count))
            return set_error(error, error_size,
                             "Moonlight fast dense block failed at layer %d",
                             layer);
    }
    if (first_sparse < config->layer_count) {
        if (!encode_attention_block(session, encoder, first_sparse, count,
                                    start_position, 1))
            return set_error(error, error_size,
                             "Moonlight fast sparse attention failed at layer %d",
                             first_sparse);
    } else if (!encode_final_logits(session, encoder, count)) {
        return set_error(error, error_size,
                         "Moonlight fast final projection failed");
    }
    [encoder endEncoding];
    double wait_start = monotonic_seconds();
    if (!finish_command(session, command))
        return set_error(error, error_size,
                         "Moonlight fast initial command failed");
    if (profile)
        fprintf(stderr,
                "moonlight profile rows=%d initial encode=%.3f wait=%.3f gpu=%.3f ms\n",
                count, (wait_start - command_start) * 1000.0,
                (monotonic_seconds() - wait_start) * 1000.0,
                (command.GPUEndTime - command.GPUStartTime) * 1000.0);

    for (int layer = first_sparse; layer < config->layer_count; ++layer) {
        int unique_count;
        double route_start = monotonic_seconds();
        if (!prepare_route_tables(session, layer, count, &unique_count))
            return set_error(error, error_size,
                             "Moonlight fast route table failed at layer %d",
                             layer);
        double route_end = monotonic_seconds();
        command_start = route_end;
        command = [session->model->queue commandBuffer];
        encoder = [command computeCommandEncoder];
        if (!command || !encoder ||
            !encode_moe_residual_buffer(session, encoder, layer, count,
                                        unique_count))
            return set_error(error, error_size,
                             "Moonlight fast MoE failed at layer %d", layer);
        if (layer + 1 < config->layer_count) {
            if (!encode_attention_block(session, encoder, layer + 1, count,
                                        start_position, 1))
                return set_error(
                    error, error_size,
                    "Moonlight fast sparse attention failed at layer %d",
                    layer + 1);
        } else if (!encode_final_logits(session, encoder, count)) {
            return set_error(error, error_size,
                             "Moonlight fast final projection failed");
        }
        [encoder endEncoding];
        wait_start = monotonic_seconds();
        if (!finish_command(session, command))
            return set_error(error, error_size,
                             "Moonlight fast command failed at layer %d",
                             layer);
        if (profile)
            fprintf(stderr,
                    "moonlight profile rows=%d layer=%d experts=%d route=%.3f encode=%.3f wait=%.3f gpu=%.3f ms\n",
                    count, layer, unique_count,
                    (route_end - route_start) * 1000.0,
                    (wait_start - command_start) * 1000.0,
                    (monotonic_seconds() - wait_start) * 1000.0,
                    (command.GPUEndTime - command.GPUStartTime) * 1000.0);
    }
    memcpy(last_logits, session->logits.contents,
           (size_t)config->vocab_size * sizeof(*last_logits));
    session->position = start_position + count;
    session->last_forward_rows = count;
    return 1;
}

int moonlight_session_prefill(MoonlightSession *session, const int *ids,
                              int count, float *last_logits,
                              char *error, size_t error_size) {
    if (!session || !ids || !last_logits || count <= 0 ||
        count > session->options.max_batch ||
        count > session->options.context_size)
        return set_error(error, error_size, "invalid Moonlight prefill request");
    if (session->position != 0)
        return set_error(error, error_size,
                         "Moonlight prefill requires an empty session");
    double start = monotonic_seconds();
    if (!run_tokens_fast(session, ids, count, 0, last_logits,
                         error, error_size))
        return 0;
    session->stats.prefill_tokens += (uint64_t)count;
    session->stats.prefill_ms += (monotonic_seconds() - start) * 1000.0;
    return 1;
}

int moonlight_session_decode(MoonlightSession *session, int token,
                             float *logits, char *error, size_t error_size) {
    if (!session || !logits || session->position < 0 ||
        session->position >= session->options.context_size)
        return set_error(error, error_size, "invalid Moonlight decode request");
    int position = session->position;
    double start = monotonic_seconds();
    if (!run_tokens_fast(session, &token, 1, position, logits,
                         error, error_size))
        return 0;
    session->stats.decode_tokens++;
    session->stats.decode_ms += (monotonic_seconds() - start) * 1000.0;
    return 1;
}

int moonlight_test_prefill_layers(MoonlightSession *session, const int *ids,
                                  int count, float *layer_outputs,
                                  float *last_logits,
                                  char *error, size_t error_size) {
    if (!session || !ids || !layer_outputs || !last_logits || count <= 0 ||
        count > session->options.max_batch || session->position != 0)
        return set_error(error, error_size,
                         "invalid Moonlight layer trace request");
    double start = monotonic_seconds();
    if (!run_tokens(session, ids, count, 0, layer_outputs, last_logits,
                    error, error_size)) return 0;
    session->stats.prefill_tokens += (uint64_t)count;
    session->stats.prefill_ms += (monotonic_seconds() - start) * 1000.0;
    return 1;
}

int moonlight_test_copy_routes(const MoonlightSession *session, int layer,
                               int *route_ids, int capacity) {
    if (!session || !route_ids || capacity < session->last_forward_rows)
        return 0;
    const MoonlightConfig *config =
        moonlight_model_data_config(session->model->data);
    if (layer < config->first_dense_layer_count ||
        layer >= config->layer_count) return 0;
    memcpy(route_ids,
           session->host_route_trace +
               (int64_t)layer * session->options.max_batch *
               config->experts_per_token,
           (size_t)session->last_forward_rows * config->experts_per_token *
               sizeof(*route_ids));
    return 1;
}
