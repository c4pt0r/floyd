#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

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
    __strong id<MTLComputePipelineState> matmul_q4;
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
    MoonlightStats stats;
    int position;
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
    model->matmul_q4 = make_pipeline(model, @"moonlight_matmul_q4", &metal_error);
    if (!model->noop || !model->embed_f32 || !model->rmsnorm ||
        !model->matmul_f32 || !model->matmul_q8 || !model->matmul_q4) {
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
    model->matmul_q4 = nil;
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
        !multiply_size(logit_size, sizeof(float), &logit_size))
        return set_error(error, error_size, "Moonlight session size overflow");

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
    if (!session->activation_a || !session->activation_b || !session->latent_kv ||
        !session->rope_kv || !session->token_ids || !session->logits) {
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
    free(session);
}

void moonlight_session_reset(MoonlightSession *session) {
    if (!session) return;
    session->position = 0;
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
