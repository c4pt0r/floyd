#include <metal_stdlib>

using namespace metal;

kernel void moonlight_noop(device uint *state [[buffer(0)]],
                           uint index [[thread_position_in_grid]]) {
    if (index == 0) state[0] = state[0];
}

kernel void moonlight_embed_f32(device const float *weight [[buffer(0)]],
                                device const int *ids [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant uint *shape [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    uint count = shape[0];
    uint width = shape[1];
    if (index >= count * width) return;
    uint row = index / width;
    uint column = index - row * width;
    output[index] = weight[uint(ids[row]) * width + column];
}

kernel void moonlight_rmsnorm(device const float *input [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant uint *shape [[buffer(3)]],
                              constant float &epsilon [[buffer(4)]],
                              uint row [[thread_position_in_grid]]) {
    uint rows = shape[0];
    uint width = shape[1];
    if (row >= rows) return;
    float sum = 0.0f;
    uint offset = row * width;
    for (uint column = 0; column < width; ++column) {
        float value = input[offset + column];
        sum += value * value;
    }
    float scale = rsqrt(sum / float(width) + epsilon);
    for (uint column = 0; column < width; ++column)
        output[offset + column] = input[offset + column] * scale * weight[column];
}

kernel void moonlight_matmul_f32(device const float *weight [[buffer(0)]],
                                 device const float *input [[buffer(1)]],
                                 device float *output [[buffer(2)]],
                                 constant uint *shape [[buffer(3)]],
                                 uint2 index [[thread_position_in_grid]]) {
    uint output_width = shape[0];
    uint input_width = shape[1];
    uint rows = shape[2];
    uint column = index.x;
    uint row = index.y;
    if (column >= output_width || row >= rows) return;
    float sum = 0.0f;
    uint input_offset = row * input_width;
    uint weight_offset = column * input_width;
    for (uint inner = 0; inner < input_width; ++inner)
        sum += input[input_offset + inner] * weight[weight_offset + inner];
    output[row * output_width + column] = sum;
}

kernel void moonlight_matmul_q8(device const char *weight [[buffer(0)]],
                                device const float *scale [[buffer(1)]],
                                device const float *input [[buffer(2)]],
                                device float *output [[buffer(3)]],
                                constant uint *shape [[buffer(4)]],
                                uint2 index [[thread_position_in_grid]]) {
    uint output_width = shape[0];
    uint input_width = shape[1];
    uint rows = shape[2];
    uint column = index.x;
    uint row = index.y;
    if (column >= output_width || row >= rows) return;
    float sum = 0.0f;
    uint input_offset = row * input_width;
    uint weight_offset = column * input_width;
    for (uint inner = 0; inner < input_width; ++inner)
        sum += input[input_offset + inner] * float(weight[weight_offset + inner]);
    output[row * output_width + column] = sum * scale[column];
}

kernel void moonlight_matmul_q4(device const uchar *weight [[buffer(0)]],
                                device const float *scale [[buffer(1)]],
                                device const float *input [[buffer(2)]],
                                device float *output [[buffer(3)]],
                                constant uint *shape [[buffer(4)]],
                                uint2 index [[thread_position_in_grid]]) {
    uint output_width = shape[0];
    uint input_width = shape[1];
    uint rows = shape[2];
    uint column = index.x;
    uint row = index.y;
    if (column >= output_width || row >= rows) return;
    float sum = 0.0f;
    uint input_offset = row * input_width;
    uint packed_width = (input_width + 1) / 2;
    uint weight_offset = column * packed_width;
    for (uint inner = 0; inner < input_width; ++inner) {
        uchar packed = weight[weight_offset + inner / 2];
        int value = (inner & 1) ? int(packed >> 4) - 8
                                : int(packed & 15) - 8;
        sum += input[input_offset + inner] * float(value);
    }
    output[row * output_width + column] = sum * scale[column];
}
