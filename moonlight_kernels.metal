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

kernel void moonlight_cache_append(device const float *compressed [[buffer(0)]],
                                   device const float *norm_weight [[buffer(1)]],
                                   device float *latent_cache [[buffer(2)]],
                                   device float *rope_cache [[buffer(3)]],
                                   constant uint *shape [[buffer(4)]],
                                   constant float &epsilon [[buffer(5)]],
                                   constant float &theta [[buffer(6)]],
                                   uint row [[thread_position_in_grid]]) {
    uint rows = shape[0];
    uint latent_width = shape[1];
    uint rope_width = shape[2];
    uint context_size = shape[3];
    uint layer = shape[4];
    uint position = shape[5];
    if (row >= rows) return;
    uint source_offset = row * (latent_width + rope_width);
    uint latent_offset = (layer * context_size + position + row) * latent_width;
    uint rope_offset = (layer * context_size + position + row) * rope_width;
    float sum = 0.0f;
    for (uint column = 0; column < latent_width; ++column) {
        float value = compressed[source_offset + column];
        sum += value * value;
    }
    float norm_scale = rsqrt(sum / float(latent_width) + epsilon);
    for (uint column = 0; column < latent_width; ++column)
        latent_cache[latent_offset + column] =
            compressed[source_offset + column] * norm_scale * norm_weight[column];
    uint half_width = rope_width / 2;
    for (uint pair = 0; pair < half_width; ++pair) {
        float angle = float(position + row) *
                      pow(theta, -2.0f * float(pair) / float(rope_width));
        float cosine = cos(angle);
        float sine = sin(angle);
        float first = compressed[source_offset + latent_width + 2 * pair];
        float second = compressed[source_offset + latent_width + 2 * pair + 1];
        rope_cache[rope_offset + pair] = first * cosine - second * sine;
        rope_cache[rope_offset + half_width + pair] = second * cosine + first * sine;
    }
}

kernel void moonlight_rope_query(device float *query [[buffer(0)]],
                                 constant uint *shape [[buffer(1)]],
                                 constant float &theta [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
    uint rows = shape[0];
    uint heads = shape[1];
    uint nope_width = shape[2];
    uint rope_width = shape[3];
    uint position = shape[4];
    if (index >= rows * heads) return;
    uint row = index / heads;
    uint head = index - row * heads;
    uint head_width = nope_width + rope_width;
    uint offset = (row * heads + head) * head_width + nope_width;
    float source[256];
    for (uint column = 0; column < rope_width; ++column)
        source[column] = query[offset + column];
    uint half_width = rope_width / 2;
    for (uint pair = 0; pair < half_width; ++pair) {
        float angle = float(position + row) *
                      pow(theta, -2.0f * float(pair) / float(rope_width));
        float cosine = cos(angle);
        float sine = sin(angle);
        float first = source[2 * pair];
        float second = source[2 * pair + 1];
        query[offset + pair] = first * cosine - second * sine;
        query[offset + half_width + pair] = second * cosine + first * sine;
    }
}

static inline float moonlight_weight_value(device const uchar *weight,
                                           device const float *weight_scale,
                                           uint format, uint row, uint column,
                                           uint width) {
    if (format == 0)
        return reinterpret_cast<device const float *>(weight)[row * width + column];
    if (format == 1)
        return float(reinterpret_cast<device const char *>(weight)[row * width + column]) *
               weight_scale[row];
    uint packed_width = (width + 1) / 2;
    uchar packed = weight[row * packed_width + column / 2];
    int value = (column & 1) ? int(packed >> 4) - 8
                             : int(packed & 15) - 8;
    return float(value) * weight_scale[row];
}

kernel void moonlight_attention_absorbed(
        device const float *query [[buffer(0)]],
        device const float *latent_cache [[buffer(1)]],
        device const float *rope_cache [[buffer(2)]],
        device const uchar *kv_b_weight [[buffer(3)]],
        device const float *kv_b_scale [[buffer(4)]],
        device float *context [[buffer(5)]],
        constant uint *shape [[buffer(6)]],
        constant float &attention_scale [[buffer(7)]],
        uint index [[thread_position_in_grid]]) {
    uint rows = shape[0];
    uint heads = shape[1];
    uint nope_width = shape[2];
    uint rope_width = shape[3];
    uint value_width = shape[4];
    uint latent_width = shape[5];
    uint context_size = shape[6];
    uint layer = shape[7];
    uint position = shape[8];
    uint weight_format = shape[9];
    if (index >= rows * heads) return;
    uint row = index / heads;
    uint head = index - row * heads;
    uint head_width = nope_width + rope_width;
    uint query_offset = (row * heads + head) * head_width;
    uint weight_row = head * (nope_width + value_width);
    uint cache_base = layer * context_size;
    uint key_count = position + row + 1;
    float query_absorbed[512];
    float latent_context[512];

    for (uint latent = 0; latent < latent_width; ++latent) {
        float sum = 0.0f;
        for (uint column = 0; column < nope_width; ++column)
            sum += query[query_offset + column] *
                   moonlight_weight_value(kv_b_weight, kv_b_scale,
                                          weight_format, weight_row + column,
                                          latent, latent_width);
        query_absorbed[latent] = sum;
        latent_context[latent] = 0.0f;
    }

    float maximum = -3.402823466e+38f;
    for (uint key = 0; key < key_count; ++key) {
        uint latent_offset = (cache_base + key) * latent_width;
        uint rope_offset = (cache_base + key) * rope_width;
        float score = 0.0f;
        for (uint latent = 0; latent < latent_width; ++latent)
            score += query_absorbed[latent] * latent_cache[latent_offset + latent];
        for (uint column = 0; column < rope_width; ++column)
            score += query[query_offset + nope_width + column] *
                     rope_cache[rope_offset + column];
        maximum = max(maximum, score * attention_scale);
    }
    float denominator = 0.0f;
    for (uint key = 0; key < key_count; ++key) {
        uint latent_offset = (cache_base + key) * latent_width;
        uint rope_offset = (cache_base + key) * rope_width;
        float score = 0.0f;
        for (uint latent = 0; latent < latent_width; ++latent)
            score += query_absorbed[latent] * latent_cache[latent_offset + latent];
        for (uint column = 0; column < rope_width; ++column)
            score += query[query_offset + nope_width + column] *
                     rope_cache[rope_offset + column];
        denominator += exp(score * attention_scale - maximum);
    }
    for (uint key = 0; key < key_count; ++key) {
        uint latent_offset = (cache_base + key) * latent_width;
        uint rope_offset = (cache_base + key) * rope_width;
        float score = 0.0f;
        for (uint latent = 0; latent < latent_width; ++latent)
            score += query_absorbed[latent] * latent_cache[latent_offset + latent];
        for (uint column = 0; column < rope_width; ++column)
            score += query[query_offset + nope_width + column] *
                     rope_cache[rope_offset + column];
        float probability = exp(score * attention_scale - maximum) / denominator;
        for (uint latent = 0; latent < latent_width; ++latent)
            latent_context[latent] += probability * latent_cache[latent_offset + latent];
    }
    uint context_offset = (row * heads + head) * value_width;
    for (uint value = 0; value < value_width; ++value) {
        float sum = 0.0f;
        for (uint latent = 0; latent < latent_width; ++latent)
            sum += latent_context[latent] *
                   moonlight_weight_value(kv_b_weight, kv_b_scale,
                                          weight_format,
                                          weight_row + nope_width + value,
                                          latent, latent_width);
        context[context_offset + value] = sum;
    }
}
