#include <metal_stdlib>
using namespace metal;

/* q8: w[o,i] int8, scale per riga s[o]. y[t,o] = (sum_i w[o,i]*x[t,i]) * s[o] */
kernel void matmul_q8(device const char  *w [[buffer(0)]],
                      device const float *s [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float       *y [[buffer(3)]],
                      constant uint3     &d [[buffer(4)]],   // O, I, S
                      uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    device const char  *wr = w + (ulong)o * d.y;
    device const float *xr = x + (ulong)t * d.y;
    float acc = 0.0f;
    for (uint i = 0; i < d.y; i++) acc += (float)wr[i] * xr[i];
    y[(ulong)t * d.x + o] = acc * s[o];
}

/* q4: int4 nibble in ogni byte, low nibble prima, offset -8. */
kernel void matmul_q4(device const uchar *w [[buffer(0)]],
                      device const float *s [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float       *y [[buffer(3)]],
                      constant uint3     &d [[buffer(4)]],
                      uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    uint Ib = (d.y + 1) / 2;
    device const uchar *wr = w + (ulong)o * Ib;
    device const float *xr = x + (ulong)t * d.y;
    float acc = 0.0f;
    for (uint i = 0; i + 1 < d.y; i += 2) {
        uchar b = wr[i >> 1];
        acc += (float)((int)(b & 0xF) - 8) * xr[i] + (float)((int)(b >> 4) - 8) * xr[i + 1];
    }
    if (d.y & 1) { uchar b = wr[d.y >> 1]; acc += (float)((int)(b & 0xF) - 8) * xr[d.y - 1]; }
    y[(ulong)t * d.x + o] = acc * s[o];
}

static inline float deepseek_v4_e8m0(uchar code) {
    return code == 255 ? NAN : exp2((float)((int)code - 127));
}

static inline float deepseek_v4_e4m3(uchar code) {
    int sign = (code & 0x80) ? -1 : 1;
    int exponent = (code >> 3) & 0x0f;
    int mantissa = code & 0x07;
    if (exponent == 0x0f && mantissa == 0x07) return NAN;
    float value = exponent == 0
        ? ((float)mantissa / 8.0f) * exp2(-6.0f)
        : (1.0f + (float)mantissa / 8.0f) * exp2((float)(exponent - 7));
    return (float)sign * value;
}

kernel void matmul_deepseek_v4_fp8(device const uchar *w [[buffer(0)]],
                          device const uchar *s [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device float       *y [[buffer(3)]],
                          constant uint3     &d [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    uint scale_columns = d.y / 128;
    device const uchar *wr = w + (ulong)o * d.y;
    device const float *xr = x + (ulong)t * d.y;
    device const uchar *sr = s + (ulong)(o / 128) * scale_columns;
    float acc = 0.0f;
    for (uint i = 0; i < d.y; i++)
        acc += xr[i] * deepseek_v4_e4m3(wr[i]) * deepseek_v4_e8m0(sr[i / 128]);
    y[(ulong)t * d.x + o] = acc;
}

kernel void matmul_deepseek_v4_fp4(device const uchar *w [[buffer(0)]],
                          device const uchar *s [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device float       *y [[buffer(3)]],
                          constant uint3     &d [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    uint row_bytes = d.y / 2;
    uint scale_columns = d.y / 32;
    device const uchar *wr = w + (ulong)o * row_bytes;
    device const uchar *sr = s + (ulong)o * scale_columns;
    device const float *xr = x + (ulong)t * d.y;
    float acc = 0.0f;
    for (uint i = 0; i < d.y; i++) {
        uchar packed = wr[i >> 1];
        uchar code = (i & 1) ? packed >> 4 : packed & 0x0f;
        acc += xr[i] * values[code] * deepseek_v4_e8m0(sr[i / 32]);
    }
    y[(ulong)t * d.x + o] = acc;
}
