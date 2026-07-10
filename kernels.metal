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
