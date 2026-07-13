#include <metal_stdlib>

using namespace metal;

kernel void moonlight_noop(device uint *state [[buffer(0)]],
                           uint index [[thread_position_in_grid]]) {
    if (index == 0) state[0] = state[0];
}
