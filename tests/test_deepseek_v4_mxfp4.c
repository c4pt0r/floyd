#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "quants.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static const float fp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float e8m0(uint8_t value) {
    return ldexpf(1.0f, (int)value - 127);
}

int main(void) {
    uint8_t packed[32];
    uint8_t scales[2] = {127, 128};
    uint8_t mxfp4[34];
    for (int i = 0; i < 32; i++) {
        uint8_t lo = (uint8_t)((2 * i) & 15);
        uint8_t hi = (uint8_t)((2 * i + 1) & 15);
        packed[i] = (uint8_t)(lo | (hi << 4));
    }

    CHECK(ds4q_deepseek_fp4_to_mxfp4(mxfp4, packed, scales, 1, 64));
    for (int block = 0; block < 2; block++) {
        CHECK(mxfp4[block * 17] == scales[block]);
        for (int column = 0; column < 32; column++) {
            int source_column = block * 32 + column;
            uint8_t source_byte = packed[source_column / 2];
            uint8_t source_code = (source_column & 1) ? source_byte >> 4
                                                     : source_byte & 15;
            uint8_t target_byte = mxfp4[block * 17 + 1 + column % 16];
            uint8_t target_code = column < 16 ? target_byte & 15
                                              : target_byte >> 4;
            float source_value = fp4_values[source_code] * e8m0(scales[block]);
            float target_value = fp4_values[target_code] * e8m0(mxfp4[block * 17]);
            CHECK(source_value == target_value);
        }
    }
    puts("DeepSeek V4 MXFP4 transcode: exact 64/64");
    return 0;
}
