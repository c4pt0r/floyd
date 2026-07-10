#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../v4_quant.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(fabsf((actual) - (expected)) <= (tolerance))

static int test_all_fp4_codes_and_nibble_order(void) {
    const float expected[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    uint8_t packed[8];
    for (int i = 0; i < 8; i++) packed[i] = (uint8_t)((2 * i) | ((2 * i + 1) << 4));

    for (int i = 0; i < 16; i++) {
        CHECK(v4_fp4_code_value((uint8_t)i) == expected[i]);
        CHECK(v4_fp4_packed_value(packed, i) == expected[i]);
    }
    return 0;
}

static int test_e8m0_values(void) {
    CHECK(v4_e8m0_to_f32(0) == ldexpf(1.0f, -127));
    CHECK(v4_e8m0_to_f32(125) == 0.25f);
    CHECK(v4_e8m0_to_f32(126) == 0.5f);
    CHECK(v4_e8m0_to_f32(127) == 1.0f);
    CHECK(v4_e8m0_to_f32(128) == 2.0f);
    CHECK(v4_e8m0_to_f32(254) == ldexpf(1.0f, 127));
    CHECK(isnan(v4_e8m0_to_f32(255)));
    return 0;
}

static int test_dequant_row_changes_scale_at_32_values(void) {
    uint8_t packed[32];
    uint8_t scales[2] = {126, 128};
    float out[64];
    for (int i = 0; i < 32; i++) packed[i] = 0x22;

    v4_fp4_dequant_row(out, packed, scales, 64);

    for (int i = 0; i < 32; i++) CHECK_NEAR(out[i], 0.5f, 0.0f);
    for (int i = 32; i < 64; i++) CHECK_NEAR(out[i], 2.0f, 0.0f);
    return 0;
}

int main(void) {
    CHECK(test_all_fp4_codes_and_nibble_order() == 0);
    CHECK(test_e8m0_values() == 0);
    CHECK(test_dequant_row_changes_scale_at_32_values() == 0);
    puts("v4 quant tests: ok");
    return 0;
}
