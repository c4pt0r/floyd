#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    CHECK(st_dtype_code("BF16") == 0);
    CHECK(st_dtype_code("F16") == 1);
    CHECK(st_dtype_code("F32") == 2);
    CHECK(st_dtype_code("U8") == 3);
    CHECK(st_dtype_code("I8") == 3);
    CHECK(st_dtype_code("I64") == 4);
    CHECK(st_dtype_code("F8_E8M0") == 5);
    CHECK(st_dtype_code("F8_E4M3") == 6);
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));

    puts("safetensors primitive tests: ok");
    return 0;
}
