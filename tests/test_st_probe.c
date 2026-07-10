#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../st.h"
#include "../st_probe.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    st_tensor tensors[6] = {
        {.dtype = ST_DTYPE_BF16},
        {.dtype = ST_DTYPE_F32},
        {.dtype = ST_DTYPE_I64},
        {.dtype = ST_DTYPE_F8_E8M0},
        {.dtype = ST_DTYPE_F8_E8M0},
        {.dtype = ST_DTYPE_F8_E4M3},
    };
    shards source = {.t = tensors, .n = 6};
    uint64_t counts[ST_DTYPE_COUNT];

    st_dtype_counts(&source, counts);

    CHECK(counts[ST_DTYPE_BF16] == 1);
    CHECK(counts[ST_DTYPE_F32] == 1);
    CHECK(counts[ST_DTYPE_I64] == 1);
    CHECK(counts[ST_DTYPE_F8_E8M0] == 2);
    CHECK(counts[ST_DTYPE_F8_E4M3] == 1);
    CHECK(strcmp(st_dtype_name(ST_DTYPE_F8_E8M0), "F8_E8M0") == 0);
    CHECK(strcmp(st_dtype_name(-1), "UNKNOWN") == 0);
    puts("safetensors probe tests: ok");
    return 0;
}
