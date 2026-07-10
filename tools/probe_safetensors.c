#include <stdint.h>
#include <stdio.h>

#include "../st_probe.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model-dir>\n", argv[0]);
        return 2;
    }

    shards source;
    st_init(&source, argv[1]);

    uint64_t counts[ST_DTYPE_COUNT];
    st_dtype_counts(&source, counts);
    printf("shards=%d\n", source.nfd);
    printf("tensors=%d\n", source.n);
    for (int dtype = 0; dtype < ST_DTYPE_COUNT; dtype++)
        printf("%s=%llu\n", st_dtype_name(dtype), (unsigned long long)counts[dtype]);
    return 0;
}
