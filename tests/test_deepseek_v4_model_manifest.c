#include <stdio.h>

#include "../deepseek_v4_model_manifest.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark>\n", argv[0]);
        return 2;
    }

    shards source;
    DeepSeekV4ModelManifest model;
    CHECK(deepseek_v4_model_manifest_load(&model, &source, argv[1]));
    CHECK(model.dim == 4096);
    CHECK(model.n_layers == 43 && model.loaded_layers == 43);
    CHECK(model.n_hash_layers == 3 && model.n_mtp_layers == 3);
    CHECK(model.n_experts == 256 && model.n_shared_experts == 1);
    CHECK(model.n_heads == 64 && model.head_dim == 512);
    CHECK(model.csa_layers == 21);
    CHECK(model.hca_layers == 20);
    CHECK(model.sliding_layers == 2);
    CHECK(model.resolved_tensors > 67000);
    CHECK(source.nfd == 48 && source.n == 72317);
    printf("DeepSeek V4 manifest: layers=%d/43 sliding=%d csa=%d hca=%d tensors=%d\n",
           model.loaded_layers, model.sliding_layers, model.csa_layers,
           model.hca_layers, model.resolved_tensors);
    puts("DeepSeek V4 model manifest tests: ok");
    return 0;
}
