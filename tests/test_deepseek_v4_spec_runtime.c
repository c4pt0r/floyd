#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../deepseek_v4_runtime.h"

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    shards oracle;
    st_init(&oracle, argv[2]);
    int64_t expected_first = -1, expected_input = -1, expected_ids[6];
    st_read_raw(&oracle, "base.first_id", &expected_first, 0);
    st_read_raw(&oracle, "base.input_id", &expected_input, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected_ids, 0);

    int prompt_ids[5] = {0, 128803, 33310, 128804, 128822};
    DeepSeekV4Runtime runtime;
    const float *logits = NULL;
    CHECK(deepseek_v4_runtime_init(&runtime, argv[1], 16));
    CHECK(deepseek_v4_runtime_forward(&runtime, prompt_ids, 5, &logits));
    int first = deepseek_v4_runtime_argmax(logits);
    CHECK(first == expected_first);
    CHECK(deepseek_v4_runtime_dspark_prefill(&runtime));
    CHECK(deepseek_v4_runtime_forward(&runtime, &first, 1, &logits));
    int input = deepseek_v4_runtime_argmax(logits);
    CHECK(input == expected_input);
    int64_t proposals[6];
    float confidence[5];
    CHECK(deepseek_v4_runtime_dspark_propose(&runtime, input, proposals, confidence));
    int hits = 0;
    for (int i = 0; i < 6; i++) if (proposals[i] == expected_ids[i]) hits++;
    printf("DeepSeek V4 runtime DSpark: first=%d input=%d proposals=%d/6\n",
           first, input, hits);
    CHECK(proposals[0] == deepseek_v4_runtime_argmax(logits));
    CHECK(hits == 6);
    int accepted = 0;
    int chosen = deepseek_v4_runtime_verify_token(logits, proposals[0], &accepted);
    CHECK(accepted == 1 && chosen == input);
    int64_t wrong = proposals[0] == 0 ? 1 : 0;
    chosen = deepseek_v4_runtime_verify_token(logits, wrong, &accepted);
    CHECK(accepted == 0 && chosen == input);
    deepseek_v4_runtime_free(&runtime);
    return 0;
}
