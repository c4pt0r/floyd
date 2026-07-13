#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "../st.h"

#define D 4096
#define HC 4
#define DRAFT 5
#define KV 512

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_block_q4_k;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float max_abs(const float *actual, const float *expected, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) return INFINITY;
        float error = fabsf(actual[i] - expected[i]);
        if (error > result) result = error;
    }
    return result;
}

static void similarity(const float *actual, const float *expected, size_t count,
                       float *rmse, float *cosine) {
    double error2 = 0.0, aa = 0.0, ee = 0.0, ae = 0.0;
    for (size_t i = 0; i < count; i++) {
        double delta = (double)actual[i] - expected[i];
        error2 += delta * delta;
        aa += (double)actual[i] * actual[i];
        ee += (double)expected[i] * expected[i];
        ae += (double)actual[i] * expected[i];
    }
    *rmse = (float)sqrt(error2 / count);
    *cosine = (float)(ae / sqrt(aa * ee));
}

static int test_dense_q4_k(void **mapped_out) {
    enum { ROWS = 8, COLS = 256, TOKENS = 2, MAP_BYTES = 16384 };
    void *mapped = NULL;
    CHECK(posix_memalign(&mapped, MAP_BYTES, MAP_BYTES) == 0);
    memset(mapped, 0, MAP_BYTES);
    test_block_q4_k *blocks = mapped;
    for (int row = 0; row < ROWS; row++) {
        blocks[row].d = 0x3c00;
        for (int group = 0; group < 4; group++)
            blocks[row].scales[group] = 1;
        for (int group = 4; group < 8; group++)
            blocks[row].scales[group + 4] = 1;
        for (int pair = 0; pair < 4; pair++) {
            for (int col = 0; col < 32; col++) {
                const uint8_t low = (uint8_t)((row + pair + col) & 15);
                const uint8_t high = (uint8_t)((15 - row + pair + col) & 15);
                blocks[row].qs[pair * 32 + col] =
                    (uint8_t)(low | (high << 4));
            }
        }
    }

    float input[TOKENS * COLS];
    float expected[TOKENS * ROWS];
    float actual[TOKENS * ROWS];
    for (int token = 0; token < TOKENS; token++)
        for (int col = 0; col < COLS; col++)
            input[token * COLS + col] =
                0.01f * (float)(((token + 3) * (col + 5)) % 29 - 14);
    for (int token = 0; token < TOKENS; token++) {
        for (int row = 0; row < ROWS; row++) {
            float sum = 0.0f;
            for (int group = 0; group < 8; group++) {
                const int pair = group / 2;
                const int shift = (group & 1) * 4;
                for (int col = 0; col < 32; col++) {
                    const float value =
                        (float)((blocks[row].qs[pair * 32 + col] >> shift) & 15);
                    sum += input[token * COLS + group * 32 + col] * value;
                }
            }
            expected[token * ROWS + row] = sum;
        }
    }

    CHECK(ds4_gpu_set_model_map_range(mapped, MAP_BYTES, 0, MAP_BYTES,
                                       sizeof(test_block_q4_k) * ROWS));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    CHECK(x && out);
    CHECK(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
    CHECK(ds4_gpu_matmul_q4_k_tensor(out, mapped, MAP_BYTES, 0,
                                     COLS, ROWS, x, TOKENS));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    const float error = max_abs(actual, expected, TOKENS * ROWS);
    printf("DeepSeek V4 dense Q4_K Metal: max_abs=%.9g\n", error);
    CHECK(error < 5e-5f);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    *mapped_out = mapped;
    return 0;
}

static int test_attention_q8_hc_exact_rows(void **mapped_out) {
    enum { TOKENS = 3, IN_DIM = 2048, N_EMBD = 4096, N_HC = 4, MIX_HC = 24 };
    const size_t page_size = 16384u;
    const size_t blocks_per_row = IN_DIM / 32u;
    const size_t weight_bytes =
        (size_t)N_EMBD * blocks_per_row * sizeof(test_block_q8_0);
    const size_t map_bytes = (weight_bytes + page_size - 1u) & ~(page_size - 1u);
    void *mapped = NULL;
    CHECK(posix_memalign(&mapped, page_size, map_bytes) == 0);
    memset(mapped, 0, map_bytes);
    test_block_q8_0 *weights = mapped;
    for (int row = 0; row < N_EMBD; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            test_block_q8_0 *w = weights + (size_t)row * blocks_per_row + block;
            w->d = 0x3000u;
            for (int i = 0; i < 32; i++)
                w->qs[i] = (int8_t)(((row * 3 + (int)block * 5 + i * 7) % 15) - 7);
        }
    }

    const size_t input_count = (size_t)TOKENS * IN_DIM;
    const size_t block_count = (size_t)TOKENS * N_EMBD;
    const size_t hc_count = (size_t)TOKENS * N_HC * N_EMBD;
    const size_t split_count = (size_t)TOKENS * MIX_HC;
    float *input = malloc(input_count * sizeof(*input));
    float *residual = malloc(hc_count * sizeof(*residual));
    float *split = malloc(split_count * sizeof(*split));
    float *batch_block = malloc(block_count * sizeof(*batch_block));
    float *row_block = malloc(block_count * sizeof(*row_block));
    float *batch_hc = malloc(hc_count * sizeof(*batch_hc));
    float *row_hc = malloc(hc_count * sizeof(*row_hc));
    CHECK(input && residual && split && batch_block && row_block && batch_hc && row_hc);
    for (size_t i = 0; i < input_count; i++)
        input[i] = (float)((int)((i * 13u + 17u) % 63u) - 31) / 128.0f;
    for (size_t i = 0; i < hc_count; i++)
        residual[i] = (float)((int)((i * 11u + 3u) % 47u) - 23) / 64.0f;
    for (int token = 0; token < TOKENS; token++) {
        float *row = split + (size_t)token * MIX_HC;
        for (int i = 0; i < N_HC; i++) row[i] = 0.0f;
        for (int i = 0; i < N_HC; i++) row[N_HC + i] = 0.25f * (float)(i + 1);
        for (int dst = 0; dst < N_HC; dst++)
            for (int src = 0; src < N_HC; src++)
                row[2 * N_HC + dst * N_HC + src] =
                    dst == src ? 0.75f : 0.0625f;
    }

#define TENSOR(name, count) \
    ds4_gpu_tensor *name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(float)); \
    CHECK(name)
    TENSOR(x, input_count);
    TENSOR(res, hc_count);
    TENSOR(post, split_count);
    TENSOR(block, block_count);
    TENSOR(hc, hc_count);
    TENSOR(x_one, IN_DIM);
    TENSOR(res_one, N_HC * N_EMBD);
    TENSOR(post_one, MIX_HC);
    TENSOR(block_one, N_EMBD);
    TENSOR(hc_one, N_HC * N_EMBD);
#undef TENSOR

    CHECK(ds4_gpu_set_model_map_range(mapped, map_bytes, 0, map_bytes,
                                       weight_bytes));
    CHECK(ds4_gpu_tensor_write(x, 0, input, input_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(res, 0, residual, hc_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(post, 0, split, split_count * sizeof(float)));

    ds4_gpu_kernel_stats before, after;
    ds4_gpu_get_kernel_stats(&before);
    CHECK(ds4_gpu_matmul_q8_0_hc_expand_decode_order_batch_tensor(
        hc, block, mapped, map_bytes, 0, IN_DIM, N_EMBD, x, res, post,
        N_EMBD, N_HC, TOKENS));
    CHECK(ds4_gpu_tensor_read(block, 0, batch_block,
                              block_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(hc, 0, batch_hc, hc_count * sizeof(float)));
    for (int token = 0; token < TOKENS; token++) {
        CHECK(ds4_gpu_tensor_write(x_one, 0,
                                   input + (size_t)token * IN_DIM,
                                   (uint64_t)IN_DIM * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(res_one, 0,
                                   residual + (size_t)token * N_HC * N_EMBD,
                                   (uint64_t)N_HC * N_EMBD * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(post_one, 0,
                                   split + (size_t)token * MIX_HC,
                                   (uint64_t)MIX_HC * sizeof(float)));
        CHECK(ds4_gpu_matmul_q8_0_hc_expand_decode_order_batch_tensor(
            hc_one, block_one, mapped, map_bytes, 0, IN_DIM, N_EMBD,
            x_one, res_one, post_one, N_EMBD, N_HC, 1));
        CHECK(ds4_gpu_tensor_read(block_one, 0,
                                  row_block + (size_t)token * N_EMBD,
                                  (uint64_t)N_EMBD * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(hc_one, 0,
                                  row_hc + (size_t)token * N_HC * N_EMBD,
                                  (uint64_t)N_HC * N_EMBD * sizeof(float)));
    }
    ds4_gpu_get_kernel_stats(&after);
    const float block_error = max_abs(batch_block, row_block, block_count);
    const float hc_error = max_abs(batch_hc, row_hc, hc_count);
    printf("DeepSeek V4 attention Q8 HC exact rows Metal: "
           "block_max_abs=%.9g hc_max_abs=%.9g calls=%llu\n",
           block_error, hc_error,
           (unsigned long long)(after.tiny_batch_attn_hc_exact_rows_calls -
                                before.tiny_batch_attn_hc_exact_rows_calls));
    CHECK(block_error == 0.0f);
    CHECK(hc_error == 0.0f);
    CHECK(after.tiny_batch_attn_hc_exact_rows_calls ==
          before.tiny_batch_attn_hc_exact_rows_calls + 1);

    ds4_gpu_tensor_free(hc_one);
    ds4_gpu_tensor_free(block_one);
    ds4_gpu_tensor_free(post_one);
    ds4_gpu_tensor_free(res_one);
    ds4_gpu_tensor_free(x_one);
    ds4_gpu_tensor_free(hc);
    ds4_gpu_tensor_free(block);
    ds4_gpu_tensor_free(post);
    ds4_gpu_tensor_free(res);
    ds4_gpu_tensor_free(x);
    free(row_hc);
    free(batch_hc);
    free(row_block);
    free(batch_block);
    free(split);
    free(residual);
    free(input);
    *mapped_out = mapped;
    return 0;
}

static int test_attention_low_q8_exact_rows(void **mapped_out) {
    enum { TOKENS = 3, GROUPS = 8, RANK = 256, GROUP_DIM = 2048 };
    const size_t page_size = 16384u;
    const size_t blocks_per_row = GROUP_DIM / 32u;
    const size_t weight_bytes =
        (size_t)GROUPS * RANK * blocks_per_row * sizeof(test_block_q8_0);
    const size_t map_bytes = (weight_bytes + page_size - 1u) & ~(page_size - 1u);
    void *mapped = NULL;
    CHECK(posix_memalign(&mapped, page_size, map_bytes) == 0);
    memset(mapped, 0, map_bytes);
    test_block_q8_0 *weights = mapped;
    const size_t weight_blocks = (size_t)GROUPS * RANK * blocks_per_row;
    for (size_t block = 0; block < weight_blocks; block++) {
        weights[block].d = 0x3000u;
        for (int i = 0; i < 32; i++)
            weights[block].qs[i] =
                (int8_t)(((block * 5u + (size_t)i * 7u) % 15u) - 7);
    }

    const size_t heads_count = (size_t)TOKENS * GROUPS * GROUP_DIM;
    const size_t low_count = (size_t)TOKENS * GROUPS * RANK;
    float *heads = malloc(heads_count * sizeof(*heads));
    float *batch = malloc(low_count * sizeof(*batch));
    float *rowwise = malloc(low_count * sizeof(*rowwise));
    CHECK(heads && batch && rowwise);
    for (size_t i = 0; i < heads_count; i++)
        heads[i] = (float)((int)((i * 17u + 11u) % 67u) - 33) / 128.0f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *x_one = ds4_gpu_tensor_alloc(
        (uint64_t)GROUPS * GROUP_DIM * sizeof(float));
    ds4_gpu_tensor *low_one = ds4_gpu_tensor_alloc(
        (uint64_t)GROUPS * RANK * sizeof(float));
    CHECK(x && low && x_one && low_one);
    CHECK(ds4_gpu_set_model_map_range(mapped, map_bytes, 0, map_bytes,
                                       weight_bytes));
    CHECK(ds4_gpu_tensor_write(x, 0, heads, heads_count * sizeof(float)));

    ds4_gpu_kernel_stats before, after;
    ds4_gpu_get_kernel_stats(&before);
    CHECK(ds4_gpu_attention_output_low_q8_batch_tensor(
        low, mapped, map_bytes, 0, GROUP_DIM, RANK, GROUPS, x, TOKENS));
    CHECK(ds4_gpu_tensor_read(low, 0, batch, low_count * sizeof(float)));
    for (int token = 0; token < TOKENS; token++) {
        CHECK(ds4_gpu_tensor_write(x_one, 0,
                                   heads + (size_t)token * GROUPS * GROUP_DIM,
                                   (uint64_t)GROUPS * GROUP_DIM * sizeof(float)));
        CHECK(ds4_gpu_attention_output_low_q8_batch_tensor(
            low_one, mapped, map_bytes, 0, GROUP_DIM, RANK, GROUPS, x_one, 1));
        CHECK(ds4_gpu_tensor_read(low_one, 0,
                                  rowwise + (size_t)token * GROUPS * RANK,
                                  (uint64_t)GROUPS * RANK * sizeof(float)));
    }
    ds4_gpu_get_kernel_stats(&after);
    const float error = max_abs(batch, rowwise, low_count);
    printf("DeepSeek V4 attention low Q8 exact rows Metal: "
           "max_abs=%.9g calls=%llu\n",
           error,
           (unsigned long long)(after.tiny_batch_attn_low_exact_rows_calls -
                                before.tiny_batch_attn_low_exact_rows_calls));
    CHECK(error == 0.0f);
    CHECK(after.tiny_batch_attn_low_exact_rows_calls ==
          before.tiny_batch_attn_low_exact_rows_calls + 1);

    ds4_gpu_tensor_free(low_one);
    ds4_gpu_tensor_free(x_one);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(x);
    free(rowwise);
    free(batch);
    free(heads);
    *mapped_out = mapped;
    return 0;
}

static int test_shared_q8_exact_rows(void **mapped_out) {
    enum { TOKENS = 3, IN_DIM = 4096, HIDDEN = 2048, N_EMBD = 4096,
           N_HC = 4, MIX_HC = 24 };
    const size_t page_size = 16384u;
    const size_t gate_bytes =
        (size_t)HIDDEN * (IN_DIM / 32u) * sizeof(test_block_q8_0);
    const size_t down_bytes =
        (size_t)N_EMBD * (HIDDEN / 32u) * sizeof(test_block_q8_0);
    const size_t weight_bytes = gate_bytes * 2u + down_bytes;
    const size_t map_bytes = (weight_bytes + page_size - 1u) & ~(page_size - 1u);
    void *mapped = NULL;
    CHECK(posix_memalign(&mapped, page_size, map_bytes) == 0);
    memset(mapped, 0, map_bytes);
    test_block_q8_0 *weights = mapped;
    const size_t weight_blocks = weight_bytes / sizeof(*weights);
    for (size_t block = 0; block < weight_blocks; block++) {
        weights[block].d = 0x3000u;
        for (int i = 0; i < 32; i++)
            weights[block].qs[i] =
                (int8_t)(((block * 11u + (size_t)i * 3u) % 15u) - 7);
    }

    const size_t input_count = (size_t)TOKENS * IN_DIM;
    const size_t hidden_count = (size_t)TOKENS * HIDDEN;
    const size_t embd_count = (size_t)TOKENS * N_EMBD;
    const size_t hc_count = (size_t)TOKENS * N_HC * N_EMBD;
    const size_t split_count = (size_t)TOKENS * MIX_HC;
    float *input = malloc(input_count * sizeof(*input));
    float *routed = malloc(embd_count * sizeof(*routed));
    float *residual = malloc(hc_count * sizeof(*residual));
    float *split = malloc(split_count * sizeof(*split));
    float *batch_gate = malloc(hidden_count * sizeof(*batch_gate));
    float *batch_up = malloc(hidden_count * sizeof(*batch_up));
    float *batch_mid = malloc(hidden_count * sizeof(*batch_mid));
    float *batch_shared = malloc(embd_count * sizeof(*batch_shared));
    float *batch_hc = malloc(hc_count * sizeof(*batch_hc));
    float *row_gate = malloc(hidden_count * sizeof(*row_gate));
    float *row_up = malloc(hidden_count * sizeof(*row_up));
    float *row_mid = malloc(hidden_count * sizeof(*row_mid));
    float *row_shared = malloc(embd_count * sizeof(*row_shared));
    float *row_hc = malloc(hc_count * sizeof(*row_hc));
    CHECK(input && routed && residual && split && batch_gate && batch_up &&
          batch_mid && batch_shared && batch_hc && row_gate && row_up &&
          row_mid && row_shared && row_hc);
    for (size_t i = 0; i < input_count; i++)
        input[i] = (float)((int)((i * 19u + 5u) % 71u) - 35) / 128.0f;
    for (size_t i = 0; i < embd_count; i++)
        routed[i] = (float)((int)((i * 7u + 13u) % 43u) - 21) / 64.0f;
    for (size_t i = 0; i < hc_count; i++)
        residual[i] = (float)((int)((i * 13u + 9u) % 53u) - 26) / 64.0f;
    for (int token = 0; token < TOKENS; token++) {
        float *row = split + (size_t)token * MIX_HC;
        for (int i = 0; i < N_HC; i++) row[i] = 0.0f;
        for (int i = 0; i < N_HC; i++) row[N_HC + i] = 0.125f * (float)(i + 1);
        for (int dst = 0; dst < N_HC; dst++)
            for (int src = 0; src < N_HC; src++)
                row[2 * N_HC + dst * N_HC + src] =
                    dst == src ? 0.75f : 0.0625f;
    }

#define TENSOR(name, count) \
    ds4_gpu_tensor *name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(float)); \
    CHECK(name)
    TENSOR(x, input_count); TENSOR(routed_t, embd_count);
    TENSOR(res, hc_count); TENSOR(post, split_count);
    TENSOR(gate, hidden_count); TENSOR(up, hidden_count); TENSOR(mid, hidden_count);
    TENSOR(shared, embd_count); TENSOR(hc, hc_count);
    TENSOR(x_one, IN_DIM); TENSOR(routed_one, N_EMBD);
    TENSOR(res_one, N_HC * N_EMBD); TENSOR(post_one, MIX_HC);
    TENSOR(gate_one, HIDDEN); TENSOR(up_one, HIDDEN); TENSOR(mid_one, HIDDEN);
    TENSOR(shared_one, N_EMBD); TENSOR(hc_one, N_HC * N_EMBD);
#undef TENSOR
    CHECK(ds4_gpu_set_model_map_range(mapped, map_bytes, 0, map_bytes,
                                       weight_bytes));
    CHECK(ds4_gpu_tensor_write(x, 0, input, input_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(routed_t, 0, routed, embd_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(res, 0, residual, hc_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(post, 0, split, split_count * sizeof(float)));

    ds4_gpu_kernel_stats before, after;
    ds4_gpu_get_kernel_stats(&before);
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_decode_order_batch_tensor(
        gate, up, mid, mapped, map_bytes, 0, gate_bytes,
        IN_DIM, HIDDEN, x, TOKENS, 7.0f));
    CHECK(ds4_gpu_shared_down_hc_expand_q8_0_decode_order_batch_tensor(
        hc, shared, mapped, map_bytes, gate_bytes * 2u,
        HIDDEN, N_EMBD, mid, routed_t, res, post, N_EMBD, N_HC, TOKENS));
    CHECK(ds4_gpu_tensor_read(gate, 0, batch_gate, hidden_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(up, 0, batch_up, hidden_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(mid, 0, batch_mid, hidden_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(shared, 0, batch_shared, embd_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(hc, 0, batch_hc, hc_count * sizeof(float)));
    for (int token = 0; token < TOKENS; token++) {
        CHECK(ds4_gpu_tensor_write(x_one, 0, input + (size_t)token * IN_DIM,
                                   (uint64_t)IN_DIM * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(routed_one, 0,
                                   routed + (size_t)token * N_EMBD,
                                   (uint64_t)N_EMBD * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(res_one, 0,
                                   residual + (size_t)token * N_HC * N_EMBD,
                                   (uint64_t)N_HC * N_EMBD * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(post_one, 0,
                                   split + (size_t)token * MIX_HC,
                                   (uint64_t)MIX_HC * sizeof(float)));
        CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_decode_order_batch_tensor(
            gate_one, up_one, mid_one, mapped, map_bytes, 0, gate_bytes,
            IN_DIM, HIDDEN, x_one, 1, 7.0f));
        CHECK(ds4_gpu_shared_down_hc_expand_q8_0_decode_order_batch_tensor(
            hc_one, shared_one, mapped, map_bytes, gate_bytes * 2u,
            HIDDEN, N_EMBD, mid_one, routed_one, res_one, post_one,
            N_EMBD, N_HC, 1));
        CHECK(ds4_gpu_tensor_read(gate_one, 0, row_gate + (size_t)token * HIDDEN,
                                  (uint64_t)HIDDEN * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(up_one, 0, row_up + (size_t)token * HIDDEN,
                                  (uint64_t)HIDDEN * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(mid_one, 0, row_mid + (size_t)token * HIDDEN,
                                  (uint64_t)HIDDEN * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(shared_one, 0,
                                  row_shared + (size_t)token * N_EMBD,
                                  (uint64_t)N_EMBD * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(hc_one, 0,
                                  row_hc + (size_t)token * N_HC * N_EMBD,
                                  (uint64_t)N_HC * N_EMBD * sizeof(float)));
    }
    ds4_gpu_get_kernel_stats(&after);
    const float gate_error = max_abs(batch_gate, row_gate, hidden_count);
    const float up_error = max_abs(batch_up, row_up, hidden_count);
    const float mid_error = max_abs(batch_mid, row_mid, hidden_count);
    const float shared_error = max_abs(batch_shared, row_shared, embd_count);
    const float hc_error = max_abs(batch_hc, row_hc, hc_count);
    printf("DeepSeek V4 shared Q8 exact rows Metal: "
           "gate=%.9g up=%.9g mid=%.9g down=%.9g hc=%.9g calls=%llu/%llu\n",
           gate_error, up_error, mid_error, shared_error, hc_error,
           (unsigned long long)(after.tiny_batch_shared_gate_up_exact_rows_calls -
                                before.tiny_batch_shared_gate_up_exact_rows_calls),
           (unsigned long long)(after.tiny_batch_shared_down_exact_rows_calls -
                                before.tiny_batch_shared_down_exact_rows_calls));
    CHECK(gate_error == 0.0f && up_error == 0.0f && mid_error == 0.0f);
    CHECK(shared_error == 0.0f && hc_error == 0.0f);
    CHECK(after.tiny_batch_shared_gate_up_exact_rows_calls ==
          before.tiny_batch_shared_gate_up_exact_rows_calls + 1);
    CHECK(after.tiny_batch_shared_down_exact_rows_calls ==
          before.tiny_batch_shared_down_exact_rows_calls + 1);

    ds4_gpu_tensor_free(hc_one); ds4_gpu_tensor_free(shared_one);
    ds4_gpu_tensor_free(mid_one); ds4_gpu_tensor_free(up_one); ds4_gpu_tensor_free(gate_one);
    ds4_gpu_tensor_free(post_one); ds4_gpu_tensor_free(res_one);
    ds4_gpu_tensor_free(routed_one); ds4_gpu_tensor_free(x_one);
    ds4_gpu_tensor_free(hc); ds4_gpu_tensor_free(shared);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(post); ds4_gpu_tensor_free(res);
    ds4_gpu_tensor_free(routed_t); ds4_gpu_tensor_free(x);
    free(row_hc); free(row_shared); free(row_mid); free(row_up); free(row_gate);
    free(batch_hc); free(batch_shared); free(batch_mid); free(batch_up); free(batch_gate);
    free(split); free(residual); free(routed); free(input);
    *mapped_out = mapped;
    return 0;
}

static int test_attention_exact_batch_kv_broadcast(void **mapped_out) {
    enum { TOKENS = 3, HEADS = 8, HEAD_DIM = 512, RAW_CAP = 128,
           POS0 = 20, N_COMP = 5, RATIO = 4 };
    const size_t page_size = 16384u;
    void *mapped = NULL;
    CHECK(posix_memalign(&mapped, page_size, page_size) == 0);
    memset(mapped, 0, page_size);
    float *sinks = mapped;
    for (int head = 0; head < HEADS; head++)
        sinks[head] = (float)(head - 4) / 16.0f;

    const size_t q_count = (size_t)TOKENS * HEADS * HEAD_DIM;
    const size_t raw_count = (size_t)RAW_CAP * HEAD_DIM;
    const size_t comp_count = (size_t)N_COMP * HEAD_DIM;
    float *q = malloc(q_count * sizeof(*q));
    float *raw = malloc(raw_count * sizeof(*raw));
    float *comp = malloc(comp_count * sizeof(*comp));
    float *reference = malloc(q_count * sizeof(*reference));
    float *actual = malloc(q_count * sizeof(*actual));
    CHECK(q && raw && comp && reference && actual);
    for (size_t i = 0; i < q_count; i++)
        q[i] = (float)((int)((i * 7u + 5u) % 43u) - 21) / 64.0f;
    for (size_t i = 0; i < raw_count; i++)
        raw[i] = (float)((int)((i * 11u + 3u) % 61u) - 30) / 64.0f;
    for (size_t i = 0; i < comp_count; i++)
        comp[i] = (float)((int)((i * 13u + 9u) % 47u) - 23) / 64.0f;

    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp_t = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    CHECK(q_t && raw_t && comp_t && heads_t);
    CHECK(ds4_gpu_set_model_map_range(mapped, page_size, 0, page_size,
                                       HEADS * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(q_t, 0, q, q_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(raw_t, 0, raw, raw_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(comp_t, 0, comp, comp_count * sizeof(float)));

    setenv("DS4_METAL_DISABLE_EXACT_BATCH_KV_BROADCAST", "1", 1);
    CHECK(ds4_gpu_attention_decode_exact_batch_heads_tensor(
        heads_t, mapped, page_size, 0, q_t, raw_t, comp_t, 0,
        TOKENS, POS0, RAW_CAP, N_COMP, RAW_CAP, RATIO, HEADS, HEAD_DIM));
    CHECK(ds4_gpu_tensor_read(heads_t, 0, reference, q_count * sizeof(float)));
    unsetenv("DS4_METAL_DISABLE_EXACT_BATCH_KV_BROADCAST");

    ds4_gpu_kernel_stats before, after;
    ds4_gpu_get_kernel_stats(&before);
    CHECK(ds4_gpu_attention_decode_exact_batch_heads_tensor(
        heads_t, mapped, page_size, 0, q_t, raw_t, comp_t, 0,
        TOKENS, POS0, RAW_CAP, N_COMP, RAW_CAP, RATIO, HEADS, HEAD_DIM));
    CHECK(ds4_gpu_tensor_read(heads_t, 0, actual, q_count * sizeof(float)));
    ds4_gpu_get_kernel_stats(&after);
    const float error = max_abs(actual, reference, q_count);
    printf("DeepSeek V4 exact batch attention KV broadcast Metal: "
           "max_abs=%.9g calls=%llu\n",
           error,
           (unsigned long long)(after.tiny_batch_attn_kv_broadcast_calls -
                                before.tiny_batch_attn_kv_broadcast_calls));
    CHECK(error < 3e-7f);
    CHECK(after.tiny_batch_attn_kv_broadcast_calls ==
          before.tiny_batch_attn_kv_broadcast_calls + 1);

    ds4_gpu_tensor_free(heads_t); ds4_gpu_tensor_free(comp_t);
    ds4_gpu_tensor_free(raw_t); ds4_gpu_tensor_free(q_t);
    free(actual); free(reference); free(comp); free(raw); free(q);
    *mapped_out = mapped;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base.gguf> <dspark.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }
    enum { VOCAB = 129280 };
    float *argmax_logits = malloc((size_t)VOCAB * sizeof(*argmax_logits));
    CHECK(argmax_logits);
    for (int i = 0; i < VOCAB; i++) argmax_logits[i] = -10.0f;
    argmax_logits[17] = 3.0f;
    argmax_logits[42] = 3.0f;
    CHECK(ds4_gpu_init());
    void *q4_mapped = NULL;
    CHECK(test_dense_q4_k(&q4_mapped) == 0);
    void *attn_hc_mapped = NULL;
    CHECK(test_attention_q8_hc_exact_rows(&attn_hc_mapped) == 0);
    void *attn_low_mapped = NULL;
    CHECK(test_attention_low_q8_exact_rows(&attn_low_mapped) == 0);
    void *shared_mapped = NULL;
    CHECK(test_shared_q8_exact_rows(&shared_mapped) == 0);
    void *attn_broadcast_mapped = NULL;
    CHECK(test_attention_exact_batch_kv_broadcast(&attn_broadcast_mapped) == 0);
    ds4_gpu_kernel_stats init_stats;
    ds4_gpu_get_kernel_stats(&init_stats);
    CHECK(init_stats.exact_q8_pipeline_warmups == 3);
    ds4_gpu_tensor *argmax_input =
        ds4_gpu_tensor_alloc((uint64_t)VOCAB * sizeof(*argmax_logits));
    ds4_gpu_tensor *argmax_output = ds4_gpu_tensor_alloc(sizeof(int32_t));
    CHECK(argmax_input && argmax_output);
    CHECK(ds4_gpu_tensor_write(argmax_input, 0, argmax_logits,
                               (uint64_t)VOCAB * sizeof(*argmax_logits)));
    CHECK(ds4_gpu_argmax_tensor(argmax_output, argmax_input, VOCAB));
    int32_t argmax_id = -1;
    CHECK(ds4_gpu_tensor_read(argmax_output, 0, &argmax_id, sizeof(argmax_id)));
    CHECK(argmax_id == 17);
    ds4_gpu_kernel_stats argmax_stats_before;
    ds4_gpu_get_kernel_stats(&argmax_stats_before);
    CHECK(argmax_stats_before.fast_argmax_calls > 0);
    ds4_gpu_tensor_free(argmax_output);
    ds4_gpu_tensor_free(argmax_input);
    free(argmax_logits);

    shards oracle;
    st_init(&oracle, argv[3]);
    float *main_x = malloc(D * sizeof(float));
    float *prefill = malloc(3u * DRAFT * KV * sizeof(float));
    float *actual = malloc(3u * DRAFT * HC * D * sizeof(float));
    float *expected = malloc(DRAFT * HC * D * sizeof(float));
    int64_t input_id = -1;
    int64_t expected_ids[6];
    int output_ids[6];
    CHECK(main_x && prefill && actual && expected);
    st_read_raw(&oracle, "dspark.current_main_x", main_x, 0);
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected_ids, 0);
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.prefill_kv.%d", stage);
        st_read_raw(&oracle, name, prefill + stage * DRAFT * KV, 0);
    }

    CHECK(ds4_dspark_probe_stages(argv[1], argv[2], prefill, DRAFT,
                                  main_x, (int)input_id, actual, output_ids) == 0);
    ds4_gpu_kernel_stats argmax_stats_after;
    ds4_gpu_get_kernel_stats(&argmax_stats_after);
    CHECK(argmax_stats_after.fast_argmax_calls >=
          argmax_stats_before.fast_argmax_calls + DRAFT);
    float errors[3], rmses[3], cosines[3];
    for (int stage = 0; stage < 3; stage++) {
        char name[64];
        snprintf(name, sizeof(name), "dspark.stage.%d.output", stage);
        st_read_raw(&oracle, name, expected, 0);
        errors[stage] = max_abs(actual + (size_t)stage * DRAFT * HC * D,
                                expected, DRAFT * HC * D);
        similarity(actual + (size_t)stage * DRAFT * HC * D,
                   expected, DRAFT * HC * D, &rmses[stage], &cosines[stage]);
        CHECK(isfinite(errors[stage]));
    }
    printf("DeepSeek V4 DSpark Metal stages: max_abs=%.9g/%.9g/%.9g\n",
           errors[0], errors[1], errors[2]);
    printf("DeepSeek V4 DSpark Metal stages: rmse=%.9g/%.9g/%.9g cosine=%.9g/%.9g/%.9g\n",
           rmses[0], rmses[1], rmses[2], cosines[0], cosines[1], cosines[2]);
    int id_hits = 0;
    for (int i = 0; i < 6; i++) if (output_ids[i] == expected_ids[i]) id_hits++;
    printf("DeepSeek V4 DSpark Metal proposals: ids=%d/6\n", id_hits);
    printf("DeepSeek V4 DSpark fast argmax: calls=%llu tie_id=%d\n",
           (unsigned long long)argmax_stats_after.fast_argmax_calls,
           argmax_id);
    printf("  actual=%d,%d,%d,%d,%d,%d expected=%lld,%lld,%lld,%lld,%lld,%lld\n",
           output_ids[0], output_ids[1], output_ids[2], output_ids[3],
           output_ids[4], output_ids[5],
           (long long)expected_ids[0], (long long)expected_ids[1],
           (long long)expected_ids[2], (long long)expected_ids[3],
           (long long)expected_ids[4], (long long)expected_ids[5]);
    CHECK(rmses[0] < 0.02f);
    CHECK(rmses[1] < 0.05f);
    CHECK(rmses[2] < 7.0f);
    CHECK(cosines[0] > 0.999f);
    CHECK(cosines[1] > 0.999f);
    CHECK(cosines[2] > 0.70f);
    CHECK(id_hits == 6);
    free(expected);
    free(actual);
    free(prefill);
    free(main_x);
    free(attn_broadcast_mapped);
    free(shared_mapped);
    free(attn_low_mapped);
    free(attn_hc_mapped);
    free(q4_mapped);
    return 0;
}
