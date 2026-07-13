#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ds4.h"
#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base.gguf> <dspark.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }

    const float confidence_cases[] = {2.0f, -1.0f, 3.0f};
    CHECK(ds4_dspark_prefix1_capture_mask(20, 3) == 0);
    CHECK(ds4_dspark_prefix1_capture_mask(33, 3) == 4);
    CHECK(ds4_dspark_prefix1_capture_mask(125, 3) == (4 | 128));
    CHECK(ds4_dspark_prefix2_capture_mask(33, 3) == 4);
    CHECK(ds4_dspark_prefix2_capture_mask(125, 3) == (4 | 128));
    CHECK(ds4_dspark_snapshot_required(20, 3) == 0);
    CHECK(ds4_dspark_snapshot_required(33, 3) == 0);
    CHECK(ds4_dspark_snapshot_required(125, 3) == 0);
    CHECK(ds4_dspark_snapshot_required(125, 2) == 0);
    CHECK(ds4_dspark_snapshot_required(32, 4) == 1);
    CHECK(ds4_dspark_confident_prefix_length(
              confidence_cases, 3, 0.0f) == 3);
    CHECK(ds4_dspark_confident_prefix_length(
              confidence_cases, 3, 0.5f) == 1);
    CHECK(ds4_dspark_confident_prefix_length(
              confidence_cases, 3, 0.9f) == 0);
    CHECK(ds4_dspark_confident_prefix_length(
              confidence_cases, 3, 0.2f) == 3);
    CHECK(ds4_dspark_confident_prefix_length_staged(
              confidence_cases, 3, 0.5f, 0.2f) == 3);
    CHECK(ds4_dspark_confident_prefix_length_staged(
              confidence_cases, 3, 0.5f, 0.48f) == 1);

    shards oracle;
    st_init(&oracle, argv[3]);
    int64_t first_id = -1, input_id = -1, expected[6];
    float expected_confidence[5];
    st_read_raw(&oracle, "base.first_id", &first_id, 0);
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected, 0);
    st_read_raw(&oracle, "dspark.confidence", expected_confidence, 0);

    char lock_path[] = "/tmp/floyd-dspark-session.XXXXXX";
    int lock_fd = mkstemp(lock_path);
    CHECK(lock_fd >= 0);
    CHECK(close(lock_fd) == 0);
    CHECK(setenv("DS4_LOCK_FILE", lock_path, 1) == 0);

    ds4_engine_options options = {
        .model_path = argv[1],
        .dspark_path = argv[2],
        .backend = DS4_BACKEND_METAL,
        .mtp_draft_tokens = 4,
    };
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    char error[4096] = {0};
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(ds4_session_create(&session, engine, 64) == 0);

    int prompt_ids[] = {0, 128803, 33310, 128804, 128822};
    ds4_tokens prompt = {.v = prompt_ids, .len = 5, .cap = 5};
    CHECK(ds4_session_sync(session, &prompt, error, sizeof(error)) == 0);
    CHECK(ds4_session_argmax(session) == (int)first_id);
    CHECK(ds4_session_eval(session, (int)first_id, error, sizeof(error)) == 0);
    CHECK(ds4_session_argmax(session) == (int)input_id);

    int actual[6] = {-1, -1, -1, -1, -1, -1};
    int proposal_count = ds4_session_copy_dspark_proposals(session, actual, 6);
    CHECK(proposal_count == 4);
    int hits = 0;
    for (int i = 0; i < proposal_count; i++) if (actual[i] == expected[i]) hits++;
    printf("DeepSeek V4 DSpark resident proposals: ids=%d/%d\n",
           hits, proposal_count);
    printf("  actual=%d,%d,%d,%d,%d,%d expected=%lld,%lld,%lld,%lld,%lld,%lld\n",
           actual[0], actual[1], actual[2], actual[3], actual[4], actual[5],
           (long long)expected[0], (long long)expected[1],
           (long long)expected[2], (long long)expected[3],
           (long long)expected[4], (long long)expected[5]);
    for (int i = 0; i < proposal_count; i++) CHECK(actual[i] == expected[i]);
    for (int i = 0; i < proposal_count; i++)
        CHECK(actual[i] >= 0 && actual[i] < 129280);
    for (int i = proposal_count; i < 6; i++) CHECK(actual[i] == -1);
    float confidence[5] = {NAN, NAN, NAN, NAN, NAN};
    int confidence_count = ds4_session_copy_dspark_confidence(
        session, confidence, 5);
    CHECK(confidence_count == proposal_count - 1);
    float confidence_max_abs = 0.0f;
    for (int i = 0; i < confidence_count; i++) {
        CHECK(isfinite(confidence[i]));
        float error = fabsf(confidence[i] - expected_confidence[i]);
        if (error > confidence_max_abs) confidence_max_abs = error;
    }
    printf("DeepSeek V4 DSpark resident confidence: max_abs=%.9g\n",
           confidence_max_abs);
    CHECK(confidence_max_abs < 0.5f);

    ds4_session_free(session);
    session = NULL;
    ds4_session *greedy = NULL, *speculative = NULL;
    CHECK(ds4_session_create(&greedy, engine, 64) == 0);
    CHECK(ds4_session_create(&speculative, engine, 64) == 0);
    CHECK(ds4_session_sync(greedy, &prompt, error, sizeof(error)) == 0);
    CHECK(ds4_session_sync(speculative, &prompt, error, sizeof(error)) == 0);

    int greedy_ids[6], spec_ids[6], spec_count = 0, max_round = 0;
    int eval_rounds = 0;
    for (int i = 0; i < 6; i++) {
        greedy_ids[i] = ds4_session_argmax(greedy);
        CHECK(greedy_ids[i] >= 0);
        CHECK(ds4_session_eval(greedy, greedy_ids[i], error, sizeof(error)) == 0);
    }
    while (spec_count < 6) {
        int first = ds4_session_argmax(speculative);
        int round[6];
        int count = ds4_session_eval_speculative_argmax(
            speculative, first, 6 - spec_count, -1,
            round, 6, error, sizeof(error));
        CHECK(count > 0);
        eval_rounds++;
        if (count > max_round) max_round = count;
        for (int i = 0; i < count; i++) spec_ids[spec_count++] = round[i];
    }
    int stream_hits = 0;
    for (int i = 0; i < 6; i++) if (spec_ids[i] == greedy_ids[i]) stream_hits++;
    printf("DeepSeek V4 DSpark resident verify: greedy=%d/6 max_round=%d\n",
           stream_hits, max_round);
    CHECK(stream_hits == 6);
    CHECK(max_round > 1);

    ds4_session_spec_stats spec_stats;
    CHECK(ds4_session_get_spec_stats(speculative, &spec_stats));
    CHECK(spec_stats.rounds > 0);
    CHECK(spec_stats.proposed_tokens == 4);
    CHECK(spec_stats.proposed_tokens >= spec_stats.accepted_tokens);
    CHECK(spec_stats.accepted_tokens == (uint64_t)(spec_count - eval_rounds));
    CHECK(spec_stats.target_ms > 0.0);
    CHECK(spec_stats.proposal_ms > 0.0);
    CHECK(spec_stats.verify_ms > 0.0);
    CHECK(spec_stats.verify_layer_encode_ms > 0.0);
    CHECK(spec_stats.verify_layer_execute_ms > 0.0);
    CHECK(spec_stats.verify_head_ms > 0.0);
    CHECK(spec_stats.verify_read_ms > 0.0);
    CHECK(spec_stats.replay_ms == 0.0);

    ds4_session_kernel_stats kernel_stats;
    CHECK(ds4_session_get_kernel_stats(speculative, &kernel_stats));
    CHECK(kernel_stats.tiny_batch_pair_swiglu_calls >= 43);
    CHECK(kernel_stats.tiny_batch_activation_fallback_calls == 0);
    CHECK(kernel_stats.tiny_batch_shared_swiglu_calls >= 43);
    CHECK(kernel_stats.tiny_batch_router_calls >= 43);
    CHECK(kernel_stats.tiny_batch_attn_hc_calls >= 43);
    CHECK(kernel_stats.tiny_batch_raw_store_calls >= 43);
    CHECK(kernel_stats.tiny_batch_exact_attn_calls >= 43);
    CHECK(kernel_stats.tiny_batch_exact_q8_calls > 0);
    printf("DeepSeek V4 DSpark tiny batch fusion: moe_calls=%llu "
           "moe_fallback=%llu shared_calls=%llu router_calls=%llu "
           "attn_hc_calls=%llu raw_store_calls=%llu exact_attn_calls=%llu "
           "exact_q8_calls=%llu\n",
           (unsigned long long)kernel_stats.tiny_batch_pair_swiglu_calls,
           (unsigned long long)kernel_stats.tiny_batch_activation_fallback_calls,
           (unsigned long long)kernel_stats.tiny_batch_shared_swiglu_calls,
           (unsigned long long)kernel_stats.tiny_batch_router_calls,
           (unsigned long long)kernel_stats.tiny_batch_attn_hc_calls,
           (unsigned long long)kernel_stats.tiny_batch_raw_store_calls,
           (unsigned long long)kernel_stats.tiny_batch_exact_attn_calls,
           (unsigned long long)kernel_stats.tiny_batch_exact_q8_calls);
    printf("DeepSeek V4 DSpark timing: target=%.3f proposal=%.3f "
           "verify=%.3f layer_encode=%.3f layer_execute=%.3f "
           "head=%.3f read=%.3f replay=%.3f proposed=%llu accepted=%llu\n",
           spec_stats.target_ms, spec_stats.proposal_ms,
           spec_stats.verify_ms, spec_stats.verify_layer_encode_ms,
           spec_stats.verify_layer_execute_ms, spec_stats.verify_head_ms,
           spec_stats.verify_read_ms, spec_stats.replay_ms,
           (unsigned long long)spec_stats.proposed_tokens,
           (unsigned long long)spec_stats.accepted_tokens);

    ds4_session_free(speculative);
    ds4_session_free(greedy);
    ds4_engine_close(engine);
    CHECK(unlink(lock_path) == 0);
    unsetenv("DS4_LOCK_FILE");
    return 0;
}
