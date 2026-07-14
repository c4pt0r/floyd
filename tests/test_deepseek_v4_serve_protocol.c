#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_serve.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int fake_handler(void *user_data,
                        const DeepSeekV4ServeRequest *request,
                        DeepSeekV4ServeResponse *response,
                        char *error, size_t error_size) {
    int *calls = user_data;
    (*calls)++;
    (void)error;
    (void)error_size;
    response->text = request->prompt ? request->prompt : "messages";
    response->prompt_tokens = 12;
    response->cached_tokens = *calls == 2 ? 8 : 0;
    response->completion_tokens = 1;
    response->prompt_ms = 2.5;
    response->decode_ms = 4.0;
    response->cache_hit = *calls == 2;
    response->cache_prefix_tokens = response->cached_tokens;
    response->cache_entries = 1;
    response->cache_bytes = 128;
    return 1;
}

static int test_prompt_request(void) {
    DeepSeekV4ServeRequest request = {0};
    char error[256] = {0};
    CHECK(deepseek_v4_serve_parse_request(
        "{\"id\":\"req-1\",\"system\":\"be concise\","
        "\"prompt\":\"hello\",\"max_tokens\":3,"
        "\"temperature\":0.5,\"top_p\":0.8,\"draft\":0}",
        &request, error, sizeof(error)));
    CHECK(strcmp(request.id_json, "\"req-1\"") == 0);
    CHECK(strcmp(request.system, "be concise") == 0);
    CHECK(strcmp(request.prompt, "hello") == 0);
    CHECK(request.message_count == 0);
    CHECK(request.max_tokens == 3);
    CHECK(request.temperature == 0.5f);
    CHECK(request.top_p == 0.8f);
    CHECK(request.draft == 0);
    deepseek_v4_serve_request_free(&request);
    return 0;
}

static int test_messages_request(void) {
    DeepSeekV4ServeRequest request = {0};
    char error[256] = {0};
    CHECK(deepseek_v4_serve_parse_request(
        "{\"id\":42,\"messages\":["
        "{\"role\":\"system\",\"content\":\"rules\"},"
        "{\"role\":\"user\",\"content\":\"question\"}]}",
        &request, error, sizeof(error)));
    CHECK(strcmp(request.id_json, "42") == 0);
    CHECK(request.prompt == NULL);
    CHECK(request.message_count == 2);
    CHECK(strcmp(request.messages[0].role, "system") == 0);
    CHECK(strcmp(request.messages[1].content, "question") == 0);
    CHECK(request.max_tokens == -1);
    CHECK(request.temperature == 0.0f);
    CHECK(request.top_p == 1.0f);
    CHECK(request.draft == -1);
    deepseek_v4_serve_request_free(&request);
    return 0;
}

static int test_invalid_requests(void) {
    const char *invalid[] = {
        "{\"prompt\":\"missing id\"}",
        "{\"id\":1,\"prompt\":\"x\",\"messages\":[]}",
        "{\"id\":1.5,\"prompt\":\"x\"}",
        "{\"id\":1,\"prompt\":7}",
        "{\"id\":1,\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}",
        "{\"id\":1,\"prompt\":\"x\",\"max_tokens\":0}",
        "{\"id\":1,\"prompt\":\"x\",\"temperature\":-1}",
        "{\"id\":1,\"prompt\":\"x\",\"top_p\":1.1}",
        "{\"id\":1,\"prompt\":\"x\",\"draft\":17}",
        "{\"id\":1,\"prompt\":\"x\"} trailing",
        "{\"id\":"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        DeepSeekV4ServeRequest request = {0};
        char error[256] = {0};
        CHECK(!deepseek_v4_serve_parse_request(
            invalid[i], &request, error, sizeof(error)));
        CHECK(error[0] != 0);
        deepseek_v4_serve_request_free(&request);
    }
    return 0;
}

static int test_response_and_stdio(void) {
    char *encoded = NULL;
    size_t encoded_size = 0;
    FILE *output = open_memstream(&encoded, &encoded_size);
    CHECK(output != NULL);
    DeepSeekV4ServeResponse response = {
        .id_json = "\"quote-id\"",
        .text = "line 1\n\"中文\"",
        .prompt_tokens = 10,
        .cached_tokens = 6,
        .completion_tokens = 2,
        .prompt_ms = 1.25,
        .decode_ms = 3.5,
        .cache_hit = 1,
        .cache_prefix_tokens = 6,
        .cache_entries = 2,
        .cache_bytes = 4096,
    };
    CHECK(deepseek_v4_serve_write_response(output, &response));
    CHECK(fclose(output) == 0);
    CHECK(strstr(encoded, "\"id\":\"quote-id\"") != NULL);
    CHECK(strstr(encoded, "line 1\\n\\\"中文\\\"") != NULL);
    CHECK(strstr(encoded, "\"cache_hit\":true") != NULL);
    CHECK(strstr(encoded, "\"cache_prefix_tokens\":6") != NULL);
    CHECK(strstr(encoded, "\"error\":null") != NULL);
    free(encoded);

    FILE *input = tmpfile();
    output = tmpfile();
    CHECK(input && output);
    fputs("{broken\n", input);
    fputs("{\"id\":1,\"prompt\":\"first\"}\n", input);
    fputs("{\"id\":2,\"prompt\":\"second\"}\n", input);
    rewind(input);
    int calls = 0;
    CHECK(deepseek_v4_serve_stdio(input, output, fake_handler, &calls) == 0);
    CHECK(calls == 2);
    rewind(output);
    char line[4096];
    CHECK(fgets(line, sizeof(line), output) != NULL);
    CHECK(strstr(line, "\"error\":{") != NULL);
    CHECK(fgets(line, sizeof(line), output) != NULL);
    CHECK(strstr(line, "\"id\":1") != NULL);
    CHECK(strstr(line, "\"cache_hit\":false") != NULL);
    CHECK(fgets(line, sizeof(line), output) != NULL);
    CHECK(strstr(line, "\"id\":2") != NULL);
    CHECK(strstr(line, "\"cache_hit\":true") != NULL);
    CHECK(fgets(line, sizeof(line), output) == NULL);
    fclose(input);
    fclose(output);
    return 0;
}

int main(void) {
    CHECK(deepseek_v4_openai_effective_draft(3, 0.0f) == 3);
    CHECK(deepseek_v4_openai_effective_draft(3, 0.5f) == 1);
    CHECK(deepseek_v4_openai_effective_draft(0, 0.5f) == 1);
    CHECK(deepseek_v4_openai_is_client_error(
        "prompt exceeds DS4 context"));
    CHECK(!deepseek_v4_openai_is_client_error(
        "failed to read /sensitive/model/path"));
    CHECK(test_prompt_request() == 0);
    CHECK(test_messages_request() == 0);
    CHECK(test_invalid_requests() == 0);
    CHECK(test_response_and_stdio() == 0);
    puts("DeepSeek V4 stdio protocol tests: ok");
    return 0;
}
