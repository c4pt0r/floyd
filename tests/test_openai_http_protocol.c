#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../openai_http.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int parse_invalid(const char *body, const char *model) {
    OpenAIChatRequest request = {0};
    char error[256] = {0};
    int ok = openai_chat_request_parse(
        body, model, &request, error, sizeof(error));
    openai_chat_request_free(&request);
    return !ok && error[0] != 0;
}

static int test_valid_request(void) {
    const char *body =
        "{\"model\":\"deepseek-v4\",\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be concise.\"},"
        "{\"role\":\"user\",\"content\":\"Say hello.\"},"
        "{\"role\":\"assistant\",\"content\":\"Hello.\"}],"
        "\"max_completion_tokens\":17,\"temperature\":0.5,"
        "\"top_p\":0.75,\"stream\":true,"
        "\"stream_options\":{\"include_usage\":true},\"n\":1}";
    OpenAIChatRequest request = {0};
    char error[256] = {0};

    CHECK(openai_chat_request_parse(
        body, "deepseek-v4", &request, error, sizeof(error)));
    CHECK(strcmp(request.model, "deepseek-v4") == 0);
    CHECK(request.message_count == 3);
    CHECK(strcmp(request.messages[0].role, "system") == 0);
    CHECK(strcmp(request.messages[0].content, "Be concise.") == 0);
    CHECK(strcmp(request.messages[1].role, "user") == 0);
    CHECK(strcmp(request.messages[2].role, "assistant") == 0);
    CHECK(request.max_tokens == 17);
    CHECK(request.temperature == 0.5f);
    CHECK(request.top_p == 0.75f);
    CHECK(request.stream == 1);
    CHECK(request.include_usage == 1);
    openai_chat_request_free(&request);
    return 0;
}

static int test_defaults_and_model_match(void) {
    OpenAIChatRequest request = {0};
    char error[256] = {0};
    const char *body =
        "{\"model\":\"moonlight\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hello\"}]}";

    CHECK(openai_chat_request_parse(
        body, "moonlight", &request, error, sizeof(error)));
    CHECK(request.max_tokens == -1);
    CHECK(request.temperature == 0.0f);
    CHECK(request.top_p == 1.0f);
    CHECK(request.stream == 0);
    CHECK(request.include_usage == 0);
    openai_chat_request_free(&request);

    CHECK(parse_invalid(body, "Moonlight"));
    CHECK(parse_invalid(
        "{\"model\":\"moonlight-extra\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hello\"}]}",
        "moonlight"));
    CHECK(parse_invalid(
        "{\"model\":\"moonlight\\u0000-extra\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hello\"}]}",
        "moonlight"));
    return 0;
}

static int test_invalid_requests(void) {
    const char *model = "test-model";
    const char *invalid[] = {
        "{}",
        "{\"model\":\"test-model\",\"messages\":[]}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"tool\",\"content\":\"x\"}]}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":[{\"type\":\"text\","
            "\"text\":\"x\"}]}]}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[]}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"tool_choice\":\"none\"}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"logprobs\":false}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],"
            "\"response_format\":{\"type\":\"text\"}}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],"
            "\"max_tokens\":2,\"max_completion_tokens\":2}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":0}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":1.5}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":-0.1}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":2.1}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"top_p\":0}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"top_p\":1.1}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"stream\":1}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],"
            "\"stream_options\":{\"include_usage\":true}}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"hello\\u0000tail\"}]}",
        "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\\u0000tool\",\"content\":\"x\"}]}",
    };

    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
        CHECK(parse_invalid(invalid[i], model));
    return 0;
}

static char *build_large_request(size_t field_count, int duplicate) {
    if (field_count > (SIZE_MAX - 512) / 32) return NULL;
    size_t capacity = field_count * 32 + 512;
    char *body = malloc(capacity);
    if (!body) return NULL;
    size_t offset = 0;
    int written = snprintf(body, capacity, "{");
    if (written < 0 || (size_t)written >= capacity) goto fail;
    offset = (size_t)written;
    for (size_t i = 0; i < field_count; i++) {
        written = snprintf(body + offset, capacity - offset,
                           "\"extra_%06zu\":0,", i);
        if (written < 0 || (size_t)written >= capacity - offset) goto fail;
        offset += (size_t)written;
    }
    if (duplicate) {
        written = snprintf(body + offset, capacity - offset,
                           "\"extra_%06d\":1,", 0);
        if (written < 0 || (size_t)written >= capacity - offset) goto fail;
        offset += (size_t)written;
    }
    written = snprintf(
        body + offset, capacity - offset,
        "\"model\":\"large-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hello\"}]}");
    if (written < 0 || (size_t)written >= capacity - offset) goto fail;
    return body;

fail:
    free(body);
    return NULL;
}

static int test_large_object_duplicate_detection(void) {
    OpenAIChatRequest request = {0};
    char error[256] = {0};
    char *body = build_large_request(8192, 0);
    CHECK(body != NULL);
    CHECK(openai_chat_request_parse(
        body, "large-model", &request, error, sizeof(error)));
    openai_chat_request_free(&request);
    free(body);

    body = build_large_request(8192, 1);
    CHECK(body != NULL);
    CHECK(parse_invalid(body, "large-model"));
    free(body);
    return 0;
}

static int test_literal_nul_escape_text(void) {
    OpenAIChatRequest request = {0};
    char error[256] = {0};
    CHECK(openai_chat_request_parse(
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"literal \\\\u0000 text\"}]}",
        "test-model", &request, error, sizeof(error)));
    CHECK(strcmp(request.messages[0].content, "literal \\u0000 text") == 0);
    openai_chat_request_free(&request);
    return 0;
}

static int test_json_formatters(void) {
    char *json = NULL;
    size_t json_size = 0;
    OpenAIGenerationResult result = {
        .prompt_tokens = 11,
        .cached_tokens = 7,
        .completion_tokens = 3,
        .prompt_ms = 1.25,
        .decode_ms = 2.5,
        .cache_entries = 2,
        .cache_bytes = 4096,
        .finish_reason = "length",
    };

    CHECK(openai_format_models_json("model\"one", &json, &json_size));
    CHECK(json_size == strlen(json));
    CHECK(strstr(json, "\"object\":\"list\"") != NULL);
    CHECK(strstr(json, "\"id\":\"model\\\"one\"") != NULL);
    free(json);

    json = NULL;
    CHECK(openai_format_completion_json(
        "chatcmpl-test", "model\"one", "line 1\n\"hello\"",
        strlen("line 1\n\"hello\""), &result,
        &json, &json_size));
    CHECK(json_size == strlen(json));
    CHECK(strstr(json, "\"object\":\"chat.completion\"") != NULL);
    CHECK(strstr(json, "\"created\":0") == NULL);
    CHECK(strstr(json, "\"choices\":[{\"index\":0,\"message\":{"
                       "\"role\":\"assistant\"") != NULL);
    CHECK(strstr(json, "line 1\\n\\\"hello\\\"") != NULL);
    CHECK(strstr(json, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(json, "\"prompt_tokens\":11") != NULL);
    CHECK(strstr(json, "\"completion_tokens\":3") != NULL);
    CHECK(strstr(json, "\"total_tokens\":14") != NULL);
    CHECK(strstr(json,
                 "\"prompt_tokens_details\":{\"cached_tokens\":7}") != NULL);
    free(json);

    json = NULL;
    static const char nul_content[] = {'A', '\0', 'B'};
    CHECK(openai_format_completion_json(
        "chatcmpl-test", "model-one", nul_content, sizeof(nul_content),
        &result, &json, &json_size));
    CHECK(strstr(json, "\"content\":\"A\\u0000B\"") != NULL);
    free(json);

    json = NULL;
    CHECK(openai_format_error_json(
        "bad \"field\"", "invalid_request_error", "messages", "bad_value",
        &json, &json_size));
    CHECK(json_size == strlen(json));
    CHECK(strstr(json, "\"message\":\"bad \\\"field\\\"\"") != NULL);
    CHECK(strstr(json, "\"type\":\"invalid_request_error\"") != NULL);
    CHECK(strstr(json, "\"param\":\"messages\"") != NULL);
    CHECK(strstr(json, "\"code\":\"bad_value\"") != NULL);
    free(json);
    return 0;
}

int main(void) {
    CHECK(test_valid_request() == 0);
    CHECK(test_defaults_and_model_match() == 0);
    CHECK(test_invalid_requests() == 0);
    CHECK(test_large_object_duplicate_detection() == 0);
    CHECK(test_literal_nul_escape_text() == 0);
    CHECK(test_json_formatters() == 0);
    puts("OpenAI HTTP protocol tests: ok");
    return 0;
}
