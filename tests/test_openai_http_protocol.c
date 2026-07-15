#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../openai_http.h"
#include "../openai_responses.h"

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

static int parse_invalid_with_error(
    const char *body, const char *model, const char *expected) {
    OpenAIChatRequest request = {0};
    char error[256] = {0};
    int ok = openai_chat_request_parse(
        body, model, &request, error, sizeof(error));
    openai_chat_request_free(&request);
    return !ok && strstr(error, expected) != NULL;
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

static int test_root_field_allowlist(void) {
    static const char *unknown[] = {
        "tools", "tool_choice", "verbosity", "web_search_options",
        "moderation", "prompt_cache_key", "prompt_cache_retention",
        "logprobs", "response_format", "extra_body_field",
    };
    for (size_t i = 0; i < sizeof(unknown) / sizeof(unknown[0]); i++) {
        char body[512];
        int size = snprintf(
            body, sizeof(body),
            "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],\"%s\":null}",
            unknown[i]);
        CHECK(size > 0 && (size_t)size < sizeof(body));
        CHECK(parse_invalid_with_error(body, "test-model", unknown[i]));
    }
    CHECK(parse_invalid_with_error(
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,"
        "\"stream_options\":{\"include_usage\":true,\"extra\":1}}",
        "test-model", "extra"));
    return 0;
}

static char *build_nested_request(size_t depth) {
    size_t prefix_size = strlen(
        "{\"model\":\"test-model\",\"messages\":");
    size_t suffix_size = strlen("}");
    if (depth > (SIZE_MAX - prefix_size - suffix_size - 2) / 2) return NULL;
    size_t capacity = prefix_size + depth * 2 + suffix_size + 2;
    char *body = malloc(capacity);
    if (!body) return NULL;
    size_t offset = 0;
    memcpy(body + offset,
           "{\"model\":\"test-model\",\"messages\":", prefix_size);
    offset += prefix_size;
    memset(body + offset, '[', depth);
    offset += depth;
    body[offset++] = '0';
    memset(body + offset, ']', depth);
    offset += depth;
    body[offset++] = '}';
    body[offset] = 0;
    return body;
}

static char *build_node_limit_request(size_t count) {
    static const char prefix[] =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],\"extra\":[";
    static const char suffix[] = "]}";
    if (count > (SIZE_MAX - sizeof(prefix) - sizeof(suffix)) / 2) return NULL;
    size_t capacity = sizeof(prefix) + count * 2 + sizeof(suffix);
    char *body = malloc(capacity);
    if (!body) return NULL;
    size_t offset = 0;
    memcpy(body + offset, prefix, sizeof(prefix) - 1);
    offset += sizeof(prefix) - 1;
    for (size_t i = 0; i < count; i++) {
        if (i) body[offset++] = ',';
        body[offset++] = '0';
    }
    memcpy(body + offset, suffix, sizeof(suffix));
    return body;
}

static int test_strict_json_limits_and_numbers(void) {
    char *nested = build_nested_request(129);
    CHECK(nested != NULL);
    CHECK(parse_invalid_with_error(nested, "test-model", "nesting depth"));
    free(nested);

    char *nodes = build_node_limit_request(1048577);
    CHECK(nodes != NULL);
    CHECK(parse_invalid_with_error(nodes, "test-model", "node limit"));
    free(nodes);

    static const char *numbers[] = {
        "+1", ".5", "01", "1.", "1e", "1e+", "1e-", "NaN", "Infinity",
    };
    for (size_t i = 0; i < sizeof(numbers) / sizeof(numbers[0]); i++) {
        char body[256];
        int size = snprintf(
            body, sizeof(body),
            "{\"model\":\"test-model\",\"messages\":["
            "{\"role\":\"user\",\"content\":\"x\"}],"
            "\"temperature\":%s}", numbers[i]);
        CHECK(size > 0 && (size_t)size < sizeof(body));
        CHECK(parse_invalid_with_error(body, "test-model", "JSON"));
    }
    return 0;
}

static int test_strict_json_utf8(void) {
    static const unsigned char invalid[][4] = {
        {0xc0, 0xaf, 0, 0},
        {0xe0, 0x80, 0x80, 0},
        {0xed, 0xa0, 0x80, 0},
        {0xe2, 0x28, 0xa1, 0},
        {0xf4, 0x90, 0x80, 0x80},
        {0xf5, 0x80, 0x80, 0x80},
    };
    static const size_t invalid_size[] = {2, 3, 3, 3, 4, 4};
    static const char prefix[] =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"";
    static const char suffix[] = "\"}]}";
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        char body[256];
        size_t offset = 0;
        memcpy(body + offset, prefix, sizeof(prefix) - 1);
        offset += sizeof(prefix) - 1;
        memcpy(body + offset, invalid[i], invalid_size[i]);
        offset += invalid_size[i];
        memcpy(body + offset, suffix, sizeof(suffix));
        CHECK(parse_invalid_with_error(body, "test-model", "UTF-8"));
    }

    OpenAIChatRequest request = {0};
    char error[256] = {0};
    CHECK(openai_chat_request_parse(
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"\xf0\x9f\x98\x80"
        " \\ud83d\\ude00\"}]}",
        "test-model", &request, error, sizeof(error)));
    CHECK(strcmp(request.messages[0].content,
                 "\xf0\x9f\x98\x80 \xf0\x9f\x98\x80") == 0);
    openai_chat_request_free(&request);
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
    char *body = build_large_request(8192, 0);
    CHECK(body != NULL);
    CHECK(parse_invalid_with_error(body, "large-model", "extra_000000"));
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

static int test_pie_responses_request_and_replay(void) {
    static const char body[] =
        "{\"model\":\"deepseek-v4-flash\",\"input\":["
        "{\"role\":\"system\",\"content\":[{\"type\":\"input_text\","
        "\"text\":\"System\"}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"input_text\","
        "\"text\":\"Inspect\"}]},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\","
        "\"text\":\"I will inspect.\"}]},"
        "{\"type\":\"function_call\",\"call_id\":\"call_1\","
        "\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
        "\"output\":\"Floyd README\"}],"
        "\"tools\":[{\"type\":\"function\",\"name\":\"read\","
        "\"description\":\"Read a file\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"path\":{\"type\":\"string\"}}}}],"
        "\"stream\":true,\"store\":false,\"max_output_tokens\":64,"
        "\"temperature\":0,\"prompt_cache_key\":\"session-1\"}"
        ;
    OpenAIResponsesRequest request = {0};
    char error[256] = {0};
    CHECK(openai_responses_request_parse(
        body, "deepseek-v4", &request, error, sizeof(error)));
    CHECK(strcmp(request.chat.model, "deepseek-v4-flash") == 0);
    CHECK(request.chat.stream == 1);
    CHECK(request.chat.max_tokens == 64);
    CHECK(request.chat.message_count == 4);
    CHECK(strcmp(request.chat.messages[0].role, "system") == 0);
    CHECK(strstr(request.chat.messages[0].content, "## Tools") != NULL);
    CHECK(strstr(request.chat.messages[0].content, "System") != NULL);
    CHECK(strcmp(request.chat.messages[1].content, "Inspect") == 0);
    CHECK(strcmp(request.chat.messages[2].role, "assistant") == 0);
    CHECK(strstr(request.chat.messages[2].content, "I will inspect.") != NULL);
    CHECK(strstr(request.chat.messages[2].content,
                 "<｜DSML｜invoke name=\"read\">") != NULL);
    CHECK(strstr(request.chat.messages[2].content,
                 "name=\"path\" string=\"true\">README.md") != NULL);
    CHECK(strcmp(request.chat.messages[3].role, "user") == 0);
    CHECK(strcmp(request.chat.messages[3].content,
                 "<tool_result>Floyd README</tool_result>") == 0);
    openai_responses_request_free(&request);

    static const char *invalid[] = {
        "{\"model\":\"other\",\"input\":[{\"role\":\"user\","
        "\"content\":\"x\"}]}",
        "{\"model\":\"deepseek-v4-flash\",\"input\":[{\"role\":\"user\","
        "\"content\":[{\"type\":\"input_image\",\"image_url\":\"x\"}]}]}",
        "{\"model\":\"deepseek-v4-flash\",\"input\":[{\"role\":\"user\","
        "\"content\":\"x\"}],\"store\":true}",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        memset(&request, 0, sizeof(request));
        memset(error, 0, sizeof(error));
        CHECK(!openai_responses_request_parse(
            invalid[i], "deepseek-v4", &request, error, sizeof(error)));
        CHECK(error[0] != 0);
        openai_responses_request_free(&request);
    }
    return 0;
}

static int test_responses_dsml_projection(void) {
    static const char raw[] =
        "Checking.\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd &amp;&amp; ls"
        "</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    OpenAIResponsesOutput output = {0};
    CHECK(openai_responses_output_parse(
        raw, sizeof(raw) - 1, &output));
    CHECK(strcmp(output.text, "Checking.") == 0);
    CHECK(output.tool_call_count == 1);
    CHECK(strcmp(output.tool_calls[0].name, "bash") == 0);
    CHECK(strcmp(output.tool_calls[0].arguments,
                 "{\"command\":\"pwd && ls\",\"timeout\":10}") == 0);
    openai_responses_output_free(&output);
    return 0;
}

int main(void) {
    CHECK(test_valid_request() == 0);
    CHECK(test_defaults_and_model_match() == 0);
    CHECK(test_invalid_requests() == 0);
    CHECK(test_root_field_allowlist() == 0);
    CHECK(test_strict_json_limits_and_numbers() == 0);
    CHECK(test_strict_json_utf8() == 0);
    CHECK(test_large_object_duplicate_detection() == 0);
    CHECK(test_literal_nul_escape_text() == 0);
    CHECK(test_json_formatters() == 0);
    CHECK(test_pie_responses_request_and_replay() == 0);
    CHECK(test_responses_dsml_projection() == 0);
    puts("OpenAI HTTP protocol tests: ok");
    return 0;
}
