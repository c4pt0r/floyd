#define _GNU_SOURCE
#include "openai_responses.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static void responses_error(char *error, size_t error_size,
                            const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static char *responses_strdup(const char *value) {
    if (!value) return NULL;
    size_t size = strlen(value) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, value, size);
    return copy;
}

static char *responses_strndup(const char *value, size_t size) {
    char *copy = malloc(size + 1);
    if (!copy) return NULL;
    memcpy(copy, value, size);
    copy[size] = 0;
    return copy;
}

static int responses_unique_keys(const jval *object) {
    if (!object || object->t != J_OBJ) return 0;
    for (int i = 0; i < object->len; i++)
        for (int j = i + 1; j < object->len; j++)
            if (strcmp(object->keys[i], object->keys[j]) == 0) return 0;
    return 1;
}

static int responses_json_string_n(FILE *output,
                                   const char *value, size_t size) {
    if (fputc('"', output) == EOF) return 0;
    for (size_t i = 0; i < size; i++) {
        unsigned char c = (unsigned char)value[i];
        switch (c) {
            case '"': if (fputs("\\\"", output) == EOF) return 0; break;
            case '\\': if (fputs("\\\\", output) == EOF) return 0; break;
            case '\b': if (fputs("\\b", output) == EOF) return 0; break;
            case '\f': if (fputs("\\f", output) == EOF) return 0; break;
            case '\n': if (fputs("\\n", output) == EOF) return 0; break;
            case '\r': if (fputs("\\r", output) == EOF) return 0; break;
            case '\t': if (fputs("\\t", output) == EOF) return 0; break;
            default:
                if (c < 0x20) {
                    if (fprintf(output, "\\u%04x", (unsigned)c) < 0) return 0;
                } else if (fputc(c, output) == EOF) return 0;
                break;
        }
    }
    return fputc('"', output) != EOF;
}

static int responses_json_string(FILE *output, const char *value) {
    return responses_json_string_n(output, value ? value : "",
                                   strlen(value ? value : ""));
}

static int responses_json_value(FILE *output, const jval *value) {
    if (!value) return 0;
    switch (value->t) {
        case J_NULL: return fputs("null", output) != EOF;
        case J_BOOL: return fputs(value->boolean ? "true" : "false", output) != EOF;
        case J_NUM: return isfinite(value->num) && fprintf(output, "%.17g", value->num) >= 0;
        case J_STR: return responses_json_string(output, value->str);
        case J_ARR:
            if (fputc('[', output) == EOF) return 0;
            for (int i = 0; i < value->len; i++) {
                if (i && fputc(',', output) == EOF) return 0;
                if (!responses_json_value(output, value->kids[i])) return 0;
            }
            return fputc(']', output) != EOF;
        case J_OBJ:
            if (fputc('{', output) == EOF) return 0;
            for (int i = 0; i < value->len; i++) {
                if (i && fputc(',', output) == EOF) return 0;
                if (!responses_json_string(output, value->keys[i]) ||
                    fputc(':', output) == EOF ||
                    !responses_json_value(output, value->kids[i])) return 0;
            }
            return fputc('}', output) != EOF;
    }
    return 0;
}

static char *responses_json_value_string(const jval *value) {
    char *json = NULL;
    size_t size = 0;
    FILE *stream = open_memstream(&json, &size);
    if (!stream) return NULL;
    int ok = responses_json_value(stream, value);
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(json);
        return NULL;
    }
    return json;
}

static int responses_add_message(OpenAIChatRequest *chat,
                                 const char *role, const char *content) {
    if (chat->message_count == SIZE_MAX / sizeof(*chat->messages)) return 0;
    size_t count = chat->message_count + 1;
    OpenAIMessage *grown = realloc(chat->messages, count * sizeof(*grown));
    if (!grown) return 0;
    chat->messages = grown;
    OpenAIMessage *message = &chat->messages[chat->message_count];
    memset(message, 0, sizeof(*message));
    message->role = responses_strdup(role);
    message->content = responses_strdup(content ? content : "");
    if (!message->role || !message->content) {
        free(message->role);
        free(message->content);
        memset(message, 0, sizeof(*message));
        return 0;
    }
    chat->message_count = count;
    return 1;
}

static int responses_append_message_content(OpenAIMessage *message,
                                            const char *suffix) {
    size_t prefix_size = strlen(message->content);
    size_t suffix_size = strlen(suffix ? suffix : "");
    if (suffix_size > SIZE_MAX - prefix_size - 1) return 0;
    char *grown = realloc(message->content, prefix_size + suffix_size + 1);
    if (!grown) return 0;
    memcpy(grown + prefix_size, suffix ? suffix : "", suffix_size + 1);
    message->content = grown;
    return 1;
}

static char *responses_content_text(const jval *content,
                                    char *error, size_t error_size) {
    if (!content) {
        responses_error(error, error_size, "input message content is required");
        return NULL;
    }
    if (content->t == J_STR) return responses_strdup(content->str);
    if (content->t != J_ARR) {
        responses_error(error, error_size,
                        "input message content must be a string or array");
        return NULL;
    }
    char *text = NULL;
    size_t text_size = 0;
    FILE *stream = open_memstream(&text, &text_size);
    if (!stream) return NULL;
    int ok = 1;
    for (int i = 0; i < content->len; i++) {
        jval *part = content->kids[i];
        jval *type = part && part->t == J_OBJ ? json_get(part, "type") : NULL;
        jval *value = part && part->t == J_OBJ ? json_get(part, "text") : NULL;
        if (!part || part->t != J_OBJ || !responses_unique_keys(part) ||
            !type || type->t != J_STR || !value || value->t != J_STR ||
            (strcmp(type->str, "input_text") != 0 &&
             strcmp(type->str, "output_text") != 0)) {
            responses_error(error, error_size,
                            "only input_text and output_text content are supported");
            ok = 0;
            break;
        }
        if (fputs(value->str, stream) == EOF) {
            ok = 0;
            break;
        }
    }
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(text);
        return NULL;
    }
    return text;
}

static char *responses_reasoning_text(const jval *item,
                                      char *error, size_t error_size) {
    jval *summary = json_get((jval *)item, "summary");
    if (!summary || summary->t != J_ARR) {
        responses_error(error, error_size,
                        "reasoning summary must be an array");
        return NULL;
    }
    char *text = NULL;
    size_t text_size = 0;
    FILE *stream = open_memstream(&text, &text_size);
    if (!stream) return NULL;
    int ok = 1;
    for (int i = 0; i < summary->len && ok; i++) {
        jval *part = summary->kids[i];
        jval *type = part && part->t == J_OBJ ? json_get(part, "type") : NULL;
        jval *value = part && part->t == J_OBJ ? json_get(part, "text") : NULL;
        if (!part || part->t != J_OBJ || !responses_unique_keys(part) ||
            !type || type->t != J_STR || strcmp(type->str, "summary_text") ||
            !value || value->t != J_STR) {
            responses_error(error, error_size,
                            "reasoning summary only supports summary_text");
            ok = 0;
            break;
        }
        if (i && fputc('\n', stream) == EOF) ok = 0;
        if (ok && fputs(value->str, stream) == EOF) ok = 0;
    }
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(text);
        return NULL;
    }
    return text;
}

static int responses_dsml_attr(FILE *output, const char *value) {
    for (const unsigned char *p = (const unsigned char *)(value ? value : "");
         *p; p++) {
        const char *escaped = NULL;
        if (*p == '&') escaped = "&amp;";
        else if (*p == '<') escaped = "&lt;";
        else if (*p == '>') escaped = "&gt;";
        else if (*p == '"') escaped = "&quot;";
        if (escaped) {
            if (fputs(escaped, output) == EOF) return 0;
        } else if (fputc(*p, output) == EOF) return 0;
    }
    return 1;
}

static int responses_dsml_text(FILE *output, const char *value,
                               const char *closing_tag) {
    size_t closing_size = strlen(closing_tag);
    const char *p = value ? value : "";
    while (*p) {
        if (strncmp(p, closing_tag, closing_size) == 0) {
            if (fputs("&lt;", output) == EOF) return 0;
            p++;
        } else if (fputc((unsigned char)*p++, output) == EOF) return 0;
    }
    return 1;
}

static char *responses_render_function_call(const char *name,
                                            const char *arguments) {
    char parse_error[128] = {0};
    jval *args = json_parse_full(arguments ? arguments : "{}",
                                 parse_error, sizeof(parse_error));
    char *rendered = NULL;
    size_t rendered_size = 0;
    FILE *stream = open_memstream(&rendered, &rendered_size);
    if (!stream) {
        json_free(args);
        return NULL;
    }
    int ok = fputs("\n\n<｜DSML｜tool_calls>\n"
                   "<｜DSML｜invoke name=\"", stream) != EOF &&
             responses_dsml_attr(stream, name) &&
             fputs("\">\n", stream) != EOF;
    if (ok && args && args->t == J_OBJ) {
        for (int i = 0; i < args->len && ok; i++) {
            jval *value = args->kids[i];
            ok = fputs("<｜DSML｜parameter name=\"", stream) != EOF &&
                 responses_dsml_attr(stream, args->keys[i]) &&
                 fputs(value->t == J_STR
                           ? "\" string=\"true\">"
                           : "\" string=\"false\">", stream) != EOF;
            if (ok && value->t == J_STR) {
                ok = responses_dsml_text(
                    stream, value->str, "</｜DSML｜parameter>");
            } else if (ok) {
                char *json = responses_json_value_string(value);
                ok = json && responses_dsml_text(
                    stream, json, "</｜DSML｜parameter>");
                free(json);
            }
            if (ok) ok = fputs("</｜DSML｜parameter>\n", stream) != EOF;
        }
    } else if (ok) {
        ok = fputs("<｜DSML｜parameter name=\"arguments\" string=\"true\">",
                   stream) != EOF &&
             responses_dsml_text(stream, arguments ? arguments : "{}",
                                 "</｜DSML｜parameter>") &&
             fputs("</｜DSML｜parameter>\n", stream) != EOF;
    }
    if (ok) ok = fputs("</｜DSML｜invoke>\n</｜DSML｜tool_calls>", stream) != EOF;
    if (fclose(stream) != 0) ok = 0;
    json_free(args);
    if (!ok) {
        free(rendered);
        return NULL;
    }
    return rendered;
}

static char *responses_render_tool_output(const char *output) {
    char *rendered = NULL;
    size_t rendered_size = 0;
    FILE *stream = open_memstream(&rendered, &rendered_size);
    if (!stream) return NULL;
    int ok = fputs("<tool_result>", stream) != EOF &&
             responses_dsml_text(stream, output ? output : "",
                                 "</tool_result>") &&
             fputs("</tool_result>", stream) != EOF;
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(rendered);
        return NULL;
    }
    return rendered;
}

static char *responses_tools_prompt(const jval *tools) {
    if (!tools) return responses_strdup("");
    if (tools->t != J_ARR) return NULL;
    char *prompt = NULL;
    size_t prompt_size = 0;
    FILE *stream = open_memstream(&prompt, &prompt_size);
    if (!stream) return NULL;
    int ok = fputs(
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
        "$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal "
        "string characters with XML or HTML entity escapes. Only if a string value "
        "itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write "
        "that tag as `&lt;/｜DSML｜parameter>` inside the value. For all other types "
        "(numbers, booleans, arrays, objects), pass the value in JSON format and set "
        "`string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your "
        "complete reasoning inside <think>...</think> BEFORE any tool calls or final "
        "response.\n\nOtherwise, output directly after </think> with tool calls or "
        "final response.\n\n### Available Tool Schemas\n\n", stream) != EOF;
    for (int i = 0; i < tools->len && ok; i++) {
        jval *tool = tools->kids[i];
        jval *type = tool && tool->t == J_OBJ ? json_get(tool, "type") : NULL;
        jval *name = tool && tool->t == J_OBJ ? json_get(tool, "name") : NULL;
        if (!tool || tool->t != J_OBJ || !responses_unique_keys(tool) ||
            !type || type->t != J_STR || strcmp(type->str, "function") != 0 ||
            !name || name->t != J_STR || !name->str[0]) {
            ok = 0;
            break;
        }
        if (i && fputc('\n', stream) == EOF) ok = 0;
        if (ok) ok = responses_json_value(stream, tool);
    }
    if (ok) ok = fputs(
        "\n\nYou MUST strictly follow the above defined tool name and parameter "
        "schemas to invoke tool calls. Use the exact parameter names from the schemas.",
        stream) != EOF;
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(prompt);
        return NULL;
    }
    return prompt;
}

static int responses_prepend_system(OpenAIChatRequest *chat,
                                    const char *prompt) {
    if (!prompt || !prompt[0]) return 1;
    if (chat->message_count &&
        strcmp(chat->messages[0].role, "system") == 0) {
        OpenAIMessage *system = &chat->messages[0];
        size_t prompt_size = strlen(prompt);
        size_t content_size = strlen(system->content);
        if (prompt_size > SIZE_MAX - content_size - 3) return 0;
        char *combined = malloc(prompt_size + content_size + 3);
        if (!combined) return 0;
        memcpy(combined, prompt, prompt_size);
        memcpy(combined + prompt_size, "\n\n", 2);
        memcpy(combined + prompt_size + 2, system->content, content_size + 1);
        free(system->content);
        system->content = combined;
        return 1;
    }
    if (chat->message_count == SIZE_MAX / sizeof(*chat->messages)) return 0;
    OpenAIMessage *grown = realloc(
        chat->messages, (chat->message_count + 1) * sizeof(*grown));
    if (!grown) return 0;
    chat->messages = grown;
    memmove(chat->messages + 1, chat->messages,
            chat->message_count * sizeof(*chat->messages));
    memset(&chat->messages[0], 0, sizeof(chat->messages[0]));
    chat->messages[0].role = responses_strdup("system");
    chat->messages[0].content = responses_strdup(prompt);
    if (!chat->messages[0].role || !chat->messages[0].content) {
        free(chat->messages[0].role);
        free(chat->messages[0].content);
        memmove(chat->messages, chat->messages + 1,
                chat->message_count * sizeof(*chat->messages));
        return 0;
    }
    chat->message_count++;
    return 1;
}

static int responses_render_prompt(OpenAIChatRequest *chat, int thinking) {
    char *rendered = NULL;
    size_t rendered_size = 0;
    FILE *stream = open_memstream(&rendered, &rendered_size);
    if (!stream) return 0;
    int ok = fputs("<｜begin▁of▁sentence｜>", stream) != EOF;
    int wrote_system = 0;
    for (size_t i = 0; i < chat->message_count && ok; i++) {
        if (strcmp(chat->messages[i].role, "system") != 0) continue;
        if (wrote_system) ok = fputs("\n\n", stream) != EOF;
        if (ok) ok = fputs(chat->messages[i].content, stream) != EOF;
        wrote_system = 1;
    }
    for (size_t i = 0; i < chat->message_count && ok; i++) {
        OpenAIMessage *message = &chat->messages[i];
        if (!strcmp(message->role, "system")) continue;
        if (i + 1 == chat->message_count) {
            if (fflush(stream) != 0) {
                ok = 0;
                break;
            }
            chat->rendered_anchor_bytes = rendered_size;
        }
        if (!strcmp(message->role, "user")) {
            ok = fputs("<｜User｜>", stream) != EOF &&
                 fputs(message->content, stream) != EOF;
        } else if (!strcmp(message->role, "assistant")) {
            ok = fputs("<｜Assistant｜>", stream) != EOF;
            if (ok && thinking)
                ok = fputs("<think>", stream) != EOF &&
                     fputs(message->reasoning ? message->reasoning : "", stream) != EOF &&
                     fputs("</think>", stream) != EOF;
            else if (ok)
                ok = fputs("</think>", stream) != EOF;
            ok = ok && fputs(message->content, stream) != EOF &&
                 fputs("<｜end▁of▁sentence｜>", stream) != EOF;
        } else {
            ok = 0;
        }
    }
    if (ok) ok = fputs(thinking ? "<｜Assistant｜><think>"
                                : "<｜Assistant｜></think>", stream) != EOF;
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(rendered);
        chat->rendered_anchor_bytes = 0;
        return 0;
    }
    chat->rendered_prompt = rendered;
    chat->rendered_thinking = thinking;
    return 1;
}

static int responses_model_matches(const char *served, const char *requested) {
    if (strcmp(served, requested) == 0) return 1;
    return (!strcmp(served, "deepseek-v4") &&
            !strcmp(requested, "deepseek-v4-flash")) ||
           (!strcmp(served, "deepseek-v4-flash") &&
            !strcmp(requested, "deepseek-v4"));
}

static int responses_parse_number(const jval *root, const char *name,
                                  double missing, double min, double max,
                                  double *output) {
    jval *value = json_get((jval *)root, name);
    if (!value) {
        *output = missing;
        return 1;
    }
    if (value->t != J_NUM || !isfinite(value->num) ||
        value->num < min || value->num > max) return 0;
    *output = value->num;
    return 1;
}

static int responses_thinking_enabled(const jval *root, int *enabled) {
    *enabled = 1;
    jval *reasoning = json_get((jval *)root, "reasoning");
    if (!reasoning || reasoning->t == J_NULL) return 1;
    if (reasoning->t != J_OBJ || !responses_unique_keys(reasoning)) return 0;
    jval *effort = json_get(reasoning, "effort");
    if (!effort || effort->t == J_NULL) return 1;
    if (effort->t != J_STR) return 0;
    static const char *accepted[] = {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    };
    for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); i++) {
        if (strcmp(effort->str, accepted[i]) != 0) continue;
        *enabled = strcmp(effort->str, "none") != 0;
        return 1;
    }
    return 0;
}

void openai_responses_request_free(OpenAIResponsesRequest *request) {
    if (!request) return;
    openai_chat_request_free(&request->chat);
    memset(request, 0, sizeof(*request));
}

int openai_responses_request_parse(
    const char *body, const char *served_model,
    OpenAIResponsesRequest *request, char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!body || !served_model || !served_model[0] || !request) {
        responses_error(error, error_size, "body and served model are required");
        return 0;
    }
    memset(request, 0, sizeof(*request));
    request->chat.max_tokens = -1;
    request->chat.temperature = 0.0f;
    request->chat.top_p = 1.0f;
    if (!openai_json_body_validate(body, error, error_size)) return 0;

    char parse_error[256] = {0};
    jval *root = json_parse_full(body, parse_error, sizeof(parse_error));
    char *pending_reasoning = NULL;
    if (!root) {
        responses_error(error, error_size, parse_error);
        return 0;
    }
    if (root->t != J_OBJ || !responses_unique_keys(root)) {
        responses_error(error, error_size,
                        "Responses request must be an object with unique keys");
        goto fail;
    }
    static const char *allowed[] = {
        "model", "input", "stream", "store", "max_output_tokens",
        "temperature", "top_p", "tools", "tool_choice",
        "parallel_tool_calls", "reasoning", "include", "service_tier",
        "prompt_cache_key", "prompt_cache_retention",
    };
    for (int i = 0; i < root->len; i++) {
        int accepted = 0;
        for (size_t j = 0; j < sizeof(allowed) / sizeof(allowed[0]); j++)
            if (!strcmp(root->keys[i], allowed[j])) accepted = 1;
        if (!accepted) {
            responses_error(error, error_size,
                            "Responses request contains an unsupported field");
            goto fail;
        }
    }

    jval *model = json_get(root, "model");
    if (!model || model->t != J_STR || !model->str[0] ||
        !responses_model_matches(served_model, model->str)) {
        responses_error(error, error_size,
                        "model does not match the served model");
        goto fail;
    }
    request->chat.model = responses_strdup(model->str);
    if (!request->chat.model) goto oom;

    jval *stream = json_get(root, "stream");
    if (stream && stream->t != J_BOOL) {
        responses_error(error, error_size, "stream must be a boolean");
        goto fail;
    }
    request->chat.stream = stream ? stream->boolean : 0;
    request->chat.include_usage = 1;
    jval *store = json_get(root, "store");
    if (store && (store->t != J_BOOL || store->boolean)) {
        responses_error(error, error_size, "store must be false");
        goto fail;
    }
    double number;
    if (!responses_parse_number(root, "max_output_tokens", -1, 1, INT_MAX,
                                &number) ||
        (number >= 0 && floor(number) != number)) {
        responses_error(error, error_size,
                        "max_output_tokens must be a positive integer");
        goto fail;
    }
    request->chat.max_tokens = (int)number;
    if (!responses_parse_number(root, "temperature", 0, 0, 2, &number)) {
        responses_error(error, error_size, "temperature is outside its range");
        goto fail;
    }
    request->chat.temperature = (float)number;
    if (!responses_parse_number(root, "top_p", 1, 0.000001, 1, &number)) {
        responses_error(error, error_size, "top_p is outside its range");
        goto fail;
    }
    request->chat.top_p = (float)number;

    jval *input = json_get(root, "input");
    if (!input || input->t != J_ARR || input->len <= 0) {
        responses_error(error, error_size, "input must be a non-empty array");
        goto fail;
    }
    for (int i = 0; i < input->len; i++) {
        jval *item = input->kids[i];
        if (!item || item->t != J_OBJ || !responses_unique_keys(item)) {
            responses_error(error, error_size,
                            "each input item must be an object with unique keys");
            goto fail;
        }
        jval *type = json_get(item, "type");
        const char *kind = type && type->t == J_STR ? type->str : "message";
        if (!strcmp(kind, "message")) {
            jval *role = json_get(item, "role");
            if (!role || role->t != J_STR ||
                (strcmp(role->str, "system") && strcmp(role->str, "developer") &&
                 strcmp(role->str, "user") && strcmp(role->str, "assistant"))) {
                responses_error(error, error_size,
                                "input message has an unsupported role");
                goto fail;
            }
            char *text = responses_content_text(
                json_get(item, "content"), error, error_size);
            if (!text) goto fail;
            const char *role_name = !strcmp(role->str, "developer")
                ? "system" : role->str;
            int added = responses_add_message(&request->chat, role_name, text);
            free(text);
            if (!added) goto oom;
            if (!strcmp(role_name, "assistant") && pending_reasoning) {
                request->chat.messages[request->chat.message_count - 1].reasoning =
                    pending_reasoning;
                pending_reasoning = NULL;
            }
        } else if (!strcmp(kind, "reasoning")) {
            free(pending_reasoning);
            pending_reasoning = responses_reasoning_text(
                item, error, error_size);
            if (!pending_reasoning) goto fail;
        } else if (!strcmp(kind, "function_call")) {
            jval *name = json_get(item, "name");
            jval *arguments = json_get(item, "arguments");
            if (!name || name->t != J_STR || !name->str[0] ||
                !arguments || arguments->t != J_STR) {
                responses_error(error, error_size,
                                "function_call requires name and arguments");
                goto fail;
            }
            if (!request->chat.message_count ||
                strcmp(request->chat.messages[
                    request->chat.message_count - 1].role, "assistant")) {
                if (!responses_add_message(&request->chat, "assistant", ""))
                    goto oom;
            }
            if (pending_reasoning &&
                !request->chat.messages[
                    request->chat.message_count - 1].reasoning) {
                request->chat.messages[
                    request->chat.message_count - 1].reasoning = pending_reasoning;
                pending_reasoning = NULL;
            }
            char *dsml = responses_render_function_call(
                name->str, arguments->str);
            if (!dsml || !responses_append_message_content(
                    &request->chat.messages[request->chat.message_count - 1],
                    dsml)) {
                free(dsml);
                goto oom;
            }
            free(dsml);
        } else if (!strcmp(kind, "function_call_output")) {
            jval *output = json_get(item, "output");
            if (!output || output->t != J_STR) {
                responses_error(error, error_size,
                                "function_call_output requires string output");
                goto fail;
            }
            char *tool_result = responses_render_tool_output(output->str);
            if (!tool_result) goto oom;
            int added;
            if (request->chat.message_count &&
                !strcmp(request->chat.messages[
                    request->chat.message_count - 1].role, "user") &&
                strstr(request->chat.messages[
                    request->chat.message_count - 1].content,
                    "<tool_result>") != NULL) {
                added = responses_append_message_content(
                    &request->chat.messages[request->chat.message_count - 1],
                    tool_result);
            } else {
                added = responses_add_message(
                    &request->chat, "user", tool_result);
            }
            free(tool_result);
            if (!added) goto oom;
        } else {
            responses_error(error, error_size,
                            "input item type is not supported");
            goto fail;
        }
    }
    free(pending_reasoning);
    pending_reasoning = NULL;

    char *tools_prompt = responses_tools_prompt(json_get(root, "tools"));
    if (!tools_prompt) {
        responses_error(error, error_size, "tools must contain function tools");
        goto fail;
    }
    int prepended = responses_prepend_system(&request->chat, tools_prompt);
    free(tools_prompt);
    if (!prepended) goto oom;
    if (!request->chat.message_count ||
        strcmp(request->chat.messages[
            request->chat.message_count - 1].role, "user")) {
        responses_error(error, error_size,
                        "the last Responses input must be user or tool output");
        goto fail;
    }
    int thinking = 1;
    if (!responses_thinking_enabled(root, &thinking)) {
        responses_error(error, error_size, "reasoning effort is not supported");
        goto fail;
    }
    if (!responses_render_prompt(&request->chat, thinking)) goto oom;

    json_free(root);
    return 1;

oom:
    responses_error(error, error_size, "out of memory parsing Responses request");
fail:
    free(pending_reasoning);
    json_free(root);
    openai_responses_request_free(request);
    return 0;
}

static char *responses_xml_unescape(const char *value, size_t size) {
    char *result = calloc(size + 1, 1);
    if (!result) return NULL;
    size_t written = 0;
    for (size_t i = 0; i < size;) {
        const char *replacement = NULL;
        size_t consumed = 0;
        if (size - i >= 5 && !memcmp(value + i, "&amp;", 5)) {
            replacement = "&"; consumed = 5;
        } else if (size - i >= 4 && !memcmp(value + i, "&lt;", 4)) {
            replacement = "<"; consumed = 4;
        } else if (size - i >= 4 && !memcmp(value + i, "&gt;", 4)) {
            replacement = ">"; consumed = 4;
        } else if (size - i >= 6 && !memcmp(value + i, "&quot;", 6)) {
            replacement = "\""; consumed = 6;
        } else if (size - i >= 6 && !memcmp(value + i, "&apos;", 6)) {
            replacement = "'"; consumed = 6;
        }
        if (replacement) {
            result[written++] = replacement[0];
            i += consumed;
        } else {
            result[written++] = value[i++];
        }
    }
    result[written] = 0;
    return result;
}

static char *responses_xml_attr(const char *tag, size_t tag_size,
                                const char *name) {
    char pattern[96];
    int pattern_size = snprintf(pattern, sizeof(pattern), "%s=\"", name);
    if (pattern_size <= 0 || (size_t)pattern_size >= sizeof(pattern)) return NULL;
    const char *found = NULL;
    for (size_t i = 0; i + (size_t)pattern_size <= tag_size; i++)
        if (!memcmp(tag + i, pattern, (size_t)pattern_size)) {
            found = tag + i + pattern_size;
            break;
        }
    if (!found) return NULL;
    const char *limit = tag + tag_size;
    const char *end = memchr(found, '"', (size_t)(limit - found));
    return end ? responses_xml_unescape(found, (size_t)(end - found)) : NULL;
}

static const char *responses_skip_space(const char *cursor) {
    while (*cursor == ' ' || *cursor == '\t' ||
           *cursor == '\r' || *cursor == '\n') cursor++;
    return cursor;
}

static const char *responses_tool_start(const char *raw,
                                        const char **end_tag,
                                        const char **invoke_start,
                                        const char **invoke_end,
                                        const char **param_start,
                                        const char **param_end) {
    static const char *starts[] = {
        "<｜DSML｜tool_calls>", "<DSML｜tool_calls>", "<tool_calls>",
    };
    static const char *ends[] = {
        "</｜DSML｜tool_calls>", "</DSML｜tool_calls>", "</tool_calls>",
    };
    static const char *invokes[] = {
        "<｜DSML｜invoke", "<DSML｜invoke", "<invoke",
    };
    static const char *invoke_ends[] = {
        "</｜DSML｜invoke>", "</DSML｜invoke>", "</invoke>",
    };
    static const char *params[] = {
        "<｜DSML｜parameter", "<DSML｜parameter", "<parameter",
    };
    static const char *param_ends[] = {
        "</｜DSML｜parameter>", "</DSML｜parameter>", "</parameter>",
    };
    const char *best = NULL;
    size_t best_index = 0;
    for (size_t i = 0; i < sizeof(starts) / sizeof(starts[0]); i++) {
        const char *found = strstr(raw, starts[i]);
        if (found && (!best || found < best)) {
            best = found;
            best_index = i;
        }
    }
    if (!best) {
        static const char *loose_starts[] = {
            "<｜DSML｜tool_cls>", "<DSML｜tool_cls>", "<tool_cls>",
        };
        const char *loose = NULL;
        for (size_t i = 0; i < sizeof(loose_starts) / sizeof(loose_starts[0]); i++) {
            const char *found = strstr(raw, loose_starts[i]);
            if (found && (!loose || found < loose)) loose = found;
        }
        if (!loose) return NULL;
        for (size_t i = 0; i < sizeof(invokes) / sizeof(invokes[0]); i++) {
            const char *found = strstr(loose, invokes[i]);
            if (found && (!best || found < best)) {
                best = found;
                best_index = i;
            }
        }
        if (!best) return NULL;
        *end_tag = NULL;
        *invoke_start = invokes[best_index];
        *invoke_end = invoke_ends[best_index];
        *param_start = params[best_index];
        *param_end = param_ends[best_index];
        return best;
    }
    *end_tag = ends[best_index];
    *invoke_start = invokes[best_index];
    *invoke_end = invoke_ends[best_index];
    *param_start = params[best_index];
    *param_end = param_ends[best_index];
    return best + strlen(starts[best_index]);
}

static const char *responses_invoke_close(const char *cursor,
                                          const char *invoke_end) {
    size_t exact_size = strlen(invoke_end);
    if (!strncmp(cursor, invoke_end, exact_size)) return cursor + exact_size;
    static const char *short_closes[] = {
        "</｜DSML｜inv", "</DSML｜inv", "</inv",
    };
    for (size_t i = 0; i < sizeof(short_closes) / sizeof(short_closes[0]); i++) {
        size_t size = strlen(short_closes[i]);
        if (strncmp(cursor, short_closes[i], size)) continue;
        return cursor[size] == '>' ? cursor + size + 1 : NULL;
    }
    return NULL;
}

static int responses_output_add_call(OpenAIResponsesOutput *output,
                                     char *name, char *arguments) {
    if (output->tool_call_count == SIZE_MAX / sizeof(*output->tool_calls))
        return 0;
    size_t count = output->tool_call_count + 1;
    OpenAIResponsesToolCall *grown = realloc(
        output->tool_calls, count * sizeof(*grown));
    if (!grown) return 0;
    output->tool_calls = grown;
    output->tool_calls[output->tool_call_count].name = name;
    output->tool_calls[output->tool_call_count].arguments = arguments;
    output->tool_call_count = count;
    return 1;
}

void openai_responses_output_free(OpenAIResponsesOutput *output) {
    if (!output) return;
    free(output->text);
    free(output->reasoning);
    for (size_t i = 0; i < output->tool_call_count; i++) {
        free(output->tool_calls[i].name);
        free(output->tool_calls[i].arguments);
    }
    free(output->tool_calls);
    memset(output, 0, sizeof(*output));
}

int openai_responses_output_parse(
    const char *raw, size_t raw_size, int thinking,
    OpenAIResponsesOutput *output) {
    if (!output || (!raw && raw_size)) return 0;
    memset(output, 0, sizeof(*output));
    char *copy = responses_strndup(raw ? raw : "", raw_size);
    if (!copy) return 0;
    const char *visible = copy;
    if (thinking) {
        const char *reasoning_start = !strncmp(visible, "<think>", 7)
            ? visible + 7 : visible;
        const char *reasoning_end = strstr(reasoning_start, "</think>");
        if (reasoning_end) {
            output->reasoning = responses_strndup(
                reasoning_start, (size_t)(reasoning_end - reasoning_start));
            if (!output->reasoning) goto fail;
            visible = reasoning_end + 8;
        } else {
            output->reasoning = responses_strdup(reasoning_start);
            output->text = responses_strdup("");
            output->text_size = 0;
            free(copy);
            return output->reasoning && output->text;
        }
    }
    const char *end_tag = NULL;
    const char *invoke_start = NULL;
    const char *invoke_end = NULL;
    const char *param_start = NULL;
    const char *param_end = NULL;
    const char *cursor = responses_tool_start(
        visible, &end_tag, &invoke_start, &invoke_end, &param_start, &param_end);
    if (!cursor) {
        output->text = responses_strdup(visible);
        output->text_size = strlen(visible);
        free(copy);
        return output->text != NULL;
    }
    const char *block_start = cursor;
    while (block_start > visible && block_start[-1] != '<') block_start--;
    if (block_start > visible) block_start--;
    size_t text_size = (size_t)(block_start - visible);
    while (text_size && (visible[text_size - 1] == ' ' ||
                         visible[text_size - 1] == '\t' ||
                         visible[text_size - 1] == '\r' ||
                         visible[text_size - 1] == '\n')) text_size--;
    while (text_size && (*visible == ' ' || *visible == '\t' ||
                         *visible == '\r' || *visible == '\n')) {
        visible++;
        text_size--;
    }
    output->text = responses_strndup(visible, text_size);
    output->text_size = text_size;
    if (!output->text) goto fail;

    for (;;) {
        cursor = responses_skip_space(cursor);
        if (end_tag && !strncmp(cursor, end_tag, strlen(end_tag))) break;
        if (strncmp(cursor, invoke_start, strlen(invoke_start))) {
            if (!end_tag && output->tool_call_count > 0) break;
            goto fail;
        }
        const char *tag_end = strchr(cursor, '>');
        if (!tag_end) goto fail;
        char *name = responses_xml_attr(
            cursor, (size_t)(tag_end - cursor + 1), "name");
        if (!name || !name[0]) {
            free(name);
            goto fail;
        }
        cursor = tag_end + 1;
        char *arguments = NULL;
        size_t arguments_size = 0;
        FILE *args = open_memstream(&arguments, &arguments_size);
        if (!args) {
            free(name);
            goto fail;
        }
        int ok = fputc('{', args) != EOF;
        int parameter_count = 0;
        while (ok) {
            cursor = responses_skip_space(cursor);
            const char *after_invoke = responses_invoke_close(
                cursor, invoke_end);
            if (after_invoke) {
                cursor = after_invoke;
                break;
            }
            if (strncmp(cursor, param_start, strlen(param_start))) {
                ok = 0;
                break;
            }
            tag_end = strchr(cursor, '>');
            if (!tag_end) {
                ok = 0;
                break;
            }
            char *param_name = responses_xml_attr(
                cursor, (size_t)(tag_end - cursor + 1), "name");
            char *is_string = responses_xml_attr(
                cursor, (size_t)(tag_end - cursor + 1), "string");
            const char *value_start = tag_end + 1;
            const char *value_end = strstr(value_start, param_end);
            if (!param_name || !value_end) {
                free(param_name);
                free(is_string);
                ok = 0;
                break;
            }
            char *value = responses_xml_unescape(
                value_start, (size_t)(value_end - value_start));
            if (!value) {
                free(param_name);
                free(is_string);
                ok = 0;
                break;
            }
            if (parameter_count++ && fputc(',', args) == EOF) ok = 0;
            if (ok) ok = responses_json_string(args, param_name) &&
                         fputc(':', args) != EOF;
            if (ok && (!is_string || strcmp(is_string, "false")))
                ok = responses_json_string(args, value);
            else if (ok) {
                char value_error[128] = {0};
                jval *parsed = json_parse_full(value, value_error,
                                               sizeof(value_error));
                ok = parsed && responses_json_value(args, parsed);
                json_free(parsed);
            }
            free(param_name);
            free(is_string);
            free(value);
            cursor = value_end + strlen(param_end);
        }
        if (ok) ok = fputc('}', args) != EOF;
        if (fclose(args) != 0) ok = 0;
        if (!ok || !responses_output_add_call(output, name, arguments)) {
            free(name);
            free(arguments);
            goto fail;
        }
    }
    free(copy);
    return output->tool_call_count > 0;

fail:
    openai_responses_output_free(output);
    output->text = copy;
    output->text_size = raw_size;
    return 1;
}
