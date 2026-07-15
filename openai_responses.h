#ifndef OPENAI_RESPONSES_H
#define OPENAI_RESPONSES_H

#include <stddef.h>

#include "openai_http.h"

typedef struct {
    OpenAIChatRequest chat;
} OpenAIResponsesRequest;

typedef struct {
    char *name;
    char *arguments;
} OpenAIResponsesToolCall;

typedef struct {
    char *text;
    size_t text_size;
    char *reasoning;
    OpenAIResponsesToolCall *tool_calls;
    size_t tool_call_count;
} OpenAIResponsesOutput;

int openai_responses_request_parse(
    const char *body, const char *served_model,
    OpenAIResponsesRequest *request, char *error, size_t error_size);
void openai_responses_request_free(OpenAIResponsesRequest *request);

int openai_responses_output_parse(
    const char *raw, size_t raw_size, int thinking,
    OpenAIResponsesOutput *output);
void openai_responses_output_free(OpenAIResponsesOutput *output);

#endif
