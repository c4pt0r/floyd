#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_chat.h"
#include "moonlight_chat.h"

typedef enum {
    COMMAND_CHAT,
    COMMAND_RUN,
    COMMAND_SERVE,
} Command;

typedef struct {
    Command command;
    const char *model_dir;
    const char *prompt;
    const char *system_prompt;
    const char *ds4_model;
    const char *ds4_support;
    int max_context;
    int max_new_tokens;
    int draft;
    float temperature;
    float top_p;
    uint64_t prefix_cache_mb;
    int context_set;
    int new_tokens_set;
    int draft_set;
    int temperature_set;
    int top_p_set;
    int system_set;
    int prefix_cache_set;
    int stdio;
    const char *host;
    int port;
    const char *api_key;
    const char *served_model_name;
    char *default_served_model_name;
    int host_set;
    int port_set;
    int api_key_set;
    int served_model_name_set;
    int trace;
} CliOptions;

static void usage(FILE *stream) {
    fputs(
        "floyd - Metal inference for Moonlight and DeepSeek V4 DS4\n"
        "usage: floyd [chat] --model DIR [flags]\n"
        "       floyd run --model DIR --prompt TEXT [flags]\n"
        "       floyd serve --model DIR [flags]\n\n"
        "commands:\n"
        "  chat   interactive conversation (default)\n"
        "  run    one-shot generation\n"
        "  serve  persistent OpenAI HTTP or JSONL service\n"
        "  help   show this text\n\n"
        "common flags:\n"
        "  --model DIR        model checkpoint directory\n"
        "  --ctx N            context length\n"
        "  --ngen N           maximum generated tokens\n"
        "  --draft N          DeepSeek V4 speculative draft length (0..16)\n"
        "  --ds4-model FILE   prepared DeepSeek V4 base GGUF\n"
        "  --ds4-support FILE prepared DeepSeek V4 speculative GGUF\n"
        "  --trace            print generated token IDs and timing details\n\n"
        "Moonlight flags:\n"
        "  --temp T           sampling temperature (default: 0.7)\n"
        "  --top-p P          nucleus probability (default: 0.90)\n"
        "  --system TEXT      system prompt\n\n"
        "run flags:\n"
        "  --prompt TEXT      prompt to generate from\n\n"
        "serve flags:\n"
        "  --stdio            read and write one JSON object per line\n"
        "  --host HOST        HTTP listen host (default: 127.0.0.1)\n"
        "  --port N           HTTP listen port (default: 8080)\n"
        "  --api-key KEY      require an Authorization Bearer token\n"
        "  --served-model-name NAME  model ID advertised over HTTP\n"
        "  --prefix-cache-mb N  prefix snapshot budget; 0 disables cache\n",
        stream);
}

static void fail_usage(const char *message) {
    fprintf(stderr, "%s\n\n", message);
    usage(stderr);
    exit(2);
}

static const char *next_value(int argc, char **argv, int *index,
                              const char *option) {
    if (++*index >= argc) {
        char message[128];
        snprintf(message, sizeof(message), "%s requires a value", option);
        fail_usage(message);
    }
    return argv[*index];
}

static int parse_integer(const char *text, int minimum, int maximum,
                         const char *option) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !text[0] || *end || value < minimum || value > maximum) {
        char message[160];
        snprintf(message, sizeof(message), "%s must be an integer in %d..%d",
                 option, minimum, maximum);
        fail_usage(message);
    }
    return (int)value;
}

static float parse_float(const char *text, float minimum, float maximum,
                         int include_minimum, const char *option) {
    char *end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (errno || !text[0] || *end || !isfinite(value) ||
        (include_minimum ? value < minimum : value <= minimum) ||
        value > maximum) {
        char message[160];
        snprintf(message, sizeof(message), "%s is outside its supported range",
                 option);
        fail_usage(message);
    }
    return value;
}

static uint64_t parse_megabytes(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !text[0] || text[0] == '-' || *end ||
        value > UINT64_MAX / (UINT64_C(1024) * 1024))
        fail_usage("prefix-cache-mb must be a non-negative integer");
    return (uint64_t)value;
}

static char *model_basename(const char *path) {
    const char *end = path + strlen(path);
    while (end > path && end[-1] == '/') end--;
    const char *start = end;
    while (start > path && start[-1] != '/') start--;
    if (start == end) return NULL;
    size_t size = (size_t)(end - start);
    char *name = malloc(size + 1);
    if (!name) return NULL;
    memcpy(name, start, size);
    name[size] = 0;
    return name;
}

static CliOptions parse_cli(int argc, char **argv) {
    CliOptions options = {
        .command = COMMAND_CHAT,
        .draft = 3,
        .temperature = 0.7f,
        .top_p = 0.90f,
        .prefix_cache_mb = 256,
        .host = "127.0.0.1",
        .port = 8080,
    };
    if (argc < 2) fail_usage("missing --model");

    int first = 1;
    if (!strcmp(argv[1], "help") || !strcmp(argv[1], "--help") ||
        !strcmp(argv[1], "-h")) {
        usage(stdout);
        exit(0);
    }
    if (!strcmp(argv[1], "chat")) {
        first = 2;
    } else if (!strcmp(argv[1], "run")) {
        options.command = COMMAND_RUN;
        first = 2;
    } else if (!strcmp(argv[1], "serve")) {
        options.command = COMMAND_SERVE;
        first = 2;
    } else if (strncmp(argv[1], "--", 2)) {
        char message[160];
        snprintf(message, sizeof(message), "unknown command: %s", argv[1]);
        fail_usage(message);
    }

    for (int index = first; index < argc; ++index) {
        const char *option = argv[index];
        if (!strcmp(option, "--model")) {
            options.model_dir = next_value(argc, argv, &index, option);
        } else if (!strcmp(option, "--prompt")) {
            options.prompt = next_value(argc, argv, &index, option);
        } else if (!strcmp(option, "--ctx")) {
            const char *value = next_value(argc, argv, &index, option);
            options.max_context = parse_integer(value, 1, 1 << 20, option);
            options.context_set = 1;
        } else if (!strcmp(option, "--ngen")) {
            const char *value = next_value(argc, argv, &index, option);
            options.max_new_tokens = parse_integer(value, 1, 1 << 20, option);
            options.new_tokens_set = 1;
        } else if (!strcmp(option, "--draft")) {
            const char *value = next_value(argc, argv, &index, option);
            options.draft = parse_integer(value, 0, 16, option);
            options.draft_set = 1;
        } else if (!strcmp(option, "--temp")) {
            const char *value = next_value(argc, argv, &index, option);
            options.temperature = parse_float(value, 0.0f, 2.0f, 1, option);
            options.temperature_set = 1;
        } else if (!strcmp(option, "--top-p")) {
            const char *value = next_value(argc, argv, &index, option);
            options.top_p = parse_float(value, 0.0f, 1.0f, 0, option);
            options.top_p_set = 1;
        } else if (!strcmp(option, "--system")) {
            options.system_prompt = next_value(argc, argv, &index, option);
            options.system_set = 1;
        } else if (!strcmp(option, "--ds4-model")) {
            options.ds4_model = next_value(argc, argv, &index, option);
        } else if (!strcmp(option, "--ds4-support")) {
            options.ds4_support = next_value(argc, argv, &index, option);
        } else if (!strcmp(option, "--trace")) {
            options.trace = 1;
        } else if (!strcmp(option, "--stdio")) {
            options.stdio = 1;
        } else if (!strcmp(option, "--host")) {
            options.host = next_value(argc, argv, &index, option);
            options.host_set = 1;
        } else if (!strcmp(option, "--port")) {
            const char *value = next_value(argc, argv, &index, option);
            options.port = parse_integer(value, 1, 65535, option);
            options.port_set = 1;
        } else if (!strcmp(option, "--api-key")) {
            options.api_key = next_value(argc, argv, &index, option);
            options.api_key_set = 1;
        } else if (!strcmp(option, "--served-model-name")) {
            options.served_model_name = next_value(argc, argv, &index, option);
            options.served_model_name_set = 1;
        } else if (!strcmp(option, "--prefix-cache-mb")) {
            const char *value = next_value(argc, argv, &index, option);
            options.prefix_cache_mb = parse_megabytes(value);
            options.prefix_cache_set = 1;
        } else {
            char message[160];
            snprintf(message, sizeof(message), "unknown option: %s", option);
            fail_usage(message);
        }
    }

    if (!options.model_dir) fail_usage("missing --model");
    if (options.command == COMMAND_RUN && !options.prompt)
        fail_usage("run requires --prompt");
    if (options.command != COMMAND_RUN && options.prompt)
        fail_usage("--prompt is only valid for run");
    if (options.command != COMMAND_SERVE && options.stdio)
        fail_usage("--stdio is only valid for serve");
    int http_flags_set = options.host_set || options.port_set ||
                         options.api_key_set || options.served_model_name_set;
    if (options.command != COMMAND_SERVE && http_flags_set)
        fail_usage("HTTP flags are only valid for serve");
    if (options.stdio && http_flags_set)
        fail_usage("HTTP flags cannot be used with --stdio");
    if (options.command != COMMAND_SERVE && options.prefix_cache_set)
        fail_usage("--prefix-cache-mb is only valid for serve");
    if (options.command == COMMAND_SERVE && !options.host[0])
        fail_usage("--host must be non-empty");
    if (options.command == COMMAND_SERVE && options.served_model_name_set &&
        !options.served_model_name[0])
        fail_usage("--served-model-name must be non-empty");
    if (options.command == COMMAND_SERVE && !options.served_model_name) {
        options.default_served_model_name = model_basename(options.model_dir);
        if (!options.default_served_model_name)
            fail_usage("could not derive served model name from --model");
        options.served_model_name = options.default_served_model_name;
    }
    return options;
}

static void configure_ds4(const CliOptions *options) {
    if (options->ds4_model)
        setenv("FLOYD_DEEPSEEK_V4_DS4_GGUF", options->ds4_model, 1);
    else
        unsetenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    if (options->ds4_support)
        setenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK", options->ds4_support, 1);
    else
        unsetenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");
    char draft[16];
    snprintf(draft, sizeof(draft), "%d", options->draft);
    setenv("DRAFT", draft, 1);
    if (options->trace)
        setenv("DEEPSEEK_V4_CHAT_TRACE", "1", 1);
    else
        unsetenv("DEEPSEEK_V4_CHAT_TRACE");
}

static int run_deepseek_v4(CliOptions *options) {
    if (options->temperature_set)
        fail_usage("DeepSeek V4 chat does not support --temp");
    if (options->top_p_set)
        fail_usage("DeepSeek V4 chat does not support --top-p");
    if (options->system_set)
        fail_usage("DeepSeek V4 chat does not support --system");
    configure_ds4(options);
    if (!options->context_set) options->max_context = 512;
    if (!options->new_tokens_set)
        options->max_new_tokens = options->command == COMMAND_RUN ? 256 : 16;
    if (options->command == COMMAND_SERVE) {
        DeepSeekV4ServeOptions serve = {
            .model_dir = options->model_dir,
            .max_context = options->max_context,
            .max_new_tokens = options->max_new_tokens,
            .draft = options->draft,
            .prefix_cache_bytes = options->prefix_cache_mb * UINT64_C(1024) * 1024,
            .stdio = options->stdio,
            .host = options->host,
            .port = options->port,
            .api_key = options->api_key,
            .served_model_name = options->served_model_name,
        };
        return deepseek_v4_serve_run(&serve);
    }
    DeepSeekV4ChatOptions chat = {
        .model_dir = options->model_dir,
        .prompt = options->command == COMMAND_RUN ? options->prompt : NULL,
        .max_context = options->max_context,
        .max_new_tokens = options->max_new_tokens,
        .use_spec = options->draft > 1,
    };
    return deepseek_v4_chat_run(&chat);
}

static int run_moonlight(CliOptions *options) {
    if (options->command == COMMAND_SERVE && options->stdio) {
        fprintf(stderr, "floyd serve currently requires DeepSeek V4\n");
        return 2;
    }
    if (options->draft_set && options->draft > 0)
        fail_usage("Moonlight does not support --draft");
    if (options->ds4_model || options->ds4_support)
        fail_usage("DS4 model flags require a DeepSeek V4 checkpoint");
    if (!options->context_set) options->max_context = 4096;
    if (!options->new_tokens_set)
        options->max_new_tokens = options->command == COMMAND_RUN ? 256 : 512;
    if (options->command == COMMAND_SERVE) {
        MoonlightServeOptions serve = {
            .model_dir = options->model_dir,
            .max_context = options->max_context,
            .max_new_tokens = options->max_new_tokens,
            .host = options->host,
            .port = options->port,
            .api_key = options->api_key,
            .served_model_name = options->served_model_name,
        };
        return moonlight_serve_run(&serve);
    }
    MoonlightChatOptions chat = {
        .model_dir = options->model_dir,
        .prompt = options->command == COMMAND_RUN ? options->prompt : NULL,
        .system_prompt = options->system_prompt
            ? options->system_prompt : "You are a helpful assistant",
        .max_context = options->max_context,
        .max_new_tokens = options->max_new_tokens,
        .temperature = options->temperature,
        .top_p = options->top_p,
    };
    return moonlight_chat_run(&chat);
}

int main(int argc, char **argv) {
    CliOptions options = parse_cli(argc, argv);
    int status;
    if (deepseek_v4_model_dir(options.model_dir))
        status = run_deepseek_v4(&options);
    else if (moonlight_metal_model_dir(options.model_dir))
        status = run_moonlight(&options);
    else {
        fprintf(stderr, "unsupported model directory: %s\n", options.model_dir);
        status = 2;
    }
    free(options.default_served_model_name);
    return status;
}
