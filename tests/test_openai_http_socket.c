#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../openai_http.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define HEADER_LIMIT (32u * 1024u)
#define BODY_LIMIT (8u * 1024u * 1024u)

typedef struct {
    int calls;
} FakeContext;

typedef struct {
    int client_fd;
    int calls;
    int sink_result;
} DisconnectContext;

typedef struct {
    const char *finish_reason;
    int calls;
} FinishContext;

typedef struct {
    int fd;
    const OpenAIHttpConfig *config;
    OpenAIGenerateHandler handler;
    void *user_data;
    int result;
    double elapsed_ms;
} ServerThread;

typedef struct {
    char *piece;
    size_t piece_size;
    int calls;
} LargeOutputContext;

typedef struct {
    int calls;
    int emit_tool_call;
} PieResponsesContext;

static int fake_generate(void *ctx, const OpenAIChatRequest *request,
                         OpenAITokenSink sink, void *sink_data,
                         OpenAIGenerationResult *result,
                         char *error, size_t error_size) {
    FakeContext *fake = ctx;
    (void)error;
    (void)error_size;
    CHECK(request != NULL);
    CHECK(strcmp(request->model, "test-model") == 0);
    fake->calls++;
    CHECK(sink(1, "Hel", 3, sink_data));
    CHECK(sink(2, "lo", 2, sink_data));
    result->prompt_tokens = 5;
    result->cached_tokens = 2;
    result->completion_tokens = 2;
    result->finish_reason = "stop";
    return 1;
}

static int utf8_generate(void *ctx, const OpenAIChatRequest *request,
                         OpenAITokenSink sink, void *sink_data,
                         OpenAIGenerationResult *result,
                         char *error, size_t error_size) {
    FakeContext *fake = ctx;
    static const char embedded_nul[] = {'A', '\0', 'B'};
    (void)error;
    (void)error_size;
    CHECK(request != NULL);
    fake->calls++;
    CHECK(sink(1, "\xe2", 1, sink_data));
    CHECK(sink(2, "\x82", 1, sink_data));
    CHECK(sink(3, "\xac", 1, sink_data));
    CHECK(sink(4, embedded_nul, sizeof(embedded_nul), sink_data));
    CHECK(sink(5, "\xff", 1, sink_data));
    CHECK(sink(6, "\xf0\x9f", 2, sink_data));
    result->prompt_tokens = 3;
    result->completion_tokens = 6;
    result->finish_reason = "stop";
    return 1;
}

static int finish_generate(void *ctx, const OpenAIChatRequest *request,
                           OpenAITokenSink sink, void *sink_data,
                           OpenAIGenerationResult *result,
                           char *error, size_t error_size) {
    FinishContext *finish = ctx;
    (void)request;
    (void)error;
    (void)error_size;
    finish->calls++;
    CHECK(sink(1, "x", 1, sink_data));
    result->prompt_tokens = 1;
    result->completion_tokens = 1;
    result->finish_reason = finish->finish_reason;
    return 1;
}

static int disconnect_generate(void *ctx, const OpenAIChatRequest *request,
                               OpenAITokenSink sink, void *sink_data,
                               OpenAIGenerationResult *result,
                               char *error, size_t error_size) {
    DisconnectContext *disconnect = ctx;
    (void)request;
    (void)result;
    (void)error;
    (void)error_size;
    disconnect->calls++;
    struct linger reset = {.l_onoff = 1, .l_linger = 0};
    setsockopt(disconnect->client_fd, SOL_SOCKET, SO_LINGER,
               &reset, sizeof(reset));
    close(disconnect->client_fd);
    disconnect->client_fd = -1;
    disconnect->sink_result = sink(1, "ignored", 7, sink_data);
    return 0;
}

static int pretoken_client_error_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    FakeContext *fake = ctx;
    (void)request;
    (void)sink;
    (void)sink_data;
    (void)result;
    fake->calls++;
    snprintf(error, error_size, "prompt exceeds model context");
    return OPENAI_GENERATE_CLIENT_ERROR;
}

static int posttoken_internal_error_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    FakeContext *fake = ctx;
    (void)request;
    (void)result;
    fake->calls++;
    CHECK(sink(1, "partial", 7, sink_data));
    snprintf(error, error_size, "failed to read /sensitive/model/path");
    return OPENAI_GENERATE_INTERNAL_ERROR;
}

static int zero_token_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    FakeContext *fake = ctx;
    (void)request;
    (void)sink;
    (void)sink_data;
    (void)error;
    (void)error_size;
    fake->calls++;
    result->prompt_tokens = 1;
    result->finish_reason = "stop";
    return 1;
}

static int delayed_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    FakeContext *fake = ctx;
    (void)request;
    (void)error;
    (void)error_size;
    fake->calls++;
    usleep(80000);
    CHECK(sink(1, "ok", 2, sink_data));
    result->prompt_tokens = 1;
    result->completion_tokens = 1;
    result->finish_reason = "stop";
    return 1;
}

static int large_output_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    LargeOutputContext *large = ctx;
    (void)request;
    (void)result;
    (void)error;
    (void)error_size;
    large->calls++;
    return sink(1, large->piece, large->piece_size, sink_data) ? 1 : 0;
}

static int pie_responses_generate(
    void *ctx, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    PieResponsesContext *pie = ctx;
    (void)error;
    (void)error_size;
    CHECK(request != NULL);
    CHECK(strcmp(request->model, "deepseek-v4-flash") == 0);
    CHECK(request->message_count == 2);
    CHECK(strcmp(request->messages[0].role, "system") == 0);
    CHECK(strstr(request->messages[0].content, "## Tools") != NULL);
    CHECK(strstr(request->messages[0].content, "You are Pie.") != NULL);
    CHECK(strstr(request->messages[0].content, "\"name\":\"ls\"") != NULL);
    CHECK(strcmp(request->messages[1].role, "user") == 0);
    CHECK(strcmp(request->messages[1].content, "Reply OK") == 0);
    pie->calls++;
    if (pie->emit_tool_call) {
        static const char dsml[] =
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"ls\">\n"
            "<｜DSML｜parameter name=\"path\" string=\"true\">."
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>";
        CHECK(sink(1, dsml, sizeof(dsml) - 1, sink_data));
    } else {
        CHECK(sink(1, "OK", 2, sink_data));
    }
    result->prompt_tokens = 23;
    result->cached_tokens = 7;
    result->completion_tokens = 1;
    result->finish_reason = "stop";
    return OPENAI_GENERATE_OK;
}

static double monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static void *serve_connection(void *opaque) {
    ServerThread *thread = opaque;
    double started = monotonic_ms();
    thread->result = openai_http_handle_connection(
        thread->fd, thread->config, thread->handler, thread->user_data);
    thread->elapsed_ms = monotonic_ms() - started;
    close(thread->fd);
    return NULL;
}

static int socket_send_all(int fd, const char *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
#ifdef MSG_NOSIGNAL
        ssize_t sent = send(fd, data + offset, size - offset, MSG_NOSIGNAL);
#else
        ssize_t sent = send(fd, data + offset, size - offset, 0);
#endif
        if (sent > 0) {
            offset += (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) continue;
        if (sent < 0 && (errno == EPIPE || errno == ECONNRESET)) return 1;
        return 0;
    }
    return 1;
}

static char *socket_read_all(int fd) {
    size_t size = 0;
    size_t capacity = 4096;
    char *response = malloc(capacity);
    if (!response) return NULL;
    for (;;) {
        if (size + 1 == capacity) {
            if (capacity > SIZE_MAX / 2) {
                free(response);
                return NULL;
            }
            capacity *= 2;
            char *grown = realloc(response, capacity);
            if (!grown) {
                free(response);
                return NULL;
            }
            response = grown;
        }
        ssize_t received = recv(fd, response + size, capacity - size - 1, 0);
        if (received > 0) {
            size += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) continue;
        if (received < 0) {
            free(response);
            return NULL;
        }
        response[size] = 0;
        return response;
    }
}

static char *exchange(const OpenAIHttpConfig *config,
                      OpenAIGenerateHandler handler, void *user_data,
                      const char *request, size_t request_size,
                      int *handle_result) {
    int pair[2] = {-1, -1};
    pthread_t worker;
    ServerThread thread = {
        .fd = -1,
        .config = config,
        .handler = handler,
        .user_data = user_data,
        .result = 0,
    };
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return NULL;

    struct timeval timeout = {.tv_sec = 5, .tv_usec = 0};
    setsockopt(pair[0], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(pair[0], SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    setsockopt(pair[0], SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif

    thread.fd = pair[1];
    if (pthread_create(&worker, NULL, serve_connection, &thread) != 0) {
        close(pair[0]);
        close(pair[1]);
        return NULL;
    }
    if (!socket_send_all(pair[0], request, request_size)) {
        close(pair[0]);
        pthread_join(worker, NULL);
        return NULL;
    }
    char *response = socket_read_all(pair[0]);
    close(pair[0]);
    pthread_join(worker, NULL);
    if (handle_result) *handle_result = thread.result;
    return response;
}

static char *post_path_request_handler(
    const OpenAIHttpConfig *config, OpenAIGenerateHandler handler,
    void *user_data, const char *path, const char *body) {
    size_t capacity = strlen(body) + 256;
    char *request = malloc(capacity);
    if (!request) return NULL;
    int size = snprintf(
        request, capacity,
        "POST %s HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "authorization: Bearer secret\r\n"
        "content-type: application/json\r\n"
        "content-length: %zu\r\n\r\n%s",
        path, strlen(body), body);
    if (size < 0 || (size_t)size >= capacity) {
        free(request);
        return NULL;
    }
    char *response = exchange(
        config, handler, user_data, request, (size_t)size, NULL);
    free(request);
    return response;
}

static char *post_request_handler(
    const OpenAIHttpConfig *config, OpenAIGenerateHandler handler,
    void *user_data, const char *body) {
    return post_path_request_handler(
        config, handler, user_data, "/v1/chat/completions", body);
}

static char *post_request(const OpenAIHttpConfig *config, FakeContext *fake,
                          const char *body) {
    return post_request_handler(config, fake_generate, fake, body);
}

static int has_status(const char *response, int status) {
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "HTTP/1.1 %d ", status);
    return response && strncmp(response, prefix, strlen(prefix)) == 0;
}

static int test_models_and_authentication(void) {
    OpenAIHttpConfig keyed = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *valid =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"
        "Authorization: Bearer secret\r\n\r\n";
    char *response = exchange(
        &keyed, fake_generate, &fake, valid, strlen(valid), NULL);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "Content-Type: application/json") != NULL);
    CHECK(strstr(response, "\"object\":\"list\"") != NULL);
    CHECK(strstr(response, "\"id\":\"test-model\"") != NULL);
    CHECK(fake.calls == 0);
    free(response);

    const char *invalid =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"
        "Authorization: Bearer wrong\r\n\r\n";
    response = exchange(
        &keyed, fake_generate, &fake, invalid, strlen(invalid), NULL);
    CHECK(has_status(response, 401));
    CHECK(strstr(response, "Content-Type: application/json") != NULL);
    CHECK(strstr(response, "\"error\"") != NULL);
    free(response);

    const char *missing =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    response = exchange(
        &keyed, fake_generate, &fake, missing, strlen(missing), NULL);
    CHECK(has_status(response, 401));
    free(response);

    const char *invalid_values[] = {
        "bearer secret",
        "BEARER secret",
        "Bearer  secret",
        "Bearer secret ",
        " Bearer secret",
        "\tBearer secret",
        "Bearer secret-suffix",
        "prefix-Bearer secret",
    };
    for (size_t i = 0;
         i < sizeof(invalid_values) / sizeof(invalid_values[0]); i++) {
        char request[256];
        int request_size = snprintf(
            request, sizeof(request),
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"
            "Authorization: %s\r\n\r\n", invalid_values[i]);
        CHECK(request_size > 0 && (size_t)request_size < sizeof(request));
        response = exchange(
            &keyed, fake_generate, &fake,
            request, (size_t)request_size, NULL);
        CHECK(has_status(response, 401));
        free(response);
    }

    OpenAIHttpConfig open = keyed;
    open.api_key = NULL;
    response = exchange(
        &open, fake_generate, &fake, invalid, strlen(invalid), NULL);
    CHECK(has_status(response, 200));
    free(response);

    OpenAIHttpConfig empty = keyed;
    empty.api_key = "";
    response = exchange(
        &empty, fake_generate, &fake, valid, strlen(valid), NULL);
    CHECK(response != NULL && response[0] == 0);
    free(response);
    return 0;
}

static int test_host_header_requirements(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *requests[] = {
        "GET /v1/models HTTP/1.1\r\n\r\n",
        "GET /v1/models HTTP/1.1\r\nHost: \r\n\r\n",
        "GET /v1/models HTTP/1.1\r\nHost:\t\r\n\r\n",
        "GET /v1/models HTTP/1.1\r\nHost: one\r\nHost: two\r\n\r\n",
    };
    for (size_t i = 0; i < sizeof(requests) / sizeof(requests[0]); i++) {
        char *response = exchange(
            &config, fake_generate, &fake,
            requests[i], strlen(requests[i]), NULL);
        CHECK(has_status(response, 400));
        CHECK(strstr(response, "Host") != NULL);
        free(response);
    }
    return 0;
}

static int reserve_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t address_size = sizeof(address);
    if (getsockname(fd, (struct sockaddr *)&address, &address_size) != 0) {
        close(fd);
        return -1;
    }
    int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

static int connect_loopback(pid_t child, int port) {
    for (int attempt = 0; attempt < 200; attempt++) {
        int status = 0;
        if (waitpid(child, &status, WNOHANG) == child) return -1;
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        struct sockaddr_in address;
        memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons((uint16_t)port);
        if (connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0)
            return fd;
        close(fd);
        usleep(10000);
    }
    return -1;
}

static int wait_for_child(pid_t child, int *status) {
    for (int attempt = 0; attempt < 200; attempt++) {
        pid_t waited = waitpid(child, status, WNOHANG);
        if (waited == child) return 1;
        if (waited < 0) return 0;
        usleep(10000);
    }
    kill(child, SIGKILL);
    waitpid(child, status, 0);
    return 0;
}

typedef struct {
    int signal_number;
} SignalReceiver;

typedef struct {
    pid_t child;
    int signal_number;
    int result;
    pthread_mutex_t *mutex;
    pthread_cond_t *condition;
    int *ready;
    int *start;
} SignalSender;

static void *receive_termination_signal(void *opaque) {
    SignalReceiver *receiver = opaque;
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, receiver->signal_number);
    pthread_sigmask(SIG_UNBLOCK, &signals, NULL);
    for (;;) pause();
    return NULL;
}

static void *send_termination_signal(void *opaque) {
    SignalSender *sender = opaque;
    pthread_mutex_lock(sender->mutex);
    (*sender->ready)++;
    pthread_cond_broadcast(sender->condition);
    while (!*sender->start)
        pthread_cond_wait(sender->condition, sender->mutex);
    pthread_mutex_unlock(sender->mutex);
    sender->result = kill(sender->child, sender->signal_number);
    return NULL;
}

static int send_concurrent_termination(pid_t child) {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    int ready = 0;
    int start = 0;
    SignalSender senders[2] = {
        {.child = child, .signal_number = SIGINT, .result = -1,
         .mutex = &mutex, .condition = &condition,
         .ready = &ready, .start = &start},
        {.child = child, .signal_number = SIGTERM, .result = -1,
         .mutex = &mutex, .condition = &condition,
         .ready = &ready, .start = &start},
    };
    pthread_t threads[2];
    if (pthread_create(&threads[0], NULL,
                       send_termination_signal, &senders[0]) != 0) {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&condition);
        return 0;
    }
    if (pthread_create(&threads[1], NULL,
                       send_termination_signal, &senders[1]) != 0) {
        pthread_mutex_lock(&mutex);
        start = 1;
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&mutex);
        pthread_join(threads[0], NULL);
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&condition);
        return 0;
    }
    pthread_mutex_lock(&mutex);
    while (ready != 2) pthread_cond_wait(&condition, &mutex);
    start = 1;
    pthread_cond_broadcast(&condition);
    pthread_mutex_unlock(&mutex);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&condition);
    return senders[0].result == 0 && senders[1].result == 0;
}

static int run_serve_shutdown_case(
    const char *partial, size_t partial_size,
    int termination_signal, int repeat_serve) {
    int port = reserve_loopback_port();
    if (port < 0) return 0;
    pid_t child = fork();
    if (child < 0) return 0;
    if (child == 0) {
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGINT);
        sigaddset(&signals, SIGTERM);
        if (pthread_sigmask(SIG_BLOCK, &signals, NULL) != 0) _exit(3);
        SignalReceiver receivers[2] = {
            {.signal_number = SIGINT},
            {.signal_number = SIGTERM},
        };
        pthread_t signal_threads[2];
        for (int i = 0; i < 2; i++) {
            if (pthread_create(&signal_threads[i], NULL,
                               receive_termination_signal,
                               &receivers[i]) != 0) _exit(3);
            pthread_detach(signal_threads[i]);
        }
        OpenAIHttpConfig config = {
            .host = "127.0.0.1",
            .port = port,
            .api_key = NULL,
            .model_name = "test-model",
        };
        FakeContext fake = {0};
        char error[256] = {0};
        int serve_count = repeat_serve ? 2 : 1;
        for (int i = 0; i < serve_count; i++) {
            if (!openai_http_serve(
                    &config, fake_generate, &fake,
                    error, sizeof(error))) _exit(2);
            if (repeat_serve && i == 0) usleep(20000);
        }
        _exit(0);
    }

    int first = connect_loopback(child, port);
    if (first < 0) {
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        return 0;
    }
    struct timeval response_deadline = {.tv_sec = 2, .tv_usec = 0};
    if (setsockopt(first, SOL_SOCKET, SO_RCVTIMEO,
                   &response_deadline, sizeof(response_deadline)) != 0) {
        close(first);
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        return 0;
    }
    const char *models =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    int ok = socket_send_all(first, models, strlen(models));
    shutdown(first, SHUT_WR);
    char *response = ok ? socket_read_all(first) : NULL;
    close(first);
    ok = has_status(response, 200);
    free(response);
    if (!ok) {
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        return 0;
    }

    int blocked = connect_loopback(child, port);
    if (blocked < 0 || !socket_send_all(blocked, partial, partial_size)) {
        if (blocked >= 0) close(blocked);
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        return 0;
    }
    usleep(100000);
    ok = termination_signal ? kill(child, termination_signal) == 0
                            : send_concurrent_termination(child);
    if (!ok) {
        close(blocked);
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        return 0;
    }
    if (repeat_serve) {
        close(blocked);
        blocked = -1;
        usleep(100000);
        int second = connect_loopback(child, port);
        if (second < 0) {
            kill(child, SIGKILL);
            waitpid(child, NULL, 0);
            return 0;
        }
        if (setsockopt(second, SOL_SOCKET, SO_RCVTIMEO,
                       &response_deadline, sizeof(response_deadline)) != 0 ||
            !socket_send_all(second, models, strlen(models))) {
            close(second);
            kill(child, SIGKILL);
            waitpid(child, NULL, 0);
            return 0;
        }
        shutdown(second, SHUT_WR);
        response = socket_read_all(second);
        close(second);
        ok = has_status(response, 200);
        free(response);
        if (!ok || kill(child, SIGTERM) != 0) {
            kill(child, SIGKILL);
            waitpid(child, NULL, 0);
            return 0;
        }
    }
    int status = 0;
    ok = wait_for_child(child, &status) &&
         WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (blocked >= 0) close(blocked);
    return ok;
}

static int test_serve_wakeup_interrupts_partial_requests(void) {
    const char *partial_header =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Len";
    CHECK(run_serve_shutdown_case(
        partial_header, strlen(partial_header), SIGINT, 0));

    const char *partial_body =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: 100\r\n\r\n{";
    CHECK(run_serve_shutdown_case(
        partial_body, strlen(partial_body), SIGTERM, 0));
    CHECK(run_serve_shutdown_case(
        partial_body, strlen(partial_body), 0, 1));
    return 0;
}

static int test_stream_disconnect_stops_sink(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = NULL,
        .model_name = "test-model",
    };
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}";
    char request[512];
    int request_size = snprintf(
        request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n%s", strlen(body), body);
    CHECK(request_size > 0 && (size_t)request_size < sizeof(request));

    int pair[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    setsockopt(pair[1], SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
    CHECK(socket_send_all(pair[0], request, (size_t)request_size));
    shutdown(pair[0], SHUT_WR);

    DisconnectContext disconnect = {
        .client_fd = pair[0],
        .calls = 0,
        .sink_result = 1,
    };
    int handled = openai_http_handle_connection(
        pair[1], &config, disconnect_generate, &disconnect);
    close(pair[1]);
    if (disconnect.client_fd >= 0) close(disconnect.client_fd);
    CHECK(handled == 0);
    CHECK(disconnect.calls == 1);
    CHECK(disconnect.sink_result == 0);
    return 0;
}

static int test_non_stream_disconnect_stops_sink(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
    };
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char request[512];
    int request_size = snprintf(
        request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n%s", strlen(body), body);
    CHECK(request_size > 0 && (size_t)request_size < sizeof(request));

    int pair[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    CHECK(socket_send_all(pair[0], request, (size_t)request_size));
    shutdown(pair[0], SHUT_WR);
    DisconnectContext disconnect = {
        .client_fd = pair[0],
        .sink_result = 1,
    };
    int handled = openai_http_handle_connection(
        pair[1], &config, disconnect_generate, &disconnect);
    close(pair[1]);
    if (disconnect.client_fd >= 0) close(disconnect.client_fd);
    CHECK(handled == 0);
    CHECK(disconnect.calls == 1);
    CHECK(disconnect.sink_result == 0);
    return 0;
}

static int test_generation_error_responses(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
    };
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}";
    FakeContext fake = {0};
    char *response = post_request_handler(
        &config, pretoken_client_error_generate, &fake, body);
    CHECK(has_status(response, 400));
    CHECK(strstr(response, "Content-Type: application/json") != NULL);
    CHECK(strstr(response, "prompt exceeds model context") != NULL);
    CHECK(strstr(response, "text/event-stream") == NULL);
    free(response);

    response = post_request_handler(
        &config, posttoken_internal_error_generate, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "Content-Type: text/event-stream") != NULL);
    char *partial = strstr(response, "\"content\":\"partial\"");
    char *error = strstr(response, "data: {\"error\":");
    char *done = strstr(response, "data: [DONE]\n\n");
    CHECK(partial && error && done && partial < error && error < done);
    CHECK(strstr(response, "/sensitive/model/path") == NULL);
    free(response);

    response = post_request_handler(&config, zero_token_generate, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "\"role\":\"assistant\"") != NULL);
    CHECK(strstr(response, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(response, "data: [DONE]\n\n") != NULL);
    free(response);
    CHECK(fake.calls == 3);
    return 0;
}

static int extract_completion_id(
    const char *response, char *id, size_t id_size) {
    const char *start = strstr(response, "\"id\":\"chatcmpl-");
    if (!start) return 0;
    start += strlen("\"id\":\"");
    const char *end = strchr(start, '"');
    if (!end || end == start || (size_t)(end - start) >= id_size) return 0;
    memcpy(id, start, (size_t)(end - start));
    id[end - start] = 0;
    return 1;
}

static int test_unique_completion_ids(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char *first = post_request_handler(&config, fake_generate, &fake, body);
    char *second = post_request_handler(&config, fake_generate, &fake, body);
    char first_id[128], second_id[128];
    CHECK(extract_completion_id(first, first_id, sizeof(first_id)));
    CHECK(extract_completion_id(second, second_id, sizeof(second_id)));
    CHECK(strcmp(first_id, second_id) != 0);
    free(first);
    free(second);

    body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}";
    char *stream = post_request_handler(&config, fake_generate, &fake, body);
    char stream_id[128], marker[160];
    CHECK(extract_completion_id(stream, stream_id, sizeof(stream_id)));
    int marker_size = snprintf(
        marker, sizeof(marker), "\"id\":\"%s\"", stream_id);
    CHECK(marker_size > 0 && (size_t)marker_size < sizeof(marker));
    int occurrences = 0;
    for (char *found = stream; (found = strstr(found, marker)) != NULL;
         found += strlen(marker)) occurrences++;
    CHECK(occurrences == 4);
    free(stream);
    return 0;
}

static char *run_partial_timeout(
    const char *request, size_t request_size, int timeout_ms) {
    int pair[2] = {-1, -1};
    pthread_t worker;
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
        .io_timeout_ms = timeout_ms,
    };
    FakeContext fake = {0};
    ServerThread thread = {
        .fd = -1,
        .config = &config,
        .handler = fake_generate,
        .user_data = &fake,
    };
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return NULL;
    struct timeval receive_timeout = {.tv_sec = 0, .tv_usec = 500000};
    setsockopt(pair[0], SOL_SOCKET, SO_RCVTIMEO,
               &receive_timeout, sizeof(receive_timeout));
    thread.fd = pair[1];
    if (pthread_create(&worker, NULL, serve_connection, &thread) != 0) {
        close(pair[0]);
        close(pair[1]);
        return NULL;
    }
    if (!socket_send_all(pair[0], request, request_size)) {
        close(pair[0]);
        pthread_join(worker, NULL);
        return NULL;
    }
    char *response = socket_read_all(pair[0]);
    close(pair[0]);
    pthread_join(worker, NULL);
    return response;
}

static int test_io_read_deadlines(void) {
    const char *slow_header =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Len";
    char *response = run_partial_timeout(
        slow_header, strlen(slow_header), 30);
    CHECK(has_status(response, 408));
    free(response);

    const char *slow_body =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: 100\r\n\r\n{";
    response = run_partial_timeout(slow_body, strlen(slow_body), 30);
    CHECK(has_status(response, 408));
    free(response);
    return 0;
}

static int test_io_write_deadline(void) {
    int pair[2] = {-1, -1};
    pthread_t worker;
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    int send_buffer = 4096;
    setsockopt(pair[1], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
        .io_timeout_ms = 30,
    };
    LargeOutputContext large = {
        .piece_size = 4u * 1024u * 1024u,
    };
    large.piece = malloc(large.piece_size);
    CHECK(large.piece != NULL);
    memset(large.piece, 'x', large.piece_size);
    ServerThread thread = {
        .fd = pair[1],
        .config = &config,
        .handler = large_output_generate,
        .user_data = &large,
    };
    CHECK(pthread_create(&worker, NULL, serve_connection, &thread) == 0);
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}";
    char request[512];
    int request_size = snprintf(
        request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n%s", strlen(body), body);
    CHECK(request_size > 0 && (size_t)request_size < sizeof(request));
    CHECK(socket_send_all(pair[0], request, (size_t)request_size));
    shutdown(pair[0], SHUT_WR);
    usleep(200000);
    close(pair[0]);
    pthread_join(worker, NULL);
    free(large.piece);
    CHECK(large.calls == 1);
    CHECK(thread.result == 0);
    CHECK(thread.elapsed_ms >= 20.0 && thread.elapsed_ms < 150.0);
    return 0;
}

static int test_generation_is_not_io_timed(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .model_name = "test-model",
        .io_timeout_ms = 20,
    };
    FakeContext fake = {0};
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char *response = post_request_handler(
        &config, delayed_generate, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "\"content\":\"ok\"") != NULL);
    CHECK(fake.calls == 1);
    free(response);
    return 0;
}

static int test_non_streaming_completion(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char *response = post_request(&config, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "Content-Type: application/json") != NULL);
    CHECK(strstr(response, "\"object\":\"chat.completion\"") != NULL);
    CHECK(strstr(response, "\"content\":\"Hello\"") != NULL);
    CHECK(strstr(response, "\"prompt_tokens\":5") != NULL);
    CHECK(strstr(response, "\"cached_tokens\":2") != NULL);
    CHECK(strstr(response, "\"completion_tokens\":2") != NULL);
    CHECK(fake.calls == 1);
    free(response);
    return 0;
}

static int test_streaming_completion(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
    char *response = post_request(&config, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "Content-Type: text/event-stream") != NULL);
    CHECK(strstr(response, "Cache-Control: no-cache") != NULL);

    char *role = strstr(response, "\"role\":\"assistant\"");
    char *hel = strstr(response, "\"content\":\"Hel\"");
    char *lo = strstr(response, "\"content\":\"lo\"");
    char *terminal = strstr(response, "\"finish_reason\":\"stop\"");
    char *usage = strstr(response, "\"choices\":[],\"usage\"");
    char *done = strstr(response, "data: [DONE]\n\n");
    CHECK(role && hel && lo && terminal && usage && done);
    CHECK(role < hel && hel < lo && lo < terminal && terminal < usage && usage < done);
    CHECK(strstr(usage, "\"prompt_tokens\":5") != NULL);
    CHECK(strstr(usage, "\"cached_tokens\":2") != NULL);
    CHECK(fake.calls == 1);
    free(response);
    return 0;
}

static int test_pie_responses_streaming(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "deepseek-v4",
    };
    static const char body[] =
        "{\"model\":\"deepseek-v4-flash\",\"input\":["
        "{\"role\":\"system\",\"content\":[{\"type\":\"input_text\","
        "\"text\":\"You are Pie.\"}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"input_text\","
        "\"text\":\"Reply OK\"}]}],"
        "\"tools\":[{\"type\":\"function\",\"name\":\"ls\","
        "\"description\":\"List files\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"path\":{\"type\":\"string\"}}}}],"
        "\"stream\":true,\"store\":false,\"max_output_tokens\":8}"
        ;

    PieResponsesContext pie = {0};
    char *response = post_path_request_handler(
        &config, pie_responses_generate, &pie, "/v1/responses", body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response, "Content-Type: text/event-stream") != NULL);
    char *created = strstr(response, "\"type\":\"response.created\"");
    char *added = strstr(response, "\"type\":\"response.output_item.added\"");
    char *delta = strstr(response, "\"type\":\"response.output_text.delta\"");
    char *completed = strstr(response, "\"type\":\"response.completed\"");
    CHECK(created && added && delta && completed);
    CHECK(created < added && added < delta && delta < completed);
    CHECK(strstr(delta, "\"delta\":\"OK\"") != NULL);
    CHECK(strstr(completed, "\"input_tokens\":23") != NULL);
    CHECK(strstr(completed, "\"cached_tokens\":7") != NULL);
    CHECK(strstr(completed, "\"output_tokens\":1") != NULL);
    CHECK(pie.calls == 1);
    free(response);

    pie.emit_tool_call = 1;
    response = post_path_request_handler(
        &config, pie_responses_generate, &pie, "/v1/responses", body);
    CHECK(has_status(response, 200));
    char *tool_added = strstr(
        response, "\"type\":\"response.output_item.added\","
                  "\"output_index\":0,\"item\":{\"id\":\"fc_");
    char *args_delta = strstr(
        response, "\"type\":\"response.function_call_arguments.delta\"");
    char *args_done = strstr(
        response, "\"type\":\"response.function_call_arguments.done\"");
    char *tool_done = strstr(
        response, "\"type\":\"response.output_item.done\"");
    completed = strstr(response, "\"type\":\"response.completed\"");
    CHECK(tool_added && args_delta && args_done && tool_done && completed);
    CHECK(tool_added < args_delta && args_delta < args_done &&
          args_done < tool_done && tool_done < completed);
    CHECK(strstr(response, "\"name\":\"ls\"") != NULL);
    CHECK(strstr(response, "{\\\"path\\\":\\\".\\\"}") != NULL);
    CHECK(strstr(response, "DSML") == NULL);
    CHECK(pie.calls == 2);
    free(response);
    return 0;
}

static int test_token_byte_normalization(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char *response = post_request_handler(
        &config, utf8_generate, &fake, body);
    CHECK(has_status(response, 200));
    CHECK(strstr(response,
                 "\"content\":\"" "\xe2\x82\xac"
                 "A\\u0000B" "\xef\xbf\xbd\xef\xbf\xbd" "\"") != NULL);
    CHECK(strstr(response, "\"completion_tokens\":6") != NULL);
    free(response);

    body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
    response = post_request_handler(&config, utf8_generate, &fake, body);
    CHECK(has_status(response, 200));
    char *euro = strstr(
        response, "\"content\":\"" "\xe2\x82\xac" "\"");
    char *nul = strstr(response, "\"content\":\"A\\u0000B\"");
    static const char replacement_event[] =
        "\"content\":\"" "\xef\xbf\xbd" "\"";
    char *replacement = strstr(response, replacement_event);
    char *final_replacement = replacement
        ? strstr(replacement + 1, replacement_event) : NULL;
    char *terminal = strstr(response, "\"finish_reason\":\"stop\"");
    char *usage = strstr(response, "\"choices\":[],\"usage\"");
    char *done = strstr(response, "data: [DONE]\n\n");
    CHECK(euro && nul && replacement && final_replacement &&
          terminal && usage && done);
    CHECK(euro < nul && nul < replacement &&
          replacement < final_replacement && final_replacement < terminal &&
          terminal < usage && usage < done);
    CHECK(strstr(usage, "\"completion_tokens\":6") != NULL);
    CHECK(fake.calls == 2);
    free(response);
    return 0;
}

static int test_finish_reasons(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    const char *body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    const char *reasons[] = {"stop", "length"};
    for (size_t index = 0; index < 2; ++index) {
        FinishContext finish = {
            .finish_reason = reasons[index],
        };
        char *response = post_request_handler(
            &config, finish_generate, &finish, body);
        CHECK(has_status(response, 200));
        char expected[64];
        int size = snprintf(
            expected, sizeof(expected),
            "\"finish_reason\":\"%s\"", reasons[index]);
        CHECK(size > 0 && (size_t)size < sizeof(expected));
        CHECK(strstr(response, expected) != NULL);
        CHECK(finish.calls == 1);
        free(response);
    }
    return 0;
}

static int test_routes_and_methods(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "secret",
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *unknown =
        "GET /v1/unknown HTTP/1.1\r\nHost: localhost\r\n"
        "Authorization: Bearer secret\r\n\r\n";
    char *response = exchange(
        &config, fake_generate, &fake, unknown, strlen(unknown), NULL);
    CHECK(has_status(response, 404));
    free(response);

    const char *method =
        "POST /v1/models HTTP/1.1\r\nHost: localhost\r\n"
        "Authorization: Bearer secret\r\nContent-Length: 0\r\n\r\n";
    response = exchange(
        &config, fake_generate, &fake, method, strlen(method), NULL);
    CHECK(has_status(response, 405));
    CHECK(strstr(response, "Allow: GET") != NULL);
    free(response);

    const char *extra_body =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}],\"verbosity\":\"low\"}";
    response = post_request(&config, &fake, extra_body);
    CHECK(has_status(response, 400));
    CHECK(strstr(response, "\"param\":\"verbosity\"") != NULL);
    free(response);
    CHECK(fake.calls == 0);
    return 0;
}

static int test_http_framing_rejections(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = NULL,
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *missing =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n";
    char *response = exchange(
        &config, fake_generate, &fake, missing, strlen(missing), NULL);
    CHECK(has_status(response, 400));
    free(response);

    const char *conflicting =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: 1\r\ncontent-length: 2\r\n\r\n{}";
    response = exchange(
        &config, fake_generate, &fake,
        conflicting, strlen(conflicting), NULL);
    CHECK(has_status(response, 400));
    free(response);

    const char *transfer =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
    response = exchange(
        &config, fake_generate, &fake, transfer, strlen(transfer), NULL);
    CHECK(has_status(response, 400));
    free(response);

    static const char nul_body[] =
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\","
        "\"content\":\"Hi\"}]}\0ignored";
    char nul_request[512];
    int nul_header_size = snprintf(
        nul_request, sizeof(nul_request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n", sizeof(nul_body) - 1);
    CHECK(nul_header_size > 0 && (size_t)nul_header_size < sizeof(nul_request));
    size_t nul_request_size = (size_t)nul_header_size + sizeof(nul_body) - 1;
    CHECK(nul_request_size <= sizeof(nul_request));
    memcpy(nul_request + nul_header_size, nul_body, sizeof(nul_body) - 1);
    response = exchange(
        &config, fake_generate, &fake, nul_request, nul_request_size, NULL);
    CHECK(has_status(response, 400));
    free(response);
    CHECK(fake.calls == 0);
    return 0;
}

static int test_size_limits(void) {
    OpenAIHttpConfig config = {
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = NULL,
        .model_name = "test-model",
    };
    FakeContext fake = {0};
    const char *prefix =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Fill: ";
    const char *suffix = "\r\n\r\n";
    size_t prefix_size = strlen(prefix);
    size_t suffix_size = strlen(suffix);
    char *at_limit = malloc(HEADER_LIMIT);
    CHECK(at_limit != NULL);
    memcpy(at_limit, prefix, prefix_size);
    memset(at_limit + prefix_size, 'a',
           HEADER_LIMIT - prefix_size - suffix_size);
    memcpy(at_limit + HEADER_LIMIT - suffix_size, suffix, suffix_size);
    char *response = exchange(
        &config, fake_generate, &fake, at_limit, HEADER_LIMIT, NULL);
    free(at_limit);
    CHECK(has_status(response, 200));
    free(response);

    size_t oversized_size = HEADER_LIMIT + 128;
    char *oversized = malloc(oversized_size);
    CHECK(oversized != NULL);
    memcpy(oversized, prefix, prefix_size);
    memset(oversized + prefix_size, 'a', oversized_size - prefix_size);
    response = exchange(
        &config, fake_generate, &fake, oversized, oversized_size, NULL);
    free(oversized);
    CHECK(has_status(response, 431));
    free(response);

    static const char body_prefix[] =
        "{\"model\":\"test-model\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Hi\"}]}";
    char body_header[256];
    int body_header_size = snprintf(
        body_header, sizeof(body_header),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n", (size_t)BODY_LIMIT);
    CHECK(body_header_size > 0 &&
          (size_t)body_header_size < sizeof(body_header));
    char *body = malloc(BODY_LIMIT);
    CHECK(body != NULL);
    memcpy(body, body_prefix, sizeof(body_prefix) - 1);
    memset(body + sizeof(body_prefix) - 1, ' ',
           BODY_LIMIT - (sizeof(body_prefix) - 1));
    size_t at_body_limit_size = (size_t)body_header_size + BODY_LIMIT;
    char *at_body_limit = malloc(at_body_limit_size);
    if (!at_body_limit) {
        free(body);
        CHECK(at_body_limit != NULL);
    }
    memcpy(at_body_limit, body_header, (size_t)body_header_size);
    memcpy(at_body_limit + body_header_size, body, BODY_LIMIT);
    free(body);
    response = exchange(
        &config, fake_generate, &fake,
        at_body_limit, at_body_limit_size, NULL);
    free(at_body_limit);
    CHECK(has_status(response, 200));
    free(response);
    CHECK(fake.calls == 1);

    char over_body_limit[256];
    int over_body_limit_size = snprintf(
        over_body_limit, sizeof(over_body_limit),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: %zu\r\n\r\n", (size_t)BODY_LIMIT + 1);
    CHECK(over_body_limit_size > 0 &&
          (size_t)over_body_limit_size < sizeof(over_body_limit));
    response = exchange(
        &config, fake_generate, &fake,
        over_body_limit, (size_t)over_body_limit_size, NULL);
    CHECK(has_status(response, 413));
    free(response);
    CHECK(fake.calls == 1);
    return 0;
}

int main(void) {
    CHECK(test_models_and_authentication() == 0);
    CHECK(test_host_header_requirements() == 0);
    CHECK(test_non_streaming_completion() == 0);
    CHECK(test_streaming_completion() == 0);
    CHECK(test_pie_responses_streaming() == 0);
    CHECK(test_token_byte_normalization() == 0);
    CHECK(test_finish_reasons() == 0);
    CHECK(test_stream_disconnect_stops_sink() == 0);
    CHECK(test_non_stream_disconnect_stops_sink() == 0);
    CHECK(test_generation_error_responses() == 0);
    CHECK(test_unique_completion_ids() == 0);
    CHECK(test_routes_and_methods() == 0);
    CHECK(test_http_framing_rejections() == 0);
    CHECK(test_size_limits() == 0);
    CHECK(test_io_read_deadlines() == 0);
    CHECK(test_io_write_deadline() == 0);
    CHECK(test_generation_is_not_io_timed() == 0);
    CHECK(test_serve_wakeup_interrupts_partial_requests() == 0);
    puts("OpenAI HTTP socket tests: ok");
    return 0;
}
