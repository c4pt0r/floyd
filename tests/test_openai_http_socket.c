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
    int fd;
    const OpenAIHttpConfig *config;
    OpenAIGenerateHandler handler;
    void *user_data;
    int result;
} ServerThread;

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

static void *serve_connection(void *opaque) {
    ServerThread *thread = opaque;
    thread->result = openai_http_handle_connection(
        thread->fd, thread->config, thread->handler, thread->user_data);
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
    shutdown(pair[0], SHUT_WR);
    char *response = socket_read_all(pair[0]);
    close(pair[0]);
    pthread_join(worker, NULL);
    if (handle_result) *handle_result = thread.result;
    return response;
}

static char *post_request(const OpenAIHttpConfig *config, FakeContext *fake,
                          const char *body) {
    size_t capacity = strlen(body) + 256;
    char *request = malloc(capacity);
    if (!request) return NULL;
    int size = snprintf(
        request, capacity,
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "authorization: Bearer secret\r\n"
        "content-type: application/json\r\n"
        "content-length: %zu\r\n\r\n%s",
        strlen(body), body);
    if (size < 0 || (size_t)size >= capacity) {
        free(request);
        return NULL;
    }
    char *response = exchange(
        config, fake_generate, fake, request, (size_t)size, NULL);
    free(request);
    return response;
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

static void *receive_termination_signals(void *unused) {
    (void)unused;
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_UNBLOCK, &signals, NULL);
    for (;;) pause();
    return NULL;
}

static int run_serve_shutdown_case(
    const char *partial, size_t partial_size, int termination_signal) {
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
        pthread_t signal_thread;
        if (pthread_create(
                &signal_thread, NULL,
                receive_termination_signals, NULL) != 0) _exit(3);
        pthread_detach(signal_thread);
        OpenAIHttpConfig config = {
            .host = "127.0.0.1",
            .port = port,
            .api_key = NULL,
            .model_name = "test-model",
        };
        FakeContext fake = {0};
        char error[256] = {0};
        int ok = openai_http_serve(
            &config, fake_generate, &fake, error, sizeof(error));
        _exit(ok ? 0 : 2);
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
    kill(child, termination_signal);
    int status = 0;
    ok = wait_for_child(child, &status) &&
         WIFEXITED(status) && WEXITSTATUS(status) == 0;
    close(blocked);
    return ok;
}

static int test_serve_wakeup_interrupts_partial_requests(void) {
    const char *partial_header =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Len";
    CHECK(run_serve_shutdown_case(
        partial_header, strlen(partial_header), SIGINT));

    const char *partial_body =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: 100\r\n\r\n{";
    CHECK(run_serve_shutdown_case(
        partial_body, strlen(partial_body), SIGTERM));
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
    const char *prefix = "GET /v1/models HTTP/1.1\r\nX-Fill: ";
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
    CHECK(test_non_streaming_completion() == 0);
    CHECK(test_streaming_completion() == 0);
    CHECK(test_stream_disconnect_stops_sink() == 0);
    CHECK(test_routes_and_methods() == 0);
    CHECK(test_http_framing_rejections() == 0);
    CHECK(test_size_limits() == 0);
    CHECK(test_serve_wakeup_interrupts_partial_requests() == 0);
    puts("OpenAI HTTP socket tests: ok");
    return 0;
}
