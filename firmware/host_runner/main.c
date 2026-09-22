/* The native process stand-in for the microcontroller: `control_lib` plus a byte stream.
 *
 *   control_runner                 speak the protocol over stdin/stdout
 *   control_runner --tcp 5555      listen on a TCP port and serve one connection
 *
 * Nothing in here is control: it is a transport and a `bb_session_run`, which is the same call
 * the ESP-IDF application makes. Building this and running the host against it is the second
 * rung of the ladder -- the whole wire protocol under test with no board attached.
 */
/* POSIX clock_gettime and the socket calls; the runner is host-only. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include "bb_session.h"

#define BUNDLE_CAP (4u * 1024u * 1024u)

typedef struct {
    int fd_in;
    int fd_out;
} fd_transport;

static int io_read(void *ctx, uint8_t *buf, uint32_t n)
{
    fd_transport *t = (fd_transport *)ctx;
    for (;;) {
        ssize_t got = read(t->fd_in, buf, n);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        return (int)got;
    }
}

static int io_write(void *ctx, const uint8_t *buf, uint32_t n)
{
    fd_transport *t = (fd_transport *)ctx;
    uint32_t sent = 0;
    while (sent < n) {
        ssize_t put = write(t->fd_out, buf + sent, n - sent);
        if (put < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        sent += (uint32_t)put;
    }
    return (int)sent;
}

static uint64_t io_now_us(void *ctx)
{
    struct timespec ts;
    (void)ctx;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)(ts.tv_nsec / 1000);
}

static int serve_tcp(int port, fd_transport *t)
{
    struct sockaddr_in addr;
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    int fd;

    if (listener < 0) {
        return -1;
    }
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        return -1;
    }
    if (listen(listener, 1) != 0) {
        perror("listen");
        return -1;
    }
    fprintf(stderr, "control_runner: listening on 127.0.0.1:%d\n", port);
    fd = accept(listener, 0, 0);
    close(listener);
    if (fd < 0) {
        return -1;
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    t->fd_in = fd;
    t->fd_out = fd;
    return 0;
}

int main(int argc, char **argv)
{
    static bb_session session;
    fd_transport t;
    bb_transport io;
    uint8_t *bundle;
    int port = 0;
    int i, rc;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tcp") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stdio") == 0) {
            port = 0;
        } else {
            fprintf(stderr, "usage: %s [--stdio | --tcp PORT]\n", argv[0]);
            return 2;
        }
    }

    t.fd_in = 0;
    t.fd_out = 1;
    if (port > 0 && serve_tcp(port, &t) != 0) {
        return 1;
    }

    bundle = (uint8_t *)malloc(BUNDLE_CAP);
    if (bundle == 0) {
        fprintf(stderr, "control_runner: out of memory\n");
        return 1;
    }

    io.ctx = &t;
    io.read = io_read;
    io.write = io_write;
    io.now_us = io_now_us;
    bb_session_init(&session, &io, bundle, BUNDLE_CAP, 0);
    rc = bb_session_run(&session);
    free(bundle);
    return rc == 0 ? 0 : 1;
}
