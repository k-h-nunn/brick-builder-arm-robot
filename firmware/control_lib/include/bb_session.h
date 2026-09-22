/* bb_session.h: one lock-step session over a byte stream.
 *
 * The native process and the ESP-IDF application are both a transport plus a call to
 * `bb_session_run`; nothing about the control loop is written twice.
 */
#ifndef BB_SESSION_H
#define BB_SESSION_H

#include <stdint.h>
#include "bb_bus.h"
#include "bb_control.h"
#include "bb_proto.h"

typedef struct {
    void *ctx;
    /* Block for at least one byte; return how many were read, 0 on end of stream, <0 on error. */
    int (*read)(void *ctx, uint8_t *buf, uint32_t n);
    int (*write)(void *ctx, const uint8_t *buf, uint32_t n);
    /* Microseconds since some fixed point, for the tick timing in STATUS. May be null. */
    uint64_t (*now_us)(void *ctx);
} bb_transport;

typedef struct {
    bb_transport io;
    bb_decoder dec;
    bb_controller ctrl;
    bb_bus *bus;

    uint8_t *bundle_buf;
    uint32_t bundle_cap;
    uint32_t bundle_bytes;
    uint32_t bundle_have;

    int ready;
    int finished;
    uint16_t seq;
    uint32_t ticks;
    uint64_t tick_us_total;
    uint64_t tick_us_max;

    bb_bus_readings bus_state;
    bb_tick_out out;
    uint8_t frame[BB_FRAME_MAX];
    uint8_t payload[BB_MAX_PAYLOAD];
} bb_session;

/* `bundle_buf` must be 8-byte aligned and stay alive for the session. */
void bb_session_init(bb_session *s, const bb_transport *io, uint8_t *bundle_buf,
                     uint32_t bundle_cap, bb_bus *bus);

/* Drive the session until the stream ends or a RESET arrives. Returns 0 on a clean end. */
int bb_session_run(bb_session *s);

#endif /* BB_SESSION_H */
