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
    /* Controller time per vision tick, `bb_tick_begin` plus `bb_tick_end`, transport excluded. */
    uint64_t begin_us;
    uint64_t tick_us_total;
    uint64_t tick_us_max;

    /* The 50 Hz servo-bus writer's own counters, filled in by the application that paces the
     * bus (the ESP-IDF one); zero where nothing does. Reported in STATUS. */
    uint32_t bus_writes;
    uint32_t bus_dropped;
    uint32_t bus_period_us_max;

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
