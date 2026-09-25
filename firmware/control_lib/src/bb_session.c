#include "bb_session.h"

#include <string.h>

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_i32(uint8_t *p, int32_t v)
{
    put_u32(p, (uint32_t)v);
}

static void put_f64(uint8_t *p, double v)
{
    memcpy(p, &v, 8);
}

static double get_f64(const uint8_t *p)
{
    double v;
    memcpy(&v, p, 8);
    return v;
}

static void send(bb_session *s, uint8_t type, const uint8_t *payload, uint16_t len)
{
    uint32_t n = bb_frame(s->frame, type, 0, s->seq++, payload, len);
    if (n > 0) {
        s->io.write(s->io.ctx, s->frame, n);
    }
}

static void send_error(bb_session *s, int32_t code, int32_t detail)
{
    uint8_t p[8];
    put_i32(&p[0], code);
    put_i32(&p[4], detail);
    send(s, BB_MSG_ERROR, p, 8);
}

static void send_status(bb_session *s)
{
    uint8_t *p = s->payload;
    put_u32(&p[0], BB_BUNDLE_VERSION);
    put_u32(&p[4], s->ready ? s->ctrl.bundle.n_frames : 0u);
    put_u32(&p[8], s->ticks);
    put_u32(&p[12], s->dec.bad_crc);
    put_u32(&p[16], s->dec.resyncs);
    put_i32(&p[20], s->ctrl.ik_failures);
    put_f64(&p[24], s->ticks ? (double)s->tick_us_total / (double)s->ticks : 0.0);
    put_f64(&p[32], (double)s->tick_us_max);
    put_u32(&p[40], s->bus_writes);
    put_u32(&p[44], s->bus_dropped);
    put_u32(&p[48], s->bus_period_us_max);
    put_u32(&p[52], 0u);
    send(s, BB_MSG_STATUS, p, 56);
}

void bb_session_init(bb_session *s, const bb_transport *io, uint8_t *bundle_buf,
                     uint32_t bundle_cap, bb_bus *bus)
{
    memset(s, 0, sizeof(*s));
    s->io = *io;
    s->bundle_buf = bundle_buf;
    s->bundle_cap = bundle_cap;
    s->bus = bus;
    bb_decoder_init(&s->dec);
}

static void on_bundle_chunk(bb_session *s, const uint8_t *p, uint16_t len)
{
    uint32_t total, off, n;
    if (len < 8) {
        send_error(s, BB_PROTO_ERR_LENGTH, len);
        return;
    }
    total = get_u32(&p[0]);
    off = get_u32(&p[4]);
    n = (uint32_t)len - 8u;
    if (total > s->bundle_cap) {
        send_error(s, BB_PROTO_ERR_BUNDLE, (int32_t)total);
        return;
    }
    if (off + n > total) {
        send_error(s, BB_PROTO_ERR_LENGTH, (int32_t)(off + n));
        return;
    }
    if (off == 0) {
        s->bundle_bytes = total;
        s->bundle_have = 0;
        s->ready = 0;
    }
    memcpy(&s->bundle_buf[off], &p[8], n);
    s->bundle_have = off + n;
    {
        uint8_t ack[8];
        int32_t status = 0;
        if (s->bundle_have == s->bundle_bytes) {
            double q_home[BB_NJ] = {0.0, -0.5, 0.9, 1.2, 0.0};
            status = bb_control_init(&s->ctrl, s->bundle_buf, s->bundle_bytes, q_home);
            s->ready = (status == BB_OK);
            s->finished = 0;
            s->ticks = 0;
            s->tick_us_total = 0;
            s->tick_us_max = 0;
        }
        put_i32(&ack[0], status);
        put_u32(&ack[4], s->bundle_have);
        send(s, BB_MSG_BUNDLE_ACK, ack, 8);
    }
}

/* START: the joint angles the arm is at, which seed the inverse kinematics. The controller is
 * put back at the start of the sequence with them; without a START it runs from a fixed seed. */
static void on_start(bb_session *s, const uint8_t *p, uint16_t len)
{
    double q_home[BB_NJ];
    uint8_t ack[8];
    int32_t status;
    int i;

    if (len < 8u * (uint32_t)BB_NJ) {
        send_error(s, BB_PROTO_ERR_LENGTH, len);
        return;
    }
    if (s->bundle_bytes == 0 || s->bundle_have != s->bundle_bytes) {
        send_error(s, BB_PROTO_ERR_STATE, BB_MSG_START);
        return;
    }
    for (i = 0; i < BB_NJ; ++i) {
        q_home[i] = get_f64(&p[8 * i]);
    }
    status = bb_control_init(&s->ctrl, s->bundle_buf, s->bundle_bytes, q_home);
    s->ready = (status == BB_OK);
    s->finished = 0;
    s->ticks = 0;
    s->tick_us_total = 0;
    s->tick_us_max = 0;
    put_i32(&ack[0], status);
    put_u32(&ack[4], s->bundle_have);
    send(s, BB_MSG_BUNDLE_ACK, ack, 8);
}

static void on_tick_begin(bb_session *s)
{
    uint8_t *p = s->payload;
    uint64_t t0 = s->io.now_us ? s->io.now_us(s->io.ctx) : 0;
    int n, i;
    uint32_t off;

    if (!s->ready) {
        send_error(s, BB_PROTO_ERR_STATE, 0);
        return;
    }
    n = bb_tick_begin(&s->ctrl, &s->out);
    s->begin_us = s->io.now_us ? s->io.now_us(s->io.ctx) - t0 : 0;

    put_i32(&p[0], n);
    put_i32(&p[4], s->out.k);
    put_i32(&p[8], s->out.step_index);
    put_i32(&p[12], s->out.step_id);
    off = 16;
    for (i = 0; i < n; ++i) {
        const bb_command *c = &s->out.cmd[i];
        int j;
        put_f64(&p[off + 0], c->pos[0]);
        put_f64(&p[off + 8], c->pos[1]);
        put_f64(&p[off + 16], c->pos[2]);
        put_f64(&p[off + 24], c->spin);
        put_f64(&p[off + 32], c->grip);
        put_f64(&p[off + 40], c->c_vert);
        put_f64(&p[off + 48], c->c_lat);
        for (j = 0; j < BB_NJOINT; ++j) {
            put_f64(&p[off + 56 + 8 * j], c->q[j]);
        }
        put_i32(&p[off + 104], c->ik_ok);
        put_i32(&p[off + 108], 0);
        off += 112;
    }
    /* When the bus is real the controller drives it here and reports what came back as
     * telemetry; when the plant is simulated the host writes these targets itself. */
    if (s->bus != 0 && n > 0) {
        double torque[BB_NJOINT];
        int j;
        for (j = 0; j < BB_NJOINT; ++j) {
            torque[j] = 1.0;
        }
        for (i = 0; i < n; ++i) {
            s->bus->write(s->bus->ctx, s->out.cmd[i].q, torque, BB_NJOINT);
        }
        s->bus->read(s->bus->ctx, s->bus_state.position, s->bus_state.speed,
                     s->bus_state.load, BB_NJOINT);
    }
    send(s, BB_MSG_TARGETS, p, (uint16_t)off);
}

static void on_scene_state(bb_session *s, const uint8_t *p, uint16_t len)
{
    const uint32_t need = 4u + 8u + (uint32_t)(BB_K * BB_W) * 4u + (uint32_t)BB_K * 4u
                          + (uint32_t)BB_K * 4u;
    bb_slots slots;
    uint32_t off;
    uint8_t *q = s->payload;
    int i;
    uint64_t t0;

    if (len < need) {
        send_error(s, BB_PROTO_ERR_LENGTH, len);
        return;
    }
    if (!s->ready) {
        send_error(s, BB_PROTO_ERR_STATE, 0);
        return;
    }
    off = 12;
    memcpy(slots.slots, &p[off], (size_t)(BB_K * BB_W) * 4u);
    off += (uint32_t)(BB_K * BB_W) * 4u;
    memcpy(slots.depth, &p[off], (size_t)BB_K * 4u);
    off += (uint32_t)BB_K * 4u;
    memcpy(slots.ids, &p[off], (size_t)BB_K * 4u);

    t0 = s->io.now_us ? s->io.now_us(s->io.ctx) : 0;
    s->finished = bb_tick_end(&s->ctrl, &slots, &s->bus_state, &s->out);
    if (s->io.now_us) {
        uint64_t dt = s->io.now_us(s->io.ctx) - t0 + s->begin_us;
        s->tick_us_total += dt;
        if (dt > s->tick_us_max) {
            s->tick_us_max = dt;
        }
    }
    s->begin_us = 0;
    s->ticks += 1;

    put_i32(&q[0], s->out.k);
    put_i32(&q[4], s->out.step_index);
    put_i32(&q[8], s->out.step_id);
    put_i32(&q[12], s->out.holds);
    put_i32(&q[16], s->out.refs_seen);
    put_i32(&q[20], s->out.done);
    put_i32(&q[24], s->out.failure);
    put_i32(&q[28], s->out.regrasps);
    put_i32(&q[32], s->out.matched);
    put_i32(&q[36], s->ctrl.ik_failures);
    for (i = 0; i < 3; ++i) {
        put_f64(&q[40 + 8 * i], s->out.correction[i]);
    }
    put_f64(&q[64], s->out.err_mm);
    put_f64(&q[72], s->out.tol_mm);
    put_f64(&q[80], s->out.absolute_mm);
    put_f64(&q[88], s->out.predict_mm);
    put_f64(&q[96], s->out.yaw_err);
    send(s, BB_MSG_TICK_LOG, q, 104);
}

static void on_bus_readings(bb_session *s, const uint8_t *p, uint16_t len)
{
    int i;
    if (len < 3u * (uint32_t)BB_NJOINT * 8u) {
        send_error(s, BB_PROTO_ERR_LENGTH, len);
        return;
    }
    for (i = 0; i < BB_NJOINT; ++i) {
        s->bus_state.position[i] = get_f64(&p[8 * i]);
        s->bus_state.speed[i] = get_f64(&p[8 * (BB_NJOINT + i)]);
        s->bus_state.load[i] = get_f64(&p[8 * (2 * BB_NJOINT + i)]);
    }
}

static void send_hello(bb_session *s)
{
    uint8_t hello[8];
    put_u32(&hello[0], BB_BUNDLE_VERSION);
    put_u32(&hello[4], (uint32_t)s->bundle_cap);
    send(s, BB_MSG_HELLO, hello, 8);
}

/* Act on the frame the decoder holds. Returns 1 when the session is over. */
static int dispatch(bb_session *s)
{
    switch (s->dec.type) {
    case BB_MSG_HELLO:
        /* A host that opened the line after the greeting went out asks for it again. */
        send_hello(s);
        break;
    case BB_MSG_TARGET_BUNDLE:
        on_bundle_chunk(s, s->dec.payload, s->dec.len);
        break;
    case BB_MSG_START:
        on_start(s, s->dec.payload, s->dec.len);
        break;
    case BB_MSG_TICK_BEGIN:
        on_tick_begin(s);
        break;
    case BB_MSG_BUS_READINGS:
        on_bus_readings(s, s->dec.payload, s->dec.len);
        break;
    case BB_MSG_SCENE_STATE:
        on_scene_state(s, s->dec.payload, s->dec.len);
        break;
    case BB_MSG_STATUS:
        send_status(s);
        break;
    case BB_MSG_RESET:
        send_status(s);
        return 1;
    default:
        send_error(s, BB_PROTO_ERR_STATE, s->dec.type);
        break;
    }
    return 0;
}

int bb_session_run(bb_session *s)
{
    uint8_t chunk[256];

    send_hello(s);

    for (;;) {
        int got = s->io.read(s->io.ctx, chunk, (uint32_t)sizeof(chunk));
        int i;
        if (got <= 0) {
            return got < 0 ? -1 : 0;
        }
        for (i = 0; i < got; ++i) {
            int ready = bb_decoder_feed(&s->dec, chunk[i]);
            while (ready) {
                if (dispatch(s)) {
                    return 0;
                }
                ready = bb_decoder_next(&s->dec);
            }
        }
    }
}
