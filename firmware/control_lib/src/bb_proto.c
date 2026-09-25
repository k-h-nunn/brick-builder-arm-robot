#include "bb_proto.h"
#include "bb_bundle.h"

#include <string.h>

static uint16_t rd_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void bb_decoder_init(bb_decoder *d)
{
    d->have = 0;
    d->consumed = 0;
    d->bad_crc = 0;
    d->resyncs = 0;
    d->type = 0;
    d->flags = 0;
    d->seq = 0;
    d->len = 0;
    d->payload = 0;
}

static void drop(bb_decoder *d, uint32_t n)
{
    if (n >= d->have) {
        d->have = 0;
        return;
    }
    memmove(d->buf, d->buf + n, d->have - n);
    d->have -= n;
}

/* The same scan `protocol.Decoder` runs: align on the next sync pair, wait for the header and
 * the whole frame, and on a bad length or a bad CRC drop the sync pair and scan again from the
 * byte after it -- so a damaged frame never swallows a good one that starts inside it. */
static int scan(bb_decoder *d)
{
    if (d->consumed) {
        drop(d, d->consumed);
        d->consumed = 0;
    }
    for (;;) {
        uint32_t at = 0, want;
        uint16_t len;
        while (at + 1 < d->have && !(d->buf[at] == BB_SYNC0 && d->buf[at + 1] == BB_SYNC1)) {
            ++at;
        }
        if (at + 1 >= d->have) {
            /* No pair: keep the last byte, it may be the first half of one. */
            if (d->have > 1) {
                drop(d, d->have - 1);
            }
            return 0;
        }
        if (at > 0) {
            drop(d, at);
            d->resyncs += 1;
        }
        if (d->have < BB_HDR_BYTES) {
            return 0;
        }
        len = rd_u16(&d->buf[4]);
        if (len > BB_MAX_PAYLOAD) {
            drop(d, 2);
            d->resyncs += 1;
            continue;
        }
        want = (uint32_t)BB_HDR_BYTES + len + BB_CRC_BYTES;
        if (d->have < want) {
            return 0;
        }
        if (rd_u32(&d->buf[BB_HDR_BYTES + len])
            != bb_crc32(&d->buf[2], (uint32_t)(BB_HDR_BYTES - 2) + len)) {
            d->bad_crc += 1;
            drop(d, 2);
            d->resyncs += 1;
            continue;
        }
        d->type = d->buf[2];
        d->flags = d->buf[3];
        d->len = len;
        d->seq = rd_u16(&d->buf[6]);
        d->payload = &d->buf[BB_HDR_BYTES];
        d->consumed = want;
        return 1;
    }
}

int bb_decoder_feed(bb_decoder *d, uint8_t byte)
{
    if (d->consumed) {
        drop(d, d->consumed);
        d->consumed = 0;
    }
    if (d->have >= BB_FRAME_MAX) {
        /* Cannot happen with a length check on every header; kept as a bound. */
        drop(d, 2);
        d->resyncs += 1;
    }
    d->buf[d->have++] = byte;
    return scan(d);
}

int bb_decoder_next(bb_decoder *d)
{
    return scan(d);
}

uint32_t bb_frame(uint8_t *out, uint8_t type, uint8_t flags, uint16_t seq,
                  const uint8_t *payload, uint16_t len)
{
    uint32_t crc;
    if (len > BB_MAX_PAYLOAD) {
        return 0;
    }
    out[0] = BB_SYNC0;
    out[1] = BB_SYNC1;
    out[2] = type;
    out[3] = flags;
    out[4] = (uint8_t)(len & 0xFFu);
    out[5] = (uint8_t)((len >> 8) & 0xFFu);
    out[6] = (uint8_t)(seq & 0xFFu);
    out[7] = (uint8_t)((seq >> 8) & 0xFFu);
    if (len > 0 && payload != 0) {
        memcpy(&out[BB_HDR_BYTES], payload, len);
    }
    crc = bb_crc32(&out[2], (uint32_t)(BB_HDR_BYTES - 2) + len);
    out[BB_HDR_BYTES + len + 0] = (uint8_t)(crc & 0xFFu);
    out[BB_HDR_BYTES + len + 1] = (uint8_t)((crc >> 8) & 0xFFu);
    out[BB_HDR_BYTES + len + 2] = (uint8_t)((crc >> 16) & 0xFFu);
    out[BB_HDR_BYTES + len + 3] = (uint8_t)((crc >> 24) & 0xFFu);
    return (uint32_t)BB_HDR_BYTES + len + BB_CRC_BYTES;
}
