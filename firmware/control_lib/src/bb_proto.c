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
    d->want = 0;
    d->bad_crc = 0;
    d->resyncs = 0;
    d->type = 0;
    d->flags = 0;
    d->seq = 0;
    d->len = 0;
    d->payload = 0;
}

/* Drop the first byte and rescan: a frame whose header was corrupted must not swallow the next
 * good one. */
static void resync(bb_decoder *d)
{
    uint32_t i;
    d->resyncs += 1;
    if (d->have <= 1) {
        d->have = 0;
        d->want = 0;
        return;
    }
    for (i = 1; i < d->have; ++i) {
        d->buf[i - 1] = d->buf[i];
    }
    d->have -= 1;
    d->want = 0;
    /* Re-align on the next sync pair inside what is left. */
    while (d->have > 0 && d->buf[0] != BB_SYNC0) {
        for (i = 1; i < d->have; ++i) {
            d->buf[i - 1] = d->buf[i];
        }
        d->have -= 1;
    }
    if (d->have >= 2 && d->buf[1] != BB_SYNC1) {
        resync(d);
    }
}

int bb_decoder_feed(bb_decoder *d, uint8_t byte)
{
    if (d->have == 0) {
        if (byte != BB_SYNC0) {
            return 0;
        }
        d->buf[d->have++] = byte;
        return 0;
    }
    if (d->have == 1) {
        if (byte != BB_SYNC1) {
            d->have = (byte == BB_SYNC0) ? 1 : 0;
            d->buf[0] = byte;
            return 0;
        }
        d->buf[d->have++] = byte;
        return 0;
    }
    if (d->have >= BB_FRAME_MAX) {
        resync(d);
        return 0;
    }
    d->buf[d->have++] = byte;
    if (d->have == BB_HDR_BYTES) {
        uint16_t len = rd_u16(&d->buf[4]);
        if (len > BB_MAX_PAYLOAD) {
            resync(d);
            return 0;
        }
        d->want = (uint32_t)BB_HDR_BYTES + len + BB_CRC_BYTES;
    }
    if (d->want != 0 && d->have == d->want) {
        uint16_t len = rd_u16(&d->buf[4]);
        uint32_t want_crc = rd_u32(&d->buf[BB_HDR_BYTES + len]);
        uint32_t got_crc = bb_crc32(&d->buf[2], (uint32_t)(BB_HDR_BYTES - 2) + len);
        if (want_crc != got_crc) {
            d->bad_crc += 1;
            resync(d);
            return 0;
        }
        d->type = d->buf[2];
        d->flags = d->buf[3];
        d->len = len;
        d->seq = rd_u16(&d->buf[6]);
        d->payload = &d->buf[BB_HDR_BYTES];
        d->have = 0;
        d->want = 0;
        return 1;
    }
    return 0;
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
