/* bb_proto.h: the wire protocol between the host and the controller.
 *
 * Framed, CRC-checked, little-endian binary over a byte stream -- a pipe, a TCP socket or a UART,
 * all the same to this file. One frame:
 *
 *   B1 C5 | type u8 | flags u8 | len u16 | seq u16 | payload[len] | crc32 u32
 *
 * The CRC covers `type` through the end of the payload, so a frame whose sync bytes were found
 * inside a previous payload fails it and the decoder resynchronises on the next sync pair.
 *
 * Lock step: the host sends TICK_BEGIN and waits for TARGETS, runs the plant for exactly those
 * writes, then sends SCENE_STATE and BUS_READINGS and waits for TICK_LOG. The controller never
 * advances on its own clock.
 */
#ifndef BB_PROTO_H
#define BB_PROTO_H

#include <stdint.h>
#include "bb_config.h"

#define BB_SYNC0 0xB1u
#define BB_SYNC1 0xC5u
#define BB_HDR_BYTES 8
#define BB_CRC_BYTES 4
#define BB_MAX_PAYLOAD 2048u
#define BB_FRAME_MAX (BB_HDR_BYTES + BB_MAX_PAYLOAD + BB_CRC_BYTES)

/* Message types. */
#define BB_MSG_HELLO         0x01
#define BB_MSG_TARGET_BUNDLE 0x02
#define BB_MSG_BUNDLE_ACK    0x03
#define BB_MSG_SCENE_STATE   0x04
#define BB_MSG_BUS_READINGS  0x05
#define BB_MSG_TARGETS       0x06
#define BB_MSG_TICK_LOG      0x07
#define BB_MSG_STATUS        0x08
#define BB_MSG_ERROR         0x09
#define BB_MSG_TICK_BEGIN    0x0A
#define BB_MSG_RESET         0x0B
#define BB_MSG_START         0x0C   /* host -> controller: the joint angles the arm starts at */

/* Error codes carried by BB_MSG_ERROR. */
#define BB_PROTO_ERR_CRC      1
#define BB_PROTO_ERR_LENGTH   2
#define BB_PROTO_ERR_SEQ      3
#define BB_PROTO_ERR_BUNDLE   4
#define BB_PROTO_ERR_STATE    5

typedef struct {
    uint8_t buf[BB_FRAME_MAX];
    uint32_t have;
    uint32_t consumed;    /* bytes of the frame last returned, dropped on the next call */
    uint32_t bad_crc;
    uint32_t resyncs;
    /* The decoded frame, valid only on the call that returned 1. */
    uint8_t type;
    uint8_t flags;
    uint16_t seq;
    uint16_t len;
    const uint8_t *payload;
} bb_decoder;

void bb_decoder_init(bb_decoder *d);

/* Feed one byte. Returns 1 when a whole, CRC-valid frame is ready in `d`. */
int bb_decoder_feed(bb_decoder *d, uint8_t byte);

/* Return 1 when a whole frame is already buffered without another byte. After a damaged frame
 * the bytes behind it are rescanned, and they can hold more than one frame; a caller loops on
 * this after every frame `bb_decoder_feed` returned. */
int bb_decoder_next(bb_decoder *d);

/* Write a frame into `out` (at least BB_FRAME_MAX bytes). Returns the number of bytes written,
 * or 0 when the payload is too long. */
uint32_t bb_frame(uint8_t *out, uint8_t type, uint8_t flags, uint16_t seq,
                  const uint8_t *payload, uint16_t len);

#endif /* BB_PROTO_H */
