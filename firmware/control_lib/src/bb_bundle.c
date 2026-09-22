#include "bb_bundle.h"

#include <string.h>

/* CRC-32/ISO-HDLC (the zlib polynomial), computed nibble-wise so the table is 64 bytes rather
 * than a kilobyte -- the controller is not throughput-bound and the microcontroller's cache is
 * small. */
static const uint32_t bb_crc_nibble[16] = {
    0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu,
    0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
    0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
    0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu
};

uint32_t bb_crc32(const uint8_t *data, uint32_t n)
{
    uint32_t crc = 0xFFFFFFFFu;
    uint32_t i;
    for (i = 0; i < n; ++i) {
        crc ^= data[i];
        crc = (crc >> 4) ^ bb_crc_nibble[crc & 0x0Fu];
        crc = (crc >> 4) ^ bb_crc_nibble[crc & 0x0Fu];
    }
    return ~crc;
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t crc;
    uint32_t n_frames;
    uint32_t n_commands;
    uint32_t k_slots;
    uint32_t w_slots;
    uint32_t off_cfg;
    uint32_t off_arm;
    uint32_t off_frames;
    uint32_t off_commands;
    uint32_t off_states;
    uint32_t off_step_off;
    uint32_t off_steps;
    uint32_t reserved;
} bb_bundle_hdr;

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int bb_bundle_parse(const uint8_t *blob, uint32_t bytes, bb_bundle *out)
{
    bb_bundle_hdr h;
    uint32_t i;

    if (blob == 0 || bytes < sizeof(bb_bundle_hdr)) {
        return BB_ERR_TRUNCATED;
    }
    for (i = 0; i < sizeof(bb_bundle_hdr) / 4; ++i) {
        ((uint32_t *)&h)[i] = rd_u32(blob + 4 * i);
    }
    if (h.magic != BB_BUNDLE_MAGIC) {
        return BB_ERR_MAGIC;
    }
    if (h.version != BB_BUNDLE_VERSION) {
        return BB_ERR_VERSION;
    }
    if (h.total_bytes != bytes) {
        return BB_ERR_TRUNCATED;
    }
    if (h.k_slots != BB_K || h.w_slots != BB_W) {
        return BB_ERR_LAYOUT;
    }
    if (bb_crc32(blob + 16, bytes - 16) != h.crc) {
        return BB_ERR_CRC;
    }
    if (h.off_cfg + sizeof(bb_cfg) > bytes || h.off_arm + sizeof(bb_arm) > bytes) {
        return BB_ERR_TRUNCATED;
    }
    if (h.off_frames + h.n_frames * sizeof(bb_frame_rec) > bytes) {
        return BB_ERR_TRUNCATED;
    }
    if (h.off_commands + h.n_commands * sizeof(bb_command_rec) > bytes) {
        return BB_ERR_TRUNCATED;
    }
    if (h.off_states + h.n_frames * sizeof(bb_state_rec) > bytes) {
        return BB_ERR_TRUNCATED;
    }
    if (h.off_step_off + h.n_frames * 4u > bytes || h.off_steps > bytes) {
        return BB_ERR_TRUNCATED;
    }
    if ((h.off_cfg | h.off_arm | h.off_frames | h.off_commands | h.off_states
         | h.off_step_off | h.off_steps) & 7u) {
        return BB_ERR_LAYOUT;
    }

    out->base = blob;
    out->bytes = bytes;
    out->n_frames = h.n_frames;
    out->n_commands = h.n_commands;
    out->cfg = (const bb_cfg *)(const void *)(blob + h.off_cfg);
    out->arm = (const bb_arm *)(const void *)(blob + h.off_arm);
    out->frames = (const bb_frame_rec *)(const void *)(blob + h.off_frames);
    out->commands = (const bb_command_rec *)(const void *)(blob + h.off_commands);
    out->states = (const bb_state_rec *)(const void *)(blob + h.off_states);
    out->step_off = (const uint32_t *)(const void *)(blob + h.off_step_off);

    for (i = 0; i < h.n_frames; ++i) {
        const bb_frame_rec *f = &out->frames[i];
        if (f->n_commands < 0 || f->cmd_offset < 0) {
            return BB_ERR_LAYOUT;
        }
        if ((uint32_t)(f->cmd_offset + f->n_commands) > h.n_commands) {
            return BB_ERR_LAYOUT;
        }
        if (f->n_commands > BB_MAX_CMDS) {
            return BB_ERR_TOO_BIG;
        }
        if (out->step_off[i] + sizeof(bb_step_hdr) > bytes || (out->step_off[i] & 7u)) {
            return BB_ERR_LAYOUT;
        }
    }
    return BB_OK;
}

int bb_bundle_step(const bb_bundle *b, uint32_t k, bb_step *step)
{
    const uint8_t *p;
    const bb_step_hdr *h;
    uint32_t feat_bytes;

    if (k >= b->n_frames) {
        return BB_ERR_LAYOUT;
    }
    p = b->base + b->step_off[k];
    h = (const bb_step_hdr *)(const void *)p;
    step->hdr = h;
    step->feat = 0;
    step->J = 0;
    step->s_star = 0;
    step->J_brick = 0;
    step->b_star = 0;
    step->refs = 0;
    step->brick_ref = 0;
    if (!h->valid) {
        return BB_OK;
    }
    if (h->n_refs > BB_MAX_REFS || h->n_feat > BB_MAX_REFS
        || h->brick_dim > BB_MAX_BROW || h->brick_dof > BB_MAX_BDOF
        || h->j_cols < 1 || h->j_cols > 3) {
        return BB_ERR_TOO_BIG;
    }
    p += sizeof(bb_step_hdr);
    step->feat = (const int32_t *)(const void *)p;
    feat_bytes = (uint32_t)h->n_feat * 4u;
    feat_bytes = (feat_bytes + 7u) & ~7u;
    p += feat_bytes;
    step->J = (const double *)(const void *)p;
    p += (size_t)(2 * h->n_feat) * (size_t)h->j_cols * sizeof(double);
    step->s_star = (const double *)(const void *)p;
    p += (size_t)(2 * h->n_feat) * sizeof(double);
    if (h->brick_dim > 0) {
        step->J_brick = (const double *)(const void *)p;
        p += (size_t)h->brick_dim * (size_t)h->brick_dof * sizeof(double);
        step->b_star = (const double *)(const void *)p;
        p += (size_t)h->brick_dim * sizeof(double);
    }
    step->refs = (const bb_ref_rec *)(const void *)p;
    if (h->has_brick_ref) {
        step->brick_ref = &step->refs[h->n_refs];
        p += (size_t)(h->n_refs + 1) * sizeof(bb_ref_rec);
    } else {
        p += (size_t)h->n_refs * sizeof(bb_ref_rec);
    }
    if ((uint32_t)(p - b->base) > b->bytes) {
        return BB_ERR_TRUNCATED;
    }
    return BB_OK;
}
