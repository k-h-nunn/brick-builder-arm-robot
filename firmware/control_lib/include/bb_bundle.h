/* bb_bundle.h: the target bundle -- everything the digital twin computes on the host, as one
 * little-endian binary blob the controller reads in place.
 *
 * The bundle is parsed, not copied: `bb_bundle_parse` validates the header, the CRC and every
 * offset, then fills a `bb_bundle` with pointers into the caller's buffer. The buffer must stay
 * alive for the run and must be 8-byte aligned (any malloc, any static array of doubles, and the
 * mapped flash partition on the microcontroller all are).
 *
 * Layout, all sections 8-byte aligned and in this order:
 *
 *   header (64 B)  cfg  arm  frames  commands  states  step offsets  step records
 *
 * The wire format's field order is fixed by `brick_builder/hil/bundle.py`, which writes it; the
 * version number in the header changes whenever the two must be rebuilt together.
 */
#ifndef BB_BUNDLE_H
#define BB_BUNDLE_H

#include <stdint.h>
#include "bb_config.h"
#include "bb_kin.h"
#include "bb_scene.h"

#define BB_BUNDLE_MAGIC   0x42425442u   /* 'B','B','T','B' */
#define BB_BUNDLE_VERSION 1u

/* Tracking gains, tolerances and the indices the recording fixes. Doubles first, then int32s,
 * so the struct has the same layout under every C compiler this is built with. */
typedef struct {
    double lam;
    double tol_mm;
    double advance_mm;
    double tube_scale;
    double dwell_travel_mm;
    double dwell_speed;
    double min_gain;
    double max_correction;
    double stall_load;
    double grasp_yaw_deg;
    double grasp_miss;
    double pick_freeze_below;
    double grasp_z;
    double servo_max_xy;
    double servo_max_spin;
    double servo_near;
    double servo_smooth_in_hand;
    double servo_dead_band_mm;
    double servo_dead_band_single_mm;
    double servo_min_yaw_weight;
    double servo_truncate_at;
    double servo_max_in_hand;
    double delta_xy_cap;
    double delta_spin_cap;
    double jc_ki;
    double jc_clamp;
    double jc_deadzone;
    double jc_dt;
    double grip_lo;
    double grip_hi;
    int32_t jacobian_every;
    int32_t match_window;
    int32_t max_hold;
    int32_t settle_ticks;
    int32_t grasp_ticks;
    int32_t grasp_wait;
    int32_t servo_freeze_in_hand;
    int32_t servo_min_refs;
    int32_t descent_k;          /* -1 when the recording has no descent */
    int32_t grasp_k;
    int32_t freeze_index;
    int32_t pick;
    int32_t closed_loop;
    int32_t use_yaw;
    int32_t dwell_only;
    int32_t vision_every;
    int32_t jc_joints;          /* 0 switches the host-side joint trim off */
    int32_t pad0;
} bb_cfg;

typedef struct {
    double pos[3];
    double spin;
    double grip;
    double c_vert;
    double c_lat;
    double pad;
} bb_command_rec;

typedef struct {
    int32_t k;
    int32_t step_index;
    int32_t step_id;            /* index into STEP_TITLES */
    int32_t n_commands;
    int32_t cmd_offset;
    int32_t held;               /* a brick is in the hand during this step */
    int32_t is_freeze_step;     /* this frame's step is `freeze_after` */
    int32_t held_ref;           /* reference index of the held body, -1 when it has none */
    double travel;
    double pad;
} bb_frame_rec;

typedef bb_slots bb_state_rec;

typedef struct {
    double colour[3];
    double size[2];
    double colour_tol;
    double size_tol;
    double min_mass;
    double max_border;
    int32_t has_size;
    int32_t has_min_mass;
    int32_t require_inside;
    int32_t pad;
} bb_ref_rec;

typedef struct {
    int32_t valid;              /* 0: nothing to line up on at this frame */
    int32_t n_refs;
    int32_t n_feat;
    int32_t brick_dim;          /* rows of J_brick and length of b_star; 0 when there is none */
    int32_t brick_dof;          /* columns of J_brick */
    int32_t angle_mode;         /* 0 = pair, 1 = blob */
    int32_t angle_slot;         /* reference index carrying the angle in blob mode */
    int32_t angle_period;
    int32_t trusted;
    int32_t j_cols;             /* columns of J: the tool axes it was measured on */
    int32_t has_spin;
    int32_t has_brick_ref;
    double tol;
    double gain;
    double spin_gain;
    double spin_star_re;
    double spin_star_im;
    double pad;
    /* Followed by: int32 feat[n_feat] (padded to 8 bytes), double J[2*n_feat][3],
     * double s_star[2*n_feat], double J_brick[brick_dim][brick_dof], double b_star[brick_dim],
     * bb_ref_rec refs[n_refs + has_brick_ref]. */
} bb_step_hdr;

typedef struct {
    const bb_step_hdr *hdr;
    const int32_t *feat;
    const double *J;
    const double *s_star;
    const double *J_brick;
    const double *b_star;
    const bb_ref_rec *refs;
    const bb_ref_rec *brick_ref;
} bb_step;

typedef struct {
    const bb_cfg *cfg;
    const bb_arm *arm;
    const bb_frame_rec *frames;
    const bb_command_rec *commands;
    const bb_state_rec *states;
    const uint32_t *step_off;
    const uint8_t *base;
    uint32_t n_frames;
    uint32_t n_commands;
    uint32_t bytes;
} bb_bundle;

/* Validate and index `blob`. Returns BB_OK or one of the BB_ERR_* codes. */
int bb_bundle_parse(const uint8_t *blob, uint32_t bytes, bb_bundle *out);

/* Resolve frame `k`'s step record. Returns BB_OK; `step->hdr->valid` says whether it has one. */
int bb_bundle_step(const bb_bundle *b, uint32_t k, bb_step *step);

uint32_t bb_crc32(const uint8_t *data, uint32_t n);

#endif /* BB_BUNDLE_H */
