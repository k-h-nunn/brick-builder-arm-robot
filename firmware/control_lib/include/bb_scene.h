/* bb_scene.h: the scene state as the controller reads it, and the slot operations on it.
 *
 * One scene state is K slot rows of W float32 columns, a per-slot depth order and a per-slot
 * identity. Nothing here reads anything else: no image, no decode, no body pose. The layout is
 * `PLACEHOLDER_SLOT_SCHEMA`, declared once in bb_config.h.
 *
 * The source of a scene state is whatever fills `bb_scene_state`: the wire today, an on-device
 * model later. Nothing below this line changes when it moves.
 */
#ifndef BB_SCENE_H
#define BB_SCENE_H

#include <stdint.h>
#include "bb_config.h"

/* The payload itself: what a producer publishes and what the bundle's reference frames hold. */
typedef struct {
    float slots[BB_K * BB_W];
    float depth[BB_K];
    int32_t ids[BB_K];
} bb_slots;

/* One observed tick: the payload plus where it came from. */
typedef struct {
    bb_slots s;
    uint32_t frame_id;
    double host_t;
} bb_scene_state;

double bb_slot_mass(const bb_slots *s, int k);
void   bb_slot_centre(const bb_slots *s, int k, double c[2]);
void   bb_slot_moments(const bb_slots *s, int k, double m[3]);
void   bb_slot_colour(const bb_slots *s, int k, double c[3]);
double bb_slot_border(const bb_slots *s, int k);
void   bb_slot_orient4(const bb_slots *s, int k, double z[2]);
int    bb_slot_empty(const bb_slots *s, int k);

/* Orientation and (major, minor) second-moment axis lengths of slot `k`. */
void bb_slot_axes(const bb_slots *s, int k, double *angle, double axes[2]);

/* `perception.slot_ops.select_slot`, restricted to the constraints the controller uses.
 * `exclude` is a bitmask over slots. Returns the slot index or -1. */
int bb_select_slot(const bb_slots *s, const double colour[3], double colour_tol,
                   const double size[2], int has_size, double size_tol,
                   double min_mass, int has_min_mass, uint32_t exclude);

/* `action.servo.find_held`: the slot nearest where the twin says the held brick sits. */
int bb_find_held(const bb_slots *s, const double expected[2], uint32_t exclude,
                 double max_distance);

/* `action.twin.blob_orientation`: the blob's orientation as a complex number, with the symmetry
 * it belongs to (4 for a square footprint, 2 for an elongated one). */
void bb_blob_orientation(const bb_slots *s, int k, double z[2], int *fold);

/* `action.twin.orientation_error`: arg(z_obs * conj(z_target)) / fold, and how much to believe
 * it (the weaker of the two magnitudes). */
void bb_orientation_error(const double z_obs[2], const double z_target[2], int fold,
                          double *err, double *weight);

/* The held brick's whole published row, as `action.twin.brick_features` builds it. */
void bb_brick_features(const bb_slots *s, int k, double out[BB_MAX_BROW]);

#endif /* BB_SCENE_H */
