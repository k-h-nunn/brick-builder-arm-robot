/* bb_servo.h: the one-step inverse -- a feature error turned into a tool move.
 *
 * `-lambda * J^+ (s - s*)`, saturated, with the held brick's own blob subtracted so what is
 * lined up is the brick rather than the tool. `J` and `s*` are measured in the digital twin on
 * the host and arrive in the target bundle; nothing here knows the camera's intrinsics or where
 * anything is in the world.
 */
#ifndef BB_SERVO_H
#define BB_SERVO_H

#include "bb_bundle.h"
#include "bb_scene.h"

#define BB_SLOTS_PER_STEP (BB_MAX_REFS + 1)   /* the references plus the brick in the hand */

typedef struct {
    int n;
    int slot[BB_SLOTS_PER_STEP];
    double centre[BB_SLOTS_PER_STEP][2];
    double weight[BB_SLOTS_PER_STEP];
} bb_features;

typedef struct {
    double lam;
    int settle_ticks;
    double max_spin;
    double max_in_hand;
    int require_brick;
    int hold_in_hand;
    /* Carried across ticks: the in-hand reading is an average of many frames of a quantity that
     * does not change, and the hit counter is what the done test reads. */
    int hits;
    int has_d_b;
    double d_b[2];
    int n_b;
} bb_servo;

typedef struct {
    double delta[3];
    double err_m;
    int done;
    int seen;
    int blind;
    double d_r[3];
    double d_b[2];
    double yaw_weight;
    bb_features found;
    bb_features want;
    int have_want;
} bb_servo_out;

void bb_servo_init(bb_servo *s, const bb_cfg *cfg);
void bb_servo_reset_in_hand(bb_servo *s);

/* One vision tick. `reference` is the target frame's own scene state, or 0 for the step's own
 * `s_star`. `dry_run` leaves every carried average untouched. */
void bb_servo_update(bb_servo *s, const bb_cfg *cfg, const bb_step *step,
                     const bb_slots *actual, const bb_slots *reference,
                     int dry_run, int relative, bb_servo_out *out);

#endif /* BB_SERVO_H */
