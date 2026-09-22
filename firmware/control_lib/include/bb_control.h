/* bb_control.h: the controller as it runs at run time.
 *
 * One target-sequence tracking loop: the recorded action a*_k is played forward and one step of
 * feedback on (target scene state - observed scene state) is added on top, through Jacobians the
 * digital twin measured on the host and shipped in the target bundle. The progress index
 * advances only while the world is inside a tolerance tube around S*_k; holding for longer than
 * a bound is a failure, and so is finishing outside tolerance.
 *
 * Two calls per vision tick:
 *
 *   bb_tick_begin(c, out)                       -- the 50 Hz writes this tick issues
 *   bb_tick_end(c, state, readings, out)        -- the feedback, the index, the end conditions
 *
 * Between them the caller runs the plant (or the servo bus) for the commands it was handed and
 * reads the wrist camera. The controller never free-runs: nothing advances until `bb_tick_end`
 * is given a scene state and a set of bus readings.
 */
#ifndef BB_CONTROL_H
#define BB_CONTROL_H

#include <stdint.h>
#include "bb_bundle.h"
#include "bb_joint_correction.h"
#include "bb_scene.h"
#include "bb_servo.h"

/* What the servo bus reports back, in radians, radians per second and fractions of the servo's
 * own limit. */
typedef struct {
    double position[BB_NJOINT];
    double speed[BB_NJOINT];
    double load[BB_NJOINT];
} bb_bus_readings;

/* One 50 Hz write. `q` is what goes on the bus; `pos`/`spin`/`grip` are the tool target it came
 * from, kept so a host driving a simulated plant can command the same pose. */
typedef struct {
    double pos[3];
    double spin;
    double grip;
    double c_vert;
    double c_lat;
    double q[BB_NJOINT];
    int32_t ik_ok;
} bb_command;

/* Failure classes. */
#define BB_FAIL_NONE        0
#define BB_FAIL_STALL_BLIND 1   /* nothing to line up on for most of a long hold */
#define BB_FAIL_STALL_ERROR 2   /* an error the loop can see and cannot reduce */
#define BB_FAIL_GRASP_SHORT 3   /* the grasp stopped short twice */

typedef struct {
    int32_t n_commands;
    bb_command cmd[BB_MAX_CMDS];
    int32_t k;
    int32_t step_index;
    int32_t step_id;
    int32_t holds;
    int32_t refs_seen;
    int32_t vision;
    int32_t done;
    int32_t failure;
    int32_t regrasps;
    int32_t matched;
    double correction[3];
    double err_mm;
    double tol_mm;
    double absolute_mm;
    double predict_mm;
    double yaw_err;
} bb_tick_out;

typedef struct {
    bb_bundle bundle;
    const bb_cfg *cfg;
    bb_servo servo;
    bb_joint_correction jc;
    bb_step step;
    int has_step;
    int servo_created;
    int trusted;
    double last_position[BB_NJOINT];

    double correction[3];
    double frozen_correction[3];
    int32_t k;
    int32_t holds;
    int32_t blind_holds;
    int32_t max_hold_seen;
    int32_t matched;
    int32_t lag;
    int32_t settled;
    int32_t step_index;
    int32_t jacobian_at;
    int32_t regrasps;
    int32_t unmatched;
    int32_t blind_frames;
    int32_t failure;
    int32_t done;
    int32_t dwelling;
    int32_t dwell_ticks;
    int32_t frozen;
    int32_t pick_frozen;
    int32_t descending;
    int32_t pressing;
    int32_t in_hand_reset;
    int32_t grasp_settled;
    int32_t gating_grasp;
    int32_t grasp_checked;
    int32_t regrasping;
    int32_t ticks;
    double gain;
    double yaw_err;
    double absolute_mm;
    double pred_mm;
    int has_predicted;
    double predicted[3];

    double q_goal[BB_NJ];
    double grip_last;
    int32_t ik_failures;
} bb_controller;

/* Index `blob` (which must outlive the controller and be 8-byte aligned) and put the controller
 * at the start of the sequence. `q_home` seeds the inverse kinematics. */
int bb_control_init(bb_controller *c, const uint8_t *blob, uint32_t bytes,
                    const double q_home[BB_NJ]);

/* The writes this vision tick issues. Returns the number of commands. */
int bb_tick_begin(bb_controller *c, bb_tick_out *out);

/* The feedback, the progress index and the end conditions. Returns 1 when the run is over. */
int bb_tick_end(bb_controller *c, const bb_slots *state, const bb_bus_readings *bus,
                bb_tick_out *out);

/* Struct sizes for this build, indexed as bb_control.c lists them. */
int32_t bb_struct_size(int which);

/* Forward kinematics of the reported joint angles -- the tool height the grasp check reads. */
void bb_control_tcp(const bb_controller *c, const double q[BB_NJ], double pos[3]);

#endif /* BB_CONTROL_H */
