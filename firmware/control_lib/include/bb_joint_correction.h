/* bb_joint_correction.h: one clamped integrator per joint, between the commanded joint angles
 * and the servo bus. The C99 transcription of `robot/joint_correction.py`: fixed-size arrays,
 * one explicit loop, no allocation, and no branch that depends on anything but the numbers
 * passed in.
 *
 * `freeze` holds the correction exactly where it is, so an error the arm is meant to have --
 * a tool pushed into a brick, jaws squeezing past a face -- does not wind the trim up and then
 * release it into a lurch when the contact ends.
 */
#ifndef BB_JOINT_CORRECTION_H
#define BB_JOINT_CORRECTION_H

#include "bb_config.h"

typedef struct {
    int n;
    double clamp;
    double deadzone;
    double gain;
    double correction[BB_NJ];
    double target[BB_NJ];
} bb_joint_correction;

/* `ki` is the integral gain [1/s], `clamp_ticks` and `deadzone_ticks` are encoder counts. */
void bb_jc_init(bb_joint_correction *c, int n_joints, double ki, double dt,
                double clamp_ticks, double deadzone_ticks, int ticks_per_rev);
void bb_jc_reset(bb_joint_correction *c);

/* One 50 Hz tick: integrate `desired - measured` and write `desired + correction`. */
void bb_jc_update(bb_joint_correction *c, const double *desired, const double *measured,
                  int freeze);

#endif /* BB_JOINT_CORRECTION_H */
