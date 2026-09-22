#include "bb_joint_correction.h"

void bb_jc_init(bb_joint_correction *c, int n_joints, double ki, double dt,
                double clamp_ticks, double deadzone_ticks, int ticks_per_rev)
{
    double tick = 2.0 * BB_PI / (double)ticks_per_rev;
    c->n = n_joints > BB_NJ ? BB_NJ : n_joints;
    c->clamp = clamp_ticks * tick;
    c->deadzone = deadzone_ticks * tick;
    c->gain = ki * dt;
    bb_jc_reset(c);
}

void bb_jc_reset(bb_joint_correction *c)
{
    int i;
    for (i = 0; i < BB_NJ; ++i) {
        c->correction[i] = 0.0;
        c->target[i] = 0.0;
    }
}

void bb_jc_update(bb_joint_correction *c, const double *desired, const double *measured,
                  int freeze)
{
    const double clamp = c->clamp;
    const double deadzone = c->deadzone;
    const double gain = c->gain;
    int i;

    for (i = 0; i < c->n; ++i) {
        double want = desired[i];
        double state = c->correction[i];
        if (!freeze) {
            double error = want - measured[i];
            if (error > deadzone) {
                state += gain * (error - deadzone);
            } else if (error < -deadzone) {
                state += gain * (error + deadzone);
            }
            if (state > clamp) {
                state = clamp;
            } else if (state < -clamp) {
                state = -clamp;
            }
            c->correction[i] = state;
        }
        c->target[i] = want + state;
    }
}
