/* policy.h: the exported actor's public interface.
 *
 * Plain C99, no dynamic allocation, no libm dependence beyond `tanhf` (and none at all when
 * built with -DPOLICY_TANH_TABLE). This is the same file that will be built into the
 * ESP32-P4 firmware; see docs/guide/07-export-and-firmware.md.
 */
#ifndef POLICY_H
#define POLICY_H

#include "policy_theta.h"

/* obs must point to POLICY_OBS_DIM floats; act must point to POLICY_ACT_DIM floats. Pure
 * function of its input: no state carried between calls (the actor has no memory). */
void policy_act(const float *obs, float *act);

#endif /* POLICY_H */
