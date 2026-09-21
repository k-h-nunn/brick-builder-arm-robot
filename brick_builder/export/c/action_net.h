/* action_net.h: the two entry points the firmware calls once per vision tick.
 *
 * `action_inverse(x, y)`: x is ACTION_INV_IN_DIM slot-state differences, mechanical state and
 * flags (brick_builder/action/models/features.py: encode_inverse); y is ACTION_INV_OUT_DIM
 * numbers, the correction increment in metres of tool motion in x and y and radians of spin.
 * The caller saturates and integrates it exactly as it does the analytic term.
 *
 * `action_predict(x, y)`: x is ACTION_PRED_IN_DIM slot rows, mechanical state, the action and one
 * flag (encode_predictor); y is ACTION_PRED_OUT_DIM numbers, the change of the same slot rows
 * over one vision tick.
 *
 * Dimensions come from the generated action_theta.h; include it for them.
 */
#ifndef ACTION_NET_H
#define ACTION_NET_H

void action_inverse(const float *x, float *y);
void action_predict(const float *x, float *y);

#endif /* ACTION_NET_H */
