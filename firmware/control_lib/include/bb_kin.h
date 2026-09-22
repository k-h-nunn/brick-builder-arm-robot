/* bb_kin.h: forward kinematics, the site Jacobian and the damped-least-squares inverse.
 *
 * The chain is the five non-gripper joints of the arm, each a hinge at its own body origin. The
 * geometry -- parent-relative body frames, joint axes, the tool site's offset and the joint
 * ranges -- is data the host measures once off the model and puts in the target bundle, so this
 * file carries no hard-coded link lengths.
 *
 * Orientation convention and residual rows follow `robot/kinematics.py`: the target frame's yaw
 * is pinned to the arm's own current heading, so the residual's angular rows are the world x and
 * y components of the tool's orientation error and their exact derivative is minus the first two
 * angular rows of the site Jacobian.
 */
#ifndef BB_KIN_H
#define BB_KIN_H

#include "bb_config.h"

typedef struct {
    double body_pos[BB_NLINK][3];
    double body_quat[BB_NLINK][4];
    double axis[BB_NLINK][3];
    double site_pos[3];
    double site_quat[4];
    double jnt_lo[BB_NJ];
    double jnt_hi[BB_NJ];
} bb_arm;

/* Tool site position and rotation matrix (row-major 3x3) at joint angles `q`. */
void bb_fk(const bb_arm *arm, const double q[BB_NJ], double pos[3], double rot[9]);

/* Site Jacobian: `jacp` and `jacr` are 3 x BB_NJ row-major, world frame. Also returns the site
 * pose, since it is computed on the way. */
void bb_jac(const bb_arm *arm, const double q[BB_NJ], double pos[3], double rot[9],
            double jacp[3 * BB_NJ], double jacr[3 * BB_NJ]);

typedef struct {
    double q[BB_NJ];
    int converged;
    double pos_error;
    double ang_error;
    double yaw_error;
    int iterations;
} bb_ik_result;

/* Per-tick differential solve, warm-started from `q0`. `target_yaw` is used when `use_yaw`. */
void bb_ik_diff(const bb_arm *arm, const double target_pos[3], double target_pitch,
                double target_roll, const double q0[BB_NJ], int use_yaw, double target_yaw,
                int iters, double damping, double pos_tol, double ang_tol, double yaw_tol,
                double yaw_weight, bb_ik_result *out);

/* Global Levenberg-Marquardt solve for position plus tool pitch and roll. */
void bb_ik(const bb_arm *arm, const double target_pos[3], double target_pitch, double target_roll,
           const double q0[BB_NJ], double damping, int iters, double pos_tol, double ang_tol,
           bb_ik_result *out);

/* `envs/base_env._pose_ik`: the differential solve, falling back to the global one with the
 * fixed-point yaw chase on wrist_roll. Returns 1 when the answer is usable. */
int bb_pose_ik(const bb_arm *arm, const double pos[3], double tilt, double spin,
               const double q0[BB_NJ], double q_out[BB_NJ]);

/* `scene.compose.tool_down_ik`: the second fallback `ArmDriver` runs when `bb_pose_ik` gives up
 * on a tool-down target. Returns 1 on success. */
int bb_tool_down_ik(const bb_arm *arm, const double pos[3], double yaw, const double q0[BB_NJ],
                    double q_out[BB_NJ]);

/* Decompose `rot` as Rz(yaw) Ry(pitch) Rx(roll). */
void bb_euler_zyx(const double rot[9], double *yaw, double *pitch, double *roll);

#endif /* BB_KIN_H */
