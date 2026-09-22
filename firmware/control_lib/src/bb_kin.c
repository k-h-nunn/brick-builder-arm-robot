#include "bb_kin.h"
#include "bb_math.h"

#include <math.h>
#include <string.h>

/* Tool-down targets, mirroring the constants in `scene/compose.py`. */
#define BB_TOOL_DOWN_ROLL  BB_PI
#define BB_POSE_IK_POS_TOL 2e-4
#define BB_POSE_IK_ANG_TOL (0.2 * BB_PI / 180.0)
#define BB_POSE_IK_YAW_TOL (0.1 * BB_PI / 180.0)
#define BB_POSE_IK_ANG_ACC (2.0 * BB_PI / 180.0)
#define BB_POSE_IK_ITERS   2
#define BB_YAW_WEIGHT      0.15

/* ------------------------------------------------------------------ quaternion arithmetic --- */

static void quat_mul(const double a[4], const double b[4], double r[4])
{
    double t0 = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    double t1 = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    double t2 = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    double t3 = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

static void quat_norm(double q[4])
{
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-15) {
        q[0] = 1.0; q[1] = 0.0; q[2] = 0.0; q[3] = 0.0;
        return;
    }
    q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
}

static void quat2mat(const double q[4], double m[9])
{
    double q00, q01, q02, q03, q11, q12, q13, q22, q23, q33;
    if (q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0) {
        m[0] = 1.0; m[1] = 0.0; m[2] = 0.0;
        m[3] = 0.0; m[4] = 1.0; m[5] = 0.0;
        m[6] = 0.0; m[7] = 0.0; m[8] = 1.0;
        return;
    }
    q00 = q[0] * q[0]; q01 = q[0] * q[1]; q02 = q[0] * q[2]; q03 = q[0] * q[3];
    q11 = q[1] * q[1]; q12 = q[1] * q[2]; q13 = q[1] * q[3];
    q22 = q[2] * q[2]; q23 = q[2] * q[3];
    q33 = q[3] * q[3];
    m[0] = q00 + q11 - q22 - q33;
    m[1] = 2.0 * (q12 - q03);
    m[2] = 2.0 * (q13 + q02);
    m[3] = 2.0 * (q12 + q03);
    m[4] = q00 - q11 + q22 - q33;
    m[5] = 2.0 * (q23 - q01);
    m[6] = 2.0 * (q13 - q02);
    m[7] = 2.0 * (q23 + q01);
    m[8] = q00 - q11 - q22 + q33;
}

static void mat_vec3(const double m[9], const double v[3], double r[3])
{
    double a = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    double b = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    double c = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    r[0] = a; r[1] = b; r[2] = c;
}

static void axis_angle_quat(const double axis[3], double angle, double q[4])
{
    double s;
    if (angle == 0.0) {
        q[0] = 1.0; q[1] = 0.0; q[2] = 0.0; q[3] = 0.0;
        return;
    }
    s = sin(angle * 0.5);
    q[0] = cos(angle * 0.5);
    q[1] = axis[0] * s;
    q[2] = axis[1] * s;
    q[3] = axis[2] * s;
}

/* ---------------------------------------------------------------------------- the chain ----- */

typedef struct {
    double xpos[BB_NLINK][3];
    double xmat[BB_NLINK][9];
    double xaxis[BB_NLINK][3];
    double site_pos[3];
    double site_rot[9];
} bb_chain;

static void chain_eval(const bb_arm *arm, const double q[BB_NJ], bb_chain *ch)
{
    double ppos[3] = {0.0, 0.0, 0.0};
    double pmat[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double pquat[4] = {1.0, 0.0, 0.0, 0.0};
    double xquat[BB_NLINK][4];
    double squat[4];
    int i;

    for (i = 0; i < BB_NLINK; ++i) {
        double t[3];
        double qloc[4];

        mat_vec3(pmat, arm->body_pos[i], t);
        ch->xpos[i][0] = ppos[0] + t[0];
        ch->xpos[i][1] = ppos[1] + t[1];
        ch->xpos[i][2] = ppos[2] + t[2];
        quat_mul(pquat, arm->body_quat[i], xquat[i]);

        /* The joint axis is read off the frame BEFORE the joint rotation is applied, and the
         * hinge sits at the body origin so the position is untouched by it. */
        quat2mat(xquat[i], ch->xmat[i]);
        mat_vec3(ch->xmat[i], arm->axis[i], ch->xaxis[i]);

        axis_angle_quat(arm->axis[i], q[i], qloc);
        quat_mul(xquat[i], qloc, xquat[i]);
        quat_norm(xquat[i]);
        quat2mat(xquat[i], ch->xmat[i]);

        ppos[0] = ch->xpos[i][0];
        ppos[1] = ch->xpos[i][1];
        ppos[2] = ch->xpos[i][2];
        memcpy(pmat, ch->xmat[i], sizeof(pmat));
        memcpy(pquat, xquat[i], sizeof(pquat));
    }

    {
        double t[3];
        mat_vec3(pmat, arm->site_pos, t);
        ch->site_pos[0] = ppos[0] + t[0];
        ch->site_pos[1] = ppos[1] + t[1];
        ch->site_pos[2] = ppos[2] + t[2];
        quat_mul(pquat, arm->site_quat, squat);
        quat2mat(squat, ch->site_rot);
    }
}

void bb_fk(const bb_arm *arm, const double q[BB_NJ], double pos[3], double rot[9])
{
    bb_chain ch;
    chain_eval(arm, q, &ch);
    if (pos) {
        pos[0] = ch.site_pos[0];
        pos[1] = ch.site_pos[1];
        pos[2] = ch.site_pos[2];
    }
    if (rot) {
        memcpy(rot, ch.site_rot, sizeof(ch.site_rot));
    }
}

void bb_jac(const bb_arm *arm, const double q[BB_NJ], double pos[3], double rot[9],
            double jacp[3 * BB_NJ], double jacr[3 * BB_NJ])
{
    bb_chain ch;
    int i;

    chain_eval(arm, q, &ch);
    if (pos) {
        pos[0] = ch.site_pos[0];
        pos[1] = ch.site_pos[1];
        pos[2] = ch.site_pos[2];
    }
    if (rot) {
        memcpy(rot, ch.site_rot, sizeof(ch.site_rot));
    }
    for (i = 0; i < BB_NJ; ++i) {
        double d[3];
        const double *ax = ch.xaxis[i];
        d[0] = ch.site_pos[0] - ch.xpos[i][0];
        d[1] = ch.site_pos[1] - ch.xpos[i][1];
        d[2] = ch.site_pos[2] - ch.xpos[i][2];
        jacp[0 * BB_NJ + i] = ax[1] * d[2] - ax[2] * d[1];
        jacp[1 * BB_NJ + i] = ax[2] * d[0] - ax[0] * d[2];
        jacp[2 * BB_NJ + i] = ax[0] * d[1] - ax[1] * d[0];
        jacr[0 * BB_NJ + i] = ax[0];
        jacr[1 * BB_NJ + i] = ax[1];
        jacr[2 * BB_NJ + i] = ax[2];
    }
}

void bb_euler_zyx(const double rot[9], double *yaw, double *pitch, double *roll)
{
    if (pitch) {
        *pitch = atan2(-rot[6], hypot(rot[7], rot[8]));
    }
    if (yaw) {
        *yaw = atan2(rot[3], rot[0]);
    }
    if (roll) {
        *roll = atan2(rot[7], rot[8]);
    }
}

static void clip_q(const bb_arm *arm, double q[BB_NJ])
{
    int i;
    for (i = 0; i < BB_NJ; ++i) {
        if (q[i] < arm->jnt_lo[i]) {
            q[i] = arm->jnt_lo[i];
        } else if (q[i] > arm->jnt_hi[i]) {
            q[i] = arm->jnt_hi[i];
        }
    }
}

/* Fill the five-row task error at `q`; returns the tool rotation in `rot`. */
static void diff_residual(const bb_arm *arm, const double q[BB_NJ], const double target_pos[3],
                          double pitch, double roll, double out[6], double rot[9],
                          double jacp[3 * BB_NJ], double jacr[3 * BB_NJ])
{
    double pos[3];
    double yaw, cz, sz, cy, sy, cx, sx;
    double d00, d01, d02, d10, d11, d12, d20, d21, d22;

    bb_jac(arm, q, pos, rot, jacp, jacr);
    out[0] = target_pos[0] - pos[0];
    out[1] = target_pos[1] - pos[1];
    out[2] = target_pos[2] - pos[2];

    yaw = atan2(rot[3], rot[0]);
    cz = cos(yaw); sz = sin(yaw);
    cy = cos(pitch); sy = sin(pitch);
    cx = cos(roll); sx = sin(roll);
    d00 = cz * cy; d01 = cz * sy * sx - sz * cx; d02 = cz * sy * cx + sz * sx;
    d10 = sz * cy; d11 = sz * sy * sx + cz * cx; d12 = sz * sy * cx - cz * sx;
    d20 = -sy;     d21 = cy * sx;                d22 = cy * cx;

    out[3] = 0.5 * ((rot[3] * d20 - rot[6] * d10)
                    + (rot[4] * d21 - rot[7] * d11)
                    + (rot[5] * d22 - rot[8] * d12));
    out[4] = 0.5 * ((rot[6] * d00 - rot[0] * d20)
                    + (rot[7] * d01 - rot[1] * d21)
                    + (rot[8] * d02 - rot[2] * d22));
    out[5] = 0.0;
}

void bb_ik_diff(const bb_arm *arm, const double target_pos[3], double target_pitch,
                double target_roll, const double q0[BB_NJ], int use_yaw, double target_yaw,
                int iters, double damping, double pos_tol, double ang_tol, double yaw_tol,
                double yaw_weight, bb_ik_result *out)
{
    double q[BB_NJ];
    double residual[6], rot[9];
    double jacp[3 * BB_NJ], jacr[3 * BB_NJ];
    double jac[6 * BB_NJ];
    double scale[6];
    int rows = use_yaw ? 6 : 5;
    int it, i, j, k;
    double pos_err = 0.0, ang_err = 0.0, yaw_err = 0.0;

    for (i = 0; i < BB_NJ; ++i) {
        q[i] = q0[i];
    }
    clip_q(arm, q);

    scale[0] = scale[1] = scale[2] = 1.0 / pos_tol;
    scale[3] = scale[4] = 1.0 / ang_tol;
    scale[5] = yaw_weight / ang_tol;

    for (it = 0; it <= iters; ++it) {
        double jw[6 * BB_NJ], rw[6], jjt[36], rhs[6], dq[BB_NJ];
        double lam, trace = 0.0;

        diff_residual(arm, q, target_pos, target_pitch, target_roll, residual, rot, jacp, jacr);
        pos_err = sqrt(residual[0] * residual[0] + residual[1] * residual[1]
                       + residual[2] * residual[2]);
        ang_err = fabs(residual[3]) > fabs(residual[4]) ? fabs(residual[3]) : fabs(residual[4]);
        if (!use_yaw) {
            yaw_err = 0.0;
        } else {
            yaw_err = remainder(target_yaw - atan2(rot[3], rot[0]), 2.0 * BB_PI);
            residual[5] = yaw_err;
        }
        if (pos_err < pos_tol && ang_err < ang_tol && fabs(yaw_err) < yaw_tol) {
            for (i = 0; i < BB_NJ; ++i) {
                out->q[i] = q[i];
            }
            out->converged = 1;
            out->pos_error = pos_err;
            out->ang_error = ang_err;
            out->yaw_error = yaw_err;
            out->iterations = it;
            return;
        }
        if (it == iters) {
            break;
        }

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < BB_NJ; ++j) {
                jac[i * BB_NJ + j] = -jacp[i * BB_NJ + j];
            }
        }
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < BB_NJ; ++j) {
                jac[(3 + i) * BB_NJ + j] = -jacr[i * BB_NJ + j];
            }
        }
        if (use_yaw) {
            for (j = 0; j < BB_NJ; ++j) {
                jac[5 * BB_NJ + j] = -jacr[2 * BB_NJ + j];
            }
        }

        for (i = 0; i < rows; ++i) {
            for (j = 0; j < BB_NJ; ++j) {
                jw[i * BB_NJ + j] = jac[i * BB_NJ + j] * scale[i];
            }
            rw[i] = residual[i] * scale[i];
        }
        for (i = 0; i < rows; ++i) {
            for (j = 0; j < rows; ++j) {
                double s = 0.0;
                for (k = 0; k < BB_NJ; ++k) {
                    s += jw[i * BB_NJ + k] * jw[j * BB_NJ + k];
                }
                jjt[i * rows + j] = s;
            }
            rhs[i] = rw[i];
        }
        for (i = 0; i < rows; ++i) {
            trace += jjt[i * rows + i];
        }
        lam = damping * damping * trace / (double)rows;
        for (i = 0; i < rows; ++i) {
            jjt[i * rows + i] += lam;
        }
        if (bb_solve(jjt, rhs, rows) != BB_OK) {
            break;
        }
        for (j = 0; j < BB_NJ; ++j) {
            double s = 0.0;
            for (i = 0; i < rows; ++i) {
                s += jw[i * BB_NJ + j] * rhs[i];
            }
            dq[j] = -s;
            q[j] += dq[j];
        }
        clip_q(arm, q);
    }

    for (i = 0; i < BB_NJ; ++i) {
        out->q[i] = q[i];
    }
    out->converged = 0;
    out->pos_error = pos_err;
    out->ang_error = ang_err;
    out->yaw_error = yaw_err;
    out->iterations = iters;
}

/* The global solver's residual: position plus the LOCAL pitch and roll of the orientation
 * error, exactly as `ArmKinematics.ik` evaluates it. */
static double lm_evaluate(const bb_arm *arm, const double q[BB_NJ], const double target_pos[3],
                          double pitch, double roll, double residual[5], double *pos_e,
                          double *pitch_e, double *roll_e)
{
    double pos[3], rot[9];
    double yaw_now, cz, sz, cy, sy, cx, sx;
    double rd[9], ew[3], el[3];
    int i;
    double n = 0.0;

    bb_fk(arm, q, pos, rot);
    yaw_now = atan2(rot[3], rot[0]);
    cz = cos(yaw_now); sz = sin(yaw_now);
    cy = cos(pitch);   sy = sin(pitch);
    cx = cos(roll);    sx = sin(roll);
    rd[0] = cz * cy; rd[1] = cz * sy * sx - sz * cx; rd[2] = cz * sy * cx + sz * sx;
    rd[3] = sz * cy; rd[4] = sz * sy * sx + cz * cx; rd[5] = sz * sy * cx - cz * sx;
    rd[6] = -sy;     rd[7] = cy * sx;                rd[8] = cy * cx;

    /* e_world = 0.5 * sum_i R[:, i] x Rd[:, i] */
    ew[0] = 0.5 * ((rot[3] * rd[6] - rot[6] * rd[3])
                   + (rot[4] * rd[7] - rot[7] * rd[4])
                   + (rot[5] * rd[8] - rot[8] * rd[5]));
    ew[1] = 0.5 * ((rot[6] * rd[0] - rot[0] * rd[6])
                   + (rot[7] * rd[1] - rot[1] * rd[7])
                   + (rot[8] * rd[2] - rot[2] * rd[8]));
    ew[2] = 0.5 * ((rot[0] * rd[3] - rot[3] * rd[0])
                   + (rot[1] * rd[4] - rot[4] * rd[1])
                   + (rot[2] * rd[5] - rot[5] * rd[2]));
    /* e_local = Rd^T e_world */
    for (i = 0; i < 3; ++i) {
        el[i] = rd[0 * 3 + i] * ew[0] + rd[1 * 3 + i] * ew[1] + rd[2 * 3 + i] * ew[2];
    }

    residual[0] = target_pos[0] - pos[0];
    residual[1] = target_pos[1] - pos[1];
    residual[2] = target_pos[2] - pos[2];
    residual[3] = el[1];
    residual[4] = el[0];
    *pos_e = sqrt(residual[0] * residual[0] + residual[1] * residual[1]
                  + residual[2] * residual[2]);
    *pitch_e = el[1];
    *roll_e = el[0];
    for (i = 0; i < 5; ++i) {
        n += residual[i] * residual[i];
    }
    return sqrt(n);
}

void bb_ik(const bb_arm *arm, const double target_pos[3], double target_pitch, double target_roll,
           const double q0[BB_NJ], double damping, int iters, double pos_tol, double ang_tol,
           bb_ik_result *out)
{
    double q[BB_NJ];
    double residual[5], res_norm;
    double pos_err, pitch_err, roll_err;
    double lam = damping * damping;
    const double fd_eps = 1e-6;
    int it = 0, i, j, tries;

    for (i = 0; i < BB_NJ; ++i) {
        q[i] = q0[i];
    }
    clip_q(arm, q);
    res_norm = lm_evaluate(arm, q, target_pos, target_pitch, target_roll, residual,
                           &pos_err, &pitch_err, &roll_err);

    for (it = 1; it <= iters; ++it) {
        double jac[5 * BB_NJ];
        int accepted = 0;

        if (pos_err < pos_tol && fabs(pitch_err) < ang_tol && fabs(roll_err) < ang_tol) {
            for (i = 0; i < BB_NJ; ++i) {
                out->q[i] = q[i];
            }
            out->converged = 1;
            out->pos_error = pos_err;
            out->ang_error = fabs(pitch_err) > fabs(roll_err) ? fabs(pitch_err) : fabs(roll_err);
            out->yaw_error = 0.0;
            out->iterations = it - 1;
            return;
        }

        for (j = 0; j < BB_NJ; ++j) {
            double qp[BB_NJ], rp[5], pe, pie, roe;
            for (i = 0; i < BB_NJ; ++i) {
                qp[i] = q[i];
            }
            qp[j] += fd_eps;
            lm_evaluate(arm, qp, target_pos, target_pitch, target_roll, rp, &pe, &pie, &roe);
            for (i = 0; i < 5; ++i) {
                jac[i * BB_NJ + j] = (rp[i] - residual[i]) / fd_eps;
            }
        }

        for (tries = 0; tries < 30; ++tries) {
            double jjt[25], rhs[5], q_try[BB_NJ], r_try[5];
            double n_try, pe, pie, roe;
            int a, b, c;
            for (a = 0; a < 5; ++a) {
                for (b = 0; b < 5; ++b) {
                    double s = 0.0;
                    for (c = 0; c < BB_NJ; ++c) {
                        s += jac[a * BB_NJ + c] * jac[b * BB_NJ + c];
                    }
                    jjt[a * 5 + b] = s + (a == b ? lam : 0.0);
                }
                rhs[a] = residual[a];
            }
            if (bb_solve(jjt, rhs, 5) != BB_OK) {
                break;
            }
            for (j = 0; j < BB_NJ; ++j) {
                double s = 0.0;
                for (a = 0; a < 5; ++a) {
                    s += jac[a * BB_NJ + j] * rhs[a];
                }
                q_try[j] = q[j] - s;
            }
            clip_q(arm, q_try);
            n_try = lm_evaluate(arm, q_try, target_pos, target_pitch, target_roll, r_try,
                                &pe, &pie, &roe);
            if (n_try < res_norm) {
                for (j = 0; j < BB_NJ; ++j) {
                    q[j] = q_try[j];
                }
                for (a = 0; a < 5; ++a) {
                    residual[a] = r_try[a];
                }
                res_norm = n_try;
                pos_err = pe;
                pitch_err = pie;
                roll_err = roe;
                lam = lam * 0.5 > 1e-10 ? lam * 0.5 : 1e-10;
                accepted = 1;
                break;
            }
            lam *= 4.0;
        }
        if (!accepted) {
            break;
        }
    }

    for (i = 0; i < BB_NJ; ++i) {
        out->q[i] = q[i];
    }
    out->converged = 0;
    out->pos_error = pos_err;
    out->ang_error = fabs(pitch_err) > fabs(roll_err) ? fabs(pitch_err) : fabs(roll_err);
    out->yaw_error = 0.0;
    out->iterations = it;
}

int bb_pose_ik(const bb_arm *arm, const double pos[3], double tilt, double spin,
               const double q0[BB_NJ], double q_out[BB_NJ])
{
    bb_ik_result diff, result, final;
    double q[BB_NJ];
    double rot[9], achieved, err = 0.0;
    int i, pass;

    bb_ik_diff(arm, pos, tilt, BB_TOOL_DOWN_ROLL, q0, 1, spin, BB_POSE_IK_ITERS, 1e-2,
               BB_POSE_IK_POS_TOL, BB_POSE_IK_ANG_TOL, BB_POSE_IK_YAW_TOL, BB_YAW_WEIGHT, &diff);
    if (diff.pos_error < BB_POSE_IK_POS_TOL && diff.ang_error < BB_POSE_IK_ANG_ACC) {
        for (i = 0; i < BB_NJ; ++i) {
            q_out[i] = diff.q[i];
        }
        return 1;
    }

    for (i = 0; i < BB_NJ; ++i) {
        q[i] = q0[i];
    }
    for (pass = 0; pass < 2; ++pass) {
        bb_ik(arm, pos, tilt, BB_TOOL_DOWN_ROLL, q, 1e-2, 6, BB_POSE_IK_POS_TOL,
              BB_POSE_IK_ANG_TOL, &result);
        for (i = 0; i < BB_NJ; ++i) {
            q[i] = result.q[i];
        }
        bb_fk(arm, q, 0, rot);
        achieved = atan2(rot[3], rot[0]);
        err = remainder(spin - achieved, 2.0 * BB_PI);
        if (fabs(err) < BB_POSE_IK_YAW_TOL) {
            break;
        }
        q[4] += err;
        if (q[4] < arm->jnt_lo[4]) {
            q[4] = arm->jnt_lo[4];
        } else if (q[4] > arm->jnt_hi[4]) {
            q[4] = arm->jnt_hi[4];
        }
    }
    bb_ik(arm, pos, tilt, BB_TOOL_DOWN_ROLL, q, 1e-2, 0, BB_POSE_IK_POS_TOL,
          BB_POSE_IK_ANG_TOL, &final);
    for (i = 0; i < BB_NJ; ++i) {
        q_out[i] = q[i];
    }
    return (final.pos_error < BB_POSE_IK_POS_TOL && final.ang_error < BB_POSE_IK_ANG_ACC) ? 1 : 0;
}

int bb_tool_down_ik(const bb_arm *arm, const double pos[3], double yaw, const double q0[BB_NJ],
                    double q_out[BB_NJ])
{
    bb_ik_result result;
    double q[BB_NJ], rot[9], achieved, err;
    int i, pass;
    int have = 0;

    for (i = 0; i < BB_NJ; ++i) {
        q[i] = q0[i];
    }
    result.converged = 0;
    for (pass = 0; pass < 8; ++pass) {
        bb_ik(arm, pos, 0.0, BB_TOOL_DOWN_ROLL, q, 1e-2, 200, 1e-5, 0.05 * BB_PI / 180.0, &result);
        have = 1;
        for (i = 0; i < BB_NJ; ++i) {
            q[i] = result.q[i];
        }
        bb_fk(arm, q, 0, rot);
        achieved = atan2(rot[3], rot[0]);
        err = remainder(yaw - achieved, 2.0 * BB_PI);
        if (fabs(err) < 0.05 * BB_PI / 180.0) {
            break;
        }
        q[4] += err;
        if (q[4] < arm->jnt_lo[4]) {
            q[4] = arm->jnt_lo[4];
        } else if (q[4] > arm->jnt_hi[4]) {
            q[4] = arm->jnt_hi[4];
        }
    }
    for (i = 0; i < BB_NJ; ++i) {
        q_out[i] = q[i];
    }
    return (have && result.converged) ? 1 : 0;
}
