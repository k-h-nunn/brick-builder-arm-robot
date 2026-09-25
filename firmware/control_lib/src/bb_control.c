#include "bb_control.h"
#include "bb_proto.h"
#include "bb_math.h"

#include <math.h>
#include <string.h>

/* Step titles, in the order `action.place.STEP_TITLES` declares them. */
#define STEP_APPROACH 0
#define STEP_GRASP    1
#define STEP_LIFT     2
#define STEP_ALIGN    4

static void prepare_step(bb_controller *c, const bb_frame_rec *f)
{
    bb_bundle_step(&c->bundle, (uint32_t)f->k, &c->step);
    if (!c->step.hdr->valid) {
        c->has_step = 0;
        c->trusted = 0;
        return;
    }
    c->has_step = 1;
    /* `TrackingEpisode._settle_servo`: the servo is built on the first usable step without a
     * brick reference; every later step is swapped in together with its own. */
    c->servo.brick = c->servo_created ? c->step.hdr->has_brick_ref : 0;
    c->servo_created = 1;
    c->gain = c->step.hdr->gain;
    c->trusted = c->step.hdr->trusted;
    c->servo.require_brick = f->held;
    if (f->is_freeze_step && !c->in_hand_reset) {
        c->in_hand_reset = 1;
        /* Not on a pick: there the in-hand reading was already taken at the lift dwell, against
         * the bare table and with the arm at rest. */
        if (!c->cfg->pick) {
            bb_servo_reset_in_hand(&c->servo);
        }
    }
}

int bb_control_init(bb_controller *c, const uint8_t *blob, uint32_t bytes,
                    const double q_home[BB_NJ])
{
    int rc = bb_bundle_parse(blob, bytes, &c->bundle);
    int i;

    if (rc != BB_OK) {
        return rc;
    }
    c->cfg = c->bundle.cfg;
    bb_servo_init(&c->servo, c->cfg);
    bb_jc_init(&c->jc, c->cfg->jc_joints > 0 ? c->cfg->jc_joints : BB_NJ, c->cfg->jc_ki,
               c->cfg->jc_dt, c->cfg->jc_clamp, c->cfg->jc_deadzone, 4096);

    for (i = 0; i < 3; ++i) {
        c->correction[i] = 0.0;
        c->frozen_correction[i] = 0.0;
        c->predicted[i] = 0.0;
    }
    c->has_step = 0;
    c->servo_created = 0;
    c->trusted = 0;
    for (i = 0; i < BB_NJOINT; ++i) {
        c->last_position[i] = 0.0;
    }
    c->k = 0;
    c->holds = 0;
    c->blind_holds = 0;
    c->max_hold_seen = 0;
    c->matched = 0;
    c->lag = 0;
    c->settled = 0;
    c->step_index = -1;
    c->jacobian_at = -1000000;
    c->regrasps = 0;
    c->unmatched = 0;
    c->blind_frames = 0;
    c->failure = BB_FAIL_NONE;
    c->done = 0;
    c->dwelling = 0;
    c->dwell_ticks = 0;
    c->frozen = 0;
    c->pick_frozen = 0;
    c->descending = 0;
    c->pressing = 0;
    c->in_hand_reset = 0;
    c->grasp_settled = 0;
    c->gating_grasp = 0;
    c->grasp_checked = 0;
    c->regrasping = 0;
    c->ticks = 0;
    c->gain = 0.0;
    c->yaw_err = NAN;
    c->absolute_mm = NAN;
    c->pred_mm = NAN;
    c->has_predicted = 0;
    c->ik_failures = 0;
    c->grip_last = 0.0;
    for (i = 0; i < BB_NJ; ++i) {
        c->q_goal[i] = q_home[i];
    }
    return BB_OK;
}

void bb_control_tcp(const bb_controller *c, const double q[BB_NJ], double pos[3])
{
    bb_fk(c->bundle.arm, q, pos, 0);
}

int bb_tick_begin(bb_controller *c, bb_tick_out *out)
{
    const bb_frame_rec *f;
    const bb_cfg *cfg = c->cfg;
    int n, i, first;

    memset(out, 0, sizeof(*out));
    if (c->done || c->k >= (int32_t)c->bundle.n_frames) {
        out->done = 1;
        out->failure = c->failure;
        return 0;
    }
    f = &c->bundle.frames[c->k];

    if (f->step_index != c->step_index || c->k - c->jacobian_at >= cfg->jacobian_every) {
        c->step_index = f->step_index;
        c->jacobian_at = c->k;
        prepare_step(c, f);
    }

    /* A held index issues the frame's last write only, so the world runs a fifth as fast; on the
     * way back up to the hover after a missed grasp it issues the whole frame. */
    first = (c->holds == 0 || c->regrasping || c->gating_grasp) ? 0 : f->n_commands - 1;
    n = f->n_commands - first;
    if (n > BB_MAX_CMDS) {
        n = BB_MAX_CMDS;
    }

    for (i = 0; i < n; ++i) {
        const bb_command_rec *rec = &c->bundle.commands[f->cmd_offset + first + i];
        bb_command *cmd = &out->cmd[i];
        double grip = rec->grip;
        double q5[BB_NJ];
        int ok, j;

        cmd->pos[0] = rec->pos[0] + c->correction[0];
        cmd->pos[1] = rec->pos[1] + c->correction[1];
        cmd->pos[2] = rec->pos[2];
        cmd->spin = rec->spin + c->correction[2];
        if (grip < cfg->grip_lo) {
            grip = cfg->grip_lo;
        } else if (grip > cfg->grip_hi) {
            grip = cfg->grip_hi;
        }
        cmd->grip = grip;
        cmd->c_vert = rec->c_vert;
        cmd->c_lat = rec->c_lat;

        ok = bb_pose_ik(c->bundle.arm, cmd->pos, 0.0, cmd->spin, c->q_goal, q5);
        if (!ok) {
            double q_alt[BB_NJ];
            if (bb_tool_down_ik(c->bundle.arm, cmd->pos, cmd->spin, c->q_goal, q_alt)) {
                for (j = 0; j < BB_NJ; ++j) {
                    q5[j] = q_alt[j];
                }
                ok = 1;
            }
        }
        if (ok) {
            for (j = 0; j < BB_NJ; ++j) {
                c->q_goal[j] = q5[j];
            }
        } else {
            c->ik_failures += 1;
        }
        cmd->ik_ok = ok;
        /* The trim sees the plain tracking error; its output is added to the command rather
         * than replacing it. The host adds the gravity feedforward term. */
        if (cfg->jc_joints > 0) {
            bb_jc_update(&c->jc, c->q_goal, c->last_position,
                         (cmd->c_vert < 1.0 || cmd->c_lat < 1.0) ? 1 : 0);
            for (j = 0; j < BB_NJ; ++j) {
                cmd->q[j] = c->jc.target[j];
            }
        } else {
            for (j = 0; j < BB_NJ; ++j) {
                cmd->q[j] = c->q_goal[j];
            }
        }
        cmd->q[BB_NJ] = grip;
        c->grip_last = grip;
    }

    out->n_commands = n;
    out->k = c->k;
    out->step_index = f->step_index;
    out->step_id = f->step_id;
    out->holds = c->holds;
    for (i = 0; i < 3; ++i) {
        out->correction[i] = c->correction[i];
    }
    out->tol_mm = cfg->tol_mm;
    return n;
}

/* Where the placed brick's reference sits among the active step's features: its references,
 * then the brick in the hand when the servo carries one. -1 when the step does not list it. */
static int held_ref(const bb_controller *c, int32_t name)
{
    const bb_step_hdr *h = c->step.hdr;
    int i;

    if (name < 0) {
        return -1;
    }
    for (i = 0; i < h->n_refs; ++i) {
        if (c->step.refs[i].name == name) {
            return i;
        }
    }
    if (c->servo.brick && h->has_brick_ref && h->brick_dim > 0
        && c->step.brick_ref->name == name) {
        return h->n_refs;
    }
    return -1;
}

/* One step of the inverse, plus the next-frame prediction it implies. Returns the matched frame
 * index; writes the tracking error in millimetres and how many reference blobs were seen. */
static int feedback(bb_controller *c, const bb_frame_rec *f, const bb_slots *state,
                    const bb_bus_readings *bus, double *err_mm, int *seen)
{
    const bb_cfg *cfg = c->cfg;
    bb_servo_out trial, out, absolute;
    int in_hand_only, rel, lo, j, best = c->k, have = 0;
    double best_err = INFINITY;
    double measured[3], analytic[3], delta[3];
    double speed_max = 0.0;
    int i;

    c->yaw_err = NAN;
    *err_mm = NAN;
    *seen = 0;

    in_hand_only = cfg->pick && c->has_step && c->step.hdr->brick_dim > 0
                   && f->step_id == STEP_LIFT;
    if (!c->servo_created || !c->has_step || (!c->trusted && !in_hand_only)
        || c->pressing || c->descending) {
        return c->k;
    }

    for (i = 0; i < BB_NJOINT; ++i) {
        double v = fabs(bus->speed[i]);
        if (v > speed_max) {
            speed_max = v;
        }
    }
    /* Estimate only where the sequence dwells: a residual measured while the arm is moving is
     * dominated by tracking lag, not by the world. */
    c->dwelling = (!cfg->dwell_only)
                  || (1e3 * f->travel < cfg->dwell_travel_mm && speed_max < cfg->dwell_speed);
    if (cfg->pick) {
        /* One dwell, one unknown: the brick's offset in the jaws is read at the lift dwell and
         * nowhere else. */
        c->servo.hold_in_hand = !(c->dwelling && f->step_id == STEP_LIFT);
    }
    rel = c->frozen && !f->held;

    lo = c->k - cfg->match_window;
    if (lo < 0) {
        lo = 0;
    }
    for (j = lo; j <= c->k; ++j) {
        double e;
        bb_servo_update(&c->servo, cfg, &c->step, state, &c->bundle.states[j], 1, rel, &trial);
        if (trial.blind) {
            continue;
        }
        e = hypot(trial.d_r[0] + trial.d_b[0], trial.d_r[1] + trial.d_b[1]);
        if (e < best_err) {
            best_err = e;
            best = j;
            have = 1;
        }
    }
    if (!have) {
        return c->k;
    }
    /* Once the frame is chosen, take the reading again for real so the in-hand average sees
     * exactly one measurement per tick. */
    bb_servo_update(&c->servo, cfg, &c->step, state, &c->bundle.states[best], 0, rel, &out);
    bb_servo_update(&c->servo, cfg, &c->step, state, &c->bundle.states[best], 1, 0, &absolute);
    c->absolute_mm = absolute.blind ? NAN
                     : 1e3 * hypot(absolute.d_r[0] + absolute.d_b[0],
                                   absolute.d_r[1] + absolute.d_b[1]);
    if (out.blind) {
        return c->k;
    }
    if (rel && f->held_name != BB_HELD_NONE) {
        /* In relative mode the placed brick is the whole point: if it is not among the blobs
         * that matched, the number is unknown, and it is reported as unknown. */
        int r = held_ref(c, f->held_name);
        if (r < 0 || out.found.slot[r] < 0) {
            c->unmatched += 1;
            *seen = out.seen;
            return c->k;
        }
    }
    c->matched = best;
    c->yaw_err = out.d_r[2];

    measured[0] = out.d_r[0] + out.d_b[0];
    measured[1] = out.d_r[1] + out.d_b[1];
    measured[2] = out.d_r[2];
    if (c->has_predicted) {
        c->pred_mm = 1e3 * hypot(measured[0] - c->predicted[0], measured[1] - c->predicted[1]);
    }
    for (i = 0; i < 3; ++i) {
        analytic[i] = -cfg->lam * measured[i];
        delta[i] = 0.0;
    }
    if (cfg->closed_loop && !c->frozen && c->dwelling && c->trusted) {
        double n, n_c;
        for (i = 0; i < 3; ++i) {
            delta[i] = analytic[i];
        }
        n = hypot(delta[0], delta[1]);
        if (n > cfg->delta_xy_cap) {
            delta[0] *= cfg->delta_xy_cap / n;
            delta[1] *= cfg->delta_xy_cap / n;
        }
        if (delta[2] > cfg->delta_spin_cap) {
            delta[2] = cfg->delta_spin_cap;
        } else if (delta[2] < -cfg->delta_spin_cap) {
            delta[2] = -cfg->delta_spin_cap;
        }
        for (i = 0; i < 3; ++i) {
            c->correction[i] += delta[i];
        }
        n_c = hypot(c->correction[0], c->correction[1]);
        if (n_c > cfg->max_correction) {
            c->correction[0] *= cfg->max_correction / n_c;
            c->correction[1] *= cfg->max_correction / n_c;
        }
    }
    for (i = 0; i < 3; ++i) {
        c->predicted[i] = measured[i] + delta[i];
    }
    c->has_predicted = 1;
    *err_mm = 1e3 * hypot(measured[0], measured[1]);
    *seen = out.seen;
    return best;
}

/* Whether the descent onto the brick is still waiting for the hover estimate to settle. */
static int holding_for_grasp(bb_controller *c, double err_mm, double floor)
{
    const bb_cfg *cfg = c->cfg;
    int close, turned;
    double bound;

    if (!(cfg->pick && cfg->closed_loop) || c->frozen || cfg->descent_k < 0) {
        return 0;
    }
    if (c->k != cfg->descent_k - 1 || c->regrasping) {
        return 0;
    }
    bound = floor > cfg->advance_mm ? floor : cfg->advance_mm;
    close = (err_mm == err_mm) && err_mm <= bound;
    turned = (c->yaw_err == c->yaw_err)
             && fabs(c->yaw_err) <= cfg->grasp_yaw_deg * BB_PI / 180.0;
    c->grasp_settled = (close && turned) ? c->grasp_settled + 1 : 0;
    return (c->grasp_settled < cfg->grasp_ticks && c->holds < cfg->grasp_wait);
}

/* Did the jaws close somewhere other than where they were sent? If so, go back up, once. */
static int grasp_missed(bb_controller *c, const bb_bus_readings *bus)
{
    const bb_cfg *cfg = c->cfg;
    double q[BB_NJ], pos[3];
    int i;

    if (!(cfg->pick && cfg->closed_loop) || cfg->descent_k < 0) {
        return 0;
    }
    if (c->grasp_checked || c->k != cfg->grasp_k || c->regrasping) {
        return 0;
    }
    c->grasp_checked = 1;
    for (i = 0; i < BB_NJ; ++i) {
        q[i] = bus->position[i];
    }
    bb_fk(c->bundle.arm, q, pos, 0);
    if (pos[2] - cfg->grasp_z <= cfg->grasp_miss) {
        return 0;
    }
    if (c->regrasps) {
        c->failure = BB_FAIL_GRASP_SHORT;
        return 0;
    }
    c->regrasps += 1;
    c->k = cfg->descent_k - 1;
    c->holds = 0;
    c->step_index = -1;
    c->jacobian_at = -1000000;
    c->grasp_settled = 0;
    c->grasp_checked = 0;
    c->regrasping = 1;
    c->pick_frozen = 0;
    c->descending = 0;
    bb_servo_reset_in_hand(&c->servo);
    return 1;
}

int bb_tick_end(bb_controller *c, const bb_slots *state, const bb_bus_readings *bus,
                bb_tick_out *out)
{
    const bb_cfg *cfg = c->cfg;
    const bb_frame_rec *f;
    double err_mm, floor, tube;
    int seen, match, last, playing_forward, i;
    int low;

    if (c->done) {
        out->done = 1;
        out->failure = c->failure;
        return 1;
    }
    f = &c->bundle.frames[c->k];
    for (i = 0; i < BB_NJOINT; ++i) {
        c->last_position[i] = bus->position[i];
    }

    /* While the arm is leaning on something the tracking error is an error it is MEANT to have,
     * so the correction freezes and the sequence plays forward. */
    c->pressing = 0;
    for (i = 0; i < f->n_commands; ++i) {
        const bb_command_rec *rec = &c->bundle.commands[f->cmd_offset + i];
        if (rec->c_vert < 1.0 || rec->c_lat < 1.0) {
            c->pressing = 1;
        }
    }
    if (!c->frozen && f->step_index > cfg->freeze_index) {
        c->frozen = 1;
        for (i = 0; i < 3; ++i) {
            c->frozen_correction[i] = c->correction[i];
        }
    }
    low = c->bundle.commands[f->cmd_offset].pos[2] < cfg->pick_freeze_below;
    if (cfg->pick && !c->frozen && low && !c->pick_frozen) {
        c->pick_frozen = 1;
    }
    c->descending = (c->pick_frozen && !f->held
                     && (f->step_id == STEP_APPROACH || f->step_id == STEP_GRASP)) ? 1 : 0;

    match = feedback(c, f, state, bus, &err_mm, &seen);
    if (c->dwelling) {
        c->dwell_ticks += 1;
    }
    c->ticks += 1;

    out->vision = 1;
    out->k = c->k;
    out->step_index = f->step_index;
    out->step_id = f->step_id;
    out->err_mm = err_mm;
    out->tol_mm = cfg->tol_mm;
    out->absolute_mm = c->absolute_mm;
    out->predict_mm = c->pred_mm;
    out->yaw_err = c->yaw_err;
    out->refs_seen = seen;
    out->matched = c->matched;
    for (i = 0; i < 3; ++i) {
        out->correction[i] = c->correction[i];
    }

    if (grasp_missed(c, bus)) {
        out->regrasps = c->regrasps;
        out->holds = c->holds;
        out->failure = c->failure;
        return 0;
    }
    if (c->failure != BB_FAIL_NONE) {
        c->done = 1;
        out->done = 1;
        out->failure = c->failure;
        return 1;
    }

    /* The tube can never be tighter than the reading is trustworthy. */
    floor = c->servo_created
            ? (seen < 2 ? cfg->servo_dead_band_single_mm : cfg->servo_dead_band_mm)
            : cfg->advance_mm;
    tube = cfg->advance_mm;
    if (floor > tube) {
        tube = floor;
    }
    if (cfg->tube_scale * 1e3 * f->travel > tube) {
        tube = cfg->tube_scale * 1e3 * f->travel;
    }
    last = (c->k == (int32_t)c->bundle.n_frames - 1);
    c->lag = c->k - match;
    if (err_mm != err_mm) {
        c->blind_frames += 1;
    }

    /* A loop that is not going to act cannot catch up, so it does not wait. */
    playing_forward = c->frozen || c->descending || !cfg->closed_loop || !c->trusted
                      || (cfg->pick && f->step_id == STEP_LIFT)
                      || (cfg->dwell_only && !c->dwelling);
    if (c->regrasping) {
        double speed_max = 0.0;
        playing_forward = 0;
        for (i = 0; i < BB_NJOINT; ++i) {
            double v = fabs(bus->speed[i]);
            if (v > speed_max) {
                speed_max = v;
            }
        }
        if (c->k == cfg->descent_k - 1 && speed_max < cfg->dwell_speed) {
            c->regrasping = 0;
        }
    }
    c->gating_grasp = holding_for_grasp(c, err_mm, floor);

    if (!c->gating_grasp && (err_mm != err_mm || err_mm <= tube || playing_forward)) {
        if (last) {
            int inside = (err_mm != err_mm) || err_mm <= cfg->tol_mm || c->frozen;
            c->settled = inside ? c->settled + 1 : 0;
            if (c->settled >= cfg->settle_ticks) {
                c->done = 1;
                out->done = 1;
                out->failure = c->failure;
                return 1;
            }
        } else {
            c->k += 1;
        }
        c->holds = 0;
        c->blind_holds = 0;
    } else {
        c->settled = 0;
        c->holds += 1;
        if (err_mm != err_mm) {
            c->blind_holds += 1;
        }
        if (c->holds > c->max_hold_seen) {
            c->max_hold_seen = c->holds;
        }
        if (c->holds > cfg->max_hold) {
            /* Two different faults wear the same number of held ticks: an error the loop can see
             * and cannot reduce, and a reference that has left the window. */
            c->failure = (c->blind_holds * 2 > c->holds) ? BB_FAIL_STALL_BLIND
                                                         : BB_FAIL_STALL_ERROR;
            c->done = 1;
            out->done = 1;
            out->failure = c->failure;
            return 1;
        }
    }
    out->holds = c->holds;
    out->regrasps = c->regrasps;
    out->failure = c->failure;
    return 0;
}

/* Struct sizes, so the host can check its own packing against this build's rather than assume
 * the two agree. The order is fixed; `bb_struct_size` returning 0 means "unknown index". */
int32_t bb_struct_size(int which)
{
    switch (which) {
    case 0: return (int32_t)sizeof(bb_cfg);
    case 1: return (int32_t)sizeof(bb_arm);
    case 2: return (int32_t)sizeof(bb_frame_rec);
    case 3: return (int32_t)sizeof(bb_command_rec);
    case 4: return (int32_t)sizeof(bb_state_rec);
    case 5: return (int32_t)sizeof(bb_ref_rec);
    case 6: return (int32_t)sizeof(bb_step_hdr);
    case 7: return (int32_t)sizeof(bb_controller);
    case 8: return (int32_t)sizeof(bb_tick_out);
    case 9: return (int32_t)sizeof(bb_command);
    case 10: return (int32_t)sizeof(bb_bus_readings);
    case 11: return (int32_t)sizeof(bb_slots);
    case 12: return (int32_t)sizeof(bb_joint_correction);
    case 13: return (int32_t)sizeof(bb_decoder);
    default: return 0;
    }
}
