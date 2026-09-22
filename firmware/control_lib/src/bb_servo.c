#include "bb_servo.h"
#include "bb_math.h"

#include <math.h>
#include <string.h>

void bb_servo_init(bb_servo *s, const bb_cfg *cfg)
{
    s->lam = cfg->lam;
    s->settle_ticks = cfg->settle_ticks;
    s->max_spin = cfg->use_yaw ? cfg->servo_max_spin : 0.0;
    s->max_in_hand = cfg->servo_max_in_hand;
    s->require_brick = 1;
    s->hold_in_hand = 0;
    s->hits = 0;
    s->has_d_b = 0;
    s->d_b[0] = 0.0;
    s->d_b[1] = 0.0;
    s->n_b = 0;
}

void bb_servo_reset_in_hand(bb_servo *s)
{
    s->has_d_b = 0;
    s->d_b[0] = 0.0;
    s->d_b[1] = 0.0;
    s->n_b = 0;
}

/* `action.servo.reference_features`: find each reference's slot and read its centre. */
static void reference_features(const bb_slots *state, const bb_step *step,
                               bb_features *out)
{
    const bb_step_hdr *h = step->hdr;
    uint32_t taken = 0u;
    int i;

    out->n = h->n_refs;
    for (i = 0; i < h->n_refs; ++i) {
        const bb_ref_rec *r = &step->refs[i];
        int k = bb_select_slot(state, r->colour, r->colour_tol, r->size, r->has_size,
                               r->size_tol, r->min_mass, r->has_min_mass, taken);
        double border;
        out->slot[i] = -1;
        out->centre[i][0] = 0.0;
        out->centre[i][1] = 0.0;
        out->weight[i] = 0.0;
        if (k < 0) {
            continue;
        }
        border = bb_slot_border(state, k);
        if (r->require_inside && border > r->max_border) {
            continue;
        }
        taken |= 1u << k;
        out->slot[i] = k;
        bb_slot_centre(state, k, out->centre[i]);
        out->weight[i] = (1.0 - border) > 0.0 ? (1.0 - border) : 0.0;
    }
}

static void add_brick(const bb_slots *state, const bb_step *step, bb_features *f,
                      int weight_is_one)
{
    const bb_step_hdr *h = step->hdr;
    uint32_t taken = 0u;
    int i, n = h->n_refs, k;

    for (i = 0; i < n; ++i) {
        if (f->slot[i] >= 0) {
            taken |= 1u << f->slot[i];
        }
    }
    k = bb_find_held(state, step->b_star, taken, 0.45);
    f->n = n + 1;
    f->slot[n] = k;
    if (k < 0) {
        f->centre[n][0] = 0.0;
        f->centre[n][1] = 0.0;
        f->weight[n] = 0.0;
        return;
    }
    bb_slot_centre(state, k, f->centre[n]);
    if (weight_is_one) {
        f->weight[n] = 1.0;
    } else {
        double b = bb_slot_border(state, k);
        f->weight[n] = (1.0 - b) > 0.0 ? (1.0 - b) : 0.0;
    }
}

static double pair_angle(const double *s)
{
    return atan2(s[3] - s[1], s[2] - s[0]);
}

static void centre_pairs(double *v, int n_pairs)
{
    double mx = 0.0, my = 0.0;
    int i;
    for (i = 0; i < n_pairs; ++i) {
        mx += v[2 * i];
        my += v[2 * i + 1];
    }
    mx /= (double)n_pairs;
    my /= (double)n_pairs;
    for (i = 0; i < n_pairs; ++i) {
        v[2 * i] -= mx;
        v[2 * i + 1] -= my;
    }
}

static void blind_out(bb_servo_out *out, int seen)
{
    int i;
    for (i = 0; i < 3; ++i) {
        out->delta[i] = 0.0;
        out->d_r[i] = 0.0;
    }
    out->d_b[0] = 0.0;
    out->d_b[1] = 0.0;
    out->err_m = NAN;
    out->done = 0;
    out->seen = seen;
    out->blind = 1;
    out->yaw_weight = 0.0;
}

void bb_servo_update(bb_servo *s, const bb_cfg *cfg, const bb_step *step,
                     const bb_slots *actual, const bb_slots *reference,
                     int dry_run, int relative, bb_servo_out *out)
{
    const bb_step_hdr *h = step->hdr;
    int names[BB_MAX_REFS];
    int n_names = 0;
    int brick_index = h->n_refs;
    int i, j;
    double sv[3];
    double jw[BB_MAX_FEAT * 3];
    double inv[3 * BB_MAX_FEAT];
    double ew[BB_MAX_FEAT];
    double sbuf[BB_MAX_FEAT], tbuf[BB_MAX_FEAT];
    double w[BB_MAX_FEAT];
    double err_vec[3], err_m, dead, n_xy;

    reference_features(actual, step, &out->found);
    if (h->has_brick_ref && h->brick_dim > 0) {
        add_brick(actual, step, &out->found, 0);
    }
    out->have_want = 0;
    if (reference != 0) {
        reference_features(reference, step, &out->want);
        if (h->has_brick_ref && h->brick_dim > 0) {
            add_brick(reference, step, &out->want, 1);
        }
        out->have_want = 1;
    }

    for (i = 0; i < h->n_feat; ++i) {
        int r = step->feat[i];
        if (out->found.slot[r] >= 0) {
            names[n_names++] = i;
        }
    }
    if (n_names < cfg->servo_min_refs) {
        blind_out(out, n_names);
        return;
    }
    if (h->has_brick_ref && s->require_brick && out->found.slot[brick_index] < 0) {
        blind_out(out, n_names);
        return;
    }

    if (out->have_want) {
        int keep = 0;
        for (i = 0; i < n_names; ++i) {
            int r = step->feat[names[i]];
            if (out->want.slot[r] >= 0) {
                names[keep++] = names[i];
            }
        }
        n_names = keep;
        if (n_names < cfg->servo_min_refs) {
            blind_out(out, n_names);
            return;
        }
        for (i = 0; i < n_names; ++i) {
            int r = step->feat[names[i]];
            tbuf[2 * i] = out->want.centre[r][0];
            tbuf[2 * i + 1] = out->want.centre[r][1];
        }
    } else {
        for (i = 0; i < n_names; ++i) {
            int f = names[i];
            tbuf[2 * i] = step->s_star[2 * f];
            tbuf[2 * i + 1] = step->s_star[2 * f + 1];
        }
    }
    for (i = 0; i < n_names; ++i) {
        int r = step->feat[names[i]];
        sbuf[2 * i] = out->found.centre[r][0];
        sbuf[2 * i + 1] = out->found.centre[r][1];
        w[2 * i] = out->found.weight[r];
        w[2 * i + 1] = out->found.weight[r];
    }
    if (relative) {
        centre_pairs(sbuf, n_names);
        centre_pairs(tbuf, n_names);
    }

    for (i = 0; i < 2 * n_names; ++i) {
        int f = names[i / 2];
        int rowi = 2 * f + (i & 1);
        for (j = 0; j < h->j_cols; ++j) {
            jw[i * h->j_cols + j] = step->J[rowi * h->j_cols + j] * w[i];
        }
        ew[i] = (sbuf[i] - tbuf[i]) * w[i];
    }
    bb_damped_inverse(jw, 2 * n_names, h->j_cols, cfg->servo_truncate_at, inv, sv);
    for (i = 0; i < 3; ++i) {
        double acc = 0.0;
        if (i < h->j_cols) {
            for (j = 0; j < 2 * n_names; ++j) {
                acc += inv[i * (2 * n_names) + j] * ew[j];
            }
        }
        out->d_r[i] = acc;
    }
    out->d_r[2] = 0.0;

    out->yaw_weight = 0.0;
    if (h->has_spin && fabs(h->spin_gain) > 0.2) {
        double z_obs[2], z_star[2];
        int have_obs = 0, have_star = 0;
        int fold = h->angle_period;

        if (h->angle_mode == 0 && n_names >= 2) {
            double star = out->have_want ? pair_angle(tbuf) : h->spin_star_re;
            out->d_r[2] = bb_wrap_pi(pair_angle(sbuf) - star) / h->spin_gain;
            out->yaw_weight = 1.0;
        } else if (h->angle_mode == 1 && h->angle_slot >= 0
                   && out->found.slot[h->angle_slot] >= 0) {
            bb_blob_orientation(actual, out->found.slot[h->angle_slot], z_obs, &fold);
            z_star[0] = h->spin_star_re;
            z_star[1] = h->spin_star_im;
            have_obs = have_star = 1;
            if (out->have_want && out->want.slot[h->angle_slot] >= 0) {
                int f2;
                bb_blob_orientation(reference, out->want.slot[h->angle_slot], z_star, &f2);
            }
        }
        if (have_obs && have_star) {
            double err, weight;
            bb_orientation_error(z_obs, z_star, fold, &err, &weight);
            out->yaw_weight = weight;
            if (weight >= cfg->servo_min_yaw_weight) {
                out->d_r[2] = err / h->spin_gain;
            }
        }
    }

    out->d_b[0] = 0.0;
    out->d_b[1] = 0.0;
    if (h->has_brick_ref && h->brick_dim > 0 && out->found.slot[brick_index] >= 0) {
        double b[BB_MAX_BROW], full[BB_MAX_BDOF];
        double inv_b[BB_MAX_BDOF * BB_MAX_BROW], sv_b[3];
        double d_b[2], n_b;

        bb_brick_features(actual, out->found.slot[brick_index], b);
        bb_damped_inverse(step->J_brick, h->brick_dim, h->brick_dof, cfg->servo_truncate_at,
                          inv_b, sv_b);
        for (i = 0; i < h->brick_dof; ++i) {
            double acc = 0.0;
            for (j = 0; j < h->brick_dim; ++j) {
                acc += inv_b[i * h->brick_dim + j] * (b[j] - step->b_star[j]);
            }
            full[i] = acc;
        }
        d_b[0] = full[0];
        d_b[1] = h->brick_dof > 1 ? full[1] : 0.0;
        n_b = sqrt(d_b[0] * d_b[0] + d_b[1] * d_b[1]);
        if (n_b > s->max_in_hand) {
            d_b[0] *= s->max_in_hand / n_b;
            d_b[1] *= s->max_in_hand / n_b;
        }
        if (dry_run || s->hold_in_hand) {
            if (s->has_d_b) {
                d_b[0] = s->d_b[0];
                d_b[1] = s->d_b[1];
            }
        } else {
            if (!s->has_d_b) {
                s->d_b[0] = d_b[0];
                s->d_b[1] = d_b[1];
                s->has_d_b = 1;
            } else if (s->n_b < cfg->servo_freeze_in_hand) {
                s->d_b[0] = (1.0 - cfg->servo_smooth_in_hand) * s->d_b[0]
                            + cfg->servo_smooth_in_hand * d_b[0];
                s->d_b[1] = (1.0 - cfg->servo_smooth_in_hand) * s->d_b[1]
                            + cfg->servo_smooth_in_hand * d_b[1];
            }
            s->n_b += 1;
            d_b[0] = s->d_b[0];
            d_b[1] = s->d_b[1];
        }
        out->d_b[0] = d_b[0];
        out->d_b[1] = d_b[1];
    }

    err_vec[0] = out->d_r[0] + out->d_b[0];
    err_vec[1] = out->d_r[1] + out->d_b[1];
    err_vec[2] = out->d_r[2];
    err_m = sqrt(err_vec[0] * err_vec[0] + err_vec[1] * err_vec[1]);
    dead = (n_names < 2 ? cfg->servo_dead_band_single_mm : cfg->servo_dead_band_mm) * 1e-3;
    if (err_m < dead) {
        err_vec[0] = 0.0;
        err_vec[1] = 0.0;
        err_vec[2] = 0.0;
    }
    for (i = 0; i < 3; ++i) {
        out->delta[i] = -s->lam * err_vec[i];
    }
    if (err_m < cfg->servo_near * h->tol) {
        for (i = 0; i < 3; ++i) {
            out->delta[i] *= 0.5;
        }
    }
    n_xy = sqrt(out->delta[0] * out->delta[0] + out->delta[1] * out->delta[1]);
    if (n_xy > cfg->servo_max_xy) {
        out->delta[0] *= cfg->servo_max_xy / n_xy;
        out->delta[1] *= cfg->servo_max_xy / n_xy;
    }
    if (out->delta[2] > s->max_spin) {
        out->delta[2] = s->max_spin;
    } else if (out->delta[2] < -s->max_spin) {
        out->delta[2] = -s->max_spin;
    }

    if (!dry_run) {
        s->hits = (err_m <= h->tol) ? s->hits + 1 : 0;
    }
    out->err_m = err_m;
    out->done = s->hits >= s->settle_ticks;
    out->seen = n_names;
    out->blind = 0;
}
