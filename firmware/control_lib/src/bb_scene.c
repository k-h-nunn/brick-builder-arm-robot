#include "bb_scene.h"
#include "bb_math.h"

#include <math.h>

#define ROUND_ENOUGH 1.3

static const float *row(const bb_slots *s, int k)
{
    return &s->slots[k * BB_W];
}

double bb_slot_mass(const bb_slots *s, int k)
{
    return (double)row(s, k)[BB_COL_MASS];
}

void bb_slot_centre(const bb_slots *s, int k, double c[2])
{
    const float *r = row(s, k);
    c[0] = (double)r[BB_COL_CENTRE];
    c[1] = (double)r[BB_COL_CENTRE + 1];
}

void bb_slot_moments(const bb_slots *s, int k, double m[3])
{
    const float *r = row(s, k);
    m[0] = (double)r[BB_COL_MOMENTS];
    m[1] = (double)r[BB_COL_MOMENTS + 1];
    m[2] = (double)r[BB_COL_MOMENTS + 2];
}

void bb_slot_colour(const bb_slots *s, int k, double c[3])
{
    const float *r = row(s, k);
    c[0] = (double)r[BB_COL_COLOUR];
    c[1] = (double)r[BB_COL_COLOUR + 1];
    c[2] = (double)r[BB_COL_COLOUR + 2];
}

double bb_slot_border(const bb_slots *s, int k)
{
    return (double)row(s, k)[BB_COL_BORDER];
}

void bb_slot_orient4(const bb_slots *s, int k, double z[2])
{
    const float *r = row(s, k);
    z[0] = (double)r[BB_COL_ORIENT4];
    z[1] = (double)r[BB_COL_ORIENT4 + 1];
}

int bb_slot_empty(const bb_slots *s, int k)
{
    const float *r = row(s, k);
    int i;
    if (s->depth[k] != 0.5f) {
        return 0;
    }
    for (i = 0; i < BB_W; ++i) {
        if (r[i] != 0.0f) {
            return 0;
        }
    }
    return 1;
}

void bb_slot_axes(const bb_slots *s, int k, double *angle, double axes[2])
{
    double m[3];
    bb_slot_moments(s, k, m);
    bb_ellipse(m[0], m[1], m[2], angle, axes);
}

int bb_select_slot(const bb_slots *s, const double colour[3], double colour_tol,
                   const double size[2], int has_size, double size_tol,
                   double min_mass, int has_min_mass, uint32_t exclude)
{
    int best = -1;
    double best_score = 0.0;
    int k;

    for (k = 0; k < BB_K; ++k) {
        double score = 0.0;
        if ((exclude >> k) & 1u) {
            continue;
        }
        if (bb_slot_empty(s, k)) {
            continue;
        }
        if (has_min_mass && bb_slot_mass(s, k) < min_mass) {
            continue;
        }
        if (colour != 0) {
            double c[3], d, e0, e1, e2;
            bb_slot_colour(s, k, c);
            e0 = c[0] - (2.0 * colour[0] - 1.0);
            e1 = c[1] - (2.0 * colour[1] - 1.0);
            e2 = c[2] - (2.0 * colour[2] - 1.0);
            d = sqrt(e0 * e0 + e1 * e1 + e2 * e2);
            if (d > colour_tol) {
                continue;
            }
            score += d;
        }
        if (has_size) {
            double axes[2], d, a0, a1, b0, b1, t;
            bb_slot_axes(s, k, 0, axes);
            /* Both pairs sorted ascending, as `select_slot` compares them. */
            a0 = size[0]; a1 = size[1];
            if (a0 > a1) { t = a0; a0 = a1; a1 = t; }
            b0 = axes[0]; b1 = axes[1];
            if (b0 > b1) { t = b0; b0 = b1; b1 = t; }
            d = hypot(a0 - b0, a1 - b1);
            if (d > size_tol) {
                continue;
            }
            score += d;
        }
        if (best < 0 || score < best_score) {
            best = k;
            best_score = score;
        }
    }
    return best;
}

int bb_find_held(const bb_slots *s, const double expected[2], uint32_t exclude,
                 double max_distance)
{
    int best = -1;
    double best_score = 0.0;
    int k;

    for (k = 0; k < BB_K; ++k) {
        double c[2], dx, dy, d, score;
        if ((exclude >> k) & 1u) {
            continue;
        }
        if (bb_slot_empty(s, k) || bb_slot_mass(s, k) > 0.5) {
            continue;
        }
        bb_slot_centre(s, k, c);
        dx = c[0] - expected[0];
        dy = c[1] - expected[1];
        d = sqrt(dx * dx + dy * dy);
        if (d > max_distance) {
            continue;
        }
        score = d + 0.05 * (double)s->depth[k];
        if (best < 0 || score < best_score) {
            best = k;
            best_score = score;
        }
    }
    return best;
}

void bb_blob_orientation(const bb_slots *s, int k, double z[2], int *fold)
{
    double axes[2], m[3], ratio;

    bb_slot_axes(s, k, 0, axes);
    ratio = axes[0] / (axes[1] > 1e-9 ? axes[1] : 1e-9);
    if (ratio < ROUND_ENOUGH) {
        double o[2];
        bb_slot_orient4(s, k, o);
        if (hypot(o[0], o[1]) > 0.05) {
            z[0] = o[0];
            z[1] = o[1];
            *fold = 4;
            return;
        }
    }
    bb_slot_moments(s, k, m);
    z[0] = m[0] - m[2];
    z[1] = 2.0 * m[1];
    *fold = 2;
}

void bb_orientation_error(const double z_obs[2], const double z_target[2], int fold,
                          double *err, double *weight)
{
    double re, im, ao, at;

    if ((z_obs[0] == 0.0 && z_obs[1] == 0.0) || (z_target[0] == 0.0 && z_target[1] == 0.0)) {
        *err = 0.0;
        *weight = 0.0;
        return;
    }
    /* z_obs * conj(z_target) */
    re = z_obs[0] * z_target[0] + z_obs[1] * z_target[1];
    im = z_obs[1] * z_target[0] - z_obs[0] * z_target[1];
    *err = atan2(im, re) / (double)fold;
    ao = hypot(z_obs[0], z_obs[1]);
    at = hypot(z_target[0], z_target[1]);
    *weight = ao < at ? ao : at;
}

void bb_brick_features(const bb_slots *s, int k, double out[BB_MAX_BROW])
{
    double c[2], m[3], o[2];
    bb_slot_centre(s, k, c);
    bb_slot_moments(s, k, m);
    bb_slot_orient4(s, k, o);
    out[0] = c[0];
    out[1] = c[1];
    out[2] = bb_slot_mass(s, k);
    out[3] = m[0];
    out[4] = m[1];
    out[5] = m[2];
    out[6] = o[0];
    out[7] = o[1];
}
