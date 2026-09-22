#include "bb_math.h"

#include <math.h>
#include <string.h>

double bb_norm2(const double *v, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        s += v[i] * v[i];
    }
    return sqrt(s);
}

double bb_wrap_pi(double a)
{
    return remainder(a, 2.0 * BB_PI);
}

void bb_eigh2(double a, double b, double c, double eval[2], double evec0[2], double evec1[2])
{
    /* Closed form for [[a, b], [b, c]]. Eigenvalues ascending, eigenvectors unit and orthogonal,
     * which is the contract `numpy.linalg.eigh` publishes for a 2x2 symmetric input. */
    double tr = a + c;
    double diff = a - c;
    double disc = sqrt(diff * diff + 4.0 * b * b);
    double lo = 0.5 * (tr - disc);
    double hi = 0.5 * (tr + disc);
    double vx, vy, n;

    eval[0] = lo;
    eval[1] = hi;

    if (b == 0.0) {
        /* Already diagonal: the axes themselves, ordered to match the eigenvalues. */
        if (a <= c) {
            evec0[0] = 1.0; evec0[1] = 0.0;
            evec1[0] = 0.0; evec1[1] = 1.0;
        } else {
            evec0[0] = 0.0; evec0[1] = 1.0;
            evec1[0] = 1.0; evec1[1] = 0.0;
        }
        return;
    }

    /* (A - lambda I) v = 0 with the numerically larger row. */
    vx = b;
    vy = hi - a;
    n = sqrt(vx * vx + vy * vy);
    if (n <= 0.0) {
        evec1[0] = 1.0;
        evec1[1] = 0.0;
    } else {
        evec1[0] = vx / n;
        evec1[1] = vy / n;
    }
    evec0[0] = -evec1[1];
    evec0[1] = evec1[0];
}

void bb_ellipse(double sxx, double sxy, double syy, double *angle, double axes[2])
{
    double eval[2], v0[2], v1[2];
    double major, minor;

    bb_eigh2(sxx, sxy, syy, eval, v0, v1);
    /* `orientation_and_axes` sorts descending and takes the leading eigenvector's direction. */
    major = eval[1] < 0.0 ? 0.0 : eval[1];
    minor = eval[0] < 0.0 ? 0.0 : eval[0];
    if (angle != 0) {
        *angle = atan2(v1[1], v1[0]);
    }
    axes[0] = 2.0 * sqrt(major);
    axes[1] = 2.0 * sqrt(minor);
}

void bb_svd(const double *a, int m, int n, double *u, double *sv, double *vt)
{
    /* One-sided Jacobi: rotate pairs of columns until they are orthogonal. The column norms are
     * then the singular values and the accumulated rotations are V. Converges quadratically; the
     * sweep bound is a hard stop, not a tolerance to tune. */
    double w[BB_MAX_FEAT * 3];
    double v[9];
    int i, j, k, sweep, p, q;
    int idx[3];

    for (i = 0; i < m * n; ++i) {
        w[i] = a[i];
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            v[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (sweep = 0; sweep < 30; ++sweep) {
        double off = 0.0;
        for (p = 0; p < n - 1; ++p) {
            for (q = p + 1; q < n; ++q) {
                double app = 0.0, aqq = 0.0, apq = 0.0;
                double zeta, t, cs, sn;
                for (k = 0; k < m; ++k) {
                    app += w[k * n + p] * w[k * n + p];
                    aqq += w[k * n + q] * w[k * n + q];
                    apq += w[k * n + p] * w[k * n + q];
                }
                off += apq * apq;
                if (apq == 0.0) {
                    continue;
                }
                zeta = (aqq - app) / (2.0 * apq);
                t = (zeta >= 0.0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                cs = 1.0 / sqrt(1.0 + t * t);
                sn = cs * t;
                for (k = 0; k < m; ++k) {
                    double wp = w[k * n + p];
                    double wq = w[k * n + q];
                    w[k * n + p] = cs * wp - sn * wq;
                    w[k * n + q] = sn * wp + cs * wq;
                }
                for (k = 0; k < n; ++k) {
                    double vp = v[k * n + p];
                    double vq = v[k * n + q];
                    v[k * n + p] = cs * vp - sn * vq;
                    v[k * n + q] = sn * vp + cs * vq;
                }
            }
        }
        if (off <= 1e-30) {
            break;
        }
    }

    for (i = 0; i < n; ++i) {
        double s = 0.0;
        for (k = 0; k < m; ++k) {
            s += w[k * n + i] * w[k * n + i];
        }
        sv[i] = sqrt(s);
        idx[i] = i;
    }
    /* Descending, as numpy publishes them. */
    for (i = 0; i < n - 1; ++i) {
        for (j = i + 1; j < n; ++j) {
            if (sv[idx[j]] > sv[idx[i]]) {
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }

    {
        double svs[3];
        for (i = 0; i < n; ++i) {
            int c = idx[i];
            svs[i] = sv[c];
            for (k = 0; k < m; ++k) {
                u[k * n + i] = (sv[c] > 0.0) ? w[k * n + c] / sv[c] : 0.0;
            }
            for (k = 0; k < n; ++k) {
                vt[i * n + k] = v[k * n + c];
            }
        }
        for (i = 0; i < n; ++i) {
            sv[i] = svs[i];
        }
    }
}

int bb_damped_inverse(const double *a, int m, int n, double relative, double *inv, double *sv)
{
    double u[BB_MAX_FEAT * 3];
    double vt[9];
    int i, r, c, kept = 0;

    for (i = 0; i < n * m; ++i) {
        inv[i] = 0.0;
    }
    if (m <= 0 || n <= 0) {
        for (i = 0; i < n; ++i) {
            sv[i] = 0.0;
        }
        return 0;
    }
    bb_svd(a, m, n, u, sv, vt);
    for (i = 0; i < n; ++i) {
        int keep = (sv[0] > 0.0) && (sv[i] >= relative * sv[0]);
        if (!keep) {
            continue;
        }
        kept += 1;
        /* inv += outer(vt[i], u[:, i]) / sv[i] */
        for (r = 0; r < n; ++r) {
            double vr = vt[i * n + r] / sv[i];
            for (c = 0; c < m; ++c) {
                inv[r * m + c] += vr * u[c * n + i];
            }
        }
    }
    return kept;
}

int bb_solve(double *a, double *b, int n)
{
    int i, j, k;
    for (k = 0; k < n; ++k) {
        int piv = k;
        double best = fabs(a[k * n + k]);
        for (i = k + 1; i < n; ++i) {
            double v = fabs(a[i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best == 0.0) {
            return BB_ERR_LAYOUT;
        }
        if (piv != k) {
            double t;
            for (j = 0; j < n; ++j) {
                t = a[k * n + j];
                a[k * n + j] = a[piv * n + j];
                a[piv * n + j] = t;
            }
            t = b[k];
            b[k] = b[piv];
            b[piv] = t;
        }
        for (i = k + 1; i < n; ++i) {
            double f = a[i * n + k] / a[k * n + k];
            a[i * n + k] = 0.0;
            for (j = k + 1; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    for (i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (j = i + 1; j < n; ++j) {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
    return BB_OK;
}
