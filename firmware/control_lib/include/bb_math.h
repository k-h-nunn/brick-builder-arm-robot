/* bb_math.h: the small dense linear algebra the controller needs, in fixed-size buffers.
 *
 * Matrices are row-major doubles. Every routine here has a bounded iteration count and no
 * allocation, so a tick's worst case is a fixed number of multiply-accumulates.
 */
#ifndef BB_MATH_H
#define BB_MATH_H

#include "bb_config.h"

/* Eigen-decomposition of a 2x2 symmetric matrix [[a, b], [b, c]], eigenvalues ascending, as
 * LAPACK's dsyevd returns them. `evec0` is the eigenvector of the smaller eigenvalue. */
void bb_eigh2(double a, double b, double c, double eval[2], double evec0[2], double evec1[2]);

/* Orientation [rad] and (major, minor) axis lengths of the second-moment ellipse. */
void bb_ellipse(double sxx, double sxy, double syy, double *angle, double axes[2]);

/* One-sided Jacobi SVD of an m x n matrix (n <= 3, m <= BB_MAX_FEAT).
 * Writes singular values descending into `sv[n]`, the left factor into `u[m*n]` (column i is
 * u[:, i]) and the right factor into `vt[n*n]` (row i is v_i^T). */
void bb_svd(const double *a, int m, int n, double *u, double *sv, double *vt);

/* Truncated pseudo-inverse: singular values below `relative` times the largest are dropped.
 * `inv` is n x m. Returns the number of singular values kept and fills `sv[n]`. */
int bb_damped_inverse(const double *a, int m, int n, double relative, double *inv, double *sv);

/* Solve `a x = b` for an n x n system by Gaussian elimination with partial pivoting.
 * `a` is destroyed. Returns BB_OK, or BB_ERR_LAYOUT on a singular pivot. */
int bb_solve(double *a, double *b, int n);

double bb_norm2(const double *v, int n);
double bb_wrap_pi(double a);

#endif /* BB_MATH_H */
