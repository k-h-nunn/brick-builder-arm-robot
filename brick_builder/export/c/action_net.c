/* action_net.c: the closed-loop action models' forward pass, in plain C99.
 *
 * Two networks, one file: `action_inverse` is the one-step inverse the controller calls once per
 * vision tick to turn a slot-state difference into a tool correction, and `action_predict` is the
 * next-scene-state predictor. Both are the same shape -- normalize, then Linear -> tanh -> ... ->
 * Linear with no activation on the last layer, then scale the output back into physical units --
 * so one static helper serves both and only the tables differ.
 *
 * Everything checkpoint-specific lives in the generated `action_theta.h` this file includes
 * (brick_builder/export/action_to_c.py), never in here. No dynamic allocation: the two scratch
 * buffers are fixed-size stack arrays sized by ACTION_MAX_WIDTH, which the generator sets from
 * the widest layer of either network. The only libm symbol is `tanhf`, and -DACTION_TANH_TABLE
 * replaces that with the linear-interpolation lookup table the actor export already generates
 * (tanh_table.h), leaving no libm dependence at all.
 *
 * The normalizer divides rather than multiplying by a stored reciprocal, because the NumPy
 * reference (brick_builder/action/models/mlp.py) divides: the two then agree bit for bit on the
 * normalizer and parity is limited only by the order of the dot products.
 *
 * Each dot product accumulates in double. The predictor's outputs reach 2.6 in magnitude, where
 * three float32 units in the last place is already 1.2e-6, so a float accumulator puts the C and
 * the NumPy reference on opposite sides of the 1e-6 parity bound for no reason other than
 * summation order. Building with -DACTION_ACC_FLOAT accumulates in float instead, for a device
 * with no double at all: it costs a few 1e-6 on the predictor's largest outputs and nothing
 * measurable on the inverse's, whose answers are millimetres of tool motion.
 */
#include "action_net.h"
#include "action_theta.h"

#ifdef ACTION_ACC_FLOAT
typedef float action_acc_t;
#else
typedef double action_acc_t;
#endif

#ifdef ACTION_TANH_TABLE
#include "tanh_table.h"

static float action_tanh(float x)
{
    if (x <= TANH_TABLE_LO) return -1.0f;
    if (x >= TANH_TABLE_HI) return 1.0f;
    float pos = (x - TANH_TABLE_LO) * (float)(TANH_TABLE_N - 1) / (TANH_TABLE_HI - TANH_TABLE_LO);
    int i = (int)pos;
    if (i > TANH_TABLE_N - 2) i = TANH_TABLE_N - 2;
    float frac = pos - (float)i;
    return tanh_table[i] + frac * (tanh_table[i + 1] - tanh_table[i]);
}
#else
#include <math.h>

static float action_tanh(float x)
{
    return tanhf(x);
}
#endif

static void action_forward(const float *x, float *y,
                           int n_layers, const int *layer_in, const int *layer_out,
                           const float *const *weights, const float *const *biases,
                           const float *norm_mean, const float *norm_std, float clip,
                           const float *out_mean, const float *out_std)
{
    float buf_a[ACTION_MAX_WIDTH];
    float buf_b[ACTION_MAX_WIDTH];
    float *cur = buf_a;
    float *scratch = buf_b;
    int i, j, layer;

    for (i = 0; i < layer_in[0]; i++) {
        float v = (x[i] - norm_mean[i]) / norm_std[i];
        if (v > clip) v = clip;
        if (v < -clip) v = -clip;
        cur[i] = v;
    }

    for (layer = 0; layer < n_layers; layer++) {
        const float *w = weights[layer];
        const float *b = biases[layer];
        int n_in = layer_in[layer];
        int n_out = layer_out[layer];
        for (i = 0; i < n_out; i++) {
            action_acc_t acc = (action_acc_t)b[i];
            const float *row = w + (long)i * n_in;
            for (j = 0; j < n_in; j++) {
                acc += (action_acc_t)row[j] * (action_acc_t)cur[j];
            }
            scratch[i] = (layer < n_layers - 1) ? action_tanh((float)acc) : (float)acc;
        }
        {
            float *swap = cur;
            cur = scratch;
            scratch = swap;
        }
    }

    for (i = 0; i < layer_out[n_layers - 1]; i++) {
        y[i] = cur[i] * out_std[i] + out_mean[i];
    }
}

void action_inverse(const float *x, float *y)
{
    action_forward(x, y, ACTION_INV_N_LAYERS, action_inv_layer_in, action_inv_layer_out,
                   action_inv_weights, action_inv_biases, action_inv_norm_mean,
                   action_inv_norm_std, ACTION_INV_CLIP, action_inv_out_mean,
                   action_inv_out_std);
}

void action_predict(const float *x, float *y)
{
    action_forward(x, y, ACTION_PRED_N_LAYERS, action_pred_layer_in, action_pred_layer_out,
                   action_pred_weights, action_pred_biases, action_pred_norm_mean,
                   action_pred_norm_std, ACTION_PRED_CLIP, action_pred_out_mean,
                   action_pred_out_std);
}
