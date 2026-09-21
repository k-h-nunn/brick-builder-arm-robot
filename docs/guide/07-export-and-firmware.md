# Export and firmware

**Status: the policy export (`brick_builder/export/`) is done; the firmware tree is still planned.**
This page's export section below is current; its firmware sections will gain build instructions and
frame-time measurements as Stage F lands.

## Purpose

This is how a trained policy gets from a PyTorch model on a laptop into a small piece of C code
that can run on the Waveshare ESP32-P4-Pico microcontroller — the board that will eventually run
the robot's real-time perception and control loop.

## Policy export (`brick_builder/export/`)

Everything here works on any checkpoint written by `brick_builder.rl.ppo_asym.train` — the actor MLP
(`brick_builder/rl/nets.py`: tanh hidden layers, linear output, a fixed per-action `log_std`, no squashing on
the mean) plus its frozen input normalizer (`RunningNorm`: mean/var running statistics, clipped to
±`norm_clip`). The exported C code reproduces the exact computation the torch side uses at evaluation
time (`brick_builder.rl.ppo_asym.evaluate`): normalize the observation, take the Gaussian mean (no sampling),
clamp to the action bounds.

### `actor_to_c.py`

`write_header(checkpoint, out_dir)` reads a checkpoint and writes `policy_theta.h` into `out_dir`: every
layer's weight matrix and bias vector as `static const float` arrays (row-major, `(out, in)`, matching
`torch.nn.Linear.weight` exactly), the normalizer's mean and a **precomputed** `1/sqrt(var + eps)` (so the
C code never calls `sqrtf`), the action bounds, the layer-size macros `policy.c` needs to size its
buffers, and the SHA-256 digest of the checkpoint file for provenance. `write_tanh_table(out_dir)` writes
`tanh_table.h`: `tanh` at 4097 evenly spaced knots on `[-8, 8]`, independent of any checkpoint. Both are
generated files; they are written into a build directory (`brick_builder/export/_build/` by default, git-ignored),
never into `brick_builder/export/c/`, which holds only the hand-written, checkpoint-independent sources.

### The C policy (`brick_builder/export/c/policy.c`, `policy.h`)

A plain C99 forward pass, `void policy_act(const float *obs, float *act)`: normalize, then
`Linear -> tanh -> Linear -> tanh -> ... -> Linear` (no activation on the output layer) using two
fixed-size stack buffers (`POLICY_MAX_WIDTH` floats each, generated from the checkpoint's own layer sizes)
that ping-pong as the input to one layer and the output of the next — no `malloc`, no heap allocation at
all. The only libm call anywhere is `tanhf`, and even that is switchable: build with `-DPOLICY_TANH_TABLE`
and `tanhf` is replaced by a linear-interpolation lookup into `tanh_table.h`, which needs no libm symbol.
This is the same file that will later be dropped, unchanged, into the ESP32-P4 firmware tree — everything
checkpoint-specific lives in the generated `policy_theta.h` it includes, not in `policy.c` itself.

### `build.py`

`build_library(checkpoint, build_dir, tanh_table=False)` generates the headers above and compiles
`policy.c` against them with the system `gcc`:

```
gcc -O2 -std=c99 -ffp-contract=off -shared -fPIC [-DPOLICY_TANH_TABLE] -I c -I <build_dir> \
    c/policy.c -o <build_dir>/policy_float.so   # or policy_table.so, and -lm only for the float build
```

`-ffp-contract=off` turns off fused multiply-add contraction, which matters for matching torch's CPU
kernels bit-for-bit-ish rather than only to a looser tolerance.

### `CPolicy` and parity (`c_policy.py`, `parity.py`)

`CPolicy(lib_path, obs_dim, act_dim)` loads the shared library with `ctypes` and exposes `act(obs)` —
the same call the torch actor's deterministic path makes. `check_parity(checkpoint, cpolicy, n=10_000)`
draws `n` random observations from the checkpoint's own normalizer statistics (so they land in the range
the policy actually saw), runs both the torch actor and `CPolicy` on them, and reports the max and mean
absolute error. `benchmark_inference(cpolicy, obs, n_calls)` times `act()` from Python with
`time.perf_counter_ns` and returns the median nanoseconds per call.

### Measured numbers

Measured on this development machine (CPU only), with a checkpoint from a handful of PPO updates on the
`Pendulum-v1` wrapper used by `tests/rl` (spec section 5: test with this until a real Brick Builder Arm Robot checkpoint
exists) — actor shape 3 → 64 → 64 → 1, 3 `Linear` layers, 17 924 bytes of `float32` weights:

| build            | parity vs. torch (max abs error, n=10 000) | median ns / call |
|------------------|---------------------------------------------|-------------------|
| float `tanhf`    | ~3e-8                                        | ~11 400           |
| table tanh       | ~2e-7                                        | ~8 600            |

Both are far inside the 1e-5 acceptance bound from spec section 5. The lookup table itself (4097 knots,
linear interpolation, `[-8, 8]`) has a max absolute error against `math.tanh` of about 1.5e-6, sampled
densely between knots — this is `table_tanh_max_error()` in `actor_to_c.py`. The timing includes Python
`ctypes` call overhead on both sides equally; the real firmware calling `policy_act` directly from C will
not pay it, so these numbers should be read as an upper bound, and as a same-machine comparison between
the two tanh variants rather than an absolute figure for the P4. Exact reproducible numbers for any run are
printed by `tests/export/test_actor_to_c.py::test_table_tanh_error_and_speed` (`pytest -s`).

### Later: the ESP32-P4 firmware build

`brick_builder/export/c/policy.c` and `policy.h` are written to have zero dependencies beyond a C99 compiler and
(optionally) `tanhf`: no heap, no floating-point functions the table variant cannot replace, and every
checkpoint-specific number isolated in a separate generated header. That means bringing the policy onto
the P4 is expected to be a matter of copying `policy.c`, `policy.h` and a freshly generated
`policy_theta.h` into an ESP-IDF component and calling `policy_act` from the 50 Hz control task — no
change to the C source itself. Which tanh variant the firmware uses (float vs. table) is a build-time
choice, `-DPOLICY_TANH_TABLE`, made once real closed-loop success-rate numbers on hardware say whether the
table's extra ~2e-7 of error costs anything.

![parity and speed of the exported C actor, float tanhf vs. lookup-table tanh](../media/export/parity_and_speed.png)

## Sections to be filled in as the rest of the code lands

- **The perception C kernel** — the small, fixed interface the firmware calls, and how a model
  change stays cheap because everything downstream depends only on that interface.
- **Parity fixtures** — how test vectors are generated from a trained model, and what "bit-exact"
  and "1e-5 tolerance" mean at each stage of the port.
- **Firmware build** — ESP-IDF setup, chip revision considerations, and the core assignment (camera and
  perception on one core, the 50 Hz control task on the other).
- **Frame-time matrix** — measured (not estimated) timing for perception and the control loop on real
  hardware, across the variants considered (floating point vs. a lookup-table approximation, one core
  vs. two).
