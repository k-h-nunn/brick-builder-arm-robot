# Environments

**Status: base built, one task landed.** `brick_builder/envs/obs_spec.py`, `base_env.py` and
`wrappers.py` exist and are shared by every task; `reach_env.py` (`BrickBuilder/Reach-v0`) is the
first task built on them, used to prove the whole multi-rate loop end to end.
`press_env.py` (`BrickBuilder/PressFromHeld-v0`) is the second, together with the scripted skills in
`brick_builder/control/` that it is built on; both have their own sections below.
`AlignFromTokens-v0` subclasses the same `ArmEnv` described here.

!!! note "Measurement status"
    Results on this page that depend on arm positioning or press success (scripted press,
    `PressFromHeld`, `AlignFromTokens`, the positioning dead band, deflection under load) were
    measured before the servo model's encoder was moved to the output shaft
    ([Robot and servo model](03-robot-and-servo-model.md)) and have not been re-measured.
    Current closed-loop results are in [Closed-loop videos](../demos/videos.md).

## Purpose

Brick Builder Arm Robot's RL tasks are [Gymnasium](https://gymnasium.farama.org/) environments:
each defines observation, action space, reward, and termination. All tasks share one base class,
`ArmEnv`.

```python
import gymnasium as gym
import brick_builder.envs  # registers every BrickBuilder/*-v0 id as a side effect

env = gym.make("BrickBuilder/Reach-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The multi-rate control tick

Physics steps at 2 ms; the control loop runs at 20 ms (50 Hz) — ten physics steps per
`env.step()`. `BusEmulator` sets each tick's timing (design report 02, section 0); `ServoBus`
follows it. A lost write holds the previous goal for the tick; a lost reply holds over the
previous tick's reading (flagged in the critic observation, never replaced with sim truth).

```
one env.step() = one 20 ms control tick
┌───────────────────────────────────────────────────────────────────────────┐
│ BusEmulator.tick(t) -> timing (command latency, per-servo read skew,      │
│                                which of this tick's packets are "lost")   │
│                                                                            │
│ ServoBus.write(goal, torque_limit)     <- skipped if timing.write_lost   │
│                                                                            │
│  physics step 1  (2 ms): ServoBus.substep(); snap.update(); mj_step()    │
│  physics step 2  (2 ms): ServoBus.substep();                mj_step()    │
│  ...                                                                      │
│  physics step 10 (2 ms): ServoBus.substep(); snap.update(); mj_step()    │
│                                                                            │
│ ServoBus.read()   <- per servo, held over from the last tick where       │
│                       timing.reply_lost[i] is true                       │
└───────────────────────────────────────────────────────────────────────────┘
```

`snap.update()` runs every `SnapParams.snap_every` physics steps, not every step: at 500 Hz it
would dominate the tick past a few bricks. Default `snap_every = 1`: 124 µs/call for a 2×4 on an
8×8 plate (13 % of a `PressFromHeld` tick), 0.8 µs/call for a brick-free scene (`Reach`).

## Action: 8 numbers in `[-1, 1]`

| index | name | scale | meaning |
|---|---|---|---|
| 0 | `dx` | 2 mm | world-frame TCP position delta |
| 1 | `dy` | 2 mm | world-frame TCP position delta |
| 2 | `dz` | 2 mm | world-frame TCP position delta |
| 3 | `d_tilt` | 2° | tool pitch offset from tool-down, in the arm's radial plane |
| 4 | `d_spin` | 2° | tool yaw about its own approach axis |
| 5 | `d_grip` | 0.05 rad | gripper joint angle delta |
| 6 | `c_vert` | -- | torque-limit fraction in `[0.3, 1.0]` for shoulder_lift, elbow_flex, wrist_flex |
| 7 | `c_lat` | -- | torque-limit fraction in `[0.3, 1.0]` for shoulder_pan, wrist_roll |

Indices 0-5 are **residuals**: they accumulate each tick onto a running total, not onto the
measured pose. Position/tilt/spin clip to a box (±12 mm, ±20°) around the scripted primitive's
target pose for that tick (`ArmEnv.primitive_target`); with no primitive (`primitive_target`
returns `None`, e.g. `Reach`) the box centre is fixed at the reset pose, so a zero-output policy
reproduces the scripted baseline exactly. Indices 6-7 map directly to a torque-limit fraction, no
accumulation: a policy uses them to soften the arm in one direction — letting studs self-centre,
or capping press force — without force control.

A task can freeze any of `{"pos", "tilt", "spin", "grip", "compliance"}` (`ArmEnv.frozen`).
`Reach` freezes `{"grip", "compliance"}`.

### Gravity feedforward

`ServoBus` is a plain position servo (`kv=0`, no gravity term). At `p_gain=16` the arm settles
tens of millimetres off a commanded joint target under its own weight (static-sag figure,
`docs/guide/03-robot-and-servo-model.md`). Droop of a few millimetres is within a press task's
tolerance but exceeds `Reach`'s 3 mm success band. Fix: a gravity feedforward term
(`_gravity_feedforward`), command = `q + gravity_torque(q) / kp`, iterated twice. At the home
pose: static error 12 mm → 4-5 mm (`docs/media/robot/sag.png`). The remaining 4-5 mm is Coulomb
friction (`dof_frictionloss`, dominant below about 0.3 N·m), not gravity — a kinematic feedforward
cannot remove it; only closed-loop corrective actions cross a friction dead-band.

## Observations: `Dict{"actor": Box, "critic": Box}`

Every actor entry is declared in `brick_builder/envs/obs_spec.py` with a real-hardware source
(`register:<address>`, `fk`, `command_echo`, `task`, `token`, or `fsm`); `tests/envs/test_obs_spec.py`
fails on any actor entry with no source or a `"sim"` (privileged) source. Proprioceptive entries
(servo register, or forward kinematics of read positions) carry a 4-tick history, oldest first;
command-echo and task entries do not. The critic adds: which servo reads were held over from a
lost reply, this tick's command latency and read skew, whether IK converged, and this episode's
domain-randomization draw (`brick_builder/scene/randomize.py`'s `dynamics`/`command` groups, 17
numbers, always drawn; at `level=0` these equal nominal values, so critic observation size is
constant across the curriculum).

Tables below are generated from `obs_spec.markdown_table(env.actor_names)` /
`.markdown_table(env.critic_names)`; `tests/envs/test_reach_env.py::test_docs_page_matches_the_generated_observation_tables`
checks this page matches the code.

### `BrickBuilder/Reach-v0` actor observation (120 numbers)

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `joint_pos` | 6 | 4 | `register:56 (Present_Position)` | actor | ServoBus.read().position, one number per joint in servo-ID order |
| `joint_vel` | 6 | 4 | `register:58 (Present_Speed)` | actor | ServoBus.read().speed |
| `joint_load` | 6 | 4 | `register:60 (Present_Load)` | actor | ServoBus.read().load -- commanded PWM duty in [-1, 1], not a torque |
| `tcp_pos` | 3 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos, position only |
| `tcp_rot6` | 6 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos: first two columns of the rotation matrix (continuous 6D rotation representation, Zhou et al. 2019) |
| `cmd_pose` | 5 | 1 | `command_echo` | actor | our own last commanded TCP target: x, y, z, tilt, spin |
| `cmd_grip` | 1 | 1 | `command_echo` | actor | our own last commanded gripper joint angle |
| `cmd_compliance` | 2 | 1 | `command_echo` | actor | our own last commanded c_vert, c_lat torque-limit fractions |
| `reach_goal` | 3 | 1 | `task` | actor | target TCP position minus the home TCP position, in metres; drawn once at reset and handed to the policy as task context, not measured |
| `reach_time_frac` | 1 | 1 | `task` | actor | fraction of the episode horizon elapsed (tick / horizon) |

### `BrickBuilder/Reach-v0` critic observation (148 numbers)

All actor entries above, plus six sim-only entries at the end (`read_stale` onward):

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `joint_pos` | 6 | 4 | `register:56 (Present_Position)` | actor | ServoBus.read().position, one number per joint in servo-ID order |
| `joint_vel` | 6 | 4 | `register:58 (Present_Speed)` | actor | ServoBus.read().speed |
| `joint_load` | 6 | 4 | `register:60 (Present_Load)` | actor | ServoBus.read().load -- commanded PWM duty in [-1, 1], not a torque |
| `tcp_pos` | 3 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos, position only |
| `tcp_rot6` | 6 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos: first two columns of the rotation matrix (continuous 6D rotation representation, Zhou et al. 2019) |
| `cmd_pose` | 5 | 1 | `command_echo` | actor | our own last commanded TCP target: x, y, z, tilt, spin |
| `cmd_grip` | 1 | 1 | `command_echo` | actor | our own last commanded gripper joint angle |
| `cmd_compliance` | 2 | 1 | `command_echo` | actor | our own last commanded c_vert, c_lat torque-limit fractions |
| `reach_goal` | 3 | 1 | `task` | actor | target TCP position minus the home TCP position, in metres; drawn once at reset and handed to the policy as task context, not measured |
| `reach_time_frac` | 1 | 1 | `task` | actor | fraction of the episode horizon elapsed (tick / horizon) |
| `read_stale` | 6 | 1 | `sim` | critic | per servo: this tick's sync-read reply was lost and the previous reading was reused |
| `write_dropped` | 1 | 1 | `sim` | critic | this tick's sync-write was lost on the emulated bus (BusEmulator.tick().write_lost) |
| `bus_timing` | 2 | 1 | `sim` | critic | this tick's command_latency_s, read_skew_s from BusEmulator |
| `ik_failed` | 1 | 1 | `sim` | critic | this tick's IK did not converge; the previous joint goal was kept |
| `dr_params` | 17 | 1 | `sim` | critic | this episode's dynamics.* (8) and command.* (9) randomize.sample() draw, in RANGES order |
| `reach_true_error` | 1 | 1 | `sim` | critic | true (simulator) distance from the TCP to the goal, metres |

## `BrickBuilder/Reach-v0`

Moves the TCP to a random reachable target from proprioception alone: no scripted primitive, no
bricks, no vision; grip and compliance frozen. The residual clips to a box around the reset pose
(fixed, since there is no primitive); the target is drawn from inside that same box
(`GOAL_MARGIN = 0.8`).

- **Horizon:** 100 ticks (2 s).
- **Success:** true TCP position within 3 mm of the target for 10 consecutive ticks.
- **Reward:** `-20 * distance`, a small penalty on the residual's own magnitude, plus a one-off
  `+10` bonus on the tick the within-tolerance streak first reaches 10.
- **Scene:** the patched SO-101 arm, the table and the empty 8x8 baseplate; no bricks.

## Wrappers (`brick_builder/envs/wrappers.py`)

| wrapper | purpose |
|---|---|
| `HistoryStack(env, k)` | stack the last `k` raw `Dict` observations (on top of, not instead of, the 4-tick history the shared proprioceptive entries already carry) |
| `ObservationDelay(env, delay_ticks)` | return the observation from `delay_ticks` ago; the deployable use of `randomize.sample(...)["command.obs_delay_ticks"]` |
| `ActionRepeatLatency(env, repeat, latency_ticks)` | repeat an action for `repeat` ticks and/or delay when it takes effect by `latency_ticks` |
| `EpisodeStats(env, window)` | attach `info["episode"] = {"r", "l", "success", "success_rate"}` at episode end, `success_rate` over a rolling window |

## Measured numbers (Reach)

- Single-process `env.step()` rate, random actions: **1 538 steps/s**
  (`brick_builder.envs.base_env.measure_step_rate`, 2 000 steps after a 200-step warm-up, load
  average 1.1; full table and tick breakdown: [Throughput](#throughput)).
- PPO (`brick_builder.rl.ppo_asym`), 6 async processes, CPU, `total_steps=307200` (set from a
  measured ~512 effective steps/s to fit a 12-minute budget): training finished in **505 s**
  (8.4 min), mean return rising from -20.6 (update 0) to a plateau near +8.4 (update ~150 of 200;
  `docs/media/envs/reach_learning_curve.png`). Evaluated over 200 held-out episodes (seeds
  10000-10199, deterministic actions): **200/200 successes**, Wilson interval [0.981, 1.0] (above
  the 90 % target), mean final TCP error 1.02 mm, median 0.98 mm — both below the 3 mm success
  band and the roughly 4-5 mm Coulomb-friction deadband. Checkpoint, TensorBoard log, per-update
  metrics:
  `runs/reach_smoke/` (git-ignored).

## Scripted press and the press environment

`brick_builder/control/primitives.py` and `brick_builder/control/skill_fsm.py` hold the scripted
skills; `brick_builder/envs/press_env.py` wraps the press skill as `BrickBuilder/PressFromHeld-v0`.
Task: a brick already in the gripper, a few millimetres above its target cell and misaligned; the
arm presses it in blind, on contact alone. Skill behind Deliverable 1.

### Scripted primitive inputs

A primitive receives one `Feedback` per tick: forward kinematics of the servo-read joint
positions, the `Present_Load` register, the commanded gripper angle, and the overload flag —
nothing else. `brick_builder/control/primitives.py` imports no `mujoco`, scene, or environments
(enforced by test): a fair baseline for the learned skill, and portable to the arm's own
microcontroller.

Produces a `ToolTarget`: world TCP position, tool tilt and spin, gripper angle, two compliance
scalars — the primitive half of `executed target = primitive target + bounded residual`.

### `MoveTo`

Straight-line Cartesian interpolation at fixed speed, then a settle. Finishes when the tool is
within `tol` of the target or `settle_s` elapses (a soft servo never reaches zero error).

### `PressScripted`

Descends at 10 mm/s from the tool's start pose, `c_lat` reduced to 0.45 so a stud on a chamfer can
shove the tool aside and self-centre. Senses only that descent has stopped, not misalignment
direction.

| phase | what it does |
|---|---|
| `descend` | the 10 mm/s ramp. Ends `seated` when the measured tool reaches the seated height, or `lift` when the descent stalls under load well above it. |
| `rock` | optional (`PressConfig.rocking`): the same ramp held at 4° until one stud row engages, then levelled out. |
| `lift` | a 2 mm hop to unload the brick before it is skimmed sideways. |
| `search` | an outward spiral at the skim height; ends the moment the tool drops into the sockets. |
| `seated` | done, holding the pressed pose. |
| `stuck` | gave up — saturated load, searches exhausted, or the press clock ran out. Holds where it is rather than pushing harder. |

Three parameters were measured, not chosen:

- **`press_depth` = 10 mm.** No force loop: overshoot past the seated height is the only control
  on press force. At the plate centre: 6 mm → 9 N (2× the seat time); 10 mm → 14 N; 20 mm → 25 N
  (exceeds the arm's delivered force; force map 12-16 N at 7.4 V).
- **`check` phase, `check_height` = 2.5 mm.** A blind press cannot judge seating while pushing: at
  10 N the arm deflects about 2 mm (gearbox and backlash), invisible to forward kinematics of the
  read positions. Over 24 presses, 0-6 mm of error, while pressing: jammed reads -1.5 to +3.2 mm
  above the seated height, successful -1.5 to +1.7 mm — full overlap. Releasing separates them:
  seated height commanded with no overshoot, 250 ms wait → seated 2×4 holds 2.0-2.3 mm high
  (mostly weld stretch) versus 4.6-6.5 mm resting on the stud tops.
- **`spiral_pitch` = 2.5 mm** (spec: 0.5 mm). Measured capture basin: about 3 mm (2×4), 2 mm
  (2×2). A 0.5 mm pitch covers the same disc six times over — 100 mm of path, 6.7 s — exceeding
  the 200-tick episode. `docs/media/control/spiral_search.png`.

The stall detector averages descent over 0.4 s rather than differencing two ticks: the encoder
quantizes to about 0.1 mm at the TCP, so a single-tick difference is 5 mm/s of noise against a
0.7 mm/s threshold.

### `SkillFSM`: `LOOK → ALIGN → PRESS → VERIFY`

Each state has its own deadline; the FSM names whichever one ran out (`timed_out`).
`run_skill(scene, fsm)` drives it through the same multi-rate tick the environments use, without
gymnasium.

VERIFY as specified does not work on this arm. Spec: a small upward tug at reduced torque limit; a
rise in `Present_Load` means the brick is attached. But `Present_Load` is the scaled joint
position error, and this arm has a Coulomb-friction dead band of about 4-5 mm at the TCP (see
gravity feedforward, above). A tug small enough to spare a seated brick stays entirely inside that
stiction band: the tool does not move whether or not the brick is held, and the load rise measures
stiction, not grip. Over 20 trials at a 4 mm tug: correct 6-8 times out of 20, seated and loose
distributions overlapping in reverse. An 8 mm tug beats the stiction but pulls a seated 2×4 back
off the plate.

The settled tool height after the press releases (`SkillFSM.seated_by_height`) works instead: a
seated 2×4 leaves the tool 2.0-2.3 mm above the seated height, one resting on the stud tops
4.6-6.5 mm, marginal cases between. Correct 40 out of 40 on two held-out seeds. Most of the 2 mm
baseline is weld stretch, so the threshold needs recalibration for the friction grasp, and again
on hardware. Both verdicts are reported; `SkillFSM.verified` still reports the spec's load-rise
result.

## `BrickBuilder/PressFromHeld-v0`

- **Horizon:** 200 ticks (4 s).
- **Success:** every stud of the held brick at least 0.9 stud heights into a socket **of the
  correct cell**, held for 5 consecutive ticks. Seating one pitch over is a `wrong_cell` failure.
- **Primitive:** `PressScripted`. **Frozen:** `grip` (the brick is held by the soft weld that
  stands in for the grip, so the jaws change nothing).
- **Scene:** the arm, the plate, a 2×4 (or 2×2) welded into the gripper 5 mm above the seated
  height, and whatever the level puts on the plate.

!!! warning "The weld is not a grip"

    The held brick is attached by a soft weld equality with no breaking point. Hanging weight on
    it for two seconds at a time: 0.15 mm of give at 5 N, 0.23 mm at 50 N, 0.59 mm at 150 N,
    1.34 mm at 400 N — it never releases at any of these. A linear spring, not a grip; this
    environment cannot report "the policy dropped the brick."

    This separates the press and grip questions (Deliverable 1 targets the press), but the
    learned press skill has never been tested against a brick the gripper could lose. The
    real gripper holds a 2×4 upward with about a millimetre of play, does not hold it downward at
    all, and leaks about 0.1 mm/s under the brick's own weight. See
    [Brick seating with the stock gripper](02-simulator-and-bricks.md#brick-seating-with-the-stock-gripper).

### Misalignment source and observability

The arm starts exactly where it believes the target cell is; the belief is wrong by this
episode's error, modelling a mis-taught grid and a brick sitting askew in the hand. Every
deployable signal (`press_seat_error`, `cmd_pose`, the joint encoders) is measured against the
belief, so none of them contains the error; only the critic sees the truth (`press_misalign`).
Over six episodes, each 6 mm off in a different direction, the lateral part of `press_seat_error`
at reset varies by less than 1.5 mm, all of it servo tracking error.

### Levels

| level | lateral | yaw | tilt | world |
|---|---|---|---|---|
| 0 | ±1 mm | 2° | 0.4° | empty plate, target in the middle |
| 1 | ±2 mm | 4° | 0.8° | empty plate |
| 2 | ±4 mm | 8° | 1.4° | empty plate |
| 3 | ±6 mm | 12° | 2.0° | empty plate |
| 4 | ±6 mm | 12° | 2.0° | one brick on the plate; targets beside it and on top of it |
| 5 | ±6 mm | 12° | 2.0° | two bricks; near-row, far-row and layer-1 targets |

`env.unwrapped.set_level(k)` rebuilds the MuJoCo model only when world contents change (levels
4-5; about 0.3 s). Level also scales the `dynamics`, `command`, `in_hand`, `contact` and `plate`
randomization ranges (`dr_level() = level / 5`); level 0 is nominal. `PressEnv(curriculum=(rate,
window))` promotes each environment copy independently once it clears `rate` over its last
`window` episodes.

### Reward

Potentials first, penalties after (`PressReward` holds every weight):

| term | weight | what it is |
|---|---|---|
| progress | 10 | potential on mean stud depth / stud height |
| alignment | 150 / m | potential on lateral error, only before any stud is engaged |
| success | +10 | one-off |
| broken connection | −10 | and terminate |
| load | 2 | quadratic on \|load\| above 0.8, summed over joints |
| press force | 0.02 | quadratic above 14 N |
| neighbour | 200 / m | per metre a placed brick has been shoved past 0.5 mm |
| action rate / residual / time | 0.01 / 0.01 / 0.02 | the usual smallness terms |

Terminations: success, a broken connection, the brick slipping more than 8 mm in the hand, an arm
joint at its limit, an emulated overload trip, or the horizon.

The alignment potential gates off once a stud engages (else lateral centring on a half-seated
brick pays for levering it back out). `oracle_action` — a residual that reads the true
misalignment and cancels it — is the reward's shaping test: it scores clearly better than doing
nothing (measurements below), after two fixes. First, the residual accumulates, so feeding back
the full error every tick acts as a unit-gain integrator against a plant with about 100 ms of lag,
and oscillated to a worse-than-nothing score. Second, the correction must lift the brick off the
stud tops before correcting laterally, since a 5 mm hover with a 4.5 mm stud makes contact within
about 0.1 s.

### `BrickBuilder/PressFromHeld-v0` actor observation (133 numbers)

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `joint_pos` | 6 | 4 | `register:56 (Present_Position)` | actor | ServoBus.read().position, one number per joint in servo-ID order |
| `joint_vel` | 6 | 4 | `register:58 (Present_Speed)` | actor | ServoBus.read().speed |
| `joint_load` | 6 | 4 | `register:60 (Present_Load)` | actor | ServoBus.read().load -- commanded PWM duty in [-1, 1], not a torque |
| `tcp_pos` | 3 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos, position only |
| `tcp_rot6` | 6 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos: first two columns of the rotation matrix (continuous 6D rotation representation, Zhou et al. 2019) |
| `cmd_pose` | 5 | 1 | `command_echo` | actor | our own last commanded TCP target: x, y, z, tilt, spin |
| `cmd_grip` | 1 | 1 | `command_echo` | actor | our own last commanded gripper joint angle |
| `cmd_compliance` | 2 | 1 | `command_echo` | actor | our own last commanded c_vert, c_lat torque-limit fractions |
| `press_seat_error` | 3 | 1 | `task` | actor | where the tool would be if the brick were seated on the target cell, minus where forward kinematics of the read joint positions says it is, in metres; the taught grid and the known tool geometry give the first half, the encoders the second |
| `press_phase` | 8 | 1 | `fsm` | actor | one-hot of the scripted primitive's phase, in brick_builder.control.primitives.PHASES order |
| `press_context` | 5 | 1 | `task` | actor | brick footprint in metres (x, y), target layer height in metres, and the target's yaw relative to the arm's radial direction as (cos, sin) |
| `press_time_frac` | 1 | 1 | `task` | actor | fraction of the episode horizon elapsed (tick / horizon) |

### `BrickBuilder/PressFromHeld-v0` critic observation (192 numbers)

Everything the actor sees, plus the shared privileged entries and five of its own:

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `joint_pos` | 6 | 4 | `register:56 (Present_Position)` | actor | ServoBus.read().position, one number per joint in servo-ID order |
| `joint_vel` | 6 | 4 | `register:58 (Present_Speed)` | actor | ServoBus.read().speed |
| `joint_load` | 6 | 4 | `register:60 (Present_Load)` | actor | ServoBus.read().load -- commanded PWM duty in [-1, 1], not a torque |
| `tcp_pos` | 3 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos, position only |
| `tcp_rot6` | 6 | 4 | `fk` | actor | ArmKinematics.fk() of the read joint_pos: first two columns of the rotation matrix (continuous 6D rotation representation, Zhou et al. 2019) |
| `cmd_pose` | 5 | 1 | `command_echo` | actor | our own last commanded TCP target: x, y, z, tilt, spin |
| `cmd_grip` | 1 | 1 | `command_echo` | actor | our own last commanded gripper joint angle |
| `cmd_compliance` | 2 | 1 | `command_echo` | actor | our own last commanded c_vert, c_lat torque-limit fractions |
| `press_seat_error` | 3 | 1 | `task` | actor | where the tool would be if the brick were seated on the target cell, minus where forward kinematics of the read joint positions says it is, in metres; the taught grid and the known tool geometry give the first half, the encoders the second |
| `press_phase` | 8 | 1 | `fsm` | actor | one-hot of the scripted primitive's phase, in brick_builder.control.primitives.PHASES order |
| `press_context` | 5 | 1 | `task` | actor | brick footprint in metres (x, y), target layer height in metres, and the target's yaw relative to the arm's radial direction as (cos, sin) |
| `press_time_frac` | 1 | 1 | `task` | actor | fraction of the episode horizon elapsed (tick / horizon) |
| `read_stale` | 6 | 1 | `sim` | critic | per servo: this tick's sync-read reply was lost and the previous reading was reused |
| `write_dropped` | 1 | 1 | `sim` | critic | this tick's sync-write was lost on the emulated bus (BusEmulator.tick().write_lost) |
| `bus_timing` | 2 | 1 | `sim` | critic | this tick's command_latency_s, read_skew_s from BusEmulator |
| `ik_failed` | 1 | 1 | `sim` | critic | this tick's IK did not converge; the previous joint goal was kept |
| `dr_params` | 17 | 1 | `sim` | critic | this episode's dynamics.* (8) and command.* (9) randomize.sample() draw, in RANGES order |
| `press_stud_depths` | 8 | 1 | `sim` | critic | insertion depth of each of the held brick's sockets in metres, from the snap model, zero-padded to eight |
| `press_misalign` | 4 | 1 | `sim` | critic | true error of the held brick against the target cell: dx, dy (metres), dyaw, dtilt (radians) |
| `press_in_hand` | 4 | 1 | `sim` | critic | true in-hand offset of the brick against the nominal grasp: dx, dy, dz (metres) in the TCP frame, and how far it has moved since the episode started (metres) |
| `press_contact` | 3 | 1 | `sim` | critic | vertical force the grip is transmitting (N, positive pressing down), the largest distance any placed brick has been shoved (m), and how many of the held brick's studs the snap model counts as engaged |
| `press_dr_extra` | 13 | 1 | `sim` | critic | this episode's in_hand (6), contact (4) and plate (3) randomization draw, in randomize.RANGES order; the dynamics and command halves are already in dr_params |

### Measured numbers (press)

All numbers below: this machine, default snap parameters (`f_clutch` 1.0 N/stud, `curve="ramp"`,
clearance 0.20 mm), weld grasp.

- **Throughput.** **107 steps/s** single-process, level 2, random actions, idle machine —
  fourteen times slower than `Reach`. 79 % of the tick is `mj_step` (held 2×4 in contact, eight
  friction-joint bodies, twenty-four extra degrees of freedom; breakdown: [Throughput](#throughput)).
  Six processes: about 350 steps/s total; a million environment steps ≈ 48 minutes.
- **Press cost.** Clean 2×4, level 0: about 74 ticks (1.5 s), 5 mm hover to success band, peak
  12-14 N. 2×2: quicker, about 58 ticks, about 11 N (four studs, not eight).
- **Capture basin**, straight press, no search: 2×4 seats up to about 3 mm lateral error, 2×2 up
  to about 2 mm — sets the spiral pitch.
- **Yaw is less forgiving than lateral error.** Capture gate `SnapParams.capture_lateral` =
  0.3 mm. A 3° yaw on a 2×4 moves end sockets 24 mm × sin 3° = 1.2 mm off-stud, four times the
  gate: middle sockets capture and hold the brick; ends cannot come in. At 2.9° yaw: one of eight
  studs engaged, at full depth, tool at seated height, force already relaxed — a `partial_seat`
  that further pushing cannot fix. Yaw curve collapses between 2° and 4°; lateral curve decays
  smoothly over 6 mm. Largest gap between this environment and the spec's level-5 ±12° target.

**Scripted press vs. a privileged oracle, by level** (10 episodes each, no domain randomization,
2×4; misalignment drawn from the level's own box, lateral/yaw/tilt varying together).
`oracle_action` reads the true misalignment and cancels it; not a policy, but a check for whether
a learner has anything to gain:

| level | scripted | mean return | oracle | mean return |
|---|---|---|---|---|
| 0 (±1 mm, 2°) | 9/10 | +16.7 | 10/10 | +18.2 |
| 1 (±2 mm, 4°) | 3/10 | +4.2 | 8/10 | +15.0 |
| 2 (±4 mm, 8°) | 1/10 | −1.8 | 9/10 | +16.6 |
| 3 (±6 mm, 12°) | 1/10 | −1.9 | 2/10 | +1.8 |

The gap between levels 1 and 2 is large: the oracle (lift, correct, press) turns a 10-30 % task
into an 80-90 % one, using only signals the critic already has. Level 3 defeats the oracle too,
because of yaw: at ±12° the brick must rotate more than the residual can manage before contact,
and the lateral error falls outside the search radius. Pure yaw, no lateral error: scripted 0/8 at
both 2° and 4°, oracle 8/8 at both.

**Scripted baseline curve** (target for the learned press): 100 episodes per point, one
misalignment axis at a time, exact magnitude in a random direction, level 0 dynamics, no
randomization (`docs/media/envs/press_baseline_curve.png`; raw counts
`runs/press_baseline/curve.json`; 1 451 s on six processes):

| lateral | 0 mm | 1 mm | 2 mm | 3 mm | 4 mm | 5 mm | 6 mm |
|---|---|---|---|---|---|---|---|
| 2×4 | 100/100 | 100/100 | 65/100 | 33/100 | 18/100 | 14/100 | 11/100 |
| 2×2 | 100/100 | 100/100 | 66/100 | 27/100 | 14/100 | 3/100 | 0/100 |

| yaw | 2° | 4° | 8° | 12° |
|---|---|---|---|---|
| 2×4 | 0/100 | 0/100 | 0/100 | 0/100 |
| 2×2 | 20/100 | 0/100 | 0/100 | 0/100 |

Perfect alignment: 100/100 both sizes (clears the spec's ≥ 95 % gate); lateral curve decays
monotonically over 0-6 mm for both. Above 3 mm, the spiral search accounts for the tail: a
straight press without it fails at 3 mm, and the 2×4 keeps 11 % at 6 mm because the search
occasionally finds the cell. Median press time: 74 ticks (1.5 s) for a 2×4, 58 ticks (1.2 s) for a
first-try 2×2 seat, against the spec's 3 s target; a failing episode runs the full 200 ticks. Peak
press force: 7-18 N. For yaw, a 2° error already costs a 2×4 the full seat; the failure is always
`partial_seat` (middle studs in, ends out), never a miss.

**First PPO run: no usable policy.** Six async processes, CPU, self-scheduling curriculum
(`curriculum=(0.6, 20)`), 206k steps budgeted from an optimistic throughput estimate. Reached 118
updates (about 181k steps) in a 25-minute cap, killed before writing a checkpoint — no
learned-vs-scripted comparison for this run. Curve rebuilt from the TensorBoard scalars:
`docs/media/envs/press_learning_curve.png`.

The curve falls, from about +6.5 over the first thirty updates to about −4 by update ninety. Every
process starts at level 0, where the scripted press it builds on already seats 9/10 episodes at a
return near +18; clearing 60 % over twenty episodes promotes a process to level 1 then 2, where the
same scripted press scores 3/10 and 1/10. The return drop reflects task difficulty, not policy
quality, but only the return is logged, so the two are inseparable in this run. Fix: log mean
level and per-level success rate alongside return, and hold out a fixed-level evaluation set. Open
issue.

At 4 mm of lateral error: the scripted press stalls, searches, and gives up; the privileged
`oracle_action` residual lifts, re-centres, and seats.

## `BrickBuilder/AlignFromTokens-v0`

- **Horizon:** 150 ticks (3 s).
- **Success:** TCP within 3 mm and 5° of the brick's true grasp-ready pose, held for 10
  consecutive ticks.
- **Primitive:** `MoveTo`, re-aimed at the current token estimate every time a fresh one arrives,
  ramped from wherever the tool already is (never an instant jump). **Frozen:** `grip`,
  `compliance`.
- **Scene:** a single brick lying loose near the plate's centre, at a continuous (not
  grid-snapped) random `(x, y, yaw)`.

The policy never sees the brick's true pose, only a vision token from one of three interchangeable
producers (`brick_builder.perception.producers`), converted to a metric world position by
`brick_builder.perception.backproject`. The token refreshes only while the arm holds the fixed
`LOOK_TCP` viewing pose; otherwise the last token is kept and its age grows. At zero residual
scale, this environment's `MoveTo`-on-the-token controller is the scripted go-to-token controller
evaluated against noise, age and producer in [Perception](06-perception.md), "From tokens to a
target: AlignFromTokens" (grasp-symmetry and IK-warm-start fixes, and the noise/age/producer
sweep). This section gives the action/observation/reward reference and this environment's own
numbers.

### `BrickBuilder/AlignFromTokens-v0` actor observation (122 numbers)

Everything `PROPRIO_ENTRIES` gives every `ArmEnv` (`joint_pos`, `joint_vel`, `joint_load`,
`tcp_pos`, `tcp_rot6`, `cmd_pose`, `cmd_grip`, `cmd_compliance` — 116 numbers, same sources as the
Reach and PressFromHeld tables above), plus this task's own two entries:

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `align_token` | 5 | 1 | `token` | actor | x, y (m, relative to `LOOK_TCP`'s x/y), yaw (rad), confidence, age (s) of the last backprojected token; zeros with confidence 0 before the first reading ever arrives |
| `align_time_frac` | 1 | 1 | `task` | actor | fraction of the episode horizon elapsed (tick / horizon) |

### `BrickBuilder/AlignFromTokens-v0` critic observation (151 numbers)

Everything the actor sees, plus the shared privileged entries (`read_stale`, `write_dropped`,
`bus_timing`, `ik_failed`, `dr_params` — 27 numbers, same as PressFromHeld's critic table above)
and this task's own:

| name | numbers | history | source | scope | note |
|---|---|---|---|---|---|
| `align_true_error` | 2 | 1 | `sim` | critic | true TCP distance to the grasp-ready pose (m) and true yaw error (rad, brick's 180-degree grasp symmetry already folded in) |

### Reward and termination

`reward() = -20 * pos_err_m - 1.0 * yaw_err_rad`, plus a one-off `+10` once the success streak
first reaches 10 ticks. There is no early termination condition (`terminated()` is always
`False`) — an episode only ever ends by success or by running out its 150-tick horizon.

### Measured numbers (align)

- **Throughput.** **127 steps/s**, `producer="surrogate"`, single process, idle machine
  (previously 68 steps/s; see [Throughput](#throughput)). A producer that runs a model on every
  observation is not characterized well by steps/s: per-observation cost dominates the tick, so
  budget such a run from per-episode wall-clock time.
- **Scripted-controller success against noise, age and producer** is a token-pipeline property,
  not this environment's, covered in [Perception](06-perception.md), "From tokens to a target:
  AlignFromTokens." Even at zero token noise the scripted controller does not reach 100 % (the
  same Coulomb-friction deadband that limits `Reach`'s hand-written P controller); success falls
  off as noise grows, from a modest floor at 0-2 mm to zero at 8 mm.
- **No policy has been trained to convergence.** `configs/train/align.yaml`'s budget is 600k steps
  (hours-long); only smoke-budget checkpoints exist, so there is no learned-vs-scripted comparison
  yet (same open issue as `PressFromHeld`, above).

## Throughput

Numbers are `env.step()` calls per second under a uniform random policy, measured by the loop in
`tests/perf/test_env_speed.py` (`step_rate`), CPU only, `OMP_NUM_THREADS=1`, on the project machine
(16 hardware threads, 31 GB). "Before" and "after" bracket a throughput optimization pass, run
back to back in that order.

**Single process, idle machine** (load average 1.1, 2 000 steps after a 200-step warm-up). The
"previously published" column was measured on a non-idle machine; the back-to-back table below is
the one to read for the size of the change:

| environment | steps/s | target | previously published |
|---|---|---|---|
| `BrickBuilder/Reach-v0` | **1 538** | ≥ 1 200 ✔ | 208 |
| `BrickBuilder/PressFromHeld-v0`, level 0 | **117** | ≥ 150 ✘ | — |
| `BrickBuilder/PressFromHeld-v0`, level 2 | **107** | ≥ 150 ✘ | 41 |
| `BrickBuilder/AlignFromTokens-v0`, `producer="surrogate"` | **127** | — | 68 |

**Before and after, back to back, one process and six processes** (load average 1.6 rising to 8.4
under the process pools; six-process rows are pool aggregates — independent environments, not a
vector env):

| environment | 1 proc before | 1 proc after | 6 procs before | 6 procs after |
|---|---|---|---|---|
| `Reach` | 201 | **1 322** | 895 | **5 300** |
| `PressFromHeld` level 0 | 37 | **88** | 190 | **374** |
| `PressFromHeld` level 2 | 36 | **72** | 199 | **349** |
| `AlignFromTokens` surrogate | 80 | **107** | 337 | **437** |

Four processes land between the two: 4 216 (Reach), 294 / 276 (press levels 0 / 2), 342 (align).
Six processes outperform four on this machine; `Reach`'s per-process rate falls from 1 322 to
about 880 because six processes on sixteen threads share memory bandwidth, not from anything in
this code.

**Tick breakdown now**, microseconds per `env.step()`, timed directly (`cProfile` adds enough
per-call cost to change the answer), idle machine. Timer overhead: these runs measured 1 404, 102
and 129 steps/s against the 1 538, 107 and 127 above:

| | `Reach` | `PressFromHeld` level 2 | `AlignFromTokens` surrogate |
|---|---|---|---|
| `mj_step` (10 physics steps) | 164 | 7 730 | 5 094 |
| snap model (`ColletSnap.update`, 10 calls) | 8 | 1 236 | 1 216 |
| inverse kinematics (`_pose_ik`) | 152 | 154 | 936 |
| `ServoBus.substep` (10 calls) | 85 | 114 | 112 |
| gravity feedforward (2 `gravity_torque`) | 22 | 22 | 22 |
| `ServoBus.read` | 31 | 36 | 30 |
| observation assembly | 56 | 79 | 66 |
| task reward, success, sensing, reset | 195 | 451 | 281 |
| **total** | **713** | **9 822** | **7 757** |

`Reach`: cost spread thinly, physics the largest single item at 23 %. Press: `mj_step` is 79 % of
the tick, snap model 13 %, Python overhead 8 % — no code change reaches the spec's 150 steps/s
without reducing simulator load (eight friction-joint bodies, twenty-four extra degrees of
freedom, each with a friction constraint the solver visits every step; per-stud friction floor
1e-4 N, never zero, `brick_builder/bricks/snap.py`; plus contacts between the 2×4 brick and the
plate). `AlignFromTokens` sits in between: its inverse-kinematics cost is high because a random
policy walks the arm outside its own workspace on about a third of ticks (below), each costing a
full Levenberg-Marquardt solve that then fails too.

**Sources of the speedup:**

- Pose queries no longer run on the full scene. `ArmKinematics` keeps an arm-only, geometry-free
  `MjModel`; answers `fk`, `jacobian` and `gravity_torque` with `mj_kinematics` (+ `mj_comPos`,
  `mj_rne`) instead of `mj_resetData` + `mj_forward` on the whole world — identical results. Press
  scene: 1.6 µs against 108 (home pose) or 397 (pressed into the plate, contacts to solve).
- Per-tick IK no longer finite-differences. `ArmKinematics.ik_diff` takes at most two
  damped-least-squares steps with the analytic `mj_jacSite` Jacobian, warm-started from the
  previous joint goal; Levenberg-Marquardt remains for resets, waypoints and jumped targets. Old
  solver: 17 pose evaluations a tick (`cProfile`); new: two or three.
- The snap model skips bricks that are not there. A brick-free scene (`Reach`) returns
  immediately; a scene with one brick builds the depth/lateral/tilt tables only for studs within a
  stud height of a socket mouth — four of seventy-two while a 2×4 presses onto an 8×8 plate, none
  while it hovers.
- `Reach` builds a 2×2 plate instead of an 8×8 one, since it never touches the plate: 64 collision
  cylinders cost 45.9 µs a physics step against 28.1.
- `ServoBus` no longer rewrites unchanged state: the actuator force range is written only when
  `Torque_Limit` or the overload latch moves, not every tick; the rate limiter and duty estimate
  run in place.

### IK convergence behavior

`ArmEnv` keeps a running `IKStats` (`env.ik_stats`); `EpisodeResult.params` carries `ik_failures`
and `ik_fallbacks`. Over random-policy rollouts (tilt and spin residuals driven to the edges of
their ±20° box every few ticks, harder than any scripted primitive or trained policy):

| | ticks | steps per solve | fell back to LM | unreachable | mean position error | mean spin error |
|---|---|---|---|---|---|---|
| `Reach` | 12 000 | 1.79 | 0.00 % | 0.00 % | 29 µm | 2.08° |
| `PressFromHeld` level 2 | 8 000 | 1.88 | 0.55 % | 0.55 % | 59 µm | 3.44° |
| `AlignFromTokens` surrogate | 3 000 | 1.95 | 35.8 % | 35.8 % | 2.55 mm | 20.4° |
| `PressFromHeld` level 0, zero residual | 1 500 | 0.62 | 0.00 % | 0.00 % | 4.8 µm | 0.00002° |

"Fell back" and "unreachable" match in every row: the two-step solve gives up exactly where the
global solver also fails — the arm walked into a joint limit or off the edge of its workspace.
Spin is the over-determined pose component; the last row shows why this does not matter in
practice — a scripted press, tool straight down with fixed spin, resolves all six pose numbers to
within a hundred-thousandth of a degree and five micrometres. `AlignFromTokens`'s fallback rate
reflects its viewing pose and ±12 mm residual box, not the solver; the same rollouts under the
previous code reported 98.5 % of ticks as IK failures.

One behaviour changed. The old solver flagged `ik_failed` on **76 % of random-policy `Reach`
ticks, 84 % of press ticks and 98 % of align ticks**, freezing the whole joint goal — position
included — each time. Mostly a measurement bug: the check compared position and tool angles at a
joint vector taken before the final `wrist_roll` nudge that chases tool spin, scoring a pose it
did not return; the returned pose was about 0.9 mm off in position (measured median over
500 smooth ticks), since spinning `wrist_roll` moves the TCP. The new gate measures the joint
angles returned and requires the TCP within 0.2 mm and the tool axis within 2°; spin is
reported but not gated on, since five joints cannot deliver position, tool axis and spin at once
once the tool is tilted. `ArmKinematics.YAW_WEIGHT` sets where the unavoidable error goes and is
measured, not chosen: at 0.15, a 15° tilt with a 15° spin request lands the TCP 0.11 mm from where
it was asked to be.

**The scripted press is unaffected.** Zero-residual press episodes never trigger the IK gate under
either version; 50 episodes per level on held-out seeds (10000-10049, no domain randomization,
2×4) come out identical, episode for episode:

| level | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| scripted success, before the optimization pass | 39/50 | 14/50 | 4/50 | 3/50 | 3/50 | 0/50 |
| scripted success, after the optimization pass | 39/50 | 14/50 | 4/50 | 3/50 | 3/50 | 0/50 |

The match is exact, not within binomial noise: same failure taxonomy (the four `structure_damage`
episodes at level 5 are the same four) and the same median episode length. The two solvers give
different joint vectors — up to 0.36° apart on `wrist_roll` from the same seed — but differ only
in the direction the press is insensitive to: a scripted press asks for constant tool spin with
the tool pointing straight down, where the six requested pose numbers never conflict.

## Adding a new task

1. Create `brick_builder/envs/<task>_env.py`. At module scope, `obs_spec.register(...)` every
   observation entry the task itself needs (a goal, a misalignment, a token) that is not already
   in `obs_spec.PROPRIO_ENTRIES`/`CRITIC_ONLY_ENTRIES` — give each one a real source.
2. Subclass `ArmEnv` and override the hooks it needs:
   - `scene_config(rng) -> SceneConfig` — the world to build.
   - `primitive_target(tick) -> PrimitiveTarget | None` — the scripted skill, if any.
   - `task_actor_entries()`/`task_critic_entries()` — the entry names from step 1.
   - `task_obs_values()` — this tick's values for those names.
   - `task_reset(rng)` — draw whatever the episode needs.
   - `reward()`, `terminated()`, `success()`, `failure_class()`, `episode_result_extra()`.
   - `set_level(k)` if the task has its own curriculum (the default drives domain-randomization
     level 0..1; override it if `k` means something else, e.g. PressFromHeld's 0..5 misalignment
     stages).
3. Register it in `brick_builder/envs/__init__.py`'s `_TASKS` dict.
4. Regenerate this page's observation tables with `obs_spec.markdown_table(...)` and add the
   task's own section.
