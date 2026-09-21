# Environments

**Status: base built, one task landed.** `brick_builder/envs/obs_spec.py`, `base_env.py` and
`wrappers.py` exist and are shared by every task; `reach_env.py` (`BrickBuilder/Reach-v0`) is the
first task built on them, used to prove the whole multi-rate loop end to end.
`press_env.py` (`BrickBuilder/PressFromHeld-v0`) is the second, together with the scripted skills in
`brick_builder/control/` that it is built on; both have their own sections below.
`AlignFromTokens-v0` subclasses the same `ArmEnv` described here.

## Purpose

This is where Brick Builder Arm Robot's reinforcement-learning tasks are defined as
[Gymnasium](https://gymnasium.farama.org/) environments: what the policy observes, what it can
do, how it is rewarded, and when an episode ends. Every task shares one base class, `ArmEnv`, so
adding a task means writing the task's own logic, not re-deriving the control loop.

```python
import gymnasium as gym
import brick_builder.envs  # registers every BrickBuilder/*-v0 id as a side effect

env = gym.make("BrickBuilder/Reach-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The multi-rate control tick

Physics steps at 2 ms; the control loop runs at 20 ms (50 Hz), so ten physics steps happen inside
one `env.step()`. `BusEmulator` (design report 02 section 0) hands out this
tick's timing once, up front, and `ServoBus` is driven by it: a lost write leaves the previous
goal in force for the whole tick, and each servo's read is timestamped and, on a lost reply, held
over from the previous tick (flagged in the critic observation, never silently substituted with a
sim-truth value).

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

`snap.update()` runs every `SnapParams.snap_every` physics steps, not every one — the snap-fit
model is expensive enough that running it at the full 500 Hz physics rate would dominate the tick
for a scene with more than a couple of bricks. At the default `snap_every = 1` it is 124 µs a
call for a 2×4 being pressed onto an 8×8 plate, 13 % of a `PressFromHeld` tick; a scene with no
brick in it at all, like `Reach`, costs 0.8 µs a call because there is nothing that could
connect to anything.

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

Indices 0-5 are **residuals**: they accumulate onto a running total every tick (not onto the
measured pose, so a soft position servo does not droop away from where the policy last put it),
and the position/tilt/spin residual is clipped to a box (±12 mm, ±20°) around whatever the task's
scripted primitive says this tick's target pose is (`ArmEnv.primitive_target`). A task with no
scripted primitive (`primitive_target` returns `None`, as `Reach` does) holds that box's centre
fixed at the pose recorded at reset. Because the *residual*, not the absolute target, is what
accumulates and clips, a policy that always outputs zero leaves the executed target exactly equal
to the primitive's own target forever — "zero residual reproduces the scripted baseline" holds by
construction. Indices 6-7 are not residuals: they map straight from this tick's number to a
torque-limit fraction, which is how a policy makes the arm mechanically soft in one direction
(letting studs centre themselves, or capping press force) without needing force control.

A task can freeze any of `{"pos", "tilt", "spin", "grip", "compliance"}` (`ArmEnv.frozen`) so the
policy cannot move that part of the action at all — `Reach` freezes `{"grip", "compliance"}`,
since reaching needs neither.

### Gravity feedforward

`ServoBus` is a plain position servo (`kv=0`, no gravity term); measurement found
this arm settles tens of millimetres off a commanded joint target under its own weight at
`p_gain=16` (`docs/guide/03-robot-and-servo-model.md`'s static-sag figure). A few millimetres of
steady-state droop is invisible to a press task (studs have millimetres of capture tolerance) but
swamps `Reach`'s 3 mm success band, so `ArmEnv` adds a gravity feedforward term
(`_gravity_feedforward`): the command actually written to `ServoBus` is not the IK solution `q`
but `q + gravity_torque(q) / kp`, iterated twice to converge. Measured effect at this arm's home
pose: static error drops from about 12 mm to about 4-5 mm. The remaining 4-5 mm is **not**
gravity — it is this arm's own Coulomb friction (`dof_frictionloss`; frictionloss dominates
below about 0.3 N·m), which a purely kinematic feedforward
cannot remove: friction creates a genuine dead-band around the equilibrium that only a
closed-loop policy exploring small corrective (dithering) actions can beat, and the measured PPO
run below is exactly that experiment. `docs/media/robot/sag.png` is the underlying measurement;
nothing in `patch_so101.py` or `servo_model.py` changed to produce this page's number, only how
`ArmEnv` chooses what to command.

## Observations: `Dict{"actor": Box, "critic": Box}`

Every actor entry is declared in `brick_builder/envs/obs_spec.py` with its real-hardware source
(`register:<address>`, `fk`, `command_echo`, `task`, `token` or `fsm`); `tests/envs/test_obs_spec.py`
fails if any actor-scoped entry has no source, or a `"sim"` (privileged) one. A proprioceptive
entry (a servo register or forward kinematics of the read positions) carries a 4-tick history,
oldest first; a command echo or task entry does not, since it does not change meaning across
ticks the way a noisy sensor reading does. The critic sees everything the actor does, plus a
handful of `"sim"`-sourced entries: which servos' reads were held over from a lost reply, this
tick's command latency and read skew, whether IK converged, and this episode's domain-
randomization draw (`brick_builder/scene/randomize.py`'s `dynamics`/`command` groups, 17 numbers,
always drawn — at `level=0` these are exactly the nominal values, so the critic observation size
never changes with the curriculum).

The tables below are generated straight from `obs_spec` by
`obs_spec.markdown_table(env.actor_names)` / `.markdown_table(env.critic_names)`; a test
(`tests/envs/test_reach_env.py::test_docs_page_matches_the_generated_observation_tables`) checks
this page still matches, so it cannot silently drift from the code.

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

Everything the actor sees (repeated in full below, since the critic's flat vector really does
contain it), plus six privileged, sim-only entries at the end (`read_stale` onward):

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

Move the TCP to a random reachable target from proprioception alone: no scripted primitive, no
bricks, no vision, grip and compliance frozen. Since the position/orientation residual is always
clipped to a box around the (fixed, because there is no primitive) pose recorded at reset, the
target is itself drawn from inside that same box (`GOAL_MARGIN = 0.8` of it) — that is what
"reachable" means for a task with no scripted motion of its own to reach past.

- **Horizon:** 100 ticks (2 s).
- **Success:** true TCP position within 3 mm of the target for 10 consecutive ticks.
- **Reward:** `-20 * distance`, a small penalty on the residual's own magnitude (discourages
  needless motion once close), plus a one-off `+10` bonus on the tick the within-tolerance streak
  first reaches 10.
- **Scene:** the patched SO-101 arm, the table and the empty 8x8 baseplate; no bricks.

## Wrappers (`brick_builder/envs/wrappers.py`)

| wrapper | purpose |
|---|---|
| `HistoryStack(env, k)` | stack the last `k` raw `Dict` observations (on top of, not instead of, the 4-tick history the shared proprioceptive entries already carry) |
| `ObservationDelay(env, delay_ticks)` | return the observation from `delay_ticks` ago; the deployable use of `randomize.sample(...)["command.obs_delay_ticks"]` |
| `ActionRepeatLatency(env, repeat, latency_ticks)` | repeat an action for `repeat` ticks and/or delay when it takes effect by `latency_ticks` |
| `EpisodeStats(env, window)` | attach `info["episode"] = {"r", "l", "success", "success_rate"}` at episode end, `success_rate` over a rolling window |

## Measured numbers (Reach)

- Single-process `env.step()` rate, `Reach`, random actions: **1 538 steps/s**
  (`brick_builder.envs.base_env.measure_step_rate`, 2000 steps after a 200-step warm-up, this
  machine at a load average of 1.1). See [Throughput](#throughput) below for the whole table,
  what the tick is spent on now, and what it used to be spent on.
- PPO (`brick_builder.rl.ppo_asym`) on `BrickBuilder/Reach-v0`, 6 async worker processes, CPU,
  `total_steps=307200` (chosen from a measured ~512 effective steps/s so training finishes in
  about 10 of the 12-minute budget, leaving time for evaluation and media): **training finished
  in 505 s** (8.4 min), mean return rising from -20.6 (update 0) to a plateau around +8.4 (by
  update ~150 of 200); see `docs/media/envs/reach_learning_curve.png`. Evaluated over 200 held-out
  episodes (seeds 10000-10199, deterministic/mean actions): **200/200 successes, Wilson interval
  [0.981, 1.0]** (comfortably above the 90% target), mean final TCP error 1.02 mm, median 0.98 mm
  — both below the 3 mm success band and below the roughly 4-5 mm Coulomb-friction deadband a
  purely proportional controller cannot beat (see the gravity-feedforward section above): the
  learned policy's own small corrective residual actions evidently dither past the static-friction
  dead zone, which the hand-written test controller does not attempt. Checkpoint, TensorBoard log and raw per-update metrics are
  under `runs/reach_smoke/` (git-ignored).


## Scripted press and the press environment

`brick_builder/control/primitives.py` and `brick_builder/control/skill_fsm.py` hold the scripted skills;
`brick_builder/envs/press_env.py` wraps the press one in `BrickBuilder/PressFromHeld-v0`. This is the
blind, contact-rich skill Deliverable 1 is built around: a brick is already in the gripper, a few
millimetres above its target cell, misaligned, and the arm has to feel it in.

### Scripted primitive inputs

A primitive sees one `Feedback` per tick: forward kinematics of the joint positions the servo bus
*read*, the servos' own `Present_Load` register, the gripper angle we ourselves commanded, and
the overload flag. Nothing else. `brick_builder/control/primitives.py` does not import `mujoco`, the
scene or the environments, and a test checks that it stays that way — which is what makes the
scripted press a fair baseline for the learned one, and what lets the same state machine be
ported to the arm's own microcontroller later.

It produces a `ToolTarget`: world TCP position, tool tilt and spin, gripper angle, and the two
compliance scalars. That is exactly the primitive half of "executed target = primitive target +
bounded residual", so an environment can hand the same object to a policy or run it on its own.

### `MoveTo`

Straight-line Cartesian interpolation at a fixed speed, then a settle. The move finishes when the
measured tool is inside `tol` of the target **or** `settle_s` has elapsed — a soft position servo
never lands exactly, so waiting for zero error would never finish.

### `PressScripted`

Descends straight down from wherever it is handed the tool, at 10 mm/s, with `c_lat` turned down
to 0.45 so a stud landing on a chamfer can shove the tool aside and centre itself. It never
learns which way it is misaligned — only that it has stopped going down.

| phase | what it does |
|---|---|
| `descend` | the 10 mm/s ramp. Ends `seated` when the measured tool reaches the seated height, or `lift` when the descent stalls under load well above it. |
| `rock` | optional (`PressConfig.rocking`): the same ramp held at 4° until one stud row engages, then levelled out. |
| `lift` | a 2 mm hop to unload the brick before it is skimmed sideways. |
| `search` | an outward spiral at the skim height; ends the moment the tool drops into the sockets. |
| `seated` | done, holding the pressed pose. |
| `stuck` | gave up — saturated load, searches exhausted, or the press clock ran out. Holds where it is rather than pushing harder. |

Three of its numbers were measured rather than chosen, and each of them was a bug first:

- **`press_depth` = 10 mm.** The arm is a position servo with no force loop, so commanded
  overshoot past the seated height is the only throttle on press force. Measured at the plate
  centre: 6 mm gives 9 N and takes twice as long to seat, 10 mm gives 14 N, 20 mm asks for 25 N
  — more than the arm can really deliver (the force map: 12–16 N at 7.4 V).
- **The `check` phase, and `check_height` = 2.5 mm.** A blind press has to decide, from the
  encoders alone, whether the brick went in — and while it is still pushing it cannot. At 10 N
  this arm deflects about 2 mm through its gearboxes and its backlash joints, and forward
  kinematics of the *read* positions (all a real robot has) sees none of that. Measured over 24
  presses at 0–6 mm of error: jammed presses read −1.5 to +3.2 mm above the seated height,
  successful ones read −1.5 to +1.7 mm. The populations overlap completely; no threshold on a
  pressing tool separates them. Let go first and they separate cleanly — command the seated
  height with no overshoot, wait 250 ms, and a seated 2×4 holds the tool 2.0–2.3 mm high (almost
  all of that the soft weld's own stretch) while one resting on the stud tops holds it 4.6–6.5 mm
  high. So a stall is not a failure in this state machine, it is the question, and the `check`
  phase is where it gets answered.
- **`spiral_pitch` = 2.5 mm**, where the spec sketch says 0.5 mm. The measured capture basin of a
  straight press is about 3 mm for a 2×4 and 2 mm for a 2×2, so a 0.5 mm pitch covers the same
  disc six times over: 100 mm of spiral path, 6.7 s, which does not fit inside a 200-tick
  episode, let alone leave time for the press that follows it. See
  `docs/media/control/spiral_search.png`.

The stall detector averages the descent over 0.4 s rather than differencing two ticks: the
encoder quantizes to about 0.1 mm at the TCP, so a single-tick difference is 5 mm/s of pure noise
against a 0.7 mm/s threshold.

### `SkillFSM`: `LOOK → ALIGN → PRESS → VERIFY`

Each state has its own deadline and the FSM names the one that ran out (`timed_out`).
`run_skill(scene, fsm)` drives it through the same multi-rate tick the environments use, without
gymnasium — a demo or a hardware bring-up script should not have to build an RL environment to
press a brick.

**VERIFY does not work the way the spec expected, and the reason is this arm.** The spec defines
it as a small upward tug at a reduced torque limit, with a rise in `Present_Load` meaning the
brick is attached. But `Present_Load` *is* the joint position error, scaled, and this arm carries
a Coulomb-friction dead band of about 4–5 mm at the TCP (the same one the gravity-feedforward
section above describes). A tug small enough to leave a seated brick alone is therefore entirely
inside the arm's own stiction: the tool does not move whether or not anything is holding it, and
the rise measures stiction. Measured over 20 trials at a 4 mm tug: right 6–8 times out of 20,
with the seated and loose distributions overlapping *and the wrong way round*. An 8 mm tug does
beat the stiction, but it pulls a genuinely seated 2×4 back off the plate — the test destroys
what it is testing.

What does work is the height the tool settles at once the press lets go
(`SkillFSM.seated_by_height`): a seated 2×4 leaves it 2.0–2.3 mm above the seated height, a brick
resting on the stud tops 4.6–6.5 mm, and the marginal cases in between. Right 40 times out of 40
on two held-out seeds. Most of that 2 mm is the soft weld's own stretch rather than anything
about the brick, so the threshold needs recalibrating for the friction grasp and again on
hardware. Both verdicts are reported; `SkillFSM.verified` stays the spec's load-rise one so the
finding stays visible instead of being quietly redefined.

## `BrickBuilder/PressFromHeld-v0`

- **Horizon:** 200 ticks (4 s).
- **Success:** every stud of the held brick at least 0.9 stud heights into a socket **of the
  correct cell**, held for 5 consecutive ticks. Seating one pitch over is a `wrong_cell` failure.
- **Primitive:** `PressScripted`. **Frozen:** `grip` (the brick is held by the soft weld that
  stands in for the grip, so the jaws change nothing).
- **Scene:** the arm, the plate, a 2×4 (or 2×2) welded into the gripper 5 mm above the seated
  height, and whatever the level puts on the plate.

!!! warning "The weld is not a grip"

    The brick is held by a soft weld equality, and that weld has no breaking point. Measured by
    hanging weight off the held brick for two seconds at a time: 0.15 mm of give at 5 N, 0.23 mm
    at 50 N, 0.59 mm at 150 N, 1.34 mm at 400 N, and it never lets go at any of them. It is a
    linear spring, not a grip, so "the policy dropped the brick" is not a thing this environment
    can report.

    That is deliberate — Deliverable 1 is about the press, and a grip that cannot fail keeps the
    two questions apart — but it does mean the learned press skill has never been asked to seat a
    brick the gripper could lose. The real gripper holds a 2×4 upwards in about a millimetre and
    downwards not at all, and leaks about a tenth of a millimetre a second under nothing but the
    brick's own weight. See
    [Brick seating with the stock gripper](02-simulator-and-bricks.md#brick-seating-with-the-stock-gripper).

### Misalignment source and observability

The arm does not start at the target cell and get pushed off it. It starts exactly where it
*believes* the cell is, and that belief is wrong by this episode's error. That models a
mis-taught grid and a brick sitting slightly askew in the hand: every deployable signal —
`press_seat_error`, `cmd_pose`, the joint encoders — is measured against the belief, so none of
them contains the error. Only the
critic is told the truth (`press_misalign`). Measured: over six episodes each 6 mm off in a
different direction, the lateral part of `press_seat_error` at reset varies by less than 1.5 mm,
and all of that is servo tracking error.

### Levels

| level | lateral | yaw | tilt | world |
|---|---|---|---|---|
| 0 | ±1 mm | 2° | 0.4° | empty plate, target in the middle |
| 1 | ±2 mm | 4° | 0.8° | empty plate |
| 2 | ±4 mm | 8° | 1.4° | empty plate |
| 3 | ±6 mm | 12° | 2.0° | empty plate |
| 4 | ±6 mm | 12° | 2.0° | one brick on the plate; targets beside it and on top of it |
| 5 | ±6 mm | 12° | 2.0° | two bricks; near-row, far-row and layer-1 targets |

`env.unwrapped.set_level(k)` moves between them and rebuilds the MuJoCo model only when the
world's contents change (levels 4 and 5), which costs about 0.3 s. The level also sets how much
of the `dynamics`, `command`, `in_hand`, `contact` and `plate` randomization ranges are used
(`dr_level() = level / 5`), so level 0 is the exactly nominal world the scripted baseline is
measured in. `PressEnv(curriculum=(rate, window))` lets each copy of the environment promote
itself once it clears `rate` over its own last `window` episodes — a per-worker curriculum, so a
vectorized run ends up with a spread of difficulties rather than every process stepping up at the
same moment.

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

Before changing this reward, two things about it matter. The alignment potential is
gated off once a stud is engaged, because rewarding lateral centring while the brick is half in
pays for levering it back out. And `oracle_action` — a residual that reads the true misalignment
and cancels it — is the environment's own shaping test: if a residual that knows the answer does
not score clearly better than doing nothing, the reward is not pointing anywhere a learner could
follow. It does (see the measurements below), but only after two fixes that are themselves
findings: the residual *accumulates*, so feeding back the whole error every tick is an integrator
with unit gain against a plant with a hundred milliseconds of lag, and it oscillated its way to
scoring *worse* than doing nothing; and the correction has to lift the brick off the stud tops
first, since with a 5 mm hover and a 4.5 mm stud the brick is in contact within a tenth of a
second.

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

Every number below is from this machine, with the code in this repository, at default snap
parameters (`f_clutch` 1.0 N/stud, `curve="ramp"`, clearance 0.20 mm) and the weld grasp.

**Throughput.** Single-process `env.step()` rate for `PressFromHeld`, random actions, level 2:
**107 steps/s** on an idle machine. That is fourteen times slower than `Reach`, and the
difference is the bricks: 79 % of the tick is `mj_step` itself, working on a scene with a held
2×4 in contact with the plate, eight friction-joint bodies and their twenty-four extra degrees of
freedom. See [Throughput](#throughput) for the whole table and the per-tick breakdown; six worker
processes give about 350 steps/s in total, so a million environment steps is about 48 minutes.

**What one press costs.** A clean 2×4 press at level 0 takes about 74 control ticks (1.5 s) from
the 5 mm hover to the success band, with a peak press force of 12–14 N. A 2×2 is quicker (about
58 ticks) and needs about 11 N, because it has four studs to push in instead of eight.

**Capture basin of a straight press** (no search, fixed misalignment magnitude, random
direction): a 2×4 seats from up to about 3 mm of lateral error, a 2×2 from about 2 mm. This is
the number the spiral's pitch is chosen against.

**Yaw is much less forgiving than lateral error, and the snap model says why.** A socket captures
a stud only within `SnapParams.capture_lateral` = 0.3 mm. Rotating a 2×4 by 3° moves its end
sockets 24 mm × sin 3° = 1.2 mm off their studs — four times the gate — so the middle sockets
capture, that joint holds the brick where it is, and the ends can never come in. Measured: a
2.9° yaw error leaves exactly one of eight studs engaged, at full depth, with the tool sitting at
the seated height and the press force already relaxed. That is a `partial_seat`, and no amount of
pushing fixes it. It is why the yaw curve collapses between 2° and 4° while the lateral curve
decays smoothly over 6 mm, and it is the single biggest thing standing between this environment
and the spec's level-5 target of ±12°.

**Scripted press against a privileged oracle, by level** (10 episodes each, no domain
randomization, 2×4, misalignment drawn from the level's own box so lateral, yaw and tilt all
vary together). `oracle_action` is not a policy — it reads the true misalignment and cancels it
— but it is the number that says whether there is anything for a learner to win:

| level | scripted | mean return | oracle | mean return |
|---|---|---|---|---|
| 0 (±1 mm, 2°) | 9/10 | +16.7 | 10/10 | +18.2 |
| 1 (±2 mm, 4°) | 3/10 | +4.2 | 8/10 | +15.0 |
| 2 (±4 mm, 8°) | 1/10 | −1.8 | 9/10 | +16.6 |
| 3 (±6 mm, 12°) | 1/10 | −1.9 | 2/10 | +1.8 |

Two things follow. First, there is a large, real gap for a policy to earn between levels 1 and 2
— an oracle that lifts, corrects and presses turns a 10–30 % task into a 80–90 % one, and every
signal it uses to do that is something the critic already sees, so the asymmetric-critic setup
has the information it needs. Second, level 3 defeats the oracle too, and it is the yaw that
does it: at ±12° the brick has to be rotated more than the residual can manage before contact
*and* the lateral error is outside the search radius. At a pure yaw error with no lateral error the
split is as sharp as it gets — scripted 0/8 at both 2° and 4°, oracle 8/8 at both.

**The scripted baseline curve** — the number the learned press has to beat — is 100 episodes per
point, one misalignment axis at a time, at an exact magnitude in a random direction, level 0
dynamics with no randomization (`docs/media/envs/press_baseline_curve.png`, raw counts in
`runs/press_baseline/curve.json`, 1 451 s on six worker processes):

| lateral | 0 mm | 1 mm | 2 mm | 3 mm | 4 mm | 5 mm | 6 mm |
|---|---|---|---|---|---|---|---|
| 2×4 | 100/100 | 100/100 | 65/100 | 33/100 | 18/100 | 14/100 | 11/100 |
| 2×2 | 100/100 | 100/100 | 66/100 | 27/100 | 14/100 | 3/100 | 0/100 |

| yaw | 2° | 4° | 8° | 12° |
|---|---|---|---|---|
| 2×4 | 0/100 | 0/100 | 0/100 | 0/100 |
| 2×2 | 20/100 | 0/100 | 0/100 | 0/100 |

Perfect alignment seats 100 of 100 for both brick sizes, which clears the spec's ≥ 95 % gate, and
the lateral curve decays monotonically over the whole 0–6 mm range for both. The tail above 3 mm
is the spiral search earning its keep: a straight press without it dies at 3 mm, and the 2×4 keeps
11 % at 6 mm because the search sometimes finds the cell. Median press time is 74 ticks (1.5 s)
for a 2×4 and 58 ticks (1.2 s) when it seats on the first try, against the spec's 3 s target; a
failing episode runs the full 200 ticks. Peak press force stays between 7 and 18 N.

The yaw row is the finding above, restated as a curve: 2° already costs a 2×4 everything, and the
failure is always `partial_seat` (the middle studs in, the ends out) rather than a miss.

**The first PPO run did not produce a usable policy, and the reason is instructive.** Six async
workers, CPU, self-scheduling curriculum (`curriculum=(0.6, 20)`), 206 k steps budgeted from an
optimistic throughput guess. It reached 118 updates and about 181 k steps in its 25-minute cap
and was killed before writing a checkpoint, so there is no learned-versus-scripted comparison in
this round. `docs/media/envs/press_learning_curve.png` is the curve it did produce, rebuilt from
the TensorBoard scalars.

The curve falls — from about +6.5 over the first thirty updates to about −4 by update ninety —
and that is *not* evidence the policy got worse. Every worker starts at level 0, where the
scripted press it sits on top of already seats 9 episodes in 10 and the return is around +18. As
soon as a worker clears 60 % over twenty episodes it promotes itself to level 1, then 2, where
the same scripted press scores 3/10 and 1/10. The return drops because the task got harder, and
the only scalar being logged is the return, so the two effects are inseparable. That is a flaw in
*this experiment*, not in the environment: the fix is to log the mean level and the per-level
success rate alongside the return, and to hold a fixed-level evaluation set aside. It is written
up as an open issue rather than patched into the numbers.

At 4 mm of lateral error the scripted press stalls, searches and gives up; the privileged
`oracle_action` residual lifts, re-centres and seats.



## `BrickBuilder/AlignFromTokens-v0`

- **Horizon:** 150 ticks (3 s).
- **Success:** TCP within 3 mm and 5° of the brick's true grasp-ready pose, held for 10
  consecutive ticks.
- **Primitive:** `MoveTo`, re-aimed at the current token estimate every time a fresh one arrives,
  ramped from wherever the tool already is (never an instant jump). **Frozen:** `grip`,
  `compliance`.
- **Scene:** a single brick lying loose near the plate's centre, at a continuous (not
  grid-snapped) random `(x, y, yaw)`.

A single brick sits loose near the plate's centre; the policy never sees its true pose, only a
vision token from one of three interchangeable producers (`brick_builder.perception.producers`),
turned into a metric world position by `brick_builder.perception.backproject`. The token is refreshed
only while the arm sits at a fixed `LOOK_TCP` viewing pose — otherwise the last
token is kept and its age grows. With the residual scale at zero this environment's own
`MoveTo`-on-the-token controller is exactly the "scripted go-to-token controller" whose success
is measured against noise, age and producer in [Perception](06-perception.md).
That page's "From tokens to a target: AlignFromTokens" section is the place to read the full
story — why the brick sits on the plate rather than bare table, the grasp-symmetry and
IK-warm-start fixes the scripted controller needed, and the noise/age/producer sweep itself; this
section only gives the same action/observation/reward reference the other two environments get
above, plus the numbers specific to this environment rather than to the controller.

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

**Throughput.** Single-process `env.step()` rate for `AlignFromTokens`, `producer="surrogate"`,
random actions: **127 steps/s** on an idle machine (see [Throughput](#throughput)); an earlier
measurement put it at 68 steps/s, before the throughput work described there. A producer that runs
a model on every observation is not usefully characterized by a steps/s number at all: its
per-observation cost dominates the tick, so plan such a run from a per-episode wall-clock
measurement rather than a steps/s figure that would imply something much cheaper.

**Scripted-controller success against noise, age and producer** is measured and plotted in
[Perception](06-perception.md)'s "From tokens to a target: AlignFromTokens"
section rather than duplicated here, since it is a property of the token pipeline the controller
sits on, not of this environment's own mechanics. The short version: even at zero token noise the
scripted controller does not reach 100% (the same servo-level Coulomb-friction deadband that limits Reach's own
hand-written P controller), and success falls off as noise grows from a modest floor at 0-2 mm to
zero at 8 mm.

**No policy has been trained to convergence.** `configs/train/align.yaml`'s budget (600k steps) is
real and hours-long; only smoke-budget checkpoints exist as of this writing, so there is no
learned-versus-scripted comparison for this environment yet (same open issue as PressFromHeld,
above).



## Throughput

Every number here is `env.step()` calls per second under a uniform random policy, measured with
the same loop in `tests/perf/test_env_speed.py` (`step_rate`), CPU only, `OMP_NUM_THREADS=1`, on
the project machine (16 hardware threads, 31 GB). "Before" is the code as it stood before a
throughput optimization pass and "after" is the same code with that pass's changes; the two were
run back to back, in that order, so that whatever else the machine was doing hit both of them.

**Single process, machine otherwise idle** (load average 1.1, 2 000 steps after a 200-step
warm-up). The last column is the number this page carried previously, measured then on a
machine that was not idle, which is why the back-to-back table below is the one to read for the
size of the change:

| environment | steps/s | target | previously published |
|---|---|---|---|
| `BrickBuilder/Reach-v0` | **1 538** | ≥ 1 200 ✔ | 208 |
| `BrickBuilder/PressFromHeld-v0`, level 0 | **117** | ≥ 150 ✘ | — |
| `BrickBuilder/PressFromHeld-v0`, level 2 | **107** | ≥ 150 ✘ | 41 |
| `BrickBuilder/AlignFromTokens-v0`, `producer="surrogate"` | **127** | — | 68 |

**Before and after, back to back, one process and six worker processes** (load average 1.6 rising
to 8.4 as the worker pools ran; the six-worker rows are aggregates over the pool, and the workers
are independent environments, not a vector env):

| environment | 1 proc before | 1 proc after | 6 workers before | 6 workers after |
|---|---|---|---|---|
| `Reach` | 201 | **1 322** | 895 | **5 300** |
| `PressFromHeld` level 0 | 37 | **88** | 190 | **374** |
| `PressFromHeld` level 2 | 36 | **72** | 199 | **349** |
| `AlignFromTokens` surrogate | 80 | **107** | 337 | **437** |

Four workers land between the two: 4 216 (Reach), 294 / 276 (press levels 0 / 2) and 342
(align). Six workers are worth taking over four on this machine, and the per-worker rate falls
from 1 322 to about 880 for Reach because six processes on sixteen threads start sharing memory
bandwidth, not because of anything in this code.

**Where a tick goes now**, microseconds per `env.step()`, timed directly (not through
`cProfile`, which charges enough per call to change the answer) on an idle machine. The timers
themselves cost a few per cent, so these runs came out at 1 404, 102 and 129 steps/s against the
1 538, 107 and 127 above:

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

`Reach` is now spread thinly across everything, with the physics itself the largest single item
at 23 %. The press is not: `mj_step` is 79 % of its tick and the snap model another 13 %, so the
Python around the simulator is down to 8 % of it and there is no version of this work that
reaches the spec's 150 steps/s without changing what the simulator is asked to do. The press
scene carries eight friction-joint bodies with twenty-four extra degrees of freedom, each with a friction
constraint the solver visits on every step (the per-stud friction floor is 1e-4 N and never zero,
deliberately — see `brick_builder/bricks/snap.py`), plus the contacts between a 2×4 brick and the
plate it is being pressed onto. `AlignFromTokens` sits in between; its inverse-kinematics line is
large because a random policy walks that task's arm outside its own workspace on about a third of
ticks (see below), and each of those costs a full Levenberg-Marquardt solve that then fails too.

**What made the difference.** In rough order of what it bought:

- **Pose queries stopped running on the scene.** `ArmKinematics` keeps its own arm-only,
  geometry-free `MjModel` and answers `fk`, `jacobian` and `gravity_torque` with `mj_kinematics`
  (+ `mj_comPos` and `mj_rne`) instead of `mj_resetData` + `mj_forward` on the whole world. On
  the press scene that is 1.6 µs against 108 (arm at its home pose) or 397 (arm pressed into the
  plate, so the scene has contacts to solve), for bit-for-bit identical answers.
- **The per-tick IK stopped finite-differencing.** `ArmKinematics.ik_diff` takes at most two
  damped-least-squares steps with the analytic `mj_jacSite` Jacobian, warm-started from the
  previous tick's joint goal; the Levenberg-Marquardt solver is still there for resets,
  waypoints and targets that have jumped. `cProfile` counted 17 pose evaluations a tick under
  the old solver (five finite differences per Levenberg-Marquardt iteration, plus the trial
  steps); this one takes two or three.
- **The snap model stopped looking at bricks that are not there.** A scene with no brick in it
  (`Reach`) returns immediately, and a scene with one only builds the depth/lateral/tilt tables
  against the studs whose bounding box is within a stud height of a socket mouth — four out of
  seventy-two while a 2×4 is being pressed onto an 8×8 plate, none at all while it hovers.
- **`Reach` builds a 2×2 plate instead of an 8×8 one.** It never touches the plate, and 64
  collision cylinders cost 45.9 µs a physics step against 28.1.
- **`ServoBus` stopped rewriting things that had not changed.** The actuator force range is
  written when `Torque_Limit` or the overload latch moves, not ten times a tick; the rate limiter
  and the duty estimate run in place.

### IK convergence behavior

`ArmEnv` keeps a running `IKStats` (`env.ik_stats`) of what the per-tick solve actually did, and
`EpisodeResult.params` carries `ik_failures` and `ik_fallbacks`. Over random-policy rollouts —
which drive the tilt and spin residuals to the edges of their ±20° box every few ticks, far
harder than any scripted primitive or trained policy asks for:

| | ticks | steps per solve | fell back to LM | unreachable | mean position error | mean spin error |
|---|---|---|---|---|---|---|
| `Reach` | 12 000 | 1.79 | 0.00 % | 0.00 % | 29 µm | 2.08° |
| `PressFromHeld` level 2 | 8 000 | 1.88 | 0.55 % | 0.55 % | 59 µm | 3.44° |
| `AlignFromTokens` surrogate | 3 000 | 1.95 | 35.8 % | 35.8 % | 2.55 mm | 20.4° |
| `PressFromHeld` level 0, zero residual | 1 500 | 0.62 | 0.00 % | 0.00 % | 4.8 µm | 0.00002° |

"Fell back" and "unreachable" are the same number in every row, which is the point: every target
the two-step solve gives up on is one the global solver cannot reach either — the arm has been
walked into a joint limit or off the edge of its workspace. The spin column is the over-determined
one, and the last row is why it is not a problem in practice: a scripted press, which points the
tool straight down and holds its spin, gets every one of the six pose numbers to within a
hundred-thousandth of a degree and five micrometres. `AlignFromTokens`' third is a
property of that task's viewing pose and its ±12 mm residual box, not of the solver; the same
rollouts under the previous code reported 98.5 % of ticks as IK failures.

One behaviour changes as a result. The old per-tick solver reported
`ik_failed` on **76 % of random-policy `Reach` ticks, 84 % of press ticks and 98 % of align
ticks**, and on each of those the environment froze the whole joint goal — position included.
Almost all of that was a measurement bug rather than a real miss: the check compared the
position and tool angles at a joint vector taken *before* the final `wrist_roll` nudge that
chased the tool spin, so it was reporting the residual of a pose it did not return. The pose it
did return was in fact about 0.9 mm off in position (measured median over 500 smooth ticks),
because spinning `wrist_roll` moves the TCP. The new gate is measured at the joint angles
actually returned and asks for the TCP within 0.2 mm and the tool axis within 2°; the spin is
reported but not gated on, because five joints cannot deliver position, tool axis and spin at
once once the tool is tilted, and freezing the position because a degree of spin is unavailable
is not a useful behaviour. `ArmKinematics.YAW_WEIGHT` sets where the unavoidable error goes, and
is measured rather than chosen: at 0.15 a 15° tilt with a 15° spin request lands the TCP 0.11 mm
from where it was asked to be.

**This changes nothing about the scripted press.** Zero-residual press episodes never trigger
the IK gate under either version, and 50 episodes per level on held-out seeds (10000-10049, no
domain randomization, 2×4) come out identical, episode for episode:

| level | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| scripted success, before the optimization pass | 39/50 | 14/50 | 4/50 | 3/50 | 3/50 | 0/50 |
| scripted success, after the optimization pass | 39/50 | 14/50 | 4/50 | 3/50 | 3/50 | 0/50 |

Not "within binomial noise" but the same number in every cell, with the same failure taxonomy
(the four `structure_damage` episodes at level 5 are the same four) and the same median episode
length. The two solvers do give different joint vectors — up to 0.36° apart on `wrist_roll` from
the same seed — but they differ in the direction the press is insensitive to, and a scripted
press asks for a constant tool spin with the tool pointing straight down, which is exactly the
case where the six requested pose numbers do not conflict at all.



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
