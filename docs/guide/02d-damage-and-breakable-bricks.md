# Damage and breakable bricks

Guide 02c derives the material numbers. This page is what the simulator now does with them: the
contact model every scene is built with, the damage monitor every scene carries, and the
breakable brick a scene can be built with instead of a rigid one.

## 1. What is simulated

| Modelled | Not modelled |
|---|---|
| Contact stiffness derived from the ABS contact modulus | Brick deformation: every body is rigid |
| Wall bending, stud shear and bending, tube ovalisation, roof bending, marking | Stress fields, plasticity, crack growth |
| A permanent-set, cracked and broken state per brick | The geometry changing when a brick is damaged |
| Pieces that separate at their real cross-section limits | A piece deforming before it separates |
| Clutch wear from a Palmgren-Miner sum and a creep term | Ageing, temperature history, colour |

A brick is a rigid body. Nothing in this page makes one bend. What the damage model does is read
the contact forces the solver has already computed, apply a closed-form stress formula to the
member each contact lands on, and record the result. A "yielded" brick has the same geometry it
had before; what changed is a state and a clutch multiplier.

## 2. Contact model

### 2.1 Derivation

Each pair gets a contact modulus from the two materials and a patch size from a stated piece of
geometry, and the two give a flat-punch stiffness `k = 2 a E*`. A flat punch, not a Hertz sphere:
a stud tip on a roof, a wall bottom on a plate and a jaw face on a wall are all flat on flat, and
a flat contact's stiffness is set by the size of the patch rather than by the load.

`1/E* = (1 - nu_a^2)/E_a + (1 - nu_b^2)/E_b`, with the table treated as rigid because no source
says what it is made of.

| Pair | Patch | E* [GPa] | k [MN/m] | Derived solref [us] |
|---|---|---|---|---|
| brick on brick / baseplate | flat stud tip, r = 4.50 mm | 1.31 | 11.8 | 32.6 |
| jaw on brick (PLA on ABS) | jaw face, 10 x 5 mm | 1.59 | 12.7 | 31.4 |
| brick on table | wall footprint, r_eq = 9.55 mm | 2.62 | 50.1 | 15.8 |
| arm link on anything | as the jaw face | 1.59 | 12.7 | 31.4 |

MuJoCo's `solref` takes a time constant, not a stiffness, because its soft constraint is
mass-normalised. The map is the mass-spring period `t = sqrt(m / k)` against a stated mass,
`materials.CONTACT_REFERENCE_MASS_KG` = 12.52 g, the generated DUPLO 2x4.

### 2.2 The clip, twice

**Solver floor.** MuJoCo requires the time constant to be at least twice the step. At 2 ms that
is 4 ms, and every pair above derives to tens of microseconds, so **every pair clips**.

Real ABS contact is four to five orders of magnitude stiffer than anything this solver runs at a
2 ms step. The contact stiffness in this simulator is a statement about the solver, not about the
plastic, and no setting of `solref` changes that.

**Press ceiling.** The 4 ms the derivation asks for jams the press. Measured on the closed-loop
placement, sweeping the brick-on-brick setting:

| Setting | Seat depth error | Result |
|---|---|---|
| 4 ms, derived solimp | 4.50 mm | stops on the stud tips, every seed |
| 6 ms, derived solimp | 4.50 mm | stops on the stud tips |
| 8 ms, derived solimp | 4.50 mm | stops on the stud tips |
| 12 ms, derived solimp | 4.50 mm | stops on the stud tips |
| 16 ms, derived solimp | 0.20 mm | seats 3 seeds of 5 |
| 20 ms, derived solimp | 0.26 mm | seats 3 seeds of 5; one throws the brick 43 mm |
| MuJoCo's own (shipped) | 0.26 mm | seats 5 seeds of 5 |

4.50 mm is one stud height: the brick rests on the stud tips instead of going down. The cause is
the arm rather than the solver. Seating the last 4.5 mm means closing the contacts along the
insertion path against a 10 to 14 N press, and a contact 25 times stiffer needs 25 times the force
to close by the same amount. **A press-fit insertion in a rigid-body simulator runs on contact
compliance**, and this one is calibrated, through the placement policy, to MuJoCo's default.

The derivation is therefore not applied to brick-on-brick or brick-on-baseplate. The same holds for
brick-on-table: with the derived table contact the nominal closed-loop pick-and-place ends 0.39 mm
short of full depth (not seated); with MuJoCo's settings it seats at 0.00 mm, because the grasp off
the table is calibrated against that pair. These pairs keep MuJoCo's shipped `solref` and `solimp`;
the derived setting stays selectable as `contact_model: derived`.

| Pair | Applied solref [ms] | Applied solimp | Times softer than the material |
|---|---|---|---|
| brick on brick / baseplate | 20.0 (MuJoCo's) | 0.9 0.95 0.001 | 3.8e5 |
| jaw on brick | 4.0 | 0.95 0.999 0.0005 | 1.6e4 |
| brick on table | 20.0 (MuJoCo's); 4.0 with `derived` | 0.9 0.95 0.001; 0.95 0.999 0.0005 with `derived` | 1.6e6; 6.4e4 with `derived` |
| arm link on anything | 4.0 | 0.95 0.999 0.0005 | 1.6e4 |

The jaw pads carry `priority = 1`, so their own settings decide a contact with a brick outright
instead of being averaged with the brick's. Under `derived` the table carries `priority = 1` too.

### 2.3 Measured penetration

A 2x4 resting on a baseplate under its own weight, settled for 1.5 s:

| Contact model on the press pair | 2 ms | 1 ms | 0.5 ms |
|---|---|---|---|
| `material` (shipped) = `mujoco` | 15.80 um | 15.78 um | 15.81 um |
| derived solimp at 20 ms | 7.83 um | 7.83 um | 7.83 um |
| `derived` (4 ms) | 0.32 um | 0.31 um | 0.31 um |

Identical across the step in every case, because the time constant sets the stiffness and the step
does not. The contact model is fixed at `generator.CONTACT_TIMESTEP` (2 ms), so a run stepped
finer keeps the same settings. All three are well under the 86 um press-fit interference, the
condition guide 02c section 3 asks for, and the derived setting is fifty times further under it
than the one the press needs.

The press bench sets its own `solref` at `world.default.geom` and passes it down to the generator,
so the force-against-depth curve and the bench's penetration and energy checks
(`tests/bricks/test_snap_bench.py`) read exactly what they read before: the bench already ran at
4 ms and 0.95/0.999/0.0005.

### 2.4 Switching it

`bricks.contact_model` in `configs/params/default.yaml`, or `SceneConfig.contact_model`:

- `material` — the default. Solver floor on the jaw pads and the arm links; MuJoCo's own settings
  on brick-on-brick, brick-on-baseplate and brick-on-table.
- `derived` — solver floor only. What the material asks for; stops the press 4.50 mm short.
- `mujoco` — MuJoCo's shipped defaults, so the change can be switched off and measured against.

## 3. Damage states

`brick_builder.bricks.damage.STATES`, in order. Each latches: a brick never gets better.

| State | Condition | Threshold on a DUPLO 2x4 wall |
|---|---|---|
| `intact` | — | — |
| `marked` | contact pressure past the tensile yield | about 27 N on a sharp edge |
| `loosened` | clutch multiplier under 0.8 | 14,845 cycles at half the allowable |
| `yielded` | any bending mode past the tensile yield | 12 to 35 N of lateral wall load |
| `cracked` | any bending mode past the flexural strength | 19 to 56 N |
| `broken` | bending past 1.5x flexural, or a released weld | — |

`yielded` is the permanent-set rung and the one this arm reaches. It sits above `loosened` because
a bent wall is a worse functional failure than a worn clutch, and below `cracked` because the wall
is still in one piece. It is not the same thing as `marked`, which is contact *pressure* under a
sharp edge and is cosmetic.

Bending allowables are written against the flexural strength, so `yielded` is the same ratio test
taken at `yield_Pa / flexural_Pa` = 44/70 = 0.63.

A state has to persist for `DamageThresholds.persist_s` = 10 ms before it latches. The timer
measures the span between the first and the last pass that saw the condition, not the number of
passes times `dt`, so the guard means the same thing whether the monitor runs every step or once
per control tick.

## 4. The damage monitor

`DamageMonitor` is a `DamageModel` attached to the snap model's observer list. Every stepping loop
in this project already calls `scene.snap.update(model, data)` once every `snap_every` physics
steps, and the snap models call their observers at the end of that pass — after the last
`mj_step`, so the contact forces are the ones it produced, and after the clutch has been
rescheduled, so a consequence written there is not overwritten.

Nothing in a controller, an environment or a script changes. `build_scene` attaches one by
default.

### 4.1 Cost

| Rate | Pass cost | Fraction of step time |
|---|---|---|
| every step | 180 us | 17 % |
| every control tick (10 steps) | 22 us | 2.2 % |

Measured on the default one-brick scene and on a three-brick scene, in-loop, against a 1.0 to
1.3 ms step on a loaded machine. `bricks.damage_monitor_every` sets the rate; 10 is the default.
A contact carrying under `DamageThresholds.force_floor` = 1 mN is skipped, which is four and a
half orders of magnitude below the smallest load that can move any state.

### 4.2 Reading it

| Where | Field |
|---|---|
| `Scene` | `scene.monitor.state`, `.peak_wall_stress_Pa`, `.line()`, `.summary()` |
| `EpisodeResult` | `damage_state`, `peak_wall_stress_Pa` (both `None` when not reported) |
| Environment `info` | `damage_state`, `peak_wall_stress_Pa` |
| `scripts/run_placement.py` | `damage: intact \| peak wall stress 3.3 MPa` |
| `python -m brick_builder.experiments pick-and-place` | a Damage section, worst state over all trials |
| `python -m brick_builder.experiments grasp-hold` | a Damage row per brick, direction and force |

`scene.monitor.worst_state_seen` and `.worst_wall_stress_Pa` survive a reset, which is what a
many-episode run wants; `.state` and `.peak_wall_stress_Pa` are the current episode's.

## 5. Breakable bricks

`bricks.breakable` in the parameter file, or `SceneConfig.breakable`. Off by default.

A breakable 2x4 is a roof body plus one free body per wall, tube and stud — sixteen pieces —
welded to the roof at each member's own root. The weld reports its constraint force and moment
about that root, and a piece is released when

    F / F_limit + M / M_limit >= 1

Limits come from `materials.weld_limits`, from the real cross-section of the member:

| Joint | Axial limit | Moment limit | Mode |
|---|---|---|---|
| Long wall to roof | 4.1 kN | 1.7 N*m | buckling at 19.5 MPa |
| Short wall to roof | 2.2 kN | 0.9 N*m | buckling at 28.4 MPa |
| Tube to roof | 2249 N | 9.81 N*m | yield |
| Stud to roof | 942 N | 4.40 N*m | shear at the root |

A released piece stays in the scene as a free body, is repainted dark grey, and latches its brick
to `broken` in the damage model. `bricks.breakable_limit_scale` multiplies every limit at once.

**Cost: 6.8x per brick.** A breakable 2x4 steps in 716 us against 105 us for the monolithic one in
the same scene, which is why it is for validation and demonstration rather than for training.

**A broken-off stud now loses its press fit.** Each stud's site sits on the stud's own piece
rather than on the roof, and the snap model declares a connection to every stud piece
(`BrickHandle.targets`), so a stud that comes away carries its connection with it instead of
leaving the roof pretending to hold a stud that is no longer there. Socket sites stay on the roof,
because a socket is an open column through the shell rather than a part. With
`breakable_studs=False` the studs are welded into the roof and their sites stay there too.

## 6. The two failures this arm can cause

### 6.1 Collision

A 0.5 kg rod — the arm's own operational-space inertia at press poses is 0.32 to 0.62 kg — driven
into the inside of a wall 15 mm below the roof, at a 2 ms step.

| Speed | Monitor: peak wall stress | Monitor: state | Breakable: peak utilisation | Breakable: pieces released |
|---|---|---|---|---|
| 0.1 m/s | 26.2 MPa | intact | 0.18 | 0 |
| 0.3 m/s | 85.8 MPa | intact | 0.59 | 0 |
| 0.6 m/s | 173.9 MPa | loosened | 1.66 | 1 |

The monitor's peak rises with speed and crosses the 70 MPa flexural strength between 0.1 and
0.3 m/s, but no permanent state latches: a rigid-body impact at a 2 ms step is resolved in one
impulsive step, and the persistence guard exists exactly so a one-step spike cannot latch. The
instantaneous peak is also step-dependent — the same 0.3 m/s impact reads 86 MPa at 2 ms and
328 MPa at 0.5 ms, because an impulse divided by a shorter step is a larger force.

Breakable bricks are the path that catches an impact, because the weld criterion is instantaneous
by design: a piece that has come apart is not a stress that might be noise. 0.6 m/s releases a
wall and latches `broken`.

The same three speeds driven by the arm itself, recorded in one take, are in
[Brick damage video](../demos/videos.md#brick-damage-video).

### 6.2 Over-squeeze

The stock jaws closed onto a 2x4's walls at the servo's own torque limit, `grip_squeeze` 6 mm past
the brick face, held for 2.4 s.

| Build | Peak lateral force | Peak wall stress | State | Pieces released |
|---|---|---|---|---|
| rigid | 10.4 N | 1.43 MPa | intact | — |
| breakable | 12.9 N | 0.76 MPa | intact | 0 (utilisation 0.025) |

The squeeze is a force the arm has; it is not a stress the brick minds. The jaw face is a 10 by
5 mm pad, so 10 N across it bends the wall by about 1.4 MPa against a 44 MPa yield — 3 % of the
permanent-set threshold. A grip damages a brick only by concentrating on an edge, which needs a
contact width of a few millimetres rather than ten.

## 7. Switching each thing on

```yaml
bricks:
  contact_model: {value: material}     # material | derived | mujoco
  damage_monitor: {value: "on"}        # "on" | "off"
  damage_monitor_every: {value: 10}    # physics steps between passes
  breakable: {value: "off"}            # "on" | "off"
  breakable_limit_scale: {value: 1.0}
  youngs_modulus_GPa: {value: 2.3}
  tensile_yield_MPa: {value: 44.0}
  flexural_strength_MPa: {value: 70.0}
```

Or directly:

```python
from brick_builder.scene.compose import SceneConfig, build_scene

scene = build_scene(SceneConfig(breakable=True, damage_monitor_every=1))
scene.step(500)
print(scene.monitor.line())
print(scene.breakable.summary())
```

## 8. Guard

Nothing on this page moves the closed loop. Measured after the change, `scripts/run_placement.py`
on seeds 0 to 4:

| Seed | Lateral | Seat depth error | Damage |
|---|---|---|---|
| 0 | 0.08 mm | 0.01 mm | intact, 3.7 MPa peak wall stress |
| 1 | 0.02 mm | 0.00 mm | intact, 3.5 MPa |
| 2 | 0.13 mm | 0.00 mm | intact, 2.8 MPa |
| 3 | 0.13 mm | 0.00 mm | intact, 3.2 MPa |
| 4 | 0.08 mm | 0.00 mm | intact, 2.8 MPa |

The same five seeds before the change read 0.02 to 0.13 mm lateral, so the placement is
unchanged. Run `scripts/run_placement.py` and `tests/action/test_nominal_tracking.py` after any
change to the contact model; the press pair is the one that moves it.
