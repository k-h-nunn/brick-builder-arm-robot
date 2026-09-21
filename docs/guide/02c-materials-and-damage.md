# Materials and damage

**Status: analysis, plus three modules that implement it, none of them wired in.** Sections 1 to 5
are the mechanics: what the plastics actually do at room temperature, what the press fit is made
of, and what it would take to hurt a brick. Section 6 was a design proposal and is now code —
`brick_builder/bricks/materials.py` (the sourced property database and every derivation on this page),
`damage.py` (the cheap per-step estimate) and `breakable.py` (a brick built from pieces that can
come apart) — described in "How to use the material and damage models" at the end. Section 7 is
the short list of measurements that would replace the worst guesses with facts.

**What is still opt-in: all of it.** Nothing in those three modules changes a default. No scene,
env or training run in this repository constructs a `DamageModel` or asks for
`derived_snap_params()`; a scene that never imports them behaves exactly as it did before, and
the simulator's `f_clutch` is still the 1.0 N placeholder rather than the 4.8 N this page derives.
Damage is not in any reward, any failure class or any termination condition yet. Section 6.3 says
what wiring it in would mean; doing it is a separate job.

Everything here is worked in SI units at 23 °C unless another temperature is stated. Every number
carries its source. Where a number is an assumption it says so, and where the research turned up
nothing at all it says that too. Several of the estimates that were handed to this analysis come
out wrong; where they do, the correction and the reason are in the text.

## 0. The short version

- The brick plastic is ABS. Its stiffness and strength at room temperature are known to about
  ±20 %, which is good enough for everything on this page.
- The press fit is uncertain by a factor of about twenty, and more arithmetic will not fix that.
  It depends on the difference between two dimensions near 13 mm, and the LEGO System design rule
  makes the equivalent difference exactly zero. The shipped default matches the simulator's own
  placeholder, 1.0 N per stud (status `assumed`) -- it sits near the bottom of the roughly 0.5 to
  5 N plausible range, and could easily be several times too low. Nobody has measured a real
  DUPLO clutch; this page's own first-principles estimate (4.79 N, below) is kept as an upper
  scenario, not shipped as the default, because it rests on an assumed interference that has
  never been checked against a real brick.
- This arm cannot break a DUPLO brick by pressing. Crushing one needs 13.2 kN, shearing a
  stud off needs about 1 kN, and the arm's measured static press at the build plate is 10 to 14 N.
- There are three ways this arm can damage a brick in one go: scuff it, bend a wall permanently
  in a jam, and crack a wall while holding it in a collision. The last is the only real breakage,
  it needs 19 to 56 N of lateral load, and the arm reaches that only dynamically, from about
  0.3 m/s upward. A fourth way is slow: the clutch wears out over tens of thousands of cycles.
- So "brick destruction" in this project means a wall going out of shape and a connection losing
  its grip, not a brick exploding. That is worth simulating, because the loads that do it are the
  same order as the loads a press policy will be tempted to use.
- **In the simulator, a misaligned press cannot bend a wall at all.** Section 6.4 has the
  converged numbers. The studs and tubes overlap by 86 µm and jam at about 0.18 mm of
  misalignment, while the nearest wall is 1.65 mm away, so the brick stalls on the stud tips and
  the wall never sees a lateral load. Loading a wall the way section 7.7 loads a real one — a
  rounded rod against its inside face — cracks it at 64 N, which is four and a half times the
  arm's press.
- **The one number that decides whether this project works at all is the press-fit force**, and
  nobody has measured it. If it is near the 4.8 N per stud the mechanics predict, a flat press of
  a 2x4 needs 44 N against an arm that has 10 to 14 N, and only a tilted press is possible. It
  is twenty minutes with a kitchen scale. See "Seating feasibility with the stock arm" and section 7.2.

## 1. Materials at 23 °C

### 1.1 The brick plastic

LEGO and DUPLO bricks are moulded in ABS (acrylonitrile butadiene styrene), which accounts for
roughly 80 % of LEGO elements; transparent parts are polycarbonate and thin flexible parts are
polypropylene or polyethylene, none of which appear in a DUPLO brick
([BrickNerd](https://bricknerd.com/home/every-type-of-plastic-used-by-lego-5-20-22), LEGO's own
[materials page](https://www.lego.com/en-us/sustainability/product-safety/materials)). No source
found states a DUPLO-specific grade. The grade usually named for LEGO, Novodur P2H-AT, could not
be confirmed against a manufacturer datasheet (the PDFs returned 404 and 403), and secondary
sources disagree about whether INEOS Styrolution or Lanxess supplies it. Treat the grade as
unknown and the property table below as "an unfilled injection-moulding ABS".

| Property | Nominal | Randomization range | Source |
|---|---|---|---|
| Density | 1050 kg/m³ | 1030 to 1070 | Novodur P2H-AT listed at 1040 kg/m³ (plastics-database.org); generic unfilled ABS 1030 to 1070 |
| Young's modulus, tensile | 2.3 GPa | 1.8 to 2.7 | Novodur P2H-AT 2500 MPa (Albis datasheet listing); generic ABS 1.58 to 2.63 GPa (ASTM D638), 1.79 to 2.89 GPa (ISO 527-2) |
| Poisson's ratio | 0.35 | 0.34 to 0.38 | aggregated literature; ABS datasheets rarely publish it. **Low confidence.** |
| Tensile yield stress | 44 MPa | 36 to 52 | Novodur P2H-AT 44 MPa; generic 35.1 to 51.0 MPa (ASTM D638) |
| Flexural strength (modulus of rupture) | 70 MPa | 60 to 95 | generic ABS 72 to 97 MPa (MakeItFrom) |
| Shear strength (assumed 0.6 × tensile yield) | 26 MPa | 22 to 31 | **Assumption.** No measured ABS shear strength was found |
| Elongation at break | 20 % | 3 to 50 | generic ABS 3.4 to 26 % (ISO 527-2), 1.0 to 57 % (ASTM D638) |
| Notched Izod impact | 300 J/m | 70 to 370 | Novodur P2H-AT 300 J/m at 23 °C; generic range from MakeItFrom |
| Glass transition | 104 °C | 100 to 106 | DSC on an actual unaged LEGO brick ([PMC10422395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10422395/)). **High confidence**, and the only property here measured on a real brick |
| Heat deflection temperature | 94 °C at 0.45 MPa | — | plastics-database.org for Novodur P2H-AT; generic unfilled ABS 75 to 88 °C at 1.8 MPa |
| Coefficient of friction, ABS on ABS | 0.35 (the value already in the bench) | 0.2 to 0.5 | **No source exists.** See below |
| Coefficient of restitution | unknown | — | **No source exists.** See section 3.3 |
| Linear thermal expansion | 90 × 10⁻⁶ /K | 70 to 110 | generic ABS. Used only for the temperature check in 2.7 |

Two of those rows are holes rather than numbers. The friction coefficient is the worse of the
two: the Engineering ToolBox friction table does not list ABS at all, no ABS-on-ABS or ABS-on-PLA
measurement was found anywhere, and the 0.3 that has been carried through this project's design
work traces to nothing. Every clutch force on this page scales linearly with it, so it is the
single most load-bearing unsourced number in the whole model. Section 7.3 measures it with a
ramp and a brick.

Aged and coloured bricks are a separate material. UV exposure drops the glass transition of a real
brick from 104 °C to 85 °C after 1390 hours under a xenon lamp at 60 W/m², by chain scission that
also lowers toughness (PMC10422395). Dark brown, dark red and gold elements made through the 2010s
became notoriously brittle, attributed to pigments catalysing oxidative attack on the butadiene
phase, until the dyes were reformulated around 2019 (enthusiast press; no primary materials paper
found). None of this matters for a simulator run at room temperature on new bricks, but it does
mean that if the physical bricks used for calibration are old or dark brown, they are not the
material in the table above. Use new, light-coloured bricks for every measurement in section 7.

### 1.2 The arm's printed structure

The SO-101's structural parts are printed, and upstream recommends PLA+ at 15 % infill, 0.2 mm
layers, with no wall count specified
([TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/blob/main/README.md)). A
printed part is not a bulk material: it is anisotropic, and the anisotropy is severe in exactly
one direction.

| Property | Nominal | Range | Note |
|---|---|---|---|
| Density | 1240 kg/m³ | — | bulk PLA |
| Modulus, in-plane (layers parallel to the load) | 3.5 GPa | 2.5 to 3.9 | FDM tensile study, room temperature |
| Modulus, through-layer | 2.1 GPa | 1.5 to 2.5 | about 40 % below in-plane |
| Strength, in-plane | 55 MPa | 45 to 65 | |
| Strength, through-layer (interlayer bond) | 5 MPa | 3 to 15 | about 91 % below in-plane. This is what fails |
| Glass transition | 57 °C | 55 to 60 | |
| Heat deflection temperature | 55 °C | 45 to 60 | |

The through-layer strength is the number that matters. A printed part loaded so that a layer line
is in tension is roughly a tenth as strong as the same part loaded along its layers, and nothing
upstream specifies print orientation, so which case applies to any given bracket depends on
whoever printed it. The only reported real failure on this kit is consistent with that: the
gripper servo horn screw stripping the threads cut into the printed boss (Waveshare assembly
wiki), not the servo's steel gears.

Room temperature also has a different meaning on the arm than on the bricks. PLA's heat deflection
temperature is 45 to 60 °C and the servo cuts its own torque at about 70 °C internal
(Feetech datasheet, translated), with one bench test recording a 15 °C rise after ten minutes of
continuous motion under load. A motor bracket clamped around a working servo can therefore sit
above the temperature at which its own material starts to creep under sustained load, while the
bricks two centimetres away are still at 23 °C and completely unaffected. If anything in this
project softens at room temperature, it is the arm, not the brick.

### 1.3 The servo gear train

| Property | Value | Source |
|---|---|---|
| Gear material | steel gears in a glass-filled nylon case, plastic output horn | RobotShop, Seeed listings |
| Stall torque at 7.4 V | 1.9 N·m (datasheet) or 3.41 N·m (identified law at full duty) | `ArmConfig.torque_limit_nm`, `servo_stall_torque` |
| Dry friction in the train | 0.0515 N·m | BAM identification, `BAM_FRICTION_BASE` |
| Viscous friction | 0.0599 N·m·s/rad | `BAM_FRICTION_VISCOUS` |
| Apparent rotor inertia at the output | 0.0261 kg·m² | `BAM_ARMATURE` |
| Backlash | 0.5° specified, 0.87° measured | Feetech spec vs Robonine bench test |
| Overload trip | above about 80 % of stall held for about 25 ms | Feetech datasheet, translated |
| Thermal cutout | about 70 °C internal | same |

The gear train is steel and is not a candidate for damage in this task. The overload trip is a
mechanical fuse that already exists: the servo gives up before it does anything
violent, which is one more reason the arm is not a brick-breaking machine.

## 2. The press fit, from first principles

![Member stiffnesses and the resulting holding force per stud](../media/materials/press-fit-stiffness.png)

### 2.1 Press-fit geometry

Take the DUPLO numbers already in `brick_builder/bricks/spec.py`: pitch 16 mm, stud outer diameter
9.4 mm, hollow stud bore 6.5 mm, tube outer diameter 13.4 mm, tube bore 10.7 mm, wall 1.55 mm,
roof 1.5 mm, brick height 19.2 mm, stud height 4.5 mm.

A stud from the brick below rises into the square cell bounded by four tube positions and, at the
perimeter, by the walls. Its centre sits a half-diagonal from each neighbouring tube centre:

    d = pitch × √2 / 2 = 16 × 1.41421 / 2 = 11.314 mm

and the two radii that have to fit into that distance sum to

    r_stud + r_tube = 4.700 + 6.700 = 11.400 mm

so each stud-and-tube pair overlaps by **0.086 mm** on the radius. Using the caliper figure for the
tube (13.48 mm) instead of the nominal gives 0.126 mm. Both are in the figure above.

Now count contacts on a 2x4 pressed onto another 2x4. Studs sit at (±8, −24/−8/8/24) mm; the upper
brick has three tubes on its centreline at (0, −16/0/16) mm. Each of the four interior studs is a
half-diagonal from two tubes; each of the four corner studs is a half-diagonal from one. That is
**12 stud-on-tube contacts** for the whole brick, an average of 1.5 per stud.

The walls are not in contact at all. The wall inner face sits at |x| = 14.350 mm and a stud's outer
surface reaches |x| = 12.700 mm, leaving a **1.65 mm gap**, and the end walls leave the same gap.
A real DUPLO brick bridges that gap with internal ribs (the dimension table on
[Simulator and bricks](02-simulator-and-bricks.md) lists them at about 1.5 mm wide, "touching the
studs"). The generator models no ribs, so in the simulated geometry the perimeter load path does
not exist. Any analysis that puts the clutch in the walls is describing a brick this project does
not have.

That is the first correction to the first-pass estimate handed to this work. It treated the wall as
the member that deflects and got 0.6 to 1.1 N of holding force per stud, close to the simulator's
1.0 N placeholder. The wall is the right member only for the perimeter contacts of a real ribbed
brick, and it is by far the most compliant one, so that calculation lands on the bottom of the
range rather than in the middle of it.

### 2.2 Member stiffness

Three elastic members can take up the interference, and they are wildly unequal.

**The wall, as a plate strip.** The wall is built into the roof at the top and free at the bottom,
17.7 mm tall, 1.55 mm thick. A load at distance L from the roof, spread over width b, deflects it
like a cantilever strip in plane strain:

    k = 3 E' I / L³,   I = b t³ / 12,   E' = E / (1 − ν²)

With E = 2.3 GPa, ν = 0.35, t = 1.55 mm, L = 15 mm (a contact near the bottom edge) and b = 10 mm,
that is I = 3.10 × 10⁻¹² m⁴ and **k = 7.2 kN/m**. The first-pass estimate of 7 to 8 N/mm is
confirmed for those assumptions. But k goes as b/L³, so the same wall is **100 kN/m** against a
load spread over 30 mm at mid-height (L = 9 mm). A wall has no single stiffness; it has a stiffness
per load case, varying by a factor of fourteen across load cases this arm can actually produce.
Section 6 leans on that.

**The tube, as a ring.** A ring squeezed between two opposite point loads changes diameter by
Δd = 0.149 F R³/(EI), so its radial stiffness at one load point, relative to its own centre, is
2/(Δd/F). With mean radius 6.025 mm, wall 1.35 mm and a loaded band equal to the engaged stud
height (4.5 mm), I = 9.23 × 10⁻¹³ m⁴ and **k = 130 kN/m**.

**The hollow stud, as a ring.** Same formula, mean radius 3.975 mm, wall 1.45 mm, band 4.5 mm:
**k = 562 kN/m**.

In series at one stud-on-tube contact, **k = 106 kN/m**, fifteen times the compliant wall case.

**Local flattening is negligible.** Two parallel cylinders in line contact approach each other by
about 2.6 µm at 5 N over a 4.5 mm band (Johnson's line-contact expression, E* = E/2(1−ν²) =
1.31 GPa), an equivalent stiffness of 1900 kN/m. Against interferences of 86 to 126 µm, Hertzian
flattening accounts for under 3 % of the take-up. The press fit is structural flexure, not contact
deformation, which is convenient: it means simple beam and ring formulas are the right tools and a
contact-mechanics treatment is not needed.

The same calculation gives the contact stress, which is needed in section 4. At 5 N over 4.5 mm the
half-width is 55 µm and the peak pressure is 13 MPa; at 9 N it is 73 µm and 17 MPa. Both are well
under yield, so a correctly seated stud does not mark its tube. The peak bending stress around the
tube ring, σ = 0.318 F R (t/2)/I, is 12.7 MPa at 9 N, also comfortable.

### 2.3 Predicted force and its uncertainty

Normal force per contact is k δ, clutch force per stud is μ × k δ × (contacts per stud):

| Path | k | δ = 0.086 mm | δ = 0.126 mm |
|---|---|---|---|
| Stud on tube, 1.5 contacts per stud, μ = 0.3 | 106 kN/m | 4.1 N/stud, 33 N per 2x4 | 6.0 N/stud, 48 N per 2x4 |
| Wall-only path (the first-pass estimate) | 7.2 kN/m | 0.3 N/stud | 0.4 N/stud |

Then consider LEGO System, where the same arithmetic is exact rather than approximate. The System
design rule is pitch × √2 − stud diameter = tube outer diameter: 8 × 1.41421 − 4.8 = 6.514 mm,
against a quoted tube diameter of 6.51 mm. **The nominal interference is zero.** The clutch of a
System brick comes entirely from mould draft, from the tolerance band, and from elastic recovery
after ejection, not from a designed interference at all. The widely repeated "0.1 to 0.2 mm
interference" figure traces to one engineering blog interpreting LEGO's marketing claims, with no
primary source behind it.

This has a sharp consequence for DUPLO. Applying the same rule would put the DUPLO tube at
16 × 1.41421 − 9.4 = 13.227 mm, and the 13.4 mm in `spec.py` is a hobbyist dimension, not a LEGO
drawing. If DUPLO follows the System rule, the nominal interference is zero and the whole table
above collapses to whatever the tolerance happens to be. If the 13.4 mm figure is a real design
value, the table stands and the clutch is 33 to 48 N per 2x4.

So the first-principles answer for the press fit spans **0.3 to 6 N per stud**, a factor of twenty,
and the uncertainty is not in the mechanics. It is in a single geometric difference of two
numbers near 13 mm, each of which is quoted by hobbyist sources to ±0.05 mm, multiplied by a
friction coefficient for which no measurement exists. Adding more careful elasticity to this
calculation would be wasted effort. Summary:

- The simulator's 1.0 N per stud is **not wrong**, but it sits near the bottom of the range and is
  more likely several times low than high.
- A 2x4 probably takes somewhere between 8 and 80 N to pull off, taking the stud-on-tube path with
  friction anywhere in 0.2 to 0.5 and interference anywhere in the band above. The wall-only path
  would give a tenth of that, and section 2.1 is the reason it is the wrong path. There are two
  external anchors: a hobbyist rig measured a System stud at about 6.4 N minimum
  ([lambsandwich.net](https://lambsandwich.net/welcome-to-lambsandwich-net/a-comment-on-lego-clutch-power/),
  small sample, non-standard rig), and the widely quoted "3 to 5 N to pull two bricks apart"
  figure traces to an aggregator site with no stated method and should be ignored.
- Section 7.2 settles it in about twenty minutes with a bottle of water and a kitchen scale, and
  that measurement is worth more than everything in this subsection.

### 2.4 Force against insertion depth

![Predicted force against depth compared with the three curve families in the code](../media/materials/force-vs-depth.png)

For a straight interference between two parallel cylinders, the normal force per unit of engaged
length is constant, so friction is proportional to engaged length and the press force **rises
linearly with depth**. That is exactly `engagement()`'s `"ramp"` family with `exponent = 1`, which
is the simulator's default, so the default has a mechanism behind it. `"plateau"` does not: no
straight-sided interference produces full force the instant the stud engages.

Mould draft changes the answer and is the part the first-pass estimate missed. Injection-moulded
parts need draft to eject, typically 0.5° to 1.5° per side. Over the 4.5 mm of stud engagement a
1° draft changes the radius by **79 µm**, which is the same size as the entire nominal
interference. If the stud and tube are both drafted so that the fit tightens as the stud goes in,
the interference grows roughly linearly with depth, friction goes as the product, and the force
rises roughly as **depth²**. The right default is therefore not exponent 1 but an exponent
randomized over **1.0 to 2.5**, until the curve is measured.

On top of that, a chamfered or radiused stud tip gives an entry peak: the tip wedges a local
deflection larger than the steady-state one before it slides in, which is the "click". The
`"detent"` family in `engagement()` is the shape for it, and its bump parameters are what the
measurement in section 7.4 would fit. The research found no published force-against-displacement
curve for seating a LEGO brick, from any source, which the reports flag as the single most
consequential gap for this project. So the shape stays a parameter.

Sensitivity, for the randomization config:

| Perturbation | Effect on clutch force |
|---|---|
| ±0.01 mm on the tube diameter (the mould tolerance often quoted) | ±6 to ±12 %, since δ is 86 to 126 µm and the diameter change is halved onto the radius |
| ±0.05 mm (realistic caliper uncertainty on a hobbyist dimension) | ±30 to ±60 % |
| friction 0.2 to 0.5 | −33 % to +67 % |
| E across its range, 1.8 to 2.7 GPa | ±20 % |
| number of engaged contacts, 1 or 2 per stud | ±33 % |

Randomizing `f_clutch` over a factor of three either side of nominal is not excessive caution here.
It reflects what is actually known.

### 2.5 Push-on against pull-off

The model already scales the friction bound by `1 ± beta` with the direction of travel, giving
`f_clutch (1+β)` to push and `f_clutch (1−β)` to release. That asymmetry is real but it is not a
material property: it comes from the draft, which makes the fit tighter going in than coming out,
plus the difference between static and kinetic friction. With β = 0.15 the model's push-to-pull
ratio is 1.35. Nothing found measures it. A ratio between 1.2 and 1.8 is what draft of 0.5° to 1.5°
would produce, so the current value is reasonable and should be randomized over that range.

### 2.6 Stick-slip

Dry friction against an elastic member is a stick-slip system, and the per-stud friction joint
model gets that for free because MuJoCo solves dry friction properly. The characteristic length scale is the elastic
deflection at break-away, δ_break = f_clutch/k. For the tube path at 4 N per stud that is 38 µm;
for the wall path it is 550 µm. So a real press should judder on a scale of tens of microns, well
below anything the arm can either command or observe, and the simulator does not need to resolve
it. What it does need is the break-away force itself, which is what section 7.4 measures.

### 2.7 Temperature around room temperature

The brief asks specifically for room-temperature material behaviour, so this section is
precise about how much room temperature actually matters.

**Thermal expansion does almost nothing**, and this is the non-obvious part. The interference is a
difference between dimensions of two parts made of the same material at the same temperature. Both
scale by (1 + αΔT), so the difference scales by the same factor: at α = 90 × 10⁻⁶/K, a 10 K change
alters an 86 µm interference by **0.09 %**. Thermal expansion would matter only if the brick and
the plate were at different temperatures or made of different plastics, and in this task they are
neither.

**Modulus does the work.** ABS softens continuously toward its glass transition at 104 °C. No
source found gives dE/dT for ABS near 23 °C; assuming the commonly used −0.7 %/K for amorphous
thermoplastics well below Tg, an 18 to 28 °C room changes the modulus by about ±3.5 %, and the
clutch force with it. That is smaller than the tolerance spread and much smaller than the friction
uncertainty.

So at room temperature the bricks are effectively temperature-independent for this task, to within
a few percent, and a domain randomization range of ±5 % on `f_clutch` covers the whole plausible
room. The temperature effect that does matter in this project is on the arm, in section 1.2, where
a printed bracket beside a working servo can reach its own heat deflection temperature.

## 3. Contact, damping and restitution at a 2 ms time step

![Measured penetration and restitution in the simulator](../media/materials/contact-at-2ms.png)

These are measured in this repository's own solver configuration (implicit-fast integrator,
elliptic cone, `impratio` 10, Newton solver, 100 iterations) on a 12.5 g box pressed onto a plate,
which is the same configuration `BenchConfig` and `patch_so101` set up.

### 3.1 Contact softness

MuJoCo's soft constraint takes a `solref` of (time constant, damping ratio), and the time constant
must be at least twice the step, so **dt = 2 ms forces a floor of 4 ms** and the bench sits exactly
on it. Measured steady-state penetration of a brick pressed onto the plate, at the bench's
`solref = (0.004, 1.0)` and `solimp = (0.95, 0.999, 0.0005)`:

| Load | 4 ms | 10 ms | 20 ms |
|---|---|---|---|
| 0.12 N (own weight) | 2.2 µm | 13.5 µm | 52.9 µm |
| 1.1 N | 19.8 µm | 110.9 µm | 246.7 µm |
| 5.1 N | 84.9 µm | 257.6 µm | 360.1 µm |
| 14.1 N (the arm's press ceiling) | 180.7 µm | 335.3 µm | 417.3 µm |
| 60.1 N | 307.4 µm | 420.4 µm | 492.0 µm |

The number to stare at is 180.7 µm. **At the arm's own maximum press, the simulated brick sinks
further into the plate than the entire interference the clutch is made of.** The contact is also
strongly non-linear, by design: incremental stiffness runs from 57 kN/m near zero load to 547 kN/m
at 60 N, because `solimp` ramps the impedance from 0.95 to 0.999 over the first 0.5 mm.

This is the quantitative justification for the per-stud friction joint abstraction in `snap.py`. Ordinary contact at
a 2 ms step cannot hold a 0.086 mm interference, because at working loads it is already deeper than
that. Any attempt to build the clutch out of geometric interference and contact stiffness would be
resolving a fit inside its own penetration error. Putting the clutch on a dry-friction joint
sidesteps the whole problem, and the numbers above are why.

Reducing the step does not help as much as one would hope. At 0.5, 1 and 2 ms with the time
constant held at 4 ms the penetration is identical (180.7 µm each) because the time constant, not
the step, sets the stiffness. Only going to 8 ms of time constant (forced by a 4 ms step) changes
it, and in the wrong direction, to 302.7 µm.

### 3.2 Settings per contact pair

| Pair | condim | solref | solimp | friction | Confidence |
|---|---|---|---|---|---|
| brick on brick, brick on baseplate | 3 | (0.004, 1.0) | (0.95, 0.999, 0.0005) | 0.35 | friction has **no source**; randomize 0.2 to 0.5 |
| jaw pad on brick (PLA on ABS) | 4, priority 1 | (0.004, 1.0) | (0.95, 0.999, 0.0005) | currently 1.0 | 1.0 is a modelling choice to make the grasp hold, not a measurement. PLA is reported to have *lower* friction than ABS, so the physical value is more likely 0.3 to 0.5. Randomize 0.3 to 1.0 and keep the priority so the pad's value wins |
| brick on table | 3 | (0.004, 1.0) | same | 0.4 | depends entirely on the table; randomize 0.2 to 0.8 |
| arm link on anything | 3 | (0.004, 1.0) | same | default | collisions, not manipulation; nothing rests on these |

The `condim 4` with `priority 1` on the pads is already correct and should stay: it gives the pad
torsional friction, which is what stops a gripped brick from spinning about the pinch axis, and
the priority makes the pad's friction win over the brick's.

### 3.3 Restitution

MuJoCo has no restitution parameter. Whatever bounce appears is a side effect of `solref`, and it
is nearly independent of impact speed, which is the opposite of how a real elastic-plastic material
behaves (restitution falls as impact speed rises once local yielding starts).

| Damping ratio | e at 0.05 m/s | 0.2 | 1.0 | 2.0 |
|---|---|---|---|---|
| 1.0 (the project default) | 0.23 | 0.18 | 0.19 | 0.22 |
| 0.7 | 0.47 | 0.37 | 0.38 | 0.45 |
| 0.4 | 1.47 | 1.15 | 1.17 | 1.37 |

The scatter within a row is a sampling artifact: the rebound takes a handful of steps and where the
separation falls inside a 2 ms step shifts the reading. The row-to-row trend is the real result.
**A damping ratio of 0.4 creates energy**, returning more speed than went in, so it is not usable
at any step size. The default of 1.0 gives e ≈ 0.2 and that is what the simulator has.

No coefficient of restitution for ABS or for a LEGO brick was found in any source. A hard plastic
on a hard table is usually somewhere between 0.4 and 0.7, so **the simulator under-bounces**.
Since the only thing that depends on it is how far a dropped brick skitters, this is acceptable,
but it should be written down rather than discovered later: a dropped brick in this simulator stops
sooner than a real one. Section 7.6 measures the real value with a phone camera, and if it matters,
a damping ratio of 0.7 buys e ≈ 0.4 at the cost of a springier contact everywhere else.

## 4. Failure modes

![Every failure threshold against what the arm can apply](../media/materials/load-ladder.png)

Cross-sections for a 2x4, from the same shell the generator builds: four walls totalling
286.8 mm², three tubes totalling 153.3 mm², 440.1 mm² in all. A 2x2 has the same walls scaled
(2 walls at 31.8 mm, 2 at 28.7 mm) plus one tube. The generated 2x4 masses 12.52 g against the
12.2 g catalogue figure, and the generated System 2x2 masses 1.115 g against the 1.152 g weighed
in the Open University test, so the shell model and the 1050 kg/m³ density are both within 3 %.

### 4.1 Crushing

Compressive load runs down the walls and tubes. The first-pass estimate took 296 mm² of wall at
65 MPa and got 19 kN. Three corrections:

1. The three tubes carry load too; the section is 440 mm², not 287 mm².
2. 65 MPa is a flexural strength, the stress at which a bent bar ruptures. The compressive
   allowable is the yield stress, 44 MPa.
3. **Thin walls buckle before they yield.** For a plate of thickness t and width b compressed along
   its height a, σ_cr = k π² E / (12(1−ν²)) × (t/b)², with k ≈ (b/a + a/b)² for a short wide plate.
   At the idealised 63.8 mm footprint edge (a = 17.7 mm, t = 1.55 mm) that is k = 15.1 and
   **σ_cr = 19.2 MPa**, well below yield. The tubes do not buckle: a cylinder's critical stress is
   0.6 E t/R = 309 MPa, so they yield first at 44 MPa.

Each *pair* of walls has to be taken at its own critical stress, because buckling depends on the
wall's length, and the four walls of a 2x4 are not the same length. Working it that way, from the
lengths the generator actually builds:

| Member | Section | Critical stress | Load |
|---|---|---|---|
| Front and back walls | 2 × 31.8 mm × 1.55 mm = 98.6 mm² | 28.4 MPa, buckling | 2.80 kN |
| Left and right walls | 2 × 60.7 mm × 1.55 mm = 188.2 mm² (shortened at the corners) | 19.5 MPa, buckling | 3.66 kN |
| Three tubes | 153.3 mm² | 44 MPa, yield | 6.75 kN |
| **Total** | **440.1 mm²** | | **13.2 kN** |

**13.2 kN** is what `crush_load()` computes and what this page quotes everywhere. An earlier draft
said 12.2 kN, which came from applying the *softest* wall's 19.2 MPa to all 286.8 mm² of wall — a
fair conservative simplification, but 8 % low and not what the code does. A DUPLO 2x2 comes out at
**7.85 kN** the same way (2.80 kN of front/back wall at 28.4 MPa, 2.80 kN of left/right wall at
31.5 MPa, one tube at 2.25 kN); an earlier draft said 7.6 kN for the same reason.

The check: the Open University measured a System 2x2 failing at 4240 N with a slow plastic collapse
rather than a fracture, and its 82.9 mm² section makes that a mean stress of 51.1 MPa. That brick's
walls are short and stubby (k = 5.6, σ_cr = 69 MPa), so it really should have yielded rather than
buckled, and it did, at close to the ABS yield stress. The two cases agree, and they agree on the
mechanism as well as the number.

Whichever number is used, this is **three orders of magnitude beyond the arm**. Not relevant.

### 4.2 Stud sheared off the roof

The stud root is an annulus of area π(9.4² − 6.5²)/4 = 36.2 mm². At the assumed shear strength of
0.6 × yield = 26 MPa that is **942 N**, and 1412 N if the flexural figure is used instead. The
first-pass estimate of 1.1 kN was right. Also not relevant: two orders of magnitude beyond the arm.

### 4.3 A wall bent outward by a wedging load

This is the mode that matters. A misaligned stud whose chamfer rides up the inside of a wall
pushes that wall outward. Treat the wall as a cantilever strip built into the roof:

    σ = 6 F L / (b t²)

with L the distance from the contact to the roof and b the width the load spreads over. At
L = 15 mm (a contact near the bottom edge) and t = 1.55 mm:

| Effective width b | First yield, 44 MPa | Rupture, 70 MPa |
|---|---|---|
| 10 mm | 11.7 N | 18.7 N |
| 20 mm | 23.5 N | 37.4 N |
| 30 mm | 35.2 N | 56.1 N |

So a **permanent set at 12 to 35 N** and a **crack at 19 to 56 N**, depending on how far the load
spreads. The first-pass estimate put the crack at 25 to 40 N, inside that band but slightly
optimistic at the low end, and it missed that the permanent-set threshold is lower than the crack
threshold and is the one that arrives first. A wall that has taken a set never grips properly
again, which is a functional failure even though the brick is intact.

**Those numbers are nominal stresses, and the wall-to-roof junction is not a nominal place.** The
beam formula gives the stress in a prismatic strip; a real moulded corner has a fillet, and the
local stress at the root of a filleted shoulder in bending is higher than the nominal by the
stress concentration factor K_t. Published charts put K_t at roughly 1.3 to 2.0 for the fillet
radii a 1.55 mm wall would plausibly carry (r/t of about 0.1 to 0.5), falling toward 1.0 only with
a generous radius. Nobody has measured it on a DUPLO brick, and the model does not apply it: the
default is K_t = 1.0, so every number on this page is a nominal-stress number. The direction of
the error is one-sided — **a real wall is weaker than this model says, never stronger**:

| K_t | Permanent set | Crack |
|---|---|---|
| 1.0 (the default, nominal stress) | 11.7 to 35.1 N | 18.6 to 55.9 N |
| 1.3 | 9.0 to 27.0 N | 14.3 to 43.0 N |
| 2.0 | 5.9 to 17.6 N | 9.3 to 27.9 N |

At the pessimistic end the permanent-set floor is 5.9 N, which is *inside* what the arm can apply
(10 to 14 N of press, 17 N of grip) rather than at the edge of it. So K_t is not a detail: it is
the difference between "the arm cannot quite bend a wall" and "the arm can". It is a parameter of
`wall_bending_stress()` and `wall_thresholds()` with a randomization range of 1.0 to 2.0
(`materials.WALL_KT_RANGE`), and section 7.7 settles it for free — a measured force at a visible
set contains K_t whether or not anyone names it.

Where does a lateral force like that come from? A vertical press on an inclined chamfer:

    F_lateral / F_vertical = (tan θ − μ) / (1 + μ tan θ)

with θ the chamfer angle from the press axis. At 45° and μ = 0.3 the ratio is 0.54; at 60° it is
0.94; at 30° it is 0.24. So a 14 N press on a badly misaligned stud delivers 3 to 13 N sideways,
under the permanent-set threshold but within a factor of two of it. Add a sharp edge contact
instead of a spread one, and the margin is gone.

### 4.4 Roof punched through by a jaw tip or a stud tip

Punch-out shear over a perimeter p through a 1.5 mm roof at 26 MPa: a stud-sized 9.4 mm circle
gives **1152 N**, a 4 mm square jaw corner gives **624 N**. Not reachable. The roof is more
vulnerable in bending near its edge, where a 1 mm strip of it has a limiting moment of 26 mN·m, but
the jaw pads are 15 mm wide and spread their load, so this does not arise in normal handling.

### 4.5 Tube ovalisation

The ring bending stress at a stud contact is 0.318 F R (t/2)/I = 12.7 MPa at 9 N, and reaches
44 MPa at about 31 N per contact, around four times the designed contact force. A jammed press that
wedges one stud hard into one tube could get there, but only in the same jam that is already
bending a wall at a lower load, so the wall goes first.

### 4.6 Surface marking

The Hertz line contact at a designed clutch load peaks at 13 to 17 MPa, far under yield, so normal
seating leaves no mark. A box corner or a jaw edge is a different matter: the same 17 N grip spread
over 3 mm rather than 30 mm raises the wall bending stress from 13 MPa to 127 MPa, past rupture.
**Marking and cracking are both questions of contact width, not of force**, which is why section 6
insists the damage estimate read contact geometry and not just contact force.

### 4.7 Fatigue, wear and relaxation

The only cycle-life datum found is a hobbyist rig that assembled and separated System bricks
**37,112 times over about ten days** before they lost their clutch
([Philippe Cantin's own write-up](http://phillipecantin.blogspot.com/2013/02/legos-magic-number-is-37112.html),
corroborated by [Arduino](https://blog.arduino.cc/2013/04/16/lego-stress-test-magic-number/),
[Make:](https://makezine.com/article/technology/arduino/stress-testing-lego/) and
[Hackaday](https://hackaday.com/2013/04/19/measuring-the-lifespan-of-lego/); the HOTH Bricks URL
this page used to cite now 404s). Three details of that test matter and the earlier draft left all
three out:

- **The bricks were old and already worn**, not new. The author: "I used old LEGO bricks for this
  first test so they were clearly used before." The press coverage dates them to the 1970s. This
  page is otherwise careful to say "use new, light-coloured bricks for every measurement", so the
  one cycle-life anchor it has is on the opposite material.
- **They were 2x3 System bricks**, not DUPLO and not a 2x4.
- **What failed was about half the clutch, not all of it**: "both pieces failed at 50 %... Both
  bricks can still hold on (not strongly) to normal bricks but, when put together, they can't
  hold." That is the number section 6.1's constants are anchored on, and it is what "lost their
  clutch" means here.

No S-N curve for injection-moulded ABS at 23 °C was found, and no DUPLO cycle-life data exists at
all. ABS stress relaxation obeys time-temperature superposition over 0.5 to 5 % strain and 10 to
50 °C, so a room-temperature master curve can be built, but no relaxation modulus numbers were
obtainable (the relevant paper is paywalled).

For this project the practical form is: the clutch decays with cycles and with time under load, on
a scale of tens of thousands of cycles, and the simulator should represent that as a slow
multiplier rather than as an event. Section 6.1 gives the form and section 7.5 measures the rate.

### 4.8 Impact

A collision is a different regime, and it is the one that can actually break something. For a
striker of effective mass m at speed v hitting a spring of stiffness k, the peak force is
F = v √(m k).

The effective mass is measurable rather than guessable. Taking the operational-space inertia of
the patched arm at press poses over the build plate, 1/(J M⁻¹ Jᵀ)_zz comes out at 0.62 kg at
(0.17, 0, 0.03) m, 0.50 kg at (0.17, 0, 0.08) m and 0.32 kg at (0.25, 0, 0.05) m, so **0.3 to
0.6 kg** across the working region. The tool speed bound from the servo firmware's own rate
limit of 5.29 rad/s is 1.9 to 3.0 m/s vertically at those poses, though no sensible controller
commands anything like that.

| Case | m | k | v | Peak force |
|---|---|---|---|---|
| Arm drives a held brick into a wall, gentle | 0.5 kg | 8 kN/m (compliant wall) | 0.3 m/s | 19 N |
| Arm drives a held brick into a wall, fast | 0.5 kg | 8 kN/m | 1.0 m/s | 63 N |
| Arm strikes a *free* brick | 0.012 kg reduced | 8 kN/m | 1.0 m/s | 10 N |
| Brick dropped 0.3 m onto a table | 0.012 kg | 100 kN/m (edge) | 2.4 m/s | 83 N |

A free brick is mostly just knocked away: the reduced mass is the brick's own 12 g, so it absorbs
almost nothing and flies off at nearly the striking speed. A brick clamped in the jaw is backed by
the whole arm, and the peak force climbs into the wall-cracking band. **That is the one way this
arm can break a brick, and it is a speed limit rather than a force limit.**

The dropped brick at 83 N looks alarming but is not: ABS has a notched Izod impact strength of
300 J/m and DUPLO bricks survive being dropped by toddlers all day. The 0.036 J of a 0.3 m drop is
absorbed over a corner that deflects elastically. This case is in the table as a *negative*
control: any damage model that flags an ordinary drop as breakage is wrong, and that is a test
worth writing.

### 4.9 Order of failure on the arm

| Part | Limit | Reached at |
|---|---|---|
| Servo overload trip | 80 % of stall for 25 ms | routinely, in any jam. This is the real protection |
| Servo torque | 1.9 N·m (datasheet) | 10 to 14 N at the tool, measured on the force map |
| Printed jaw finger, bending at a layer line | not estimable from what is in the repository | see below |
| Printed servo-horn boss | thread stripping | the one failure actually reported on this kit |
| Steel gear teeth | far above the servo's own torque limit | never |
| Brick wall | 12 to 35 N lateral | see 4.3 |

The jaw finger deserves a note because it is tempting to compute and the computation would be
wrong. `jaw_geometry` fits axis-aligned bounding boxes to the STL, which at the finger root is
23.8 × 22.9 mm. Treated as solid at PLA's 5 MPa through-layer strength that section would take
10.4 N·m, or 231 N at the pad, but the real printed part is neither solid nor box-shaped and the
bounding box overstates it by an unknown factor. What can be said is that the finger is a stout
part, that nobody has reported breaking one, and that the failure people do report on this kit is
the horn screw stripping its printed boss. Treat the jaw as stronger than the brick wall until
someone measures it.

## 5. Damage modes and safe limits

Stated plainly, with the numbers from section 4:

1. **The arm cannot crush, shear or punch a brick.** Those modes need 0.6 to 13.2 kN and the arm
   produces 10 to 14 N of static press at the plate (measured on the force map at 7.4 V with the
   datasheet torque limit; 16 to 23 N if the identified servo law's higher ceiling is used) and
   17 N of grip.
2. **The arm can scuff a brick** wherever a sharp edge concentrates a few newtons.
3. **The arm can permanently bend a wall**, at 12 to 35 N of lateral load — 6 to 18 N if the
   fillet stress concentration of section 4.3 is at the pessimistic end of its range — which it
   reaches only in a jam where a misaligned stud wedges against a wall, or when the jaw pinches an
   edge rather than a face. The brick stays in one piece and stops gripping properly. Section 6.4
   is the caveat that matters here: in the geometry this project simulates, a misaligned stud
   cannot reach a wall at all, so the only route left in simulation is the pinched edge.
4. **The arm can crack a wall in a collision**, at 19 to 56 N, which it reaches only dynamically:
   about 0.3 m/s while holding a brick against something solid.
5. **The arm can wear the clutch out**, over tens of thousands of cycles.

Limits a press policy should respect, with a safety factor of 2 against the bottom of each band:

| Quantity | Limit | Derivation |
|---|---|---|
| Vertical press at the tool | 7 N | half of 14 N, and also half the 12 N permanent-set threshold once the wedge factor is applied |
| Lateral force on any wall | 6 N | half of the 12 N permanent-set threshold |
| Tool speed at contact | 0.15 m/s | half the 0.3 m/s that reaches 19 N with 0.5 kg of tool |
| Grip force on a brick face | 8 N | half of 17 N; also keeps a 3 mm edge contact under 60 MPa |
| Wrist torque during a tilted press | 0.5 N·m | a quarter of the 1.9 N·m servo limit, leaving room for gravity |

Those limits are derived at K_t = 1. At the pessimistic end of section 4.3's fillet range they all
halve, and the 6 N lateral limit becomes 3 N.

There is a conflict buried in those numbers and it should not be glossed over. If the clutch really
is the 4.8 N per stud the mechanics predict, a 2x4's whole-brick clutch (8 studs) is 38 N, and
seating it flat needs 1.15 times that — 44 N of push — which is more than this arm has anywhere in
its workspace at any supply voltage, and about six times the 7 N limit above. **A flat press of a
2x4 may simply not be within this arm's capability**, and the tilted press that `bench.py` already
implements stops being an optimisation and becomes a requirement. The next section is that conflict
worked out properly. (This is the upper-scenario clutch, not what the simulator ships: the default
is 1.0 N per stud, `assumed`, which this arm seats with room to spare -- see below.)

## Seating feasibility with the stock arm

**This is the single most important measurement in the project.** Not the wall strength, not the
friction coefficient, not the restitution: the press-fit force. Every other number on this page
changes how well the robot does its job. This one decides whether the job is possible. The
simulator ships with 1.0 N per stud (`assumed`, matching its own bare placeholder) precisely
because nobody has measured this yet; at the top of the range the mechanics allow — about 10 N
per stud — a straight press of a 2x4 is out of reach at every supply voltage and torque
assumption tried here, and even a tilted press only becomes possible (in the best part of the
build plate) at 7.4 V once the higher, identified-law torque estimate is assumed, reaching
everywhere on the plate only at 12 V under that same assumption. It is twenty minutes with a
kitchen scale to find out which end of that range the project is actually in.

![Push-on force against clutch, with every force the arm has](../media/materials/seating-feasibility.png)

The figure is the whole argument. Read it as follows.

**The push-on force is the clutch (the average of push-on and pull-off force per stud) times the
number of studs, times (1+β).** Pushing on is harder than pulling off because mould draft
tightens the fit going in and because static friction is higher than kinetic; the model's
β = 0.15 makes push = clutch × (1+β) and pull = clutch × (1−β) per stud, a *push-to-pull* ratio
of (1+β)/(1−β) = 1.35 (section 2.5) — a different number from the *push-per-clutch* multiplier
used here. So a 2x4 needs 8 × 1.15 = 9.2 N of push per newton of clutch force per stud, and a 2x2
needs 4 × 1.15 = 4.6 N. Each panel's x axis runs from 0.3 to 5 N per stud, which is nearly all of
section 2.3's 0.3 to 6 N band, and the caption under the figure says exactly which friction and
interference combinations put a brick where on it.

**What the arm has, both ways of counting.** From `brick_builder.robot.force_map`, over a build-plate
patch at x = 0.12 to 0.24 m, y = ±0.06 m, z = 0.02 m, tool-down. The two columns are the
unresolved disagreement of section 1.3: the Feetech datasheet's 1.9 N·m stall torque at 7.4 V
against the 3.41 N·m the identified servo law reaches at full duty. Nobody has measured which is
right on this arm.

| Supply | Datasheet, 1.9 N·m | Identified law, 3.41 N·m |
|---|---|---|
| 5.0 V | 6.9 to 10.2 N | 11.2 to 16.6 N |
| 7.4 V | 9.5 to 14.0 N | 15.7 to 23.5 N |
| 12.0 V | 14.4 to 21.4 N | 24.4 to 36.7 N |

**So where does seating become impossible?** The clutch value at which each press runs out, taking
the best cell of the build plate (the worst cell is about 30 % lower):

| | 2x4, straight | 2x4, tilted | 2x2, straight | 2x2, tilted |
|---|---|---|---|---|
| 5 V, datasheet | 1.1 N/stud | 4.4 N/stud | 2.2 N/stud | 4.4 N/stud |
| 7.4 V, datasheet | **1.5 N/stud** | 6.1 N/stud | 3.0 N/stud | 6.1 N/stud |
| 12 V, datasheet | 2.3 N/stud | 9.3 N/stud | 4.7 N/stud | 9.3 N/stud |
| 5 V, identified | 1.8 N/stud | 7.2 N/stud | 3.6 N/stud | 7.2 N/stud |
| 7.4 V, identified | 2.5 N/stud | 10.2 N/stud | 5.1 N/stud | 10.2 N/stud |
| 12 V, identified | 4.0 N/stud | 16.0 N/stud | 8.0 N/stud | 16.0 N/stud |

At the project's own supply and the datasheet torque limit, **a straight press of a 2x4 works only
if the clutch is under about 1.5 N per stud.** The first-principles estimate is 4.8 N per stud with
a band of 2.7 to 10.0. If that estimate is anywhere near right, a flat press of a 2x4 is out of
reach at every supply voltage and under both torque assumptions: it needs 44 N and the most the arm
has anywhere in the table is 37 N. A 2x2 straight is out of reach too at 7.4 V under the datasheet
torque (22 N needed against 14 N), but becomes at least partly possible once the identified law is
assumed — already in the best part of the plate at 7.4 V (22 N needed against 15.7 to 23.5 N), not
only at 12 V.

**What a tilted press changes — in this simplified accounting.** Tilting the brick so one row of
two studs enters at a time instead of all eight is modelled here as only the row that is actually
sliding resisting, so the force needed drops by the number of studs kept out of the fight — a
factor of four on a 2x4, two on a 2x2. That lifts the 2x4 ceiling from 1.5 to 6.1 N per stud at
7.4 V with the datasheet torque, comfortably past the 4.8 N first-principles estimate.

**That per-row accounting is optimistic, and it needs saying plainly.**
`tests/bricks/test_snap_bench.py` measures what `PressBench` actually does over a *complete*
tilted run, through the real per-socket mechanism (`ColletSnap`/`engagement()`), and for the
`"ramp"` curve — the shipped default — a tilted press's overall peak force comes out within about
10% of a flat press's, not divided by four
(`test_rocking_peak_against_flat_press_depends_on_the_curve_shape`). The reason is geometric, not
a bug: a rigid brick's last moments of seating bring every stud to full depth together, tilted or
not, and the ram reads the sum of every stud's force regardless of which row got there first, so
whichever row seats *last* still adds its full share right when the earlier rows are also still
near their own full force. Staging the rows is measurably real — the tilted run's sockets really
do reach full depth at different times
(`test_rocking_really_stages_the_rows_one_at_a_time`) — but that staging lowers *when* each row's
force arrives, not the *peak* summed over the whole run. So the per-row ceiling above is a real
account of the force to get the leading row started, useful for comparing configurations against
each other, but it is not a validated prediction of the peak a whole tilted run reaches; treat the
"tilted" column throughout this section as the optimistic case pending that reconciliation, not a
settled one. A tilted press also puts the tool on a longer lever than the studs it is seating,
which costs wrist torque rather than vertical force and is not counted here either.

The three curve shapes of section 2.4 change this in one way each, and the third panel of the
figure is them:

- **"ramp"** (the default, and the one with a mechanism behind it): the force grows with depth and
  peaks at full seat. A tilted press meets that peak one row at a time, but -- per the caveat
  above -- ends up carrying close to the flat press's peak by the time the last row seats.
- **"plateau"**: full force from the instant the stud engages. Same peak, but there is no gentle
  onset, so a force-limited press has no warning before it stalls. No straight-sided interference
  produces this, which is why section 2.4 says the shape has no mechanism.
- **"detent"**: an entry bump, which is the "click" a real brick is supposed to have. Its peak is
  **1.43× the seated holding force**, and it is the peak that has to be beaten. Every ceiling in
  the table above divides by 1.43 under this shape: the 2x4 tilted ceiling at 7.4 V falls from
  6.1 to 4.3 N per stud, which is *below* the first-principles estimate. Unlike "ramp", a real
  bench measurement backs a tilted benefit for "detent" (about a third lower peak than flat,
  `test_rocking_peak_against_flat_press_depends_on_the_curve_shape`), because the bump that has
  to be beaten is transient and localised in depth rather than persisting to full seat -- so if
  the curve has a detent, a tilted press is the shape where tilting is actually expected to help.

**How much does the toy itself narrow this?** Less than one would hope. DUPLO is sold for ages
18 months and up, which means an 18-month-old has to be
able to press two bricks together and pull them apart. That is a real bound on the clutch — but no
measurement of it exists. A search for a published or careful hobby measurement of DUPLO (or any
comparable large-stud brick) separation force turned up nothing: the widely repeated "3 to 5 N to
pull two bricks apart" figure traces to aggregator sites with no stated method, the one serious
hobby rig that has been built ([lambsandwich](https://lambsandwich.net/welcome-to-lambsandwich-net/a-comment-on-lego-clutch-power/),
650 gf ≈ 6.4 N per stud) measured *System*, not DUPLO, and the clutch-testing page at
[Brick Architect](https://brickarchitect.com/2021/lego-clutch-test-implements-bricks/) describes
the implements without reporting a force. Published grip-strength norms for children do not go
below about four years old, because dynamometer testing that young is not reliable.

So the toy argument is an inference, not a datum: if a 2x4's whole-brick clutch were the 38 N the
first-principles estimate implies, an 18-month-old would not be taking those bricks apart, and they
demonstrably do. That argues against the top of the 0.3 to 6 N band and says nothing useful about
the bottom of it. It narrows a factor of twenty to perhaps a factor of ten. **The measurement in
section 7.2 is still the only thing that settles it**, and it is the first one on the list for
exactly this reason.

## 6. Making damage representable

Two mechanisms, for two different jobs, plus the training interface.

### 6.1 The cheap estimate, for training

Everything needed is already in the simulator. It costs one pass over the contact list per step and
adds no bodies, no constraints and no geometry.

**Inputs already available.** `data.contact` with `mj_contactForce` gives a force and a frame for
every contact; the generator's naming makes the geometry self-describing without any lookup table
(`{brick}_col_wall_{side}`, `{brick}_col_tube_{k}`, `{brick}_col_stud_{i}_{j}`, `{brick}_col_roof`,
and the arm's `fixed_pad` / `moving_pad`). `equality_forces` in `snap_api.py` already returns the
constraint force of every connection. `ColletSnap.stud_depths` gives per-stud insertion. Contact
position in the brick's own frame gives the lever arm, and the two geoms' sizes give the contact
width, which section 4.6 showed is as important as the force.

**Stress from each contact**, one formula per mode, all of them from section 4:

| Mode | Formula | Allowable |
|---|---|---|
| Wall bending | σ = 6 F_n L / (b_eff t²), L = (height − roof) − z_contact, b_eff = min(2L + w_contact, wall length) | 44 MPa set, 70 MPa crack |
| Stud root shear | τ = F_lateral / (π(d_o² − d_i²)/4) | 26 MPa |
| Stud root bending | σ = 32 F_lateral h_engaged d_o / (π(d_o⁴ − d_i⁴)) | 70 MPa |
| Tube ovalisation | σ = 0.318 F R (t/2) / I, I = h t³/12 | 44 MPa |
| Roof bending near an edge | σ = 6 F L / (b t_roof²) | 70 MPa |
| Contact marking | p_max = 2F/(π b L) with the Hertz half-width b, or the projected pressure for an edge contact with an assumed 0.2 mm edge radius | 44 MPa |

The `b_eff = 2L + w_contact` rule is a 45° load spread from the contact up to the roof. It is an
approximation, it is the reason the wall bending band in section 4.3 is a band rather than a
number, and it should be stated as such in the code. Two more simplifications belong beside it:

- **Simultaneous contacts on one member are combined by maximum, not superposed.** Every reduction
  from per-contact rows to a per-brick number is a maximum, so an interior stud touching two tubes
  at once — which the geometry says happens, at 1.5 contacts per stud — is scored by the worse of
  the two, not by their sum. Superposing magnitudes would be right only if the two forces were
  colinear, and in the diagonal grid they are roughly 90° apart, so a sum would over-state the
  load about as often as the maximum under-states it. The maximum is also what keeps a brick
  resting on a plate from accumulating damage for doing nothing.
- **A load has to last to count.** Anything past an allowable has to persist for 10 ms before it
  can move a brick's permanent state. Section 6.4 has the reason and the arithmetic.

**Accumulation.** Per brick, per step, with `s = σ/σ_allow` the worst stress ratio over all modes:

```python
peak     = max(peak, s)                          # worst instantaneous stress ratio
fatigue += (s / 0.5) ** 8 / 37_112               # Palmgren-Miner sum, per press cycle
relax   += max(s - 0.2, 0.0) * dt / tau_relax    # sustained load
clutch_multiplier = clip(1.0 - 0.5 * fatigue - 0.3 * relax, 0.2, 1.0)
```

The exponent 8 and the 37,112 are anchored on the only cycle-life datum that exists (section 4.7).
An earlier draft of this page said the connection "should lose its clutch after about 37,000
cycles" and left it ambiguous which rung of the ladder that meant, which made the text and the
constants describe different events. Worked through, the constants say:

| At half the allowable stress | Cycles | What the source reports |
|---|---|---|
| `loosened` — clutch down to 0.8 | 14,845 | — |
| **clutch halved — 0.5** | **37,112** | **"both pieces failed at 50 %" at 37,112 cycles** |
| clutch at the 0.2 floor | 59,379 | — |

So **the constants match the anchor and the text was wrong**: `fatigue_weight = 0.5` with
`cycles_ref = 37112` puts the clutch at exactly half after exactly the cited number of cycles,
which is what Cantin measured. `loosened` is the first rung of the ladder and arrives 2.5× sooner,
at about 14,800 cycles; nothing in the source speaks to that number. Both constants are guesses in
shape as well as in value and should be randomized over at least a factor of three, and section
7.5 measures the real decay on this arm — on new DUPLO rather than worn 1970s System.

**States**, in order, each one latching:

| State | Trigger | Effect |
|---|---|---|
| intact | — | — |
| marked | contact stress past yield anywhere | visual only: darken the geom's colour |
| loosened | `clutch_multiplier < 0.8` | `f_clutch` for that brick's sockets is scaled permanently |
| cracked | any bending mode past 70 MPa | clutch multiplier drops to 0.3, visual change |
| broken | any bending mode past 1.5 × 70 MPa, or the breakable model releases a weld | the brick is removed from the buildable set |

Cost: one loop over contacts, so tens of microseconds per step against the 105 µs a plate-and-brick
scene already takes. Suitable for training.

### 6.2 The breakable brick, for validation

Build the brick from separate rigid pieces (roof, four walls, three tubes, eight studs) joined by
weld equalities to the roof body, with each weld's force and moment limits taken from the real
cross-section, and deactivate a weld at run time when its constraint force exceeds them. The weld
force is read with the same `equality_forces` helper the snap model already uses, and
`data.eq_active` is already written at run time by `ColletSnap`, so no new machinery is needed.

Limits, from the sections in 4 at 44 MPa axial and 70 MPa bending:

| Joint | Section | Axial limit | Moment limit |
|---|---|---|---|
| Long wall to roof | 63.8 × 1.55 mm, Z = 25.6 mm³ | 4351 N | 1.79 N·m |
| Short wall to roof | 28.7 × 1.55 mm, Z = 11.5 mm³ | 1957 N | 0.80 N·m |
| Tube to roof | annulus 13.4/10.7 mm, Z = 140.2 mm³ | 2249 N | 9.81 N·m |
| Stud to roof | annulus 9.4/6.5 mm, Z = 62.9 mm³ | 942 N (shear) | 4.40 N·m |

**Cost, measured on this machine.** A scene with a 4x6 baseplate and one monolithic 2x4 steps in
105.5 µs (3 bodies, 82 geoms, 6 degrees of freedom). The same scene with that brick built from
16 welded pieces steps in 716.2 µs (18 bodies, 96 degrees of freedom, 15 equalities), a **6.8×
slowdown for one brick**. A four-brick build would be unusable for training.

So: the cheap estimate for every training run, the breakable brick for three jobs only. Validating
that the cheap estimate's thresholds match where pieces actually come apart. Producing the
demonstration video of a brick failing, which is what makes the damage model credible to anyone
who did not read this page. Answering specific "what if the arm did X at full speed" questions.

### 6.3 Damage during training

- **Reward.** A penalty on the damage increment, not on the state, so a policy is punished at the
  moment it does harm rather than for the rest of the episode. Weight it so that marking a brick
  costs about as much as a small positional error and cracking one costs more than failing the
  task, which keeps the ordering sensible: a policy should prefer not to place the brick at all
  over breaking it.
- **Failure class.** "brick damaged" alongside the existing failure classes, so evaluation reports
  can separate a press that missed from a press that succeeded destructively. A run that seats the
  brick and cracks a wall is not a success.
- **Termination.** On `cracked` or worse. Not on `marked` or `loosened`, which are recoverable and
  which a policy should be allowed to trade against.
- **Randomization.** All allowables over ±30 %, which covers the material table's spread in
  section 1.1; the fatigue constants over a factor of three; the wedge geometry (chamfer angle
  0.5° to 1.5° of draft, 30° to 60° of lead-in) per episode. A policy that only works with one
  particular wall strength has learned the wrong thing.
- **Observation.** The policy must **not** observe the damage state. A real robot cannot see it.
  What it can observe, and what should therefore be in the observation if anything is, is the
  servo's own `Present_Load` register, which is the duty cycle the firmware is commanding and which
  the identified servo law turns into a torque. That is a real signal on real hardware and it
  already exists in `duty_from_error`. Damage stays a reward and termination signal only.
- **Randomize K_t too.** Section 4.3 puts the fillet stress concentration at 1.0 to 2.0 and the
  model's default at 1.0. A policy trained only against the nominal-stress wall has been trained
  against the strongest wall the brick might have.

### 6.4 Misalignment sweep results

![Wall bending stress over misalignment and press force](../media/materials/wall-stress-map.png)

The first version of the misalignment experiment was not converged, and an independent review
caught it. The same scenario read "broken" at a 2 ms step and "undamaged" at 1 ms, because the ram
— a 1 g body on a 2 N·s/m dashpot, carrying 12.5 g of brick, with 20 g of joint armature making up
most of what moved — plunged 0.8 mm past its own resting place during the first half second of the
ramp, and the damage model latched that plunge. Nothing about it was a measurement of anything. On
that dashpot the commanded 20 N implies a terminal speed of 10 m/s, which is the whole problem in
one number. What it took to fix:

- **A ram with a mass.** 0.5 kg, which is the arm's own operational-space inertia at press poses
  (section 4.8), and no armature, because the body now carries real inertia.
- **A dashpot that makes it a press rather than a dropped weight.** 1500 N·s/m, so the terminal
  speed under the commanded force is F/1500 — 30 mm/s at the 45 N top of the sweep and 1.3 mm/s on
  the 2 N approach. The rig cannot build the kinetic energy that would turn a press into an impact.
- **A velocity-limited approach, and a ramp that starts only once the brick is touching.** Three
  phases, all measured in seconds rather than in steps so that the same physical experiment runs
  at any time step: approach under 2 N alone until the first contact appears or the starting gap
  is closed (an *aligned* brick makes no geometric contact going in, because the snap model owns
  the stud-in-tube fit), then ramp to the commanded force over 0.5 s, then hold — for at least
  0.4 s and then until the ram is slower than 0.2 mm/s. Holding for a fixed time is not enough:
  seating a brick against the clutch takes about a second of creep at 15 N, and a reading taken
  while the brick is still going in is not a force balance. The reported ram velocity says how
  close to rest it actually got.
- **A contact damping ratio of 2.0 instead of 1.0.** This is the setting that decided it. At 1.0
  and a 2 ms step, a brick resting on the plate's stud tips does not settle: it runs a limit cycle
  of about ±35 µm at 20 Hz with the contact force swinging between 10 and 26 N under a steady 20 N
  press, and that wander is exactly what pushed a misaligned stud deep enough into a wall for the
  solver to report tens of newtons sideways. At 2.0 the same scene settles to within 0.4 µm.
  MuJoCo's constraint stiffness goes as 1/dampratio², so the price is contacts four times softer
  than the simulator's own — 2.4 µm of penetration at a 20 N press instead of 0.6 µm, both far
  under the 86 µm the clutch is made of.
- **A guard in the damage model against latching transients.** Any stress past an allowable now
  has to persist for 10 ms before it can move a brick's permanent state; instantaneous peaks are
  still recorded every step, unfiltered, and reported separately. The 10 ms comes from the
  material, not from taste. The gentlest collision section 4.8 says can crack a wall — 0.5 kg at
  0.3 m/s on an 8 kN/m wall — is a half-sine lasting π√(m/k) = 25 ms, of which about 14 ms is
  spent above the permanent-set load, and it carries 150 mN·s of impulse. A one-step solver
  transient lasts 2 ms and carries a few mN·s, about 2 % of that. ABS does not crack on 2 % of the
  impulse that is only just enough to crack it. `loosened` is deliberately not guarded, because it
  comes from time integrals that a single step cannot move.

**The converged answer, and it is not the one the old figure told.** Sweeping the misalignment from
0 to 2.2 mm at press forces from 5 to 45 N, at 2 ms, 1 ms and 0.5 ms:

| | Result |
|---|---|
| Wall bending stress | **0.00 MPa in every one of the 50 cells** |
| Worst stress ratio anywhere | 0.05, except one cell at 0.62 (a tube, at 45 N and 0.18 mm) |
| Damage state | `intact` at every offset, every force and every time step |
| Worst disagreement between 2, 1 and 0.5 ms | **2.2 %** on the peak stress ratio, zero on the state |
| Seating | full (4.50 mm) up to 0.15 mm of misalignment and 10 N of press; nothing at all past 0.20 mm |
| The one exception | at 0.18 mm, a 45 N press — three times what the arm has — forces the studs past the tube rims and seats the brick, loading a *tube* to 62 % of its allowable on the way |

The seating column is a second check: it is a number the model predicts
rather than measures. With the simulator's placeholder clutch of 1.0 N per stud and β = 0.15, a
2x4 needs 8 × 1.0 × 1.15 = 9.2 N to go all the way in, and the rig stalls the brick half-way at
8 N and seats it at 10 N. The rig reproduces its own press-fit model.

The geometry is the reason, and it runs counter to what section 4.3's mechanics would suggest.
**A plate stud cannot reach a wall.** The stud and the tube overlap by
86 µm on the radius, so about 0.18 mm of misalignment is enough to stop a stud entering its cell
at all; the brick then stands on the stud tips and goes no deeper whatever the press force. The
nearest wall is 1.65 mm away — nineteen times further — and a real DUPLO brick bridges that gap
with internal ribs that the generator does not model (section 2.1 says so already). So in the
geometry this project builds, the wedging load of section 4.3 has no path to a wall. Every newton
of lateral force the old rig reported was the solver resolving an overlap that a quasi-static
press cannot create.

**Where damage does start, then.** The wall is loaded the way section 7.7 loads a real one: a
4 mm rounded rod against the inside face, 15.0 mm below the roof, force-controlled and ramped.
That is well posed — the contact normal does not depend on how the solver resolves a corner — and
it reproduces σ = 6FL/(b_eff t²) to within 0.1 % at every time step, with b_eff = 34.1 mm:

| | Rod force | At K_t = 2.0 |
|---|---|---|
| 44 MPa, permanent set | 39.9 N | 20.0 N |
| 70 MPa, `cracked` | **64 N** (63.5 N by hand) | 32 N |
| 105 MPa, `broken` | **96 N** (95.3 N by hand) | 48 N |

The first row has no rung of its own, and that is a gap worth naming: `marked` is triggered by a
*contact pressure* past yield, not by a bending stress past yield, so a wall that has quietly
taken a permanent set is not a state the ladder can report. Section 6.1's table defines it that
way and this page has not changed it; it is listed here because the permanent set is the failure
that arrives first and the one a functional test would notice.

Against an arm with 14 N of press that is a margin of 4.6× at the nominal K_t and 2.3× at the
pessimistic one; against its 17 N grip, 3.8× and 1.9×. The conclusion of section 5 survives —
this arm is not a brick-breaking machine — but the *simulated evidence* for the wedging failure
mode does not: the rig cannot produce it, and what used to look like evidence was a numerical
artefact. A jaw pinching a wall edge, which section 4.6 says concentrates the same 17 N into a
3 mm contact, remains the one route this arm plausibly has to a wall, and nothing here has
exercised it yet.

## 7. Measurements to make on real bricks

In order, cheapest and least destructive first. Everything here uses the stock arm, a kitchen
scale, a caliper and a phone. No added part, fixture or sensor.

### 7.0 The one to do first, exactly as written

Twenty minutes, no arm, no fixture. It settles the number the previous section says the project
turns on. Use **new** bricks of a light colour, not from the toy box: section 1.1 explains why an
aged or dark brick is a different material.

**Push-on.** Put a 0.1 g kitchen scale on the table and a baseplate flat on the scale. Tare it.
Press a new 2x2 down onto the plate, slowly and squarely, with one finger on the middle of the
brick, until it is fully home. Read the **peak** the scale showed on the way down — most kitchen
scales settle too slowly to catch it, so film the display with a phone at 60 fps and step back
through the frames, or use a scale with a peak-hold mode. Repeat five times, on five different
stud positions if you have the room. Then do the same with a 2x4.

**Pull-off.** Press the same brick on, turn the plate upside down and hold it by the edges, and
hang a bag from the brick. Add water a little at a time until the brick lets go. Weigh the bag:
grams × 0.00981 = newtons. Repeat five times. Then repeat with only two studs of the 2x4 engaged
(hang the brick off the edge of the plate) to separate the per-stud force from the per-brick one.

Fill this in. Every row sets a simulator parameter, and the last column says which:

| Measurement | 1 | 2 | 3 | 4 | 5 | Mean | Sets |
|---|---|---|---|---|---|---|---|
| 2x2 push-on, whole brick [N] | | | | | | | `SnapParams.f_clutch` via ÷ 4 ÷ (1+β) |
| 2x4 push-on, whole brick [N] | | | | | | | the same number, via ÷ 8 ÷ (1+β): the two must agree |
| 2x2 pull-off, whole brick [N] | | | | | | | `SnapParams.f_clutch` = mean ÷ 4 ÷ (1−β) |
| 2x4 pull-off, whole brick [N] | | | | | | | `SnapParams.f_clutch` = mean ÷ 8 ÷ (1−β) |
| 2x4 pull-off, 2 studs only [N] | | | | | | | checks that the force really is per stud |
| push-on ÷ pull-off, either brick | | | | | | | `SnapParams.beta` = (r−1)/(r+1) |
| spread across the five repeats | | | | | | | the randomization width for `f_clutch` |

Three things to notice while doing it, because they are the ones the simulator is guessing at:

- **Does the force build smoothly, or is there a click?** A click means the `"detent"` curve
  family and its entry bump; a smooth build means `"ramp"`. Section 2.4 and the third panel of the
  seating figure say why the difference matters.
- **Does the 2x4 take twice the 2x2?** If it does, the force is per stud and the model's whole
  structure is right. If it does not, something else is carrying load — ribs, most likely, which
  section 2.1 says the generated brick does not have.
- **How hard is it with one hand?** If a 2x4 needs a deliberate shove, the clutch is at the high
  end and the previous section's conclusion stands: this arm will need a tilted press.

Then the rest of the list, in order.

**7.1 Dimensions and mass, non-destructive, about an hour.** Ten new 2x4 bricks of the same
colour. Caliper each one for: stud outer diameter at the tip and again at the root (the difference
over 4.5 mm gives the mould draft directly, and 0.5° of draft is 39 µm, which a 0.01 mm caliper can
just resolve), tube outer and inner diameter, wall thickness, brick height, and the rib projection
if ribs are present. Weigh each on a 0.1 g kitchen scale. This gives the tolerance spread that
feeds the randomization ranges, confirms or refutes the 13.4 mm tube diameter that section 2.3
showed the whole clutch estimate hangs on, and checks the 12.5 g the generator produces.

**7.2 Pull-off force, non-destructive, about twenty minutes.** Section 7.0 above, written out step
by step, because it is the one measurement on this list that decides whether the project is
possible rather than how well it goes. It is the inverted-plate scene that
`build_stack_scene(inverted=True)` already simulates, done physically. It gives the holding force
of n studs directly, and section 2.3 showed the first-principles answer spans a factor of twenty.

**7.3 Friction, non-destructive, ten minutes.** Put one brick flat on another and tilt them until
it slides; the tangent of the tilt angle is the static coefficient. Repeat with a printed PLA
offcut on a brick for the jaw pair. This replaces the one number on this page that has no source at
all, and it costs nothing.

**7.4 Force against depth, non-destructive, and the one that needs the arm.** Stand the baseplate
on a kitchen scale and press a brick on by hand, slowly, filming the scale. The peak reading is the
push-on force; the shape of the trace against the descent is the curve `engagement()` is guessing
at. Then do it again with the arm, which is its own force gauge: the servo's `Present_Load`
register gives the firmware duty, the identified law turns duty into torque
(35.2 mN·m per 1 % of duty at 7.4 V), and the Jacobian turns torque into force at the tool. The
gear train's dry friction of 0.0515 N·m sets the resolution floor, which at a 0.2 m lever is
**0.26 N**, fine for measuring a clutch of a few newtons. Calibrate the arm against the kitchen
scale first, then use the arm for everything else, since it can move at a known speed and log every
step.

**7.5 Cycle life, unattended, days.** Have the arm press and pull the same brick repeatedly,
re-measuring the pull-off force every hundred cycles by the method in 7.2. This measures the decay
rate that section 6.1 currently anchors on a hobbyist's 37,112-cycle System result, and it is the
only way to get a DUPLO number, because none exists.

**7.6 Restitution, ten minutes.** Drop a brick from a measured height onto a hard table, film at
high frame rate, measure the rebound height. e = √(h′/h). Fills the gap in section 3.3.

**7.7 Wall strength, destructive, three bricks.** This is the only test that costs bricks, and it
is the one that calibrates the damage model. Stand a brick on a kitchen scale and push a rounded
rod against the inside of a wall near its bottom edge, increasing until the wall takes a visible
set, then until it cracks. Record both forces. Section 4.3 predicts 12 to 35 N and 19 to 56 N, a
band wide enough that this measurement would genuinely narrow it. Do it at three contact widths
(rod, edge, flat block) to test the b_eff rule in section 6.1, which is the weakest assumption in
the damage estimate.

Nothing on this list needs a load cell, a universal testing machine or a fixture. The crush test in
section 4.1 does, and it is not worth doing: the Open University's 4240 N on a System 2x2 already
anchors that end of the ladder, and no policy will ever get near it.

## Using the material and damage models

Sections 0 to 7 are the analysis. This section is the code that implements it. Three modules,
all opt-in: nothing in them changes a default, and a scene that never imports them behaves
exactly as it did before.

| Module | What it is | When you want it |
|---|---|---|
| `brick_builder/bricks/materials.py` | the sourced property database and every derivation on this page | working out what a simulator parameter *should* be |
| `brick_builder/bricks/damage.py` | the cheap estimate of section 6.1 | every training run |
| `brick_builder/bricks/breakable.py` | the breakable brick of section 6.2 | validation and demonstrations only |

### Materials: lookup and derived values

Every property carries its value, its randomization range, its units, its source string and a
blunt confidence field. `"unsourced"` means exactly that.

```python
from brick_builder.bricks import materials as mat

mat.ABS_BRICK_23C.tensile_yield          # Prop(44e6 Pa, 36e6 to 52e6, "Novodur P2H-AT ...")
mat.ABS_BRICK_23C.E                      # 2.3e9, the nominal, as a plain float
mat.ABS_BRICK_23C.modulus_at(28.0)       # 2.22e9: a warm room, 3.5 % softer
mat.unsourced_properties()               # the short list of numbers nobody has measured
```

The derivations take a `BrickSpec` and a `Material` and give back the numbers section 2 works
out by hand:

```python
mat.press_fit_geometry()          # interference 86.3 um, wall gap 1.65 mm, 12 contacts
mat.press_fit_stiffness()         # tube 130, stud 562, series 106, soft wall 7.2 kN/m
mat.clutch_force()                # 4.79 N/stud, 38.3 N/brick, band 2.7 to 10.0 N
mat.clutch_force(path="wall")     # 0.33 N/stud: the first-pass estimate, and the wrong path
mat.draft_exponent(draft_deg=1.0) # 1.91, not 1
mat.wall_thresholds()             # set 11.7 to 35.1 N, crack 18.6 to 55.9 N (K_t = 1)
mat.wall_thresholds(k_t=2.0)      # set 5.9 to 17.6 N, crack 9.3 to 27.9 N: section 4.3
mat.crush_load()                  # 13.2 kN, by buckling (2x2: 7.85 kN)
mat.weld_limits()                 # per-piece force and moment for the breakable brick
mat.contact_settings(timestep=0.002)   # solref/solimp/friction per pair, floor respected
mat.randomization_ranges()             # everything worth randomizing, as (lo, nominal, hi)
```

The seating question of "Seating feasibility with the stock arm" is three calls, and the middle one is
the only place in this module that reaches outside the bricks - it imports
`brick_builder.robot.force_map` and calls it, read only:

```python
mat.push_on_force(4.79, n_studs=8)                  # 44.1 N to press a 2x4 on flat
mat.push_on_force(4.79, n_studs=8, tilted=True)     # 11.0 N tilted, one row of two at a time
mat.push_on_force(4.79, n_studs=8, shape="detent")  # 63.1 N if the curve has an entry bump
mat.arm_press_force_band()        # {'datasheet': {7.4: (9.5, 14.0), ...}, 'identified': {...}}
mat.seating_feasibility()         # the clutch each supply and torque law can still seat
```

`mat.print_comparison()` puts the derived values beside the simulator's current defaults. It
reads `SnapParams()` and `spec.py`; it writes to neither. Running the module directly
(`python -m brick_builder.bricks.materials`) prints that table and then the unsourced list.

To use the derived parameters, ask for them explicitly:

```python
params = mat.derived_snap_params()       # f_clutch 4.79 N, exponent 1.91, curve "ramp"
snap = ColletSnap(params)
```

Be clear about what that implies before switching it on: at 4.8 N per stud a 2x4 needs about
44 N to push on, against an arm that produces 10 to 14 N. See section 5.

### Damage: one call per step

```python
from brick_builder.bricks.damage import DamageModel

damage = DamageModel(model, data, bricks, snap=snap)   # after compile and bind
...
mujoco.mj_step(model, data)
snap.update(model, data)
damage.update()                 # one vectorised pass over the contact list
damage.apply_consequences()     # optional, and must come after snap.update
```

Queries: `damage.brick(name)` for one brick's full record, `damage.states()` for the ladder,
`damage.worst_state()` for a termination check, `damage.summary()` for an evaluation record and
`damage.reward_penalty()` for the increment since the last call. `get_state()` / `set_state()`
round-trip everything the model carries between steps, so a checkpoint restores exactly.

The two rigs that exercise it, both quasi-static and both checked at three time steps (section
6.4):

```python
from brick_builder.bricks.damage import (build_press_rig, run_press, misalignment_sweep,
                                    build_wall_rig, run_wall_push, wall_push_onset)

run_press(build_press_rig(offset_x=-0.2e-3), 20.0)   # a misaligned press onto a baseplate
misalignment_sweep(offsets_m, force_N=20.0, timestep=0.001)
run_wall_push(build_wall_rig(), 64.0)                # section 7.7's rod, simulated: cracked
wall_push_onset()["onset_N"]                         # the first force in the sweep that cracks
```

Four notes before reading any output.

**Wall bending uses the force component perpendicular to the wall's own plane.** A brick
standing on a table puts its whole weight down its walls and that is compression, not bending;
a model that used the force magnitude would accumulate damage for doing nothing.

**Simultaneous contacts on one member are combined by maximum, not superposed**, for the reason
in section 6.1. `peak_normal_N` on a settled brick is one contact's force, not the sum over the
contact patch, and it is not meant to be.

**A state change needs 10 ms of load, an instantaneous peak needs none.** `DamageThresholds.
persist_s` is the guard of section 6.4; `record.peak_ratio` and `record.peak_by_mode` are the
unfiltered instantaneous peaks and `record.longest_overload_s` is how long the worst one actually
lasted. A large `peak_ratio` with a `longest_overload_s` of zero is a transient, and reading the
first without the second is how the old misalignment sweep came to report cracked walls that were
not there.

**`apply_consequences` is a workaround, not an interface.** `ColletSnap` rewrites
`model.dof_frictionloss` from `params.f_clutch` on every update, so the only way to reduce one
damaged brick's clutch today is to multiply those dofs afterwards, every tick. A public
per-brick clutch scale on the snap model would replace it.

### Breakable bricks: for validation

```python
from brick_builder.bricks import breakable as brk

model, data, handle, manager = brk.build_crush_scene()
brk.crush(model, data, manager, force_N=20_000.0)     # the walls let go at about 14 kN
brk.drop(model, data, manager, height=1.0)            # nothing lets go, as it should not
```

`add_breakable_brick` builds the same brick the generator builds - same geoms, same names, same
sites - but with each wall, tube and stud as its own body welded to the roof, so
`ColletSnap` binds to it unchanged and the press fit still works on an intact one. It costs
about three times a monolithic brick per step in a plate-and-brick scene, which is why it is for
validation and not for training. One limitation: the stud *sites* stay on the roof, so a stud
whose weld has released still counts as attached to the snap model.

### Figures

![Derived force against depth, against the simulator's three curve families](../media/materials/derived-force-vs-depth.png)

The derived curve peaks at 44 N for a 2x4, against 9 N for the simulator at its shipped default
and 10 to 14 N for the arm. With mould draft it is also the wrong *shape*: quadratic rather than
linear.

![Push-on force against clutch, with every force the arm has](../media/materials/seating-feasibility.png)

The most important figure on this page. A straight press of a 2x4 works only if the clutch is
under about 1.5 N per stud at 7.4 V; the mechanics predict 4.8 as an upper scenario (the shipped
default is 1.0, comfortably inside every straight-press ceiling in the table above). A tilted
press's per-row estimate lifts that ceiling to 6.1 N per stud, with the caveat above about how
optimistic that estimate is. See "Seating feasibility with the stock arm".

![Wall bending stress over misalignment and press force](../media/materials/wall-stress-map.png)

The four panels of section 6.4. Wall bending stress is zero in all fifty cells of the
misalignment-and-press-force sweep, because a plate stud jams against a tube at 0.18 mm of
misalignment and the nearest wall is 1.65 mm away; the brick stands on the stud tips and nothing
bends. The third panel is the time-step convergence the review asked for (2, 1 and 0.5 ms agree to
2.2 %), and the fourth is where damage does start: the guide's own section 7.7 rod test,
simulated, cracking the wall at 64 N against an arm that has 14.

An earlier version of this figure showed a "dangerous strip about 0.2 mm wide" where a stud
overlapped a wall's inner face. That strip was a numerical artefact of an under-damped ram, it is
gone, and section 6.4 says what replaced it.

![A breakable brick crushed by a virtual ram](../media/materials/breakable-crush-sequence.png)

The walls go first, at about 14 kN against the 13.2 kN `crush_load()` predicts, and they go by
buckling rather than by yielding. The tubes hold longest. The arm produces 10 to 14 N.
