# On-screen wording

Every word that appears in a video, a still or a figure comes from this page. Viewers do not read
the source code, so nothing on screen may be a code identifier, a project-only abbreviation, or an
invented term. A name not listed below is added here first, using its maker's own name for it.

Robot names come from the SO-ARM100 repository this project vendors the model from
(`third_party/so101/`; `UPSTREAM_README.md` and the MJCF are the vendored copy) and from the
LeRobot SO-101 documentation. Brick names are LEGO's own.

## The project

| On screen | Never |
| --- | --- |
| Brick Builder Arm Robot | the package name `brick_builder`, or any short form of it |
| simulated in MuJoCo; on the simulator page, the mode labels `sim` and `twin (sim)` | simulator run, digital twin (as prose) |

The package directory in this repository is still called `brick_builder/`; that is a path, not a name a
viewer ever sees.

## The arm

The SO-101 follower arm, exactly as it comes from the standard kit.  Six STS3215 servos
(7.4 V, 1/345 gear), one per joint.

| Part | On screen |
| --- | --- |
| The whole machine | SO-101 arm, or the arm |
| What it stands on | base |
| First link off the base | shoulder |
| Next link | upper arm |
| Next link | lower arm |
| Link carrying the gripper | wrist |
| The hand | gripper |
| The half of the gripper that does not move | fixed jaw |
| The half that opens and closes | moving jaw |
| The camera on the gripper | wrist camera |
| One motor | STS3215 servo |

The joints, in the order the servos are numbered:

| Joint | On screen |
| --- | --- |
| 1 | shoulder pan |
| 2 | shoulder lift |
| 3 | elbow flex |
| 4 | wrist flex |
| 5 | wrist roll |
| 6 | gripper |

The gripper has no pads, no fingertips, no shoe and no printed attachment of any kind.  There is
no such part, so no picture may show one and no caption may mention one.

## The bricks

| Thing | On screen |
| --- | --- |
| A brick | DUPLO brick 2x4, DUPLO brick 2x2 |
| The bump on top | stud |
| The hollow pillar underneath that a stud enters | tube |
| The large flat piece bricks are built on | baseplate |
| How firmly bricks hold together | clutch force |
| The open box of spare bricks | tray |

"2x4" is written as LEGO writes it: studs across by studs along, no spaces, no multiplication sign.

## Words for what is happening

Plain English, in the present tense, describing what the picture shows:

- the arm moves over the tray, carries the brick back over the baseplate, lowers it onto the
  studs, and pushes down until the studs enter the tubes;
- the brick is held in the gripper, the gripper holds the brick;
- the tool points straight down;
- the push through the grip, in newtons;
- how far the studs have entered, in millimetres.

## Never on screen

Code identifiers and internal shorthand, including: `wrist_cam`, `grasp_cam`, `overview_cam`,
`tcp`, `TCP`, `press_shoe`, `weld`, `detent`, `snap`, `qpos`, `ctrl`, `level 2`, `residual`,
`primitive`, `FSM`, `env`, `reward`, `seed`, `DR`.  Also never: any hardware this project does not
own, and any number nobody measured.
