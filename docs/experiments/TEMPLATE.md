# Lab notebook: <short title>

Copy this file to a new one per session (e.g. `docs/experiments/2026-09-21-clutch-force.md`) and
fill it in as you go — the "Question" and "Setup" sections before you run anything, "Result" and
"Conclusion" after. This is the member-facing counterpart to
One page, one measurement or one experiment run,
plain language.

## Question

What are you trying to find out? One or two sentences, specific enough that it has a yes/no or a
numeric answer.

> Example: "Does the arm's stock gripper hold a 2x4 by friction alone while carrying it, or does
> it need the wedge direction (pushed up) instead?"

## Setup

- **Bricks / hardware used:**
- **Measurement or experiment:** which of `params`, `press-bench`, `seating-feasibility`,
  `force-map`, `grasp-hold`, `pick-and-place` (or a real-arm/real-brick measurement with no
  simulator involved).
- **Parameters file used:** the exact path, e.g. `my_params.yaml`. If you changed anything in it
  for this session specifically, say what and why here rather than only in the file's own
  `notes:` field.

## Command

```bash
.venv/bin/python -m brick_builder.experiments <name> --params my_params.yaml --out results/
```

Paste the exact command you ran, including any extra flags.

## Result

What actually happened. Quote the numbers from `report.md`/`data.json`, or paste the relevant
table/figure — not just a summary. If you ran a real-hardware measurement, put the raw readings
here too (all five repeats, not just the mean).

## Conclusion

What did this settle, and what did you do about it? Typical answers:

- "Measured `bricks.clutch_force_per_stud_N` = X N, status now `measured`, pasted into
  `my_params.yaml`."
- "Confirms the default parameters; no change."
- "Inconclusive — see next steps."

If nothing changed as a result, say so plainly rather than leaving this blank.

**Next step**, if any:
