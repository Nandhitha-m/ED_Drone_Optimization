# SkyLink ED6002 optimization workflow

This project runs the design and operations workflow from M1 through M6. The
orchestrator is [`main.py`](./main.py); it also creates module reports and a
final recommendation through [`documentation.py`](./documentation.py).

## Requirements

- Python 3.11 (tested with Python 3.11.5)
- Packages listed in [`requirements.txt`](./requirements.txt)
- Project inputs under `Projectskylink_data(1)\`
- The M3 dataset `M3\dataset_with_infill_2.csv`

In the supplied workspace, use the existing `m4_opti` environment:

```powershell
.\m4_opti\Scripts\python.exe -m pip install -r .\requirements.txt
.\m4_opti\Scripts\python.exe .\main.py
```

In a clean checkout without `m4_opti`, create and use a Python 3.11 virtual
environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\main.py
```

`main.py` uses `Projectskylink_data(1)` and `optimization_outputs` by default.
Override them with `--data-dir` and `--output-dir`. M6's search can be adjusted
with `--m6-pop-size`, `--m6-generations`, and `--m6-replications`; defaults are
10, 3, and 1, respectively. For example:

```powershell
.\m4_opti\Scripts\python.exe .\main.py `
  --m6-pop-size 24 --m6-generations 12 --m6-replications 10
```

Larger M6 searches take longer. M4 also solves integer and relaxed models for
each of the 15 M1 designs; its route and solve progress is printed to the
terminal.

## Pipeline

`main.py` runs the stages in this order. Modules are loaded and called in the
same Python process; the workflow does not launch separate command-line
programs for each stage.

1. **Baseline M1:** optimize five payloads (0.5–2.5 kg) using the baseline
   plate mass and figure of merit.
2. **M2, then M1 again:** run the hub-plate mass/compliance sweep and topology
   visualization. Use its configured 3 mm thickness and 30% volume fraction to
   calculate a plate mass of about 0.1022 kg, then rerun M1 for the five
   payloads.
3. **M3, then M1 again:** compare surrogate models using leave-one-out
   validation, optimize the selected surrogate, and rerun M1 with the
   resulting figure of merit, fixed rotor diameter, and thrust floor.
4. **M4:** pass the combined 15 M1 designs into the hub, PHC assignment, drone,
   and charging-bay MILP. M4 selects a network plan and produces the inputs
   used by the operations stages.
5. **M5:** simulate the M4-selected plan with the integrated event simulator:
   demand arrivals, queues, charging, wind constraints, deliveries, and road
   fallback.
6. **M6:** search vehicle and operating-policy choices. Each candidate is
   evaluated through an M5 simulation adapter; NSGA-II returns a Pareto front,
   a knee point, and a weighted-sum comparison.
7. **Reports:** write one Markdown report per module plus
   `reports\final_optimization.md`.

## What M1 does

M1 is implemented in [`M1.ipynb`](./M1.ipynb), not in a standalone Python
module. `main.py` reads and executes notebook cell index 9 (the tenth cell),
which defines the multidisciplinary analysis model. It then uses
`scipy.optimize.minimize` with SLSQP to maximize mission radius for each
payload while enforcing bending-stress, tip-deflection, natural-frequency,
and arm-slenderness constraints.

The design variables are rotor diameter, battery capacity, arm diameter, arm
wall thickness, and cruise speed. M1 reports the resulting range, masses,
thrust, structural margins, and design variables. The orchestrator calls this
optimization three times, producing 15 designs total:

- **Baseline:** plate mass 0.25 kg and FM 0.65.
- **M2-conditioned:** M2-derived plate mass; other M1 settings remain at their
  defaults.
- **M3-conditioned:** M2-derived plate mass, M3-predicted FM and rotor
  diameter, and M3-predicted thrust as a lower bound.

Each stage writes a design CSV in its output folder. M1's notebook is executed
in-process; it is not launched as a separate notebook application.

## Programs and modules used by `main.py`

| File | Role in the workflow |
| --- | --- |
| `M1.ipynb` | Supplies M1's analysis, objective, and structural constraints. |
| `M2.py` | Produces the configured plate-mass/compliance sweep and topology plots. |
| `M3/compare.py` | Compares the RBF and quadratic surrogate models. |
| `M3/quadratic_res_surf.py` | Implements the quadratic response-surface model. |
| `M4.py` | Builds routes and solves the M4 network MILP. |
| `M5/M5_integrated.py` | Provides the discrete-event simulator used by M5 and M6. |
| `M6.py` | Implements the multi-objective NSGA-II policy search. |
| `documentation.py` | Writes the seven Markdown reports. |

The project does not include `M3/rbf_surrogate.py`. For the comparison run,
`main.py` supplies the needed thin-plate-spline RBF adapter in memory using
`scipy.interpolate.RBFInterpolator`; no extra RBF package is required.

## Inputs and outputs

The default data folder is `Projectskylink_data(1)\`. M4/M5/M6 read the site,
demand, hourly-profile, cost, no-fly-zone, and weather files from that folder.
M3 reads `M3\dataset_with_infill_2.csv`.

Results are written under `optimization_outputs\` by default, with separate
folders for M1, M2, M3, M4, M5, M6, and the Markdown reports. Existing outputs
may be replaced when a run writes to the same output directory.

## Interpretation notes

- M2 uses its existing 3 mm plate setting; it does not calculate an optimal
  thickness.
- M3's FM is a surrogate prediction.
- M5/M6 results depend on the simulated arrival-day scenario and random seed.
  A zero road-fallback result is not a guarantee of zero fallback under
  repeated real-world demand.
- M6's default search is intentionally compact. Increase replications and
  search size, and review the model assumptions, before using results for
  operational procurement.
