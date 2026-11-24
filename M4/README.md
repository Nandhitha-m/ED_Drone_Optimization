# SkyLink M4: Hub Network and Fleet Optimization

This project implements the M4 network-planning model for the SkyLink medical drone assignment. It selects candidate hubs, assigns PHCs, sizes drones and charging bays, compares the saved M1 vehicle payload/range designs, estimates annualized costs, checks a small instance by exact enumeration, and exports route and cost visualizations.

## Project inputs

The default input directory is the sibling folder [Projectskylink_data](../Projectskylink_data/). It must contain:

- sites.csv — candidate hubs and PHC coordinates
- demand.csv — average daily demand by PHC and service class
- hourly_profile.csv — hourly arrival profiles
- wind_days.csv — synthetic hourly weather record
- nofly.geojson — planar no-fly zone geometries
- costs.yaml — annualization, operating, and fallback costs
- components.csv — parts catalogue (provided as context; M4 does not yet snap vehicle designs to catalogue parts)

The five M1 payload/range points are read from the saved outputs in the supplied [M1 notebook](../M1%20(1).ipynb) and are recorded in the source module. The notebook and supplied PDF are not executed or parsed at run time.

## Requirements and run

Python 3.10 or later is recommended. From this directory:

```powershell
python -m pip install -r requirements.txt
python m4.py
```

If running from another directory, provide explicit paths:

```powershell
python m4.py --data-dir "C:\optimization\Projectskylink_data" --output-dir "C:\optimization\m4\m4_output"
```

Useful options:

- `--year 2025` — calendar year used to infer weekday grid outages from the yearless wind record. Change this if the synthetic record is intended to correspond to a different calendar.
- `--reserve-fraction 0.90` — additional route-distance allowance as a fraction of M1's mission radius. M1 already includes its own battery reserve.

The optimization uses SciPy's MILP interface and the HiGHS solver shipped with SciPy. It uses Shapely to build visibility-graph paths around the supplied polygon geometries. No commercial optimization package is required.

## Outputs

The program creates the output directory if needed and writes:

| Output | Contents |
|---|---|
| [design_costs.csv](m4_output/design_costs.csv) | Annual cost, optimized hub set, LP-relaxation value and gap for each M1 design |
| [hub_plan.csv](m4_output/hub_plan.csv) | Open hubs, assigned PHCs, fleet, bays, operators, and estimated utilization |
| [phc_assignments.csv](m4_output/phc_assignments.csv) | PHC-to-hub mapping, route distance, estimated workload, energy, and road fallback fractions |
| [candidate_routes.csv](m4_output/candidate_routes.csv) | Shortest route distance and waypoint sequence for every candidate hub/PHC pair |
| [cost_breakdown.csv](m4_output/cost_breakdown.csv) | Annualized cost components for the selected design |
| [verification.csv](m4_output/verification.csv) | 4-hub/8-PHC enumeration check, LP gap, and big-M rationale |
| [optimized_network.svg](m4_output/optimized_network.svg) | Route map, opened hubs, PHCs, and no-fly zones |
| [annual_cost_by_design.svg](m4_output/annual_cost_by_design.svg) | Annual cost comparison across the five M1 payload/range points |
| [assumptions_and_verification.txt](m4_output/assumptions_and_verification.txt) | Data limitations, modeling assumptions and verification summary |

A complete record of modeling choices, interpretation, and the current results is in [docs/optimization_decisions.md](docs/optimization_decisions.md).

## Important interpretation

These are preliminary expected-value planning results, not a validated operational schedule. Package masses by service class are not present, so the five M1 payload points are treated as representative loaded missions; the lowest-cost result must not be interpreted as proof that a 0.50 kg design can carry all medical orders. The selected continuous M1 battery capacity is also not yet snapped to available catalogue packs. Weather direction is unspecified, and the routine hourly profile sums to about 1.1993 rather than one. The model normalizes each hourly profile and records that choice. See the documentation for all assumptions and limitations before relying on the recommendation.
