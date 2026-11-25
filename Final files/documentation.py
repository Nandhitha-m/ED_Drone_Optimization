from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportData:
    reports_dir: Path
    m4_dir: Path
    m5_dir: Path
    m6_dir: Path
    base_designs: list[dict[str, Any]]
    m2_designs: list[dict[str, Any]]
    m3_designs: list[dict[str, Any]]
    all_designs: list[dict[str, Any]]
    m2: dict[str, Any]
    m3: dict[str, Any]
    selected_cost: dict[str, str]
    selected_design: dict[str, Any]
    hub_rows: list[dict[str, str]]
    assignment_rows: list[dict[str, str]]
    m5_summary: dict[str, Any]
    m6_output: str
    front: list[dict[str, str]]
    knee: dict[str, str]
    recommended: dict[str, str]
    recommendation_reason: str
    recommendation_meets_target: bool
    weighted_ok: bool
    m6_pop_size: int
    m6_generations: int
    m6_replications: int


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(reports_dir: Path, name: str, body: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / name).write_text(body.strip() + "\n", encoding="utf-8")


def _format_design_rows(designs: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            f"{item['payload_kg']:.2f}",
            f"{item['radius_km']:.3f}",
            f"{item['rotor_diameter_m']:.3f}",
            f"{item['battery_wh']:.0f}",
            f"{item['thrust_per_rotor_n']:.2f}",
            f"{item['minimum_constraint_margin']:.4g}",
        ]
        for item in designs
    ]


def write_reports(data: ReportData) -> None:
    d = data
    write = lambda name, body: _write_report(d.reports_dir, name, body)

    write(
        "M1.md",
        f"""# M1 - Vehicle mass, power, range, and structure

## What and why
M1 maximizes still-air mission radius for each payload from 0.5 to 2.5 kg. The vehicle trades rotor diameter, battery energy, arm diameter/wall, and cruise speed against mass, power, range, bending stress, deflection, natural frequency, and slenderness.

## How
The orchestration executes the M1 notebook's model/optimizer cell and applies its SLSQP formulation to the five payload cases. Each returned point is checked against the notebook's structural inequalities before it is added to the design set.

## Main results
{_markdown_table(["Payload (kg)", "Radius (km)", "Rotor D (m)", "Battery (Wh)", "Thrust/rotor (N)", "Min. margin"], _format_design_rows(d.base_designs))}

The original notebook uses a fixed 0.25 kg hub-plate mass and FM=0.65 in this baseline. The feasible designs are the baseline inputs to the later stages.
""",
    )

    sweep_rows = [
        [
            f"{row['volume_fraction']:.2f}",
            f"{row['plate_mass_kg']:.4f}",
            f"{row['compliance_nm']:.2f}",
        ]
        for row in d.m2["sweep"]
    ]
    write(
        "M2.md",
        f"""# M2 - Hub-plate topology and M1 mass feedback

## What and why
M2 visualizes a 160 mm square aluminum hub plate and compares compliance against plate mass. This connects structural material layout to vehicle mass instead of treating the plate as a free component.

## How
The M2 module runs its volume-fraction sweep and topology visualization. Its source fixes plate thickness at 3 mm and uses a 0.30 volume fraction for the displayed topology; it does not optimize or return thickness. To avoid inventing an M2 optimum, this run uses that configured thickness/fraction and computes plate mass including the 40 g fastener allowance, then reruns M1 with that mass replacing the notebook's 0.25 kg plate assumption.

## Main results
Configured plate thickness: **{d.m2['plate_thickness_m'] * 1000:.1f} mm**. M2-fed plate mass: **{d.m2['plate_mass_kg']:.4f} kg**.

{_markdown_table(["Volume fraction", "Plate mass (kg)", "Compliance (Nm)"], sweep_rows)}

M1 was reoptimized for each payload with the M2-fed plate mass. The new five designs are appended to the baseline set:

{_markdown_table(["Payload (kg)", "Radius (km)", "Rotor D (m)", "Battery (Wh)", "Thrust/rotor (N)", "Min. margin"], _format_design_rows(d.m2_designs))}
""",
    )

    model_rows = [
        [
            model["name"],
            f"{model['rmse']:.5f}",
            f"{model['mae']:.5f}",
            f"{model['r_squared']:.5f}",
        ]
        for model in d.m3["models"]
    ]
    write(
        "M3.md",
        f"""# M3 - Rotor figure-of-merit surrogate

## What and why
The reference brief replaces the constant rotor figure of merit with an affordable surrogate trained on the supplied rotor-oracle data. M3 compares surrogate predictions using leave-one-out cross-validation (LOOCV), then searches the observed D/T domain for the highest predicted FM.

## How
The checked-in dataset has 30 observations with rotor diameter D (m), thrust T (N), and response FM. The lower-LOOCV-RMSE model is optimized with differential evolution, bounded by the observed D and T minima/maxima. The predicted point is passed back to M1 as the FM setting; M1 fixes D at the selected diameter and enforces a minimum per-rotor thrust equal to the M3 point's thrust.

## Main results
Selected model: **{d.m3['selected_model']}** (LOOCV RMSE {d.m3['selected_rmse']:.6g}). Predicted optimum: **FM={d.m3['figure_of_merit']:.5f}, D={d.m3['rotor_diameter_m']:.5f} m, T={d.m3['thrust_n']:.5f} N**.

{_markdown_table(["Surrogate", "LOOCV RMSE", "LOOCV MAE", "R-squared"], model_rows)}

M1's five M3-conditioned designs are appended to the 10 earlier designs:

{_markdown_table(["Payload (kg)", "Radius (km)", "Rotor D (m)", "Battery (Wh)", "Thrust/rotor (N)", "Min. margin"], _format_design_rows(d.m3_designs))}

The FM optimum is a surrogate prediction, not a new oracle measurement.

The workspace is missing the `rbf_surrogate.py` helper imported by the M3 comparison scripts. For this run, `main.py` supplies an in-memory standardized SciPy thin-plate-spline RBF adapter; the comparison and LOOCV results are therefore reproducible from this orchestrator, but not by running `M3/compare.py` alone.
""",
    )

    write(
        "M4.md",
        f"""# M4 - Hub network, assignments, and fleet sizing

## What and why
M4 minimizes annualized network and service cost while choosing which candidate sites to open, assigning every PHC, and sizing drones, bays, and staffing. It uses no-fly-zone detours, hourly wind, M1 range, and a 10% route-distance margin.

## How
All {len(d.all_designs)} designs from M1-M3 are converted to M4 vehicles and passed to the existing MILP. The original exact-enumeration check on a 4-hub/8-PHC subinstance is retained. Results and assumptions are saved under `{d.m4_dir}`.

## Main results
Selected vehicle: **{d.selected_cost['vehicle']}** ({d.selected_design['payload_kg']:.2f} kg payload, {d.selected_design['radius_km']:.2f} km still-air radius). Annualized cost: **INR {float(d.selected_cost['annual_cost_inr']):,.2f}**. Open hubs: {len(d.hub_rows)}; PHCs: {len(d.assignment_rows)}.

{_markdown_table(["Hub", "Drones", "Bays", "Operators/shift", "PHCs served"], [[row["hub_id"], row["drones"], row["charging_bays"], row["operators_per_shift"], row["served_phcs"]] for row in d.hub_rows])}

M4's output is an expected-demand planning solution; queueing and emergency reliability are evaluated next in M5.
""",
    )

    write(
        "M5.md",
        f"""# M5 - Discrete-event operations simulation

## What and why
M5 checks operational behavior that the M4 expected-demand MILP cannot resolve: queueing, drone availability, charging, wind grounding, emergency road fallback, and the 45-minute emergency target.

## How
The integrated M5 simulator receives the M4 selected hub plan and PHC assignments, generates hourly Poisson demand with a fixed seed, applies the project weather, and uses M4's loaded/empty power model for mission energy. The event log and JSON summary are saved under `{d.m5_dir}`.

## Main results
Vehicle: **{d.m5_summary['vehicle']}**; orders: {d.m5_summary['orders']}; emergencies: {d.m5_summary['emergencies']}; **S45={d.m5_summary['S45_percent']:.2f}%**; **P95={d.m5_summary['p95_emergency_minutes']:.2f} min**; road fallback share: {d.m5_summary['road_fallback_share']:.3f}; undelivered: {d.m5_summary['undelivered_orders']}.

This is a one-day-arrival seeded scenario, not a confidence interval. Any orders still queued after the seven-day drain are explicitly converted to road fallback (120 minutes; emergency includes the 30-minute wait). The project-specific M5 stage-9 scripts require ZIP inputs not present in the workspace; this run uses the checked-in integrated simulator with M4 outputs instead.
""",
    )

    write(
        "M6.md",
        f"""# M6 - Simulation-based operating-policy optimization

## What and why
M6 searches vehicle, fleet, charging, spare-battery, and emergency-reserve decisions across the M4 network. Its objectives are annual cost (minimize), S45 (maximize), and road fallback (minimize); it also reports emergency-delivery P95.

## How
The existing NSGA-II implementation evaluates candidates with the M5 event simulator using shared random seeds (CRN). The knee point represents a compromise across objectives. A weighted-sum slice is compared with the NSGA-II front. Run size: population {d.m6_pop_size}, generations {d.m6_generations}, CRN replications {d.m6_replications}.

## Main results
Knee candidate: **{d.knee['vehicle']}**; annual cost INR {float(d.knee['annual_cost_inr']):,.2f}; S45 {float(d.knee['s45_percent']):.2f}%; road fallback {float(d.knee['road_fallback_share']):.3f}; P95 {float(d.knee['p95_emergency_min']):.2f} min. Weighted-sum points on/ahead of the front: **{d.weighted_ok}**.

{d.m6_output.strip()}

The Pareto front and weighted-sum verification rows are in `{d.m6_dir}`. This configurable run is an exploratory search; increase the population, generations, and replications for a more stable decision.
""",
    )

    write(
        "final_optimization.md",
        f"""# SkyLink - Final optimization recommendation

## Recommended candidate
Select **{d.recommended['vehicle']}** as the current best candidate: annual cost **INR {float(d.recommended['annual_cost_inr']):,.2f}**, **{float(d.recommended['s45_percent']):.2f}% S45**, **{float(d.recommended['road_fallback_share']):.3f} road fallback**, and **{float(d.recommended['p95_emergency_min']):.2f} min P95 emergency time**. Selection rule: {d.recommendation_reason}.

{("This candidate meets the 100% S45 target in the sampled M6 results." if d.recommendation_meets_target else "This candidate still misses the 100% S45 target; treat it as the best tested design, not as deployable compliance. Increase the search/simulation budget and revise the fleet or dispatch policy before deployment.")}

## Evidence across the chain
- **M1:** {len(d.base_designs)} payload/range designs, each checked against the notebook's structural constraints.
- **M2:** configured 3 mm / 30% topology gives a mass of {d.m2['plate_mass_kg']:.4f} kg including fasteners; M1 was rerun with this plate mass.
- **M3:** {d.m3['selected_model']} selected by LOOCV; predicted FM {d.m3['figure_of_merit']:.5f} at D={d.m3['rotor_diameter_m']:.4f} m and T={d.m3['thrust_n']:.3f} N; M1 was rerun with these settings.
- **M4:** {len(d.all_designs)} M1 designs compared; {len(d.hub_rows)} hubs opened for the selected M4 plan at INR {float(d.selected_cost['annual_cost_inr']):,.2f}/year.
- **M5:** seeded event simulation reports S45 {d.m5_summary['S45_percent']:.2f}% and P95 {d.m5_summary['p95_emergency_minutes']:.2f} min for the selected M4 plan.
- **M6:** {len(d.front)} Pareto candidates; knee {d.knee['vehicle']}; weighted-sum check passed: {d.weighted_ok}.

## Decision limits
Treat this as a model-based recommendation, not a guaranteed under-45-minute service commitment. M2 does not calculate an optimum thickness; its existing 3 mm setting is used. M3's FM is a surrogate prediction. M5 uses one arrival-day realization, and M6 defaults to a compact CRN search; increase simulation and search sizes and review data assumptions before operational procurement. The original M4 weather/profile and package-weight limitations remain in its verification notes.

## Reproducibility
Run `python main.py` from this project directory. Generated module reports are in `{d.reports_dir}`; CSV, JSON, plots, and event outputs are in the adjacent M1-M6 result folders.
""",
    )
