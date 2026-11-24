"""SkyLink M6 simulation-based multi-objective optimizer.

Run from any directory with:
    python m6.py [--data-dir PATH] [--output-dir PATH]

M6 scope (from project specification):
- Decisions: vehicle design (from M1), drones and charging bays per open hub,
  spare batteries per hub, emergency reserve fraction r in [0, 1).
- Objectives: annual cost (minimise), S45 (maximise), road-fallback share (minimise).
- Reported metric: 95th-percentile emergency delivery time (P95).
- Method: NSGA-II with common random numbers (CRN), Pareto front extraction,
  knee-point identification, and weighted-sum slice comparison/verification.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure m4.py can be imported when running from another directory.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from m4 import (  # noqa: E402
    CLASSES,
    CLASS_COLUMNS,
    Vehicle,
    annual_cost_parameters,
    build_routes,
    build_vehicle_list,
    crf,
    load_csv,
    load_no_fly,
    read_weather,
    write_csv,
)


@dataclass(frozen=True)
class PolicyDecision:
    vehicle_name: str
    drones_by_hub: dict[str, int]
    bays_by_hub: dict[str, int]
    spare_batteries_by_hub: dict[str, int]
    reserve_fraction: float


@dataclass
class SimulationOutcome:
    annual_cost_inr: float
    s45_percent: float
    road_fallback_share: float
    p95_emergency_min: float


@dataclass
class EvaluatedCandidate:
    decision: PolicyDecision
    outcome: SimulationOutcome
    # Minimized objective vector: [annual_cost_inr, -s45_percent, road_fallback_share]
    objectives_min: np.ndarray


def compute_policy_annual_cost(
    decision: PolicyDecision,
    vehicle: Vehicle,
    open_hubs: list[str],
    costs: dict[str, Any],
    sim_metrics: dict[str, float],
) -> float:
    """Compute annualised cost combining M4 capital/staffing parameters and M5 operational outputs."""
    params = annual_cost_parameters(costs, vehicle)
    rate = float(costs["discount_rate"])
    battery_crf = crf(rate, int(costs["battery_pack"]["cycle_life"]))
    battery_unit_cost = float(costs["battery_pack"]["cost_per_Wh"]) * vehicle.battery_wh
    drones_per_op = float(costs["staff"]["drones_per_operator"])

    fixed_and_fleet_cost = 0.0
    for hub_id in open_hubs:
        n_j = max(0, int(decision.drones_by_hub.get(hub_id, 0)))
        b_j = max(0, int(decision.bays_by_hub.get(hub_id, 0)))
        s_j = max(0, int(decision.spare_batteries_by_hub.get(hub_id, 0)))
        # At least 1 operator per open hub; 1 operator per 4 drones per shift.
        o_j = max(1, math.ceil(n_j / drones_per_op)) if n_j > 0 else 1

        # Each drone includes 1 battery pack in M4; add spare battery packs s_j.
        spare_battery_annual = s_j * battery_unit_cost * battery_crf
        fixed_and_fleet_cost += (
            params["hub"]
            + n_j * params["drone"]
            + b_j * params["bay"]
            + o_j * params["operator"]
            + spare_battery_annual
        )

    # Operational costs from simulation outputs (in INR/day * 365 days/year).
    daily_grid_cost = float(sim_metrics.get("daily_grid_cost_inr", 0.0))
    daily_genset_cost = float(sim_metrics.get("daily_genset_cost_inr", 0.0))
    daily_battery_wear_cost = float(sim_metrics.get("daily_battery_wear_cost_inr", 0.0))
    daily_road_cost = float(sim_metrics.get("daily_road_cost_inr", 0.0))

    # Fallback if simulator supplies total energy (Wh/day) and road counts instead of pre-summed costs.
    if daily_grid_cost == 0.0 and daily_genset_cost == 0.0 and "daily_flight_energy_wh" in sim_metrics:
        eff = float(costs["charging_bay"]["charge_efficiency"])
        grid_tariff = float(costs["energy"]["grid_per_kWh"])
        flight_wh = float(sim_metrics["daily_flight_energy_wh"])
        daily_grid_cost = (flight_wh / max(eff, 1e-9)) / 1000.0 * grid_tariff

    if daily_battery_wear_cost == 0.0 and "daily_flight_energy_wh" in sim_metrics:
        flight_wh = float(sim_metrics["daily_flight_energy_wh"])
        daily_battery_wear_cost = (
            flight_wh * float(costs["battery_pack"]["cost_per_Wh"]) / float(costs["battery_pack"]["cycle_life"])
        )

    return fixed_and_fleet_cost + 365.0 * (
        daily_grid_cost + daily_genset_cost + daily_battery_wear_cost + daily_road_cost
    )


def run_m5_simulation(
    decision: PolicyDecision,
    vehicle: Vehicle,
    lanes: dict[tuple[str, str], Any],
    phcs: list[dict[str, str]],
    weather: list[dict[str, Any]],
    costs: dict[str, Any],
    crn_seed: int,
) -> SimulationOutcome:
    """Adapter hook for the M5 discrete-event simulator under Common Random Numbers (CRN).

    Replace the fallback calculation below with your call into M5_integrated.py / M5_final.py
    using `crn_seed` so candidate policies are compared on identical arrival/weather draws.
    """
    open_hubs = [h for h, cnt in decision.drones_by_hub.items() if cnt > 0]
    rng = np.random.default_rng(crn_seed)

    # Fallback analytical/CRN-perturbed approximation using M4 lanes when M5 module hook is not wired.
    total_orders_day = sum(
        float(phc[CLASS_COLUMNS[k]]) for phc in phcs for k in CLASSES
    )
    emergency_orders_day = sum(float(phc[CLASS_COLUMNS["emergency"]]) for phc in phcs)

    total_busy_min = 0.0
    total_energy_wh = 0.0
    total_road_trips_day = 0.0
    total_emergency_road_day = 0.0

    for phc in phcs:
        phc_id = phc["phc_id"]
        feasible_hubs = [h for h in open_hubs if (h, phc_id) in lanes]
        if not feasible_hubs:
            for k in CLASSES:
                total_road_trips_day += float(phc[CLASS_COLUMNS[k]])
                if k == "emergency":
                    total_emergency_road_day += float(phc[CLASS_COLUMNS["emergency"]])
            continue

        best_hub = min(feasible_hubs, key=lambda h: lanes[(h, phc_id)].distance_km)
        lane = lanes[(best_hub, phc_id)]
        n_drones = max(1, decision.drones_by_hub.get(best_hub, 1))
        n_bays = max(1, decision.bays_by_hub.get(best_hub, 1))
        n_spares = max(0, decision.spare_batteries_by_hub.get(best_hub, 0))

        # Reserve fraction r holds a share of idle fleet capacity for emergencies;
        # higher r helps emergency wait times but can increase routine/lab fallback.
        effective_reserve = min(max(decision.reserve_fraction, 0.0), max(0.0, (n_drones - 1) / max(n_drones, 1)))
        battery_factor = min(1.0, (n_drones + n_spares) / max(n_drones * 1.5, 1.0))
        bay_factor = min(1.0, n_bays / max(n_drones * 0.6, 1.0))
        service_factor = max(0.25, (1.0 - 0.25 * effective_reserve) * (0.65 + 0.35 * min(battery_factor, bay_factor)))

        # CRN noise across replications
        crn_shock = float(rng.normal(1.0, 0.03))
        busy = lane.busy_min * crn_shock / service_factor
        energy = lane.energy_wh * crn_shock
        total_busy_min += busy
        total_energy_wh += energy

        for k in CLASSES:
            orders_k = float(phc[CLASS_COLUMNS[k]])
            fb = lane.fallback_fraction.get(k, 0.0)
            if k != "emergency":
                fb = min(1.0, fb * (1.0 + 0.20 * effective_reserve))
            else:
                fb = max(0.0, fb * (1.0 - 0.35 * effective_reserve))
            trips = orders_k * fb * crn_shock
            total_road_trips_day += trips
            if k == "emergency":
                total_emergency_road_day += trips

    road_share = min(1.0, max(0.0, total_road_trips_day / max(total_orders_day, 1e-9)))
    emergency_road_share = min(1.0, max(0.0, total_emergency_road_day / max(emergency_orders_day, 1e-9)))

    # S45 increases with reserve availability and fleet/battery/bay sufficiency,
    # while emergencies sent to road take 30 min wait + 120 min road travel (> 45 min).
    s45_raw = (1.0 - emergency_road_share) * (0.72 + 0.24 * min(1.0, decision.reserve_fraction / 0.4)) * float(rng.normal(1.0, 0.015))
    s45_percent = min(100.0, max(0.0, s45_raw * 100.0))

    # P95 emergency delivery duration: rises toward 150 min as road/queue delays increase.
    base_flight_min = 2.0 * (12.0 / max(vehicle.speed_mps, 1.0)) / 60.0 + 10.0
    p95_emergency_min = max(
        base_flight_min,
        min(150.0, base_flight_min + 115.0 * emergency_road_share + 12.0 * (1.0 - s45_percent / 100.0)) * float(rng.normal(1.0, 0.02)),
    )

    road_prices = costs["road_fallback"]["cost_per_trip"]
    avg_road_price = (
        float(road_prices["emergency"]) * emergency_orders_day
        + float(road_prices["routine"]) * sum(float(p[CLASS_COLUMNS["routine"]]) for p in phcs)
        + float(road_prices["lab"]) * sum(float(p[CLASS_COLUMNS["lab"]]) for p in phcs)
    ) / max(total_orders_day, 1e-9)

    sim_metrics = {
        "daily_flight_energy_wh": total_energy_wh,
        "daily_road_cost_inr": total_road_trips_day * avg_road_price,
    }
    annual_cost = compute_policy_annual_cost(decision, vehicle, open_hubs, costs, sim_metrics)
    return SimulationOutcome(
        annual_cost_inr=annual_cost,
        s45_percent=s45_percent,
        road_fallback_share=road_share,
        p95_emergency_min=p95_emergency_min,
    )


def evaluate_policy_crn(
    decision: PolicyDecision,
    vehicles_by_name: dict[str, Vehicle],
    lanes_by_vehicle: dict[str, dict[tuple[str, str], Any]],
    phcs: list[dict[str, str]],
    weather: list[dict[str, Any]],
    costs: dict[str, Any],
    crn_seeds: list[int],
) -> EvaluatedCandidate:
    """Evaluate a policy across shared CRN seeds and average metrics."""
    vehicle = vehicles_by_name[decision.vehicle_name]
    lanes = lanes_by_vehicle[decision.vehicle_name]
    outcomes = [
        run_m5_simulation(decision, vehicle, lanes, phcs, weather, costs, seed)
        for seed in crn_seeds
    ]
    mean_cost = float(np.mean([o.annual_cost_inr for o in outcomes]))
    mean_s45 = float(np.mean([o.s45_percent for o in outcomes]))
    mean_road = float(np.mean([o.road_fallback_share for o in outcomes]))
    mean_p95 = float(np.mean([o.p95_emergency_min for o in outcomes]))

    avg_outcome = SimulationOutcome(
        annual_cost_inr=mean_cost,
        s45_percent=mean_s45,
        road_fallback_share=mean_road,
        p95_emergency_min=mean_p95,
    )
    # Objectives to minimize: [annual_cost_inr, -s45_percent, road_fallback_share]
    obj_min = np.array([mean_cost, -mean_s45, mean_road], dtype=float)
    return EvaluatedCandidate(decision=decision, outcome=avg_outcome, objectives_min=obj_min)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if minimization vector a Pareto-dominates b."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(population: list[EvaluatedCandidate]) -> list[list[int]]:
    """Fast non-dominated sorting returning Pareto fronts by index."""
    n = len(population)
    domination_count = [0] * n
    dominated_solutions: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(population[p].objectives_min, population[q].objectives_min):
                dominated_solutions[p].append(q)
            elif dominates(population[q].objectives_min, population[p].objectives_min):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(front: list[int], population: list[EvaluatedCandidate]) -> dict[int, float]:
    """Compute crowding distance for diversity preservation in NSGA-II."""
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distances[idx] = math.inf
        return distances

    num_obj = len(population[0].objectives_min)
    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda idx: population[idx].objectives_min[m])
        min_val = population[sorted_front[0]].objectives_min[m]
        max_val = population[sorted_front[-1]].objectives_min[m]
        distances[sorted_front[0]] = math.inf
        distances[sorted_front[-1]] = math.inf
        span = max_val - min_val
        if span <= 1e-12:
            continue
        for i in range(1, len(sorted_front) - 1):
            prev_val = population[sorted_front[i - 1]].objectives_min[m]
            next_val = population[sorted_front[i + 1]].objectives_min[m]
            distances[sorted_front[i]] += (next_val - prev_val) / span

    return distances


def mutate_decision(
    parent: PolicyDecision,
    vehicle_names: list[str],
    hub_ids: list[str],
    rng: random.Random,
) -> PolicyDecision:
    """Mutate vehicle, per-hub drones/bays/spare batteries, or reserve fraction r."""
    v_name = parent.vehicle_name
    if rng.random() < 0.25:
        v_name = rng.choice(vehicle_names)

    drones = dict(parent.drones_by_hub)
    bays = dict(parent.bays_by_hub)
    spares = dict(parent.spare_batteries_by_hub)

    for h in hub_ids:
        if rng.random() < 0.35:
            drones[h] = max(0, min(15, drones.get(h, 1) + rng.choice([-1, 0, 1])))
        if rng.random() < 0.35:
            bays[h] = max(0, min(12, bays.get(h, 1) + rng.choice([-1, 0, 1])))
        if rng.random() < 0.35:
            spares[h] = max(0, min(15, spares.get(h, 1) + rng.choice([-1, 0, 1])))

    r_val = parent.reserve_fraction
    if rng.random() < 0.6:
        r_val = min(0.90, max(0.0, r_val + rng.uniform(-0.12, 0.12)))

    return PolicyDecision(
        vehicle_name=v_name,
        drones_by_hub=drones,
        bays_by_hub=bays,
        spare_batteries_by_hub=spares,
        reserve_fraction=round(r_val, 4),
    )


def crossover_decision(
    p1: PolicyDecision,
    p2: PolicyDecision,
    hub_ids: list[str],
    rng: random.Random,
) -> PolicyDecision:
    """Uniform/blend crossover over two parent policies."""
    v_name = p1.vehicle_name if rng.random() < 0.5 else p2.vehicle_name
    drones = {h: (p1.drones_by_hub.get(h, 0) if rng.random() < 0.5 else p2.drones_by_hub.get(h, 0)) for h in hub_ids}
    bays = {h: (p1.bays_by_hub.get(h, 0) if rng.random() < 0.5 else p2.bays_by_hub.get(h, 0)) for h in hub_ids}
    spares = {
        h: (p1.spare_batteries_by_hub.get(h, 0) if rng.random() < 0.5 else p2.spare_batteries_by_hub.get(h, 0))
        for h in hub_ids
    }
    r_val = round(0.5 * (p1.reserve_fraction + p2.reserve_fraction), 4)
    return PolicyDecision(
        vehicle_name=v_name,
        drones_by_hub=drones,
        bays_by_hub=bays,
        spare_batteries_by_hub=spares,
        reserve_fraction=r_val,
    )


def run_nsga2_crn(
    vehicles_by_name: dict[str, Vehicle],
    lanes_by_vehicle: dict[str, dict[tuple[str, str], Any]],
    phcs: list[dict[str, str]],
    hub_ids: list[str],
    weather: list[dict[str, Any]],
    costs: dict[str, Any],
    crn_seeds: list[int],
    pop_size: int = 24,
    generations: int = 12,
    seed: int = 42,
) -> tuple[list[EvaluatedCandidate], list[EvaluatedCandidate]]:
    """Run NSGA-II with Common Random Numbers (CRN) and return (pareto_front, all_evaluated)."""
    rng = random.Random(seed)
    vehicle_names = list(vehicles_by_name.keys())

    # Seed initial population with sensible baseline configurations + random perturbations.
    base_decision = PolicyDecision(
        vehicle_name=vehicle_names[-1],
        drones_by_hub={h: 2 for h in hub_ids[:3]},
        bays_by_hub={h: 2 for h in hub_ids[:3]},
        spare_batteries_by_hub={h: 2 for h in hub_ids[:3]},
        reserve_fraction=0.20,
    )
    population_decisions = [base_decision]
    while len(population_decisions) < pop_size:
        population_decisions.append(mutate_decision(base_decision, vehicle_names, hub_ids, rng))

    evaluated = [
        evaluate_policy_crn(d, vehicles_by_name, lanes_by_vehicle, phcs, weather, costs, crn_seeds)
        for d in population_decisions
    ]
    all_seen = list(evaluated)

    for _ in range(generations):
        offspring_decisions: list[PolicyDecision] = []
        while len(offspring_decisions) < pop_size:
            a, b = rng.sample(evaluated, 2)
            p1 = a if dominates(a.objectives_min, b.objectives_min) else b
            c, d_cand = rng.sample(evaluated, 2)
            p2 = c if dominates(c.objectives_min, d_cand.objectives_min) else d_cand
            child = crossover_decision(p1.decision, p2.decision, hub_ids, rng)
            child = mutate_decision(child, vehicle_names, hub_ids, rng)
            offspring_decisions.append(child)

        offspring_eval = [
            evaluate_policy_crn(d, vehicles_by_name, lanes_by_vehicle, phcs, weather, costs, crn_seeds)
            for d in offspring_decisions
        ]
        all_seen.extend(offspring_eval)

        combined = evaluated + offspring_eval
        fronts = non_dominated_sort(combined)
        next_pop: list[EvaluatedCandidate] = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(combined[idx] for idx in front)
            else:
                cd = crowding_distance(front, combined)
                sorted_by_cd = sorted(front, key=lambda idx: cd[idx], reverse=True)
                needed = pop_size - len(next_pop)
                next_pop.extend(combined[idx] for idx in sorted_by_cd[:needed])
                break
        evaluated = next_pop

    pareto_fronts = non_dominated_sort(evaluated)
    pareto_front = [evaluated[idx] for idx in pareto_fronts[0]] if pareto_fronts else []
    return pareto_front, all_seen


def find_knee_point(front: list[EvaluatedCandidate]) -> EvaluatedCandidate | None:
    """Identify knee point on the Pareto front using maximum distance to extreme chord."""
    if not front:
        return None
    if len(front) <= 2:
        return front[0]

    pts = np.array([c.objectives_min for c in front], dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.where(maxs - mins > 1e-12, maxs - mins, 1.0)
    norm_pts = (pts - mins) / spans

    # Extreme points along first two objectives (cost and -S45).
    p_start = norm_pts[int(np.argmin(norm_pts[:, 0]))]
    p_end = norm_pts[int(np.argmin(norm_pts[:, 1]))]
    line_vec = p_end - p_start
    line_norm = float(np.linalg.norm(line_vec))
    if line_norm <= 1e-12:
        return front[0]

    unit_vec = line_vec / line_norm
    best_idx = 0
    best_dist = -1.0
    for i, pt in enumerate(norm_pts):
        rel = pt - p_start
        proj = float(np.dot(rel, unit_vec)) * unit_vec
        perp_dist = float(np.linalg.norm(rel - proj))
        if perp_dist > best_dist:
            best_dist = perp_dist
            best_idx = i
    return front[best_idx]


def run_weighted_sum_slice(
    candidates: list[EvaluatedCandidate],
    weights_cost_vs_neg_s45: list[tuple[float, float]],
) -> list[tuple[float, float, EvaluatedCandidate]]:
    """Evaluate weighted-sum scalarization on the two-objective slice (annual cost vs. -S45)."""
    pts = np.array([c.objectives_min[:2] for c in candidates], dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.where(maxs - mins > 1e-12, maxs - mins, 1.0)
    norm_pts = (pts - mins) / spans

    ws_results: list[tuple[float, float, EvaluatedCandidate]] = []
    for w_cost, w_s45 in weights_cost_vs_neg_s45:
        scores = w_cost * norm_pts[:, 0] + w_s45 * norm_pts[:, 1]
        best_i = int(np.argmin(scores))
        ws_results.append((w_cost, w_s45, candidates[best_i]))
    return ws_results


def verify_weighted_sum_against_front(
    ws_results: list[tuple[float, float, EvaluatedCandidate]],
    nsga_front: list[EvaluatedCandidate],
    tol: float = 1e-6,
) -> list[dict[str, Any]]:
    """Verify each weighted-sum point on the 2-obj slice (cost, -S45) is not strictly dominated by NSGA-II front."""
    front_2d = np.array([c.objectives_min[:2] for c in nsga_front], dtype=float)
    rows: list[dict[str, Any]] = []
    for w_cost, w_s45, cand in ws_results:
        pt = cand.objectives_min[:2]
        strictly_dominated = False
        for f_pt in front_2d:
            if np.all(f_pt <= pt + tol) and np.any(f_pt < pt - tol):
                strictly_dominated = True
                break
        rows.append({
            "weight_cost": w_cost,
            "weight_neg_s45": w_s45,
            "annual_cost_inr": round(float(cand.outcome.annual_cost_inr), 2),
            "s45_percent": round(float(cand.outcome.s45_percent), 4),
            "road_fallback_share": round(float(cand.outcome.road_fallback_share), 5),
            "p95_emergency_min": round(float(cand.outcome.p95_emergency_min), 2),
            "on_or_ahead_of_nsga2_front": not strictly_dominated,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkyLink M6 simulation-based multi-objective optimization.")
    default_data = CURRENT_DIR.parent / "Projectskylink_data"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "m6_output")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--reserve-fraction", type=float, default=0.90)
    parser.add_argument("--crn-replications", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = load_csv(args.data_dir / "sites.csv")
    phcs = [row for row in sites if row["type"] == "phc"]
    hubs = [row for row in sites if int(row["candidate_hub"]) == 1]
    hub_ids = [h["site_id"] for h in hubs]
    demand_rows = {row["phc_id"]: row for row in load_csv(args.data_dir / "demand.csv")}
    phcs = [{**site, **demand_rows[site["site_id"]]} for site in phcs]

    profile_rows = load_csv(args.data_dir / "hourly_profile.csv")
    profiles: dict[str, list[float]] = {}
    for kind in CLASSES:
        col = {"emergency": "emergency_frac", "routine": "routine_frac", "lab": "lab_frac"}[kind]
        raw = [float(row[col]) for row in profile_rows]
        total = sum(raw)
        profiles[kind] = [f / total for f in raw]

    costs = yaml.safe_load((args.data_dir / "costs.yaml").read_text(encoding="utf-8"))
    obstacles = load_no_fly(args.data_dir / "nofly.geojson")
    weather, _, _ = read_weather(args.data_dir, args.year)
    vehicles = build_vehicle_list()
    vehicles_by_name = {v.name: v for v in vehicles}
    lanes_by_vehicle, _ = build_routes(
        phcs, hubs, obstacles, vehicles, profiles, weather, costs, args.reserve_fraction
    )

    crn_seeds = [1000 + i for i in range(max(1, args.crn_replications))]
    pareto_front, all_candidates = run_nsga2_crn(
        vehicles_by_name=vehicles_by_name,
        lanes_by_vehicle=lanes_by_vehicle,
        phcs=phcs,
        hub_ids=hub_ids,
        weather=weather,
        costs=costs,
        crn_seeds=crn_seeds,
        pop_size=args.pop_size,
        generations=args.generations,
    )
    knee = find_knee_point(pareto_front)

    weights = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
    ws_results = run_weighted_sum_slice(all_candidates, weights)
    verification_rows = verify_weighted_sum_against_front(ws_results, pareto_front)

    front_rows = []
    for idx, cand in enumerate(pareto_front):
        d = cand.decision
        front_rows.append({
            "solution_id": idx,
            "is_knee": knee is not None and cand is knee,
            "vehicle": d.vehicle_name,
            "reserve_fraction": d.reserve_fraction,
            "total_drones": sum(d.drones_by_hub.values()),
            "total_bays": sum(d.bays_by_hub.values()),
            "total_spare_batteries": sum(d.spare_batteries_by_hub.values()),
            "annual_cost_inr": round(cand.outcome.annual_cost_inr, 2),
            "s45_percent": round(cand.outcome.s45_percent, 4),
            "road_fallback_share": round(cand.outcome.road_fallback_share, 5),
            "p95_emergency_min": round(cand.outcome.p95_emergency_min, 2),
        })

    write_csv(args.output_dir / "m6_pareto_front.csv", front_rows)
    write_csv(args.output_dir / "m6_weighted_sum_verification.csv", verification_rows)

    all_ok = all(r["on_or_ahead_of_nsga2_front"] for r in verification_rows)
    print(f"M6 optimization complete. Evaluated {len(all_candidates)} policies; Pareto front size: {len(pareto_front)}")
    if knee is not None:
        print(
            f"Knee policy: vehicle={knee.decision.vehicle_name}, r={knee.decision.reserve_fraction}, "
            f"cost=INR {knee.outcome.annual_cost_inr:,.2f}, S45={knee.outcome.s45_percent:.2f}%, "
            f"road_share={knee.outcome.road_fallback_share:.4f}, P95={knee.outcome.p95_emergency_min:.2f} min"
        )
    print(f"Weighted-sum vs NSGA-II front verification passed: {all_ok}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
