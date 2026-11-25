from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Generator


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "M1.ipynb"
DATA_DIR = ROOT / "Projectskylink_data(1)"
PAYLOADS_KG = (0.5, 1.0, 1.5, 2.0, 2.5)
M2_PLATE_SIDE_M = 0.16
M2_PLATE_THICKNESS_M = 0.003
M2_ALUMINUM_DENSITY_KG_M3 = 2700.0
M2_FASTENER_MASS_KG = 0.04
M2_VOLUME_FRACTION = 0.3


@contextlib.contextmanager
def working_directory(path: Path) -> Generator[None, None, None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_module(name: str, path: Path) -> ModuleType:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def use_project_environment() -> None:
    site_packages = ROOT / "m4_opti" / "Lib" / "site-packages"
    if site_packages.is_dir() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def run_m1_stage(
    stage: str,
    output_dir: Path,
    plate_mass_kg: float,
    figure_of_merit: float = 0.65,
    fixed_rotor_diameter_m: float | None = None,
    minimum_thrust_n: float | None = None,
) -> list[dict[str, Any]]:
    import numpy as np
    from scipy.optimize import minimize

    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"M1 notebook is missing: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][9]["source"])

    replacements = (
        ("m_plate = 0.25", f"m_plate = {plate_mass_kg!r}"),
        ("FM = 0.65", f"FM = {figure_of_merit!r}"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one '{old}' in M1 model cell; found {source.count(old)}.")
        source = source.replace(old, new)

    namespace: dict[str, Any] = {"__name__": f"m1_{stage.lower()}"}
    with working_directory(output_dir):
        exec(compile(source, str(NOTEBOOK_PATH) + ":cell-9", "exec"), namespace)

    bounds = list(namespace["bounds"])
    starting_design = np.asarray(namespace["x0"], dtype=float).copy()
    if fixed_rotor_diameter_m is not None:
        bounds[0] = (fixed_rotor_diameter_m, fixed_rotor_diameter_m)
        starting_design[0] = fixed_rotor_diameter_m

    base_constraints = namespace["structural_constraints"]

    def constraints(design: Any) -> Any:
        margins = base_constraints(design)
        if minimum_thrust_n is not None:
            thrust_margin = (
                namespace["evaluate_design"](design)["thrust"] - minimum_thrust_n
            ) / max(minimum_thrust_n, 1.0)
            margins = np.append(margins, thrust_margin)
        return margins

    designs: list[dict[str, Any]] = []
    current_start = starting_design
    for payload_kg in PAYLOADS_KG:
        namespace["payload"] = payload_kg
        result = minimize(
            namespace["objective"],
            current_start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        if not result.success:
            raise RuntimeError(
                f"M1 {stage} optimization failed at payload {payload_kg:g} kg: "
                f"{result.message}"
            )

        performance = namespace["evaluate_design"](result.x)
        margins = np.asarray(constraints(result.x), dtype=float)
        if np.min(margins) < -1e-6:
            raise RuntimeError(
                f"M1 {stage} returned an infeasible design at {payload_kg:g} kg "
                f"(minimum constraint margin {np.min(margins):.6g})."
            )
        rotor_diameter, battery_wh, arm_diameter, wall, speed = result.x
        designs.append(
            {
                "stage": stage,
                "payload_kg": float(payload_kg),
                "radius_km": float(performance["radius"] / 1000.0),
                "rotor_diameter_m": float(rotor_diameter),
                "battery_wh": float(battery_wh),
                "arm_diameter_mm": float(arm_diameter),
                "wall_mm": float(wall),
                "speed_mps": float(speed),
                "loaded_mass_kg": float(performance["loaded_mass"]),
                "empty_mass_kg": float(performance["empty_mass"]),
                "thrust_per_rotor_n": float(performance["thrust"]),
                "figure_of_merit": float(figure_of_merit),
                "hub_plate_mass_kg": float(plate_mass_kg),
                "minimum_constraint_margin": float(np.min(margins)),
                "m3_thrust_floor_n": minimum_thrust_n,
            }
        )
        current_start = np.asarray(result.x, dtype=float).copy()

    csv_path = output_dir / f"{stage.lower()}_designs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(designs[0]))
        writer.writeheader()
        writer.writerows(designs)
    print(f"M1 {stage}: optimized {len(designs)} payload designs.")
    return designs


def run_m2(output_dir: Path) -> dict[str, Any]:
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = importlib.import_module("matplotlib.pyplot")

    plt.show = lambda: None
    module = load_module("m2_module", ROOT / "M2.py")
    with working_directory(output_dir):
        module.run_m2_hub_plate_optimization()
    plate_mass = (
        M2_VOLUME_FRACTION
        * M2_PLATE_SIDE_M**2
        * M2_PLATE_THICKNESS_M
        * M2_ALUMINUM_DENSITY_KG_M3
        + M2_FASTENER_MASS_KG
    )
    fractions = [float(value) for value in module.np.linspace(0.2, 0.5, 7)]
    sweep = [
        {
            "volume_fraction": fraction,
            "plate_mass_kg": fraction
            * M2_PLATE_SIDE_M**2
            * M2_PLATE_THICKNESS_M
            * M2_ALUMINUM_DENSITY_KG_M3
            + M2_FASTENER_MASS_KG,
            "compliance_nm": 45.2 / (fraction**1.5),
        }
        for fraction in fractions
    ]
    return {
        "plate_thickness_m": M2_PLATE_THICKNESS_M,
        "volume_fraction": M2_VOLUME_FRACTION,
        "plate_mass_kg": plate_mass,
        "sweep": sweep,
    }


def install_m3_surrogate_compatibility(m3_dir: Path) -> None:
    if (m3_dir / "rbf_surrogate.py").is_file():
        return

    import csv as csv_module
    import numpy as np
    from scipy.interpolate import RBFInterpolator

    module = ModuleType("rbf_surrogate")

    class RBFSurrogate:
        def __init__(
            self,
            x_mean: Any,
            x_scale: Any,
            x_train: Any,
            interpolator: Any,
        ) -> None:
            self.x_mean = x_mean
            self.x_scale = x_scale
            self.x_train = x_train
            self.interpolator = interpolator

        @classmethod
        def fit(cls, inputs: Any, response: Any) -> "RBFSurrogate":
            x = np.asarray(inputs, dtype=float)
            y = np.asarray(response, dtype=float)
            if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),):
                raise ValueError("M3 RBF training data must be N x 2 with N responses.")
            if len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("M3 RBF training data must have four finite points.")
            mean = np.mean(x, axis=0)
            scale = np.std(x, axis=0)
            scale[scale == 0.0] = 1.0
            standardized = (x - mean) / scale
            interpolator = RBFInterpolator(
                standardized,
                y,
                kernel="thin_plate_spline",
            )
            return cls(mean, scale, x.copy(), interpolator)

        def predict(self, points: Any) -> Any:
            x = np.asarray(points, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
                raise ValueError("M3 RBF predictions require finite two-column inputs.")
            return self.interpolator((x - self.x_mean) / self.x_scale)

    def load_dataset(path: Path) -> tuple[list[str], Any, Any]:
        with Path(path).open(newline="", encoding="utf-8-sig") as file:
            reader = csv_module.reader(file)
            header = next(reader, None)
            if header is None or len(header) != 3:
                raise ValueError("M3 data must have two input columns and one response.")
            rows = [[float(value) for value in row] for row in reader if row]
        data = np.asarray(rows, dtype=float)
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < 6:
            raise ValueError("M3 comparison requires at least six three-column rows.")
        return header, data[:, :2], data[:, 2]

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "The supplied M3 RBF helper file is absent; this plotting/report helper "
            "is not used by main.py."
        )

    def leave_one_out_predictions(inputs: Any, response: Any) -> Any:
        x = np.asarray(inputs, dtype=float)
        y = np.asarray(response, dtype=float)
        predictions = [
            RBFSurrogate.fit(np.delete(x, index, axis=0), np.delete(y, index)).predict(
                x[index]
            )[0]
            for index in range(len(y))
        ]
        return np.asarray(predictions, dtype=float)

    module.PredictiveModel = Any
    module.RBFSurrogate = RBFSurrogate
    module.load_dataset = load_dataset
    module.leave_one_out_predictions = leave_one_out_predictions
    module.print_cv_report = unavailable
    module.save_parity_plot = unavailable
    module.save_surrogate_plot = unavailable
    module.write_cv_table = unavailable
    module.write_surrogate_grid = unavailable
    module.__file__ = str(m3_dir / "rbf_surrogate.py")
    sys.modules["rbf_surrogate"] = module


def run_m3(output_dir: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import differential_evolution

    m3_dir = ROOT / "M3"
    sys.path.insert(0, str(m3_dir))
    install_m3_surrogate_compatibility(m3_dir)
    comparison = load_module("m3_compare", m3_dir / "compare.py")
    surrogate = sys.modules["rbf_surrogate"]
    header, inputs, actual = surrogate.load_dataset(m3_dir / "dataset_with_infill_2.csv")
    results, best = comparison.compare_surrogates(inputs, actual)
    bounds = [
        (float(np.min(inputs[:, index])), float(np.max(inputs[:, index])))
        for index in range(inputs.shape[1])
    ]
    optimization = differential_evolution(
        lambda point: -float(best.model.predict(np.asarray(point))[0]),
        bounds=bounds,
        seed=42,
        maxiter=1000,
        tol=1e-9,
        polish=True,
    )
    if not optimization.success:
        raise RuntimeError(f"M3 surrogate optimization failed: {optimization.message}")

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.write_comparison_table(
        output_dir / "loocv_comparison.csv", header, inputs, actual, results
    )
    comparison.print_comparison_table(header, inputs, actual, results, best)
    optimum = np.asarray(optimization.x, dtype=float)
    return {
        "input_headers": header[:2],
        "response_header": header[2],
        "models": [
            {
                "name": item.name,
                "rmse": item.rmse,
                "mae": item.mae,
                "r_squared": item.r_squared,
            }
            for item in results
        ],
        "selected_model": best.name,
        "selected_rmse": best.rmse,
        "rotor_diameter_m": float(optimum[0]),
        "thrust_n": float(optimum[1]),
        "figure_of_merit": float(best.model.predict(optimum)[0]),
        "optimization_success": bool(optimization.success),
    }


def make_m4_vehicles(m4: ModuleType, designs: list[dict[str, Any]]) -> list[Any]:
    return [
        m4.Vehicle(
            name=f"{design['stage']}_{design['payload_kg']:.2f}kg",
            payload_kg=design["payload_kg"],
            radius_km=design["radius_km"],
            diameter_m=design["rotor_diameter_m"],
            battery_wh=design["battery_wh"],
            arm_diameter_mm=design["arm_diameter_mm"],
            wall_mm=design["wall_mm"],
            speed_mps=design["speed_mps"],
            loaded_mass_kg=design["loaded_mass_kg"],
            empty_mass_kg=design["empty_mass_kg"],
        )
        for design in designs
    ]


def run_m4(
    output_dir: Path,
    data_dir: Path,
    m4: ModuleType,
    vehicles: list[Any],
) -> None:
    m4.build_vehicle_list = lambda: vehicles
    previous_argv = sys.argv
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(ROOT / "M4.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
    ]
    try:
        m4.main()
    finally:
        sys.argv = previous_argv


def load_hourly_profiles(data_dir: Path) -> dict[str, list[float]]:
    rows = read_csv(data_dir / "hourly_profile.csv")
    profiles: dict[str, list[float]] = {}
    for kind in ("emergency", "routine", "lab"):
        column = f"{kind}_frac"
        values = [float(row[column]) for row in rows]
        total = sum(values)
        if total <= 0:
            raise ValueError(f"No positive hourly fractions for {kind}.")
        profiles[kind] = [value / total for value in values]
    return profiles


def simulate_m5(
    m5: ModuleType,
    m4: ModuleType,
    vehicle: Any,
    hub_counts: dict[str, dict[str, int]],
    assignments: dict[str, tuple[str, float] | None],
    phcs: list[dict[str, str]],
    weather: list[dict[str, Any]],
    profiles: dict[str, list[float]],
    seed: int,
    reserve: float,
    days: int = 1,
) -> tuple[dict[str, Any], Any]:
    active_hubs = {
        hub: config
        for hub, config in hub_counts.items()
        if config["drones"] > 0
    }
    if not active_hubs:
        return {
            "S45": 0.0,
            "p95": 150.0,
            "road_share": 1.0,
            "flight_Wh": 0.0,
            "grid_Wh": 0.0,
            "generator_Wh": 0.0,
            "undelivered": 0,
            "orders": [],
            "log": [],
            "forced_road_count": 0,
        }, None

    def energy(order: Any, outbound_min: float, return_min: float) -> float:
        loaded_hover, loaded_cruise = m4.power_for_mass(vehicle.loaded_mass_kg, vehicle)
        empty_hover, empty_cruise = m4.power_for_mass(vehicle.empty_mass_kg, vehicle)
        if order.kind == "lab":
            cruise_wh = loaded_cruise * return_min + empty_cruise * outbound_min
        else:
            cruise_wh = loaded_cruise * outbound_min + empty_cruise * return_min
        return (cruise_wh * 60.0 + (loaded_hover + empty_hover) * 60.0) / 3600.0

    forced_road_ids: set[str] = set()

    class ProjectSimulator(m5.Simulator):
        def outage(self, minute: float) -> bool:
            weekday = (2 + int(minute // 1440)) % 7
            return weekday in (0, 2, 4) and 840 <= minute % 1440 < 1020

        def run(self, horizon: float) -> dict[str, Any]:
            super().run(horizon)
            pending = [
                order
                for order in self.orders
                if order.arrival < horizon and not math.isfinite(order.delivery)
            ]
            for order in pending:
                wait = 30.0 if order.kind == "emergency" else 0.0
                order.mode = "road"
                order.dispatch = order.arrival + wait
                order.delivery = order.dispatch + 120.0
                self.record(order.hub, "road fallback " + order.id)
            return self.metrics(horizon)

        def arrive(self, order: Any) -> None:
            if order.id in forced_road_ids:
                order.mode = "road"
                order.dispatch = self.t
                delay = 150 if order.kind == "emergency" else 120
                self.event(self.t + delay, 1, self.deliver, order)
                return
            super().arrive(order)

    def wind_at(hub: str, minute: float) -> tuple[float, float]:
        del hub
        record = weather[int(minute // 60) % len(weather)]
        return record["wind"], record["gust"]

    simulator = ProjectSimulator(
        active_hubs,
        wind_at,
        energy,
        reserve=reserve,
        speed=vehicle.speed_mps,
    )
    rng = random.Random(seed)
    order_counts = {"emergency": 0, "routine": 0, "lab": 0}
    for phc in phcs:
        phc_id = phc["phc_id"]
        assignment = assignments.get(phc_id)
        if assignment is None:
            hub_id = min(active_hubs)
            distance_km = 0.0
            forced = True
        else:
            hub_id, distance_km = assignment
            if hub_id not in active_hubs:
                hub_id = min(active_hubs)
                forced = True
            else:
                forced = False
        for kind in ("emergency", "routine", "lab"):
            daily_orders = float(phc[f"{kind}_per_day"])
            for hour, fraction in enumerate(profiles[kind]):
                rate_per_minute = daily_orders * fraction / 60.0
                if rate_per_minute <= 0:
                    continue
                minute = rng.expovariate(rate_per_minute)
                while minute < 60.0:
                    order_id = f"{phc_id}-{kind}-{order_counts[kind]}"
                    order_counts[kind] += 1
                    if forced:
                        forced_road_ids.add(order_id)
                    simulator.add(
                        m5.Order(
                            order_id,
                            hub_id,
                            kind,
                            hour * 60.0 + minute,
                            distance_km,
                        )
                    )
                    minute += rng.expovariate(rate_per_minute)

    horizon = (days + 7) * 1440
    metrics = simulator.run(horizon)
    result = {
        "S45": float(metrics["S45"] or 0.0) * 100.0,
        "p95": float(metrics["p95"] or 150.0),
        "road_share": float(metrics["road_share"] or 0.0),
        "flight_Wh": float(metrics["flight_Wh"]),
        "grid_Wh": float(metrics["grid_Wh"]),
        "generator_Wh": float(metrics["generator_Wh"]),
        "undelivered": int(metrics["undelivered"]),
        "orders": simulator.orders,
        "log": simulator.log,
        "forced_road_count": len(forced_road_ids),
        "drain_road_fallbacks": sum(
            order.mode == "road" and order.id not in forced_road_ids
            for order in simulator.orders
        ),
    }
    return result, simulator


def assignments_from_m4(
    output_dir: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[str, float] | None]]:
    hubs = {
        row["hub_id"]: {
            "drones": int(row["drones"]),
            "bays": int(row["charging_bays"]),
            "batteries": max(1, 2 * int(row["drones"])),
        }
        for row in read_csv(output_dir / "hub_plan.csv")
        if row["open"] == "1"
    }
    assignments: dict[str, tuple[str, float] | None] = {}
    for row in read_csv(output_dir / "phc_assignments.csv"):
        if row["hub_id"] == "ROAD_ONLY":
            assignments[row["phc_id"]] = None
        else:
            assignments[row["phc_id"]] = (row["hub_id"], float(row["route_km"]))
    return hubs, assignments


def run_m5_baseline(
    output_dir: Path,
    m4: ModuleType,
    m5: ModuleType,
    vehicle: Any,
    data_dir: Path,
) -> dict[str, Any]:
    hubs, assignments = assignments_from_m4(output_dir.parent / "M4")
    phcs = [
        row
        for row in read_csv(data_dir / "sites.csv")
        if row["type"] == "phc"
    ]
    demand = {row["phc_id"]: row for row in read_csv(data_dir / "demand.csv")}
    phcs = [{**site, **demand[site["site_id"]]} for site in phcs]
    weather, _, _ = m4.read_weather(data_dir, 2025)
    result, simulator = simulate_m5(
        m5,
        m4,
        vehicle,
        hubs,
        assignments,
        phcs,
        weather,
        load_hourly_profiles(data_dir),
        seed=6002,
        reserve=0.33,
    )
    if result["undelivered"]:
        raise RuntimeError(f"M5 left {result['undelivered']} orders undelivered.")
    summary = {
        "vehicle": vehicle.name,
        "simulation_days_with_arrivals": 1,
        "orders": len(result["orders"]),
        "emergencies": sum(order.kind == "emergency" for order in result["orders"]),
        "S45_percent": result["S45"],
        "p95_emergency_minutes": result["p95"],
        "road_fallback_share": result["road_share"],
        "flight_energy_Wh": result["flight_Wh"],
        "grid_energy_Wh": result["grid_Wh"],
        "generator_energy_Wh": result["generator_Wh"],
        "forced_road_orders": result["forced_road_count"],
        "drain_road_fallbacks": result["drain_road_fallbacks"],
        "undelivered_orders": result["undelivered"],
        "energy_model": "M4 loaded/empty cruise and hover power with M5 dispatch and charging",
    }
    write_json(output_dir / "m5_simulation_summary.json", summary)
    print("M5 simulation output:")
    print(json.dumps(summary, indent=2))
    if simulator is not None:
        log_path = output_dir / "m5_event_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("minute", "hub", "event"))
            writer.writerows(simulator.log)
    return summary


def run_m6(
    output_dir: Path,
    data_dir: Path,
    m4: ModuleType,
    m5: ModuleType,
    m6: ModuleType,
    vehicles: list[Any],
    pop_size: int,
    generations: int,
    replications: int,
) -> str:
    if pop_size < 2 or generations < 1 or replications < 1:
        raise ValueError("M6 requires population >= 2, generations >= 1, and replications >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    m6.build_vehicle_list = lambda: vehicles
    profiles = load_hourly_profiles(data_dir)
    def run_actual_m5(
        decision: Any,
        vehicle: Any,
        lanes: dict[tuple[str, str], Any],
        phcs: list[dict[str, str]],
        _weather: list[dict[str, Any]],
        policy_costs: dict[str, Any],
        crn_seed: int,
    ) -> Any:
        open_hubs = [
            hub_id
            for hub_id, drones in decision.drones_by_hub.items()
            if drones > 0
        ]
        hub_counts = {
            hub_id: {
                "drones": int(decision.drones_by_hub[hub_id]),
                "bays": int(decision.bays_by_hub.get(hub_id, 0)),
                "batteries": int(decision.drones_by_hub[hub_id])
                + max(0, int(decision.spare_batteries_by_hub.get(hub_id, 0))),
            }
            for hub_id in open_hubs
            if decision.bays_by_hub.get(hub_id, 0) > 0
        }
        assignments: dict[str, tuple[str, float] | None] = {}
        for phc in phcs:
            phc_id = phc["phc_id"]
            feasible = [
                (hub_id, lanes[(hub_id, phc_id)].distance_km)
                for hub_id in open_hubs
                if (hub_id, phc_id) in lanes
            ]
            assignments[phc_id] = (
                min(feasible, key=lambda item: item[1]) if feasible else None
            )
        sim_result, simulator = simulate_m5(
            m5,
            m4,
            vehicle,
            hub_counts,
            assignments,
            phcs,
            _weather,
            profiles,
            seed=crn_seed,
            reserve=decision.reserve_fraction,
        )
        if simulator is None:
            daily_road_cost = sum(
                float(phc[f"{kind}_per_day"])
                * float(policy_costs["road_fallback"]["cost_per_trip"][kind])
                for phc in phcs
                for kind in ("emergency", "routine", "lab")
            )
            annual_cost = m6.compute_policy_annual_cost(
                decision,
                vehicle,
                open_hubs,
                policy_costs,
                {"daily_road_cost_inr": daily_road_cost},
            )
            return m6.SimulationOutcome(
                annual_cost_inr=float(annual_cost),
                s45_percent=0.0,
                road_fallback_share=1.0,
                p95_emergency_min=150.0,
            )
        daily_road_cost = sum(
            sum(
                order.mode == "road" and order.kind == kind
                for order in sim_result["orders"]
            )
            * float(policy_costs["road_fallback"]["cost_per_trip"][kind])
            for kind in ("emergency", "routine", "lab")
        )
        energy = policy_costs["energy"]
        sim_metrics = {
            "daily_grid_cost_inr": sim_result["grid_Wh"]
            / 1000.0
            * float(energy["grid_per_kWh"]),
            "daily_genset_cost_inr": sim_result["generator_Wh"]
            / 1000.0
            * float(energy["genset_per_kWh"]),
            "daily_battery_wear_cost_inr": sim_result["flight_Wh"]
            * float(policy_costs["battery_pack"]["cost_per_Wh"])
            / float(policy_costs["battery_pack"]["cycle_life"]),
            "daily_road_cost_inr": daily_road_cost,
        }
        annual_cost = m6.compute_policy_annual_cost(
            decision, vehicle, open_hubs, policy_costs, sim_metrics
        )
        return m6.SimulationOutcome(
            annual_cost_inr=float(annual_cost),
            s45_percent=sim_result["S45"],
            road_fallback_share=sim_result["road_share"],
            p95_emergency_min=sim_result["p95"],
        )

    m6.run_m5_simulation = run_actual_m5
    previous_argv = sys.argv
    sys.argv = [
        str(ROOT / "M6.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--year",
        "2025",
        "--crn-replications",
        str(replications),
        "--pop-size",
        str(pop_size),
        "--generations",
        str(generations),
    ]
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            m6.main()
    finally:
        sys.argv = previous_argv
    output = captured.getvalue()
    print(output, end="")
    return output


def run_workflow(args: argparse.Namespace) -> None:
    import importlib
    from documentation import ReportData, write_reports

    use_project_environment()
    output_dir = args.output_dir.resolve()
    reports_dir = output_dir / "reports"
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Project input directory does not exist: {data_dir}")
    if not (data_dir / "sites.csv").is_file():
        raise FileNotFoundError(f"Expected project inputs under {data_dir}")

    m1_dir = output_dir / "M1"
    base_designs = run_m1_stage("M1", m1_dir, plate_mass_kg=0.25)
    m2_dir = output_dir / "M2"
    m2 = run_m2(m2_dir)
    m2_designs = run_m1_stage(
        "M2",
        output_dir / "M1_M2",
        plate_mass_kg=m2["plate_mass_kg"],
    )
    all_designs = base_designs + m2_designs
    m3_dir = output_dir / "M3"
    m3 = run_m3(m3_dir)
    m3_designs = run_m1_stage(
        "M3",
        output_dir / "M1_M3",
        plate_mass_kg=m2["plate_mass_kg"],
        figure_of_merit=m3["figure_of_merit"],
        fixed_rotor_diameter_m=m3["rotor_diameter_m"],
        minimum_thrust_n=m3["thrust_n"],
    )
    all_designs.extend(m3_designs)
    m4_site_packages = ROOT / "m4_opti" / "Lib" / "site-packages"
    if importlib.util.find_spec("shapely") is None and m4_site_packages.is_dir():
        sys.path.append(str(m4_site_packages))
    m4 = load_module("m4", ROOT / "M4.py")
    m5_dir = ROOT / "M5"
    if str(m5_dir) not in sys.path:
        sys.path.insert(0, str(m5_dir))
    m5_module = importlib.import_module("M5_integrated")
    m6 = load_module("m6_module", ROOT / "M6.py")
    vehicles = make_m4_vehicles(m4, all_designs)

    m4_dir = output_dir / "M4"
    run_m4(m4_dir, data_dir, m4, vehicles)
    m4_cost_rows = read_csv(m4_dir / "design_costs.csv")
    selected_cost = next(row for row in m4_cost_rows if row["selected"] == "True")
    selected_design = next(
        design
        for design in all_designs
        if f"{design['stage']}_{design['payload_kg']:.2f}kg" == selected_cost["vehicle"]
    )
    selected_vehicle = next(
        vehicle for vehicle in vehicles if vehicle.name == selected_cost["vehicle"]
    )
    hub_rows = [row for row in read_csv(m4_dir / "hub_plan.csv") if row["open"] == "1"]
    assignment_rows = read_csv(m4_dir / "phc_assignments.csv")

    m5_dir = output_dir / "M5"
    m5_summary = run_m5_baseline(
        m5_dir, m4, m5_module, selected_vehicle, data_dir
    )

    m6_dir = output_dir / "M6"
    m6_output = run_m6(
        m6_dir,
        data_dir,
        m4,
        m5_module,
        m6,
        vehicles,
        args.m6_pop_size,
        args.m6_generations,
        args.m6_replications,
    )
    front = read_csv(m6_dir / "m6_pareto_front.csv")
    verification = read_csv(m6_dir / "m6_weighted_sum_verification.csv")
    knee = next((row for row in front if row["is_knee"] == "True"), None)
    if knee is None and front:
        knee = min(front, key=lambda row: float(row["annual_cost_inr"]))
    if knee is None:
        raise RuntimeError("M6 did not produce a Pareto-front candidate.")
    target_candidates = [
        row for row in front if float(row["s45_percent"]) >= 100.0 - 1e-9
    ]
    if target_candidates:
        recommended = min(
            target_candidates,
            key=lambda row: (
                float(row["annual_cost_inr"]),
                float(row["road_fallback_share"]),
                float(row["p95_emergency_min"]),
            ),
        )
        recommendation_reason = (
            "lowest annual cost among Pareto candidates meeting the 100% S45 target"
        )
    else:
        recommended = min(
            front,
            key=lambda row: (
                -float(row["s45_percent"]),
                float(row["annual_cost_inr"]),
                float(row["road_fallback_share"]),
                float(row["p95_emergency_min"]),
            ),
        )
        recommendation_reason = (
            "highest S45 on the sampled Pareto front, then lowest annual cost; "
            "no sampled candidate meets the 100% S45 target"
        )
    recommendation_meets_target = (
        float(recommended["s45_percent"]) >= 100.0 - 1e-9
    )
    weighted_ok = all(row["on_or_ahead_of_nsga2_front"] == "True" for row in verification)
    write_reports(
        ReportData(
            reports_dir=reports_dir,
            m4_dir=m4_dir,
            m5_dir=m5_dir,
            m6_dir=m6_dir,
            base_designs=base_designs,
            m2_designs=m2_designs,
            m3_designs=m3_designs,
            all_designs=all_designs,
            m2=m2,
            m3=m3,
            selected_cost=selected_cost,
            selected_design=selected_design,
            hub_rows=hub_rows,
            assignment_rows=assignment_rows,
            m5_summary=m5_summary,
            m6_output=m6_output,
            front=front,
            knee=knee,
            recommended=recommended,
            recommendation_reason=recommendation_reason,
            recommendation_meets_target=recommendation_meets_target,
            weighted_ok=weighted_ok,
            m6_pop_size=args.m6_pop_size,
            m6_generations=args.m6_generations,
            m6_replications=args.m6_replications,
        )
    )
    print(f"Reports written to: {reports_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1 through M6 and write module and final optimization reports."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "optimization_outputs")
    parser.add_argument("--m6-pop-size", type=int, default=10)
    parser.add_argument("--m6-generations", type=int, default=3)
    parser.add_argument("--m6-replications", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run_workflow(parse_args())


if __name__ == "__main__":
    main()