"""SkyLink M4 hub-network and fleet optimizer.

Run from any directory with:
	python m4.py [--data-dir PATH] [--output-dir PATH]

Formulation card
----------------
Binary y[j] opens candidate hub j; binary x[i,j] assigns PHC i to that hub;
binary road[i] selects all-road service for PHC i. Integer n[j], b[j], and
o[j] are drones, charging bays, and staffed operator positions per shift.
Exactly one hub or all-road option serves each PHC. A lane is offered only
when its no-fly-zone detour fits within 90% of the selected M1 still-air
mission radius. Flight weather is handled in the expected annual costs and
workload (gust-grounded and late emergency orders use road fallback).

Minimize annualized hub, drone, bay, operator, energy, battery replacement,
and road costs. For each hub, assigned expected drone busy minutes must be
at most 70% of available fleet minutes; charging demand must fit bay
throughput. The 90% route factor is an explicit additional operational
margin; the M1 radius itself already includes its 15% energy reserve.

Data limitations are surfaced in the output: hourly fractions are normalized
to daily totals (the routine fractions sum to 1.1993), order payload masses
are absent (each M1 payload-radius point is treated as the representative
loaded mission), wind direction convention is unspecified (wind magnitude is
treated as a headwind in both directions), and outage weekdays use 2025 as
the calendar for the yearless synthetic wind record.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from shapely.geometry import LineString, Point, shape


CLASSES = ("emergency", "routine", "lab")
CLASS_COLUMNS = {
	"emergency": "emergency_per_day",
	"routine": "routine_per_day",
	"lab": "lab_per_day",
}
PROFILE_COLUMNS = {
	"emergency": "emergency_frac",
	"routine": "routine_frac",
	"lab": "lab_frac",
}
HUB_COLORS = ["#2563eb", "#e11d48", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#a16207", "#475569"]


@dataclass(frozen=True)
class Vehicle:
	name: str
	payload_kg: float
	radius_km: float
	diameter_m: float
	battery_wh: float
	arm_diameter_mm: float
	wall_mm: float
	speed_mps: float
	loaded_mass_kg: float
	empty_mass_kg: float


@dataclass
class Lane:
	phc_id: str
	hub_id: str
	distance_km: float
	route_xy: list[tuple[float, float]]
	busy_min: float
	energy_wh: float
	road_cost_per_day: float
	electricity_cost_per_day: float
	battery_cost_per_day: float
	fallback_fraction: dict[str, float]


@dataclass
class ModelResult:
	objective: float
	values: np.ndarray
	variable_index: dict[tuple[Any, ...], int]
	constraints: LinearConstraint
	lower: np.ndarray
	upper: np.ndarray
	integrality: np.ndarray
	lanes: dict[tuple[str, str], Lane]
	hub_ids: list[str]
	phc_ids: list[str]
	status: str


def load_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline="", encoding="utf-8-sig") as handle:
		return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def crf(rate: float, years: int) -> float:
	growth = (1.0 + rate) ** years
	return rate * growth / (growth - 1.0)


def load_no_fly(path: Path) -> list[Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	return [shape(feature["geometry"]) for feature in payload["features"]]


def segment_is_clear(segment: LineString, obstacle_interiors: list[Any]) -> bool:
	for interior in obstacle_interiors:
		if not interior.is_empty and segment.intersects(interior):
			return False
	return True


def shortest_route(
	start: tuple[float, float], end: tuple[float, float], obstacles: list[Any]
) -> tuple[float, list[tuple[float, float]]]:
	"""Visibility-graph shortest path around polygonized no-fly zones."""
	if start == end:
		return 0.0, [start]
	vertices: list[tuple[float, float]] = [start, end]
	for obstacle in obstacles:
		vertices.extend((float(x), float(y)) for x, y in list(obstacle.exterior.coords)[:-1])
	# Keep node count small if the circle has many closely spaced vertices.
	vertices = list(dict.fromkeys(vertices))
	# Buffering is geometry work; do it once per route, not once per graph edge.
	obstacle_interiors = [obstacle.buffer(-1e-5) for obstacle in obstacles]
	adjacency: list[list[tuple[int, float]]] = [[] for _ in vertices]
	for i, a in enumerate(vertices):
		for j in range(i + 1, len(vertices)):
			b = vertices[j]
			edge = LineString([a, b])
			if segment_is_clear(edge, obstacle_interiors):
				distance = math.dist(a, b)
				adjacency[i].append((j, distance))
				adjacency[j].append((i, distance))

	# Dijkstra; the graph is small (typically fewer than 120 vertices).
	distances = [math.inf] * len(vertices)
	previous = [-1] * len(vertices)
	visited = [False] * len(vertices)
	distances[0] = 0.0
	for _ in vertices:
		current = min((i for i in range(len(vertices)) if not visited[i]), key=distances.__getitem__)
		if distances[current] == math.inf or current == 1:
			break
		visited[current] = True
		for neighbor, length in adjacency[current]:
			candidate = distances[current] + length
			if candidate < distances[neighbor]:
				distances[neighbor] = candidate
				previous[neighbor] = current
	if not math.isfinite(distances[1]):
		raise ValueError(f"No legal route found from {start} to {end}.")
	path: list[tuple[float, float]] = []
	node = 1
	while node != -1:
		path.append(vertices[node])
		node = previous[node]
	path.reverse()
	return distances[1], path


def read_weather(data_dir: Path, year: int) -> tuple[list[dict[str, Any]], float, float]:
	records: list[dict[str, Any]] = []
	outage_days = {"Mon": 0, "Wed": 2, "Fri": 4}
	outage_hours = 0
	grounded_hours = 0
	for row in load_csv(data_dir / "wind_days.csv"):
		# The supplied `day` field is ordinal day-of-year (1..365), not day
		# of month; e.g. month=2 has day=32..59. Confirm against month and
		# retain a fallback for conventional month/day records.
		day_value, month_value = int(row["day"]), int(row["month"])
		if day_value > 31 or (day_value <= 31 and day_value > 28 and month_value == 2):
			date = dt.date(year, 1, 1) + dt.timedelta(days=day_value - 1)
			if date.month != month_value:
				raise ValueError(f"Wind date fields disagree at month={month_value}, day={day_value}.")
		else:
			date = dt.date(year, month_value, day_value)
		hour = int(row["hour"])
		gust = float(row["gust_ms"])
		wind = float(row["mean_wind_ms"])
		outage = date.weekday() in outage_days.values() and 14 <= hour < 17
		grounded = gust >= 14.0
		outage_hours += int(outage)
		grounded_hours += int(grounded)
		records.append({"hour": hour, "wind": wind, "gust": gust, "outage": outage, "grounded": grounded})
	if not records:
		raise ValueError("wind_days.csv contains no rows")
	return records, grounded_hours / len(records), outage_hours / len(records)


def build_vehicle_list() -> list[Vehicle]:
	# Exact saved M1 payload-radius trade-off from M1 (1).ipynb, output cell.
	values = [
		(0.50, 33.2365, 2.2764, 18.3582, 17.8876, 17.3451),
		(1.00, 32.6130, 2.3306, 18.5036, 18.4523, 17.3671),
		(1.50, 32.0020, 2.3853, 18.6496, 19.0170, 17.3893),
		(2.00, 31.4037, 2.4406, 18.7960, 19.5820, 17.4116),
		(2.50, 30.8180, 2.4964, 18.9429, 20.1471, 17.4342),
	]
	return [
		Vehicle(
			name=f"M1_{payload:.2f}kg",
			payload_kg=payload,
			radius_km=radius,
			diameter_m=0.8,
			battery_wh=2000.0,
			arm_diameter_mm=40.0,
			wall_mm=wall,
			speed_mps=speed,
			loaded_mass_kg=loaded,
			empty_mass_kg=empty,
		)
		for payload, radius, wall, speed, loaded, empty in values
	]


def power_for_mass(mass_kg: float, vehicle: Vehicle) -> tuple[float, float]:
	rho, gravity, rotors = 1.225, 9.81, 4
	eta, fm, p_av, cda, tip_speed = 0.85, 0.65, 30.0, 0.08, 110.0
	weight = mass_kg * gravity
	area = math.pi * vehicle.diameter_m**2 / 4.0
	hover_velocity = math.sqrt(weight / (2 * rho * rotors * area))
	speed = vehicle.speed_mps
	forward_induced = math.sqrt(-speed**2 / 2 + math.sqrt(speed**4 / 4 + hover_velocity**4))
	hover_power = weight * hover_velocity / (fm * eta) + p_av
	advance_ratio = speed / tip_speed
	cruise_power = (
		weight * forward_induced
		+ weight * hover_velocity * (math.sqrt(1 + 4.65 * advance_ratio**2) - 1)
		+ 0.5 * rho * cda * speed**3
	) / (fm * eta) + p_av
	return hover_power, cruise_power


def make_lane(
	phc: dict[str, str],
	hub: dict[str, str],
	route_km: float,
	route_xy: list[tuple[float, float]],
	vehicle: Vehicle,
	profiles: dict[str, list[float]],
	weather: list[dict[str, Any]],
	costs: dict[str, Any],
	reserve_fraction: float,
) -> Lane | None:
	if route_km > vehicle.radius_km * reserve_fraction:
		return None
	if route_km <= 1e-9:
		return Lane(
			phc_id=phc["phc_id"],
			hub_id=hub["site_id"],
			distance_km=0.0,
			route_xy=route_xy,
			busy_min=0.0,
			energy_wh=0.0,
			road_cost_per_day=0.0,
			electricity_cost_per_day=0.0,
			battery_cost_per_day=0.0,
			fallback_fraction={kind: 0.0 for kind in CLASSES},
		)

	loaded_hover, loaded_cruise = power_for_mass(vehicle.loaded_mass_kg, vehicle)
	empty_hover, empty_cruise = power_for_mass(vehicle.empty_mass_kg, vehicle)
	class_busy: dict[str, float] = {}
	class_energy: dict[str, float] = {}
	fallback: dict[str, float] = {}
	annual_road_cost_day = 0.0
	electricity_cost_day = 0.0
	battery_cost_day = 0.0
	route_m = route_km * 1000.0
	speed = vehicle.speed_mps
	ops = costs["operations"]
	emergency_limit_min = float(ops["emergency_target_min"])
	road_prices = costs["road_fallback"]["cost_per_trip"]

	for kind in CLASSES:
		busy_per_order = 0.0
		energy_per_order_wh = 0.0
		fallback_probability = 0.0
		for item in weather:
			probability = profiles[kind][item["hour"]] / (len(weather) / 24.0)
			if probability <= 0:
				continue
			cannot_fly = item["grounded"] or item["wind"] >= speed
			if cannot_fly:
				fallback_probability += probability
				continue
			wind = item["wind"]  # conservative headwind magnitude both ways
			effective_speed = (speed**2 - wind**2) / speed
			one_way_min = route_m / effective_speed / 60.0
			round_trip_min = 2.0 * one_way_min
			emergency_arrival_min = float(ops["load_and_takeoff_min"]) + one_way_min + float(ops["drop_or_pickup_min"])
			if kind == "emergency" and emergency_arrival_min > emergency_limit_min:
				fallback_probability += probability
				continue
			busy = round_trip_min + float(ops["load_and_takeoff_min"]) + 2.0 * float(ops["drop_or_pickup_min"]) + float(ops["battery_swap_turnaround_min"])
			energy_wh = (
				(loaded_cruise + empty_cruise) * one_way_min * 60.0
				+ (loaded_hover + empty_hover) * 60.0
			) / 3600.0
			busy_per_order += probability * busy
			energy_per_order_wh += probability * energy_wh
			if item["outage"]:
				tariff = float(costs["energy"]["genset_per_kWh"])
			else:
				tariff = float(costs["energy"]["grid_per_kWh"])
			supplied_wh = energy_wh / float(costs["charging_bay"]["charge_efficiency"])
			electricity_cost_day += probability * supplied_wh / 1000.0 * tariff * float(phc[CLASS_COLUMNS[kind]])

		# Numerical normalization: tiny floating error should not exceed 1.
		fallback_probability = min(1.0, max(0.0, fallback_probability))
		class_busy[kind] = busy_per_order * float(phc[CLASS_COLUMNS[kind]])
		class_energy[kind] = energy_per_order_wh * float(phc[CLASS_COLUMNS[kind]])
		fallback[kind] = fallback_probability
		annual_road_cost_day += (
			float(phc[CLASS_COLUMNS[kind]]) * fallback_probability * float(road_prices[kind])
		)
		battery_cost_day += class_energy[kind] * float(costs["battery_pack"]["cost_per_Wh"]) / float(costs["battery_pack"]["cycle_life"])

	# Costs are per day; energy and battery values above include per-order demand.
	return Lane(
		phc_id=phc["phc_id"],
		hub_id=hub["site_id"],
		distance_km=route_km,
		route_xy=route_xy,
		busy_min=sum(class_busy.values()),
		energy_wh=sum(class_energy.values()),
		road_cost_per_day=annual_road_cost_day,
		electricity_cost_per_day=electricity_cost_day,
		battery_cost_per_day=battery_cost_day,
		fallback_fraction=fallback,
	)


def build_routes(
	phcs: list[dict[str, str]],
	hubs: list[dict[str, str]],
	obstacles: list[Any],
	vehicles: list[Vehicle],
	profiles: dict[str, list[float]],
	weather: list[dict[str, Any]],
	costs: dict[str, Any],
	reserve_fraction: float,
) -> tuple[dict[str, dict[tuple[str, str], Lane]], dict[tuple[str, str], tuple[float, list[tuple[float, float]]]]]:
	geometries: dict[tuple[str, str], tuple[float, list[tuple[float, float]]]] = {}
	total_routes = len(hubs) * len(phcs)
	print(f"M4 route generation: 0/{total_routes}", flush=True)
	completed_routes = 0
	for hub in hubs:
		for phc in phcs:
			start = (float(hub["x_km"]), float(hub["y_km"]))
			end = (float(phc["x_km"]), float(phc["y_km"]))
			geometries[(hub["site_id"], phc["phc_id"])] = shortest_route(start, end, obstacles)
			completed_routes += 1
			if completed_routes % max(1, total_routes // 20) == 0 or completed_routes == total_routes:
				print(f"M4 route generation: {completed_routes}/{total_routes}", flush=True)
	by_vehicle: dict[str, dict[tuple[str, str], Lane]] = {}
	for vehicle in vehicles:
		lanes: dict[tuple[str, str], Lane] = {}
		for hub in hubs:
			for phc in phcs:
				route_km, path = geometries[(hub["site_id"], phc["phc_id"])]
				lane = make_lane(phc, hub, route_km, path, vehicle, profiles, weather, costs, reserve_fraction)
				if lane is not None:
					lanes[(hub["site_id"], phc["phc_id"])] = lane
		by_vehicle[vehicle.name] = lanes
	return by_vehicle, geometries


def annual_cost_parameters(costs: dict[str, Any], vehicle: Vehicle) -> dict[str, float]:
	rate = float(costs["discount_rate"])
	drone_capex = float(costs["drone"]["capex_fixed"]) + float(costs["drone"]["capex_per_kg_mtow"]) * vehicle.loaded_mass_kg
	return {
		"hub": float(costs["hub"]["capex"]) * crf(rate, int(costs["hub"]["life_years"])) + float(costs["hub"]["opex_per_year"]),
		"drone": drone_capex * (crf(rate, int(costs["drone"]["life_years"])) + float(costs["drone"]["maintenance_frac_of_capex_per_year"])),
		"bay": float(costs["charging_bay"]["capex"]) * crf(rate, int(costs["charging_bay"]["life_years"])),
		"operator": float(costs["staff"]["salary_per_person_year"]) * float(costs["staff"]["shifts_per_day"]) * float(costs["staff"]["relief_factor"]),
	}


def build_and_solve(
	phcs: list[dict[str, str]],
	hubs: list[dict[str, str]],
	lanes: dict[tuple[str, str], Lane],
	vehicle: Vehicle,
	costs: dict[str, Any],
	grounded_fraction: float,
	relax: bool = False,
) -> ModelResult:
	phc_ids = [row["phc_id"] for row in phcs]
	hub_ids = [row["site_id"] for row in hubs]
	hub_costs = annual_cost_parameters(costs, vehicle)
	index: dict[tuple[Any, ...], int] = {}
	objective: list[float] = []
	lower: list[float] = []
	upper: list[float] = []
	integer: list[int] = []

	def add(key: tuple[Any, ...], obj: float, lo: float, hi: float, is_int: int) -> int:
		idx = len(objective)
		index[key] = idx
		objective.append(obj)
		lower.append(lo)
		upper.append(hi)
		integer.append(is_int)
		return idx

	for hub_id in hub_ids:
		add(("y", hub_id), hub_costs["hub"], 0, 1, 1)
	for phc_id in phc_ids:
		for hub_id in hub_ids:
			lane = lanes.get((hub_id, phc_id))
			if lane is not None:
				variable_cost = 365.0 * (lane.road_cost_per_day + lane.electricity_cost_per_day + lane.battery_cost_per_day)
				add(("x", phc_id, hub_id), variable_cost, 0, 1, 1)
		road_daily = sum(
			float(next(row for row in phcs if row["phc_id"] == phc_id)[CLASS_COLUMNS[k]]) * float(costs["road_fallback"]["cost_per_trip"][k])
			for k in CLASSES
		)
		has_feasible_hub = any((hub_id, phc_id) in lanes for hub_id in hub_ids)
		# Do not let cost alone reject a range-feasible hub assignment. Road-only
		# is available only if no candidate site can reach this PHC with reserve.
		add(("road", phc_id), 365.0 * road_daily, 0, 0 if has_feasible_hub else 1, 1)

	total_busy = sum(lane.busy_min for lane in lanes.values())
	# This may overcount PHCs across hubs, making the implied M conservative.
	per_drone_minutes = max(1e-9, 0.70 * (1.0 - grounded_fraction) * 1440.0)
	max_drones = max(1, math.ceil(total_busy / per_drone_minutes) + 1)
	total_energy = sum(lane.energy_wh for lane in lanes.values())
	bay_daily_capacity = float(costs["charging_bay"]["power_kW"]) * 1000.0 * 24.0 * float(costs["charging_bay"]["charge_efficiency"])
	max_bays = max(1, math.ceil(total_energy / max(bay_daily_capacity, 1e-9)) + 1)
	max_operators = max(1, math.ceil(max_drones / float(costs["staff"]["drones_per_operator"])))
	for hub_id in hub_ids:
		add(("n", hub_id), hub_costs["drone"], 0, max_drones, 1)
		add(("b", hub_id), hub_costs["bay"], 0, max_bays, 1)
		add(("o", hub_id), hub_costs["operator"], 0, max_operators, 1)

	rows: list[dict[int, float]] = []
	row_lower: list[float] = []
	row_upper: list[float] = []

	def constraint(coeffs: dict[int, float], lo: float = -math.inf, hi: float = math.inf) -> None:
		rows.append(coeffs)
		row_lower.append(lo)
		row_upper.append(hi)

	# Each PHC selects one feasible hub or remains fully on road.
	for phc_id in phc_ids:
		coefficients = {index[("road", phc_id)]: 1.0}
		for hub_id in hub_ids:
			key = ("x", phc_id, hub_id)
			if key in index:
				coefficients[index[key]] = 1.0
		constraint(coefficients, 1.0, 1.0)

	for hub_id in hub_ids:
		y = index[("y", hub_id)]
		n = index[("n", hub_id)]
		b = index[("b", hub_id)]
		o = index[("o", hub_id)]
		for phc_id in phc_ids:
			assignment_key = ("x", phc_id, hub_id)
			if assignment_key in index:
				constraint({index[assignment_key]: 1.0, y: -1.0}, hi=0.0)
		constraint({n: 1.0, y: -float(max_drones)}, hi=0.0)
		constraint({b: 1.0, y: -float(max_bays)}, hi=0.0)
		constraint({o: 1.0, y: -float(max_operators)}, hi=0.0)
		constraint({o: 1.0, y: -1.0}, lo=0.0)
		constraint({n: 1.0, o: -float(costs["staff"]["drones_per_operator"])}, hi=0.0)
		busy_coefficients = {n: -per_drone_minutes}
		energy_coefficients = {b: -bay_daily_capacity}
		for phc_id in phc_ids:
			lane = lanes.get((hub_id, phc_id))
			key = ("x", phc_id, hub_id)
			if lane is not None and key in index:
				busy_coefficients[index[key]] = lane.busy_min
				# Compare battery-side demand with the bay's efficiency-adjusted
				# output capacity: 1.5 kW * 24 h * charge efficiency.
				energy_coefficients[index[key]] = lane.energy_wh
		constraint(busy_coefficients, hi=0.0)
		constraint(energy_coefficients, hi=0.0)

	matrix = lil_matrix((len(rows), len(objective)), dtype=float)
	for row_idx, coefficients in enumerate(rows):
		for col_idx, value in coefficients.items():
			matrix[row_idx, col_idx] = value
	model_constraint = LinearConstraint(matrix.tocsr(), np.asarray(row_lower), np.asarray(row_upper))
	integrality = np.zeros(len(integer), dtype=int) if relax else np.asarray(integer, dtype=int)
	result = milp(
		c=np.asarray(objective),
		integrality=integrality,
		bounds=Bounds(np.asarray(lower), np.asarray(upper)),
		constraints=model_constraint,
		options={"time_limit": 300.0, "mip_rel_gap": 1e-8},
	)
	if result.x is None:
		raise RuntimeError(f"MILP did not produce a solution (status={result.message})")
	return ModelResult(
		objective=float(result.fun),
		values=result.x,
		variable_index=index,
		constraints=model_constraint,
		lower=np.asarray(lower),
		upper=np.asarray(upper),
		integrality=integrality,
		lanes=lanes,
		hub_ids=hub_ids,
		phc_ids=phc_ids,
		status=str(result.message),
	)


def value(result: ModelResult, key: tuple[Any, ...]) -> float:
	idx = result.variable_index.get(key)
	return float(result.values[idx]) if idx is not None else 0.0


def brute_force_subinstance(
	phcs: list[dict[str, str]],
	hubs: list[dict[str, str]],
	lanes: dict[tuple[str, str], Lane],
	vehicle: Vehicle,
	costs: dict[str, Any],
	grounded_fraction: float,
) -> tuple[float, int]:
	"""Exact enumeration over all 4-hub/8-PHC assignments, including road."""
	hub_cost = annual_cost_parameters(costs, vehicle)
	choices: list[list[str | None]] = []
	for phc in phcs:
		options = [hub["site_id"] for hub in hubs if (hub["site_id"], phc["phc_id"]) in lanes]
		if not options:
			options.append(None)
		choices.append(options)

	best = math.inf
	checked = 0
	daily_bay_capacity = float(costs["charging_bay"]["power_kW"]) * 1000.0 * 24 * float(costs["charging_bay"]["charge_efficiency"])
	per_drone_minutes = 0.70 * (1.0 - grounded_fraction) * 1440.0
	for assignment in itertools.product(*choices):
		checked += 1
		daily_cost = 0.0
		busy_by_hub = {hub["site_id"]: 0.0 for hub in hubs}
		energy_by_hub = {hub["site_id"]: 0.0 for hub in hubs}
		used = set()
		for phc, hub_id in zip(phcs, assignment):
			if hub_id is None:
				daily_cost += sum(float(phc[CLASS_COLUMNS[k]]) * float(costs["road_fallback"]["cost_per_trip"][k]) for k in CLASSES)
			else:
				lane = lanes[(hub_id, phc["phc_id"])]
				used.add(hub_id)
				busy_by_hub[hub_id] += lane.busy_min
				energy_by_hub[hub_id] += lane.energy_wh
				daily_cost += lane.road_cost_per_day + lane.electricity_cost_per_day + lane.battery_cost_per_day
		annual = daily_cost * 365.0
		for hub_id in used:
			drones = math.ceil(busy_by_hub[hub_id] / per_drone_minutes - 1e-12)
			bays = math.ceil(energy_by_hub[hub_id] / daily_bay_capacity - 1e-12)
			operators = max(1, math.ceil(drones / float(costs["staff"]["drones_per_operator"])))
			annual += hub_cost["hub"] + drones * hub_cost["drone"] + bays * hub_cost["bay"] + operators * hub_cost["operator"]
		if annual < best:
			best = annual
	return best, checked


def selected_plan(
	result: ModelResult,
	phcs: list[dict[str, str]],
	hubs: list[dict[str, str]],
	costs: dict[str, Any],
	grounded_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	assignment_rows: list[dict[str, Any]] = []
	hub_rows: list[dict[str, Any]] = []
	bay_daily_capacity = float(costs["charging_bay"]["power_kW"]) * 1000.0 * 24.0 * float(costs["charging_bay"]["charge_efficiency"])
	for hub in hubs:
		hub_id = hub["site_id"]
		selected = [phc_id for phc_id in result.phc_ids if value(result, ("x", phc_id, hub_id)) > 0.5]
		drones = int(round(value(result, ("n", hub_id))))
		bays = int(round(value(result, ("b", hub_id))))
		busy_min = sum(result.lanes[(hub_id, phc_id)].busy_min for phc_id in selected)
		energy_wh = sum(result.lanes[(hub_id, phc_id)].energy_wh for phc_id in selected)
		available_drone_min = drones * 1440.0 * (1.0 - grounded_fraction)
		hub_rows.append({
			"hub_id": hub_id,
			"name": hub["name"],
			"open": int(value(result, ("y", hub_id)) > 0.5),
			"drones": drones,
			"charging_bays": bays,
			"operators_per_shift": int(round(value(result, ("o", hub_id)))),
			"served_phcs": ";".join(selected),
			"daily_busy_min": round(busy_min, 3),
			"drone_utilization_percent": round(100.0 * busy_min / available_drone_min, 3) if available_drone_min else 0.0,
			"daily_energy_wh": round(energy_wh, 3),
			"bay_throughput_utilization_percent": round(100.0 * energy_wh / (bays * bay_daily_capacity), 3) if bays else 0.0,
		})
	for phc in phcs:
		phc_id = phc["phc_id"]
		chosen = next((hub_id for hub_id in result.hub_ids if value(result, ("x", phc_id, hub_id)) > 0.5), None)
		if chosen is None:
			assignment_rows.append({"phc_id": phc_id, "hub_id": "ROAD_ONLY", "route_km": "", "emergency_road_fraction": 1.0, "routine_road_fraction": 1.0, "lab_road_fraction": 1.0})
			continue
		lane = result.lanes[(chosen, phc_id)]
		assignment_rows.append({
			"phc_id": phc_id,
			"hub_id": chosen,
			"route_km": round(lane.distance_km, 4),
			"daily_drone_busy_min": round(lane.busy_min, 3),
			"daily_drone_energy_wh": round(lane.energy_wh, 3),
			"emergency_road_fraction": round(lane.fallback_fraction["emergency"], 5),
			"routine_road_fraction": round(lane.fallback_fraction["routine"], 5),
			"lab_road_fraction": round(lane.fallback_fraction["lab"], 5),
		})
	return hub_rows, assignment_rows


def cost_breakdown(
	result: ModelResult,
	phcs: list[dict[str, str]],
	hubs: list[dict[str, str]],
	costs: dict[str, Any],
	vehicle: Vehicle,
) -> list[dict[str, Any]]:
	parameters = annual_cost_parameters(costs, vehicle)
	amounts = {"hub_capital_and_opex": 0.0, "drone_capital_and_maintenance": 0.0, "charging_bay_capital": 0.0, "staff": 0.0, "electricity": 0.0, "battery_replacement": 0.0, "road_fallback": 0.0}
	for hub in hubs:
		hub_id = hub["site_id"]
		amounts["hub_capital_and_opex"] += parameters["hub"] * value(result, ("y", hub_id))
		amounts["drone_capital_and_maintenance"] += parameters["drone"] * value(result, ("n", hub_id))
		amounts["charging_bay_capital"] += parameters["bay"] * value(result, ("b", hub_id))
		amounts["staff"] += parameters["operator"] * value(result, ("o", hub_id))
	for phc in phcs:
		phc_id = phc["phc_id"]
		chosen = next((hub_id for hub_id in result.hub_ids if value(result, ("x", phc_id, hub_id)) > 0.5), None)
		if chosen is None:
			amounts["road_fallback"] += 365.0 * sum(float(phc[CLASS_COLUMNS[k]]) * float(costs["road_fallback"]["cost_per_trip"][k]) for k in CLASSES)
		else:
			lane = result.lanes[(chosen, phc_id)]
			amounts["electricity"] += 365.0 * lane.electricity_cost_per_day
			amounts["battery_replacement"] += 365.0 * lane.battery_cost_per_day
			amounts["road_fallback"] += 365.0 * lane.road_cost_per_day
	rows = [{"cost_component": name, "annual_cost_inr": round(amount, 2)} for name, amount in amounts.items()]
	rows.append({"cost_component": "TOTAL", "annual_cost_inr": round(sum(amounts.values()), 2)})
	return rows


def svg_network(path: Path, sites: list[dict[str, str]], obstacles: list[Any], hubs: list[dict[str, str]], assignments: list[dict[str, Any]], geometries: dict[tuple[str, str], tuple[float, list[tuple[float, float]]]], result: ModelResult) -> None:
	width, height, pad = 1100, 820, 70
	xs = [float(row["x_km"]) for row in sites]
	ys = [float(row["y_km"]) for row in sites]
	xmax, ymax = max(xs) + 2, max(ys) + 2
	sx = (width - 2 * pad) / xmax
	sy = (height - 2 * pad) / ymax
	scale = min(sx, sy)

	def xy(x: float, y: float) -> tuple[float, float]:
		return pad + x * scale, height - pad - y * scale

	svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
		   '<rect width="100%" height="100%" fill="#f8fafc"/>',
		   '<text x="40" y="36" font-family="Arial" font-size="23" font-weight="700" fill="#0f172a">SkyLink optimized hub-to-PHC routes</text>',
		   '<text x="40" y="59" font-family="Arial" font-size="13" fill="#475569">Shortest paths avoid shaded no-fly zones; gray lines denote road-only PHCs</text>']
	for obstacle in obstacles:
		coords = [xy(float(x), float(y)) for x, y in obstacle.exterior.coords]
		points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
		svg.append(f'<polygon points="{points}" fill="#fca5a5" fill-opacity="0.48" stroke="#b91c1c" stroke-width="2"/>')
	hub_color = {hub["site_id"]: HUB_COLORS[i % len(HUB_COLORS)] for i, hub in enumerate(hubs)}
	for row in assignments:
		hub_id = row["hub_id"]
		phc_id = row["phc_id"]
		if hub_id == "ROAD_ONLY":
			phc = next(item for item in sites if item["site_id"] == phc_id)
			x, y = xy(float(phc["x_km"]), float(phc["y_km"]))
			hx, hy = xy(float(hubs[0]["x_km"]), float(hubs[0]["y_km"]))
			svg.append(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 5"/>')
			continue
		_, route = geometries[(hub_id, phc_id)]
		points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(px, py) for px, py in route))
		color = hub_color[hub_id]
		svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-opacity="0.78"/>')
	for site in sites:
		x, y = xy(float(site["x_km"]), float(site["y_km"]))
		if int(site["candidate_hub"]):
			hub_id = site["site_id"]
			opened = value(result, ("y", hub_id)) > 0.5
			color = hub_color[hub_id] if opened else "#64748b"
			radius = 8 if opened else 5
			svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="white" stroke-width="2"/>')
			if opened:
				svg.append(f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-family="Arial" font-size="12" font-weight="700" fill="{color}">{site["site_id"]}</text>')
		else:
			svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0f172a"/>')
			svg.append(f'<text x="{x+5:.1f}" y="{y+12:.1f}" font-family="Arial" font-size="9" fill="#334155">{site["site_id"]}</text>')
	svg.append('</svg>')
	path.write_text("\n".join(svg), encoding="utf-8")


def svg_costs(path: Path, rows: list[dict[str, Any]]) -> None:
	width, height = 1000, 540
	left, right, top, bottom = 115, 35, 75, 110
	plot_w, plot_h = width - left - right, height - top - bottom
	values = [float(row["annual_cost_inr"]) for row in rows]
	maximum = max(values) if values else 1.0
	bar_w = plot_w / max(1, len(rows)) * 0.58
	svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
		   '<rect width="100%" height="100%" fill="#ffffff"/>',
		   '<text x="40" y="38" font-family="Arial" font-size="23" font-weight="700" fill="#0f172a">Annual network cost by M1 payload design</text>',
		   '<text x="40" y="61" font-family="Arial" font-size="13" fill="#475569">INR/year; each point is independently optimized over hubs, assignments, drones and bays</text>']
	for tick in range(5):
		y = top + plot_h * tick / 4
		amount = maximum * (1 - tick / 4)
		svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
		svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#475569">{amount/1e6:.1f}M</text>')
	for i, row in enumerate(rows):
		center = left + plot_w * (i + 0.5) / len(rows)
		bar_h = float(row["annual_cost_inr"]) / maximum * plot_h
		y = top + plot_h - bar_h
		color = "#16a34a" if row.get("selected") else "#3b82f6"
		svg.append(f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
		svg.append(f'<text x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#334155">{float(row["annual_cost_inr"])/1e6:.2f}M</text>')
		svg.append(f'<text x="{center:.1f}" y="{top+plot_h+23}" text-anchor="middle" font-family="Arial" font-size="11" fill="#334155">{row["vehicle"]}</text>')
	svg.append(f'<text x="25" y="{top+plot_h/2:.1f}" transform="rotate(-90 25 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#334155">Annual cost (INR)</text>')
	svg.append('</svg>')
	path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Optimize the SkyLink M4 hub network and fleet.")
	default_data = Path(__file__).resolve().parent.parent / "Projectskylink_data"
	parser.add_argument("--data-dir", type=Path, default=default_data)
	parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "m4_output")
	parser.add_argument("--year", type=int, default=2025, help="Calendar year used to infer outage weekdays for the yearless wind record.")
	parser.add_argument("--reserve-fraction", type=float, default=0.90, help="Additional route-distance fraction of M1 range (default 90%%).")
	args = parser.parse_args()
	if not 0 < args.reserve_fraction <= 1:
		parser.error("--reserve-fraction must be in (0, 1].")
	args.output_dir.mkdir(parents=True, exist_ok=True)

	sites = load_csv(args.data_dir / "sites.csv")
	phcs = [row for row in sites if row["type"] == "phc"]
	hubs = [row for row in sites if int(row["candidate_hub"]) == 1]
	demand_rows = {row["phc_id"]: row for row in load_csv(args.data_dir / "demand.csv")}
	phcs = [{**site, **demand_rows[site["site_id"]]} for site in phcs]
	profile_rows = load_csv(args.data_dir / "hourly_profile.csv")
	profiles: dict[str, list[float]] = {}
	profile_sums: dict[str, float] = {}
	for kind in CLASSES:
		raw = [float(row[PROFILE_COLUMNS[kind]]) for row in profile_rows]
		profile_sums[kind] = sum(raw)
		total = sum(raw)
		if total <= 0:
			raise ValueError(f"Hourly profile has no positive fractions for {kind}.")
		profiles[kind] = [fraction / total for fraction in raw]
	costs = yaml.safe_load((args.data_dir / "costs.yaml").read_text(encoding="utf-8"))
	obstacles = load_no_fly(args.data_dir / "nofly.geojson")
	weather, grounded_fraction, outage_fraction = read_weather(args.data_dir, args.year)
	vehicles = build_vehicle_list()
	print(
		f"M4: preparing routes for {len(hubs)} hubs, {len(phcs)} PHCs, "
		f"and {len(vehicles)} designs.",
		flush=True,
	)
	lane_sets, geometries = build_routes(phcs, hubs, obstacles, vehicles, profiles, weather, costs, args.reserve_fraction)

	results: list[tuple[Vehicle, ModelResult, ModelResult, float]] = []
	cost_rows: list[dict[str, Any]] = []
	for vehicle_number, vehicle in enumerate(vehicles, start=1):
		lanes = lane_sets[vehicle.name]
		print(f"M4 {vehicle_number}/{len(vehicles)} {vehicle.name}: solving MILP.", flush=True)
		integer_result = build_and_solve(phcs, hubs, lanes, vehicle, costs, grounded_fraction)
		print(f"M4 {vehicle_number}/{len(vehicles)} {vehicle.name}: solving LP relaxation.", flush=True)
		lp_result = build_and_solve(phcs, hubs, lanes, vehicle, costs, grounded_fraction, relax=True)
		gap = max(0.0, (integer_result.objective - lp_result.objective) / max(abs(integer_result.objective), 1.0) * 100.0)
		results.append((vehicle, integer_result, lp_result, gap))
		cost_rows.append({
			"vehicle": vehicle.name,
			"payload_kg": vehicle.payload_kg,
			"m1_range_km": vehicle.radius_km,
			"usable_route_limit_km": vehicle.radius_km * args.reserve_fraction,
			"annual_cost_inr": round(integer_result.objective, 2),
			"lp_relaxation_inr": round(lp_result.objective, 2),
			"lp_gap_percent": round(gap, 4),
			"opened_hubs": ";".join(hub_id for hub_id in integer_result.hub_ids if value(integer_result, ("y", hub_id)) > 0.5),
			"solver_status": integer_result.status,
			"selected": False,
		})

	best_index = min(range(len(results)), key=lambda index_: results[index_][1].objective)
	cost_rows[best_index]["selected"] = True
	vehicle, best, _, full_gap = results[best_index]
	write_csv(args.output_dir / "design_costs.csv", cost_rows)
	hub_rows, assignment_rows = selected_plan(best, phcs, hubs, costs, grounded_fraction)
	write_csv(args.output_dir / "hub_plan.csv", hub_rows)
	write_csv(args.output_dir / "phc_assignments.csv", assignment_rows)
	write_csv(args.output_dir / "cost_breakdown.csv", cost_breakdown(best, phcs, hubs, costs, vehicle))
	route_rows = []
	for (hub_id, phc_id), (distance, route) in geometries.items():
		route_rows.append({"hub_id": hub_id, "phc_id": phc_id, "detour_km": round(distance, 4), "waypoints_km": ";".join(f"{x:.3f}:{y:.3f}" for x, y in route)})
	write_csv(args.output_dir / "candidate_routes.csv", route_rows)
	svg_network(args.output_dir / "optimized_network.svg", sites, obstacles, hubs, assignment_rows, geometries, best)
	svg_costs(args.output_dir / "annual_cost_by_design.svg", cost_rows)

	# Required verification: exact enumeration against the same MILP on 4 hubs x 8 PHCs.
	small_hubs = hubs[:4]
	small_phcs = phcs[:8]
	small_lane_set = {
		key: lane for key, lane in lane_sets[vehicles[-1].name].items()
		if key[0] in {hub["site_id"] for hub in small_hubs} and key[1] in {phc["phc_id"] for phc in small_phcs}
	}
	small_vehicle = vehicles[-1]
	small_milp = build_and_solve(small_phcs, small_hubs, small_lane_set, small_vehicle, costs, grounded_fraction)
	brute_objective, enumerated = brute_force_subinstance(small_phcs, small_hubs, small_lane_set, small_vehicle, costs, grounded_fraction)
	mismatch = abs(small_milp.objective - brute_objective)
	tolerance = max(0.01, abs(brute_objective) * 1e-8)
	verification_rows = [{
		"subinstance": "first 4 candidate hubs x first 8 PHCs; 2.5kg M1 design",
		"milp_annual_cost_inr": round(small_milp.objective, 4),
		"brute_force_annual_cost_inr": round(brute_objective, 4),
		"absolute_difference_inr": round(mismatch, 6),
		"enumerated_assignments": enumerated,
		"matches_within_tolerance": mismatch <= tolerance,
		"full_network_lp_gap_percent_for_selected_design": round(full_gap, 6),
		"justified_big_m_drones": "ceil(sum of feasible lane busy time / per-drone 70%-utilization minutes) + 1; valid upper bound since this overcounts PHCs across hubs",
		"justified_big_m_bays": "ceil(sum of feasible lane energy / 24h single-bay throughput) + 1; valid upper bound since this overcounts PHCs across hubs",
		"justified_big_m_operators": "ceil(big-M drones / 4 drones per operator); staff is linked by n <= 4 operators and operators <= this bound times y",
	}]
	write_csv(args.output_dir / "verification.csv", verification_rows)
	if mismatch > tolerance:
		raise AssertionError(f"Brute-force verification failed: MILP={small_milp.objective}, enumeration={brute_objective}")

	notes = [
		"M4 annual cost comparison and hub/fleet optimization completed.",
		f"Selected vehicle: {vehicle.name}; annual cost INR {best.objective:,.2f}; LP gap {full_gap:.4f}%.",
		f"Additional path reserve: {args.reserve_fraction:.1%} of M1 radius (M1 already includes 15% battery reserve).",
		f"Observed hourly profile sums before normalization: {profile_sums}.",
		f"Weather grounded-hour fraction: {grounded_fraction:.4%}; outage-hour fraction (weekday inferred using {args.year}): {outage_fraction:.4%}.",
		"Weather direction is unspecified, so wind magnitude is conservatively applied as headwind in both directions.",
		"Package weights by class are absent; each M1 payload-radius point is treated as its representative loaded mission.",
		f"Enumeration check tested {enumerated:,} assignments and matched the MILP within INR {tolerance:.4f}.",
		"LP gap = (integer optimum - continuous relaxation optimum) / integer optimum.",
		"Drone/bay big-M bounds are computed from total feasible workload/energy and deliberately overcount PHCs; no arbitrary large constants are used.",
		"Expected demand is Poisson-average planning, not a queueing simulation. Emergency service reliability and peak congestion need M5 simulation.",
	]
	(args.output_dir / "assumptions_and_verification.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

	print("\nSkyLink M4 optimization complete")
	print(f"Selected design: {vehicle.name} ({vehicle.payload_kg:.2f} kg payload, {vehicle.radius_km:.2f} km M1 radius)")
	print(f"Annualized cost: INR {best.objective:,.2f}; LP-relaxation gap: {full_gap:.4f}%")
	print("Open hubs:")
	for row in hub_rows:
		if row["open"]:
			print(f"  {row['hub_id']}: {row['drones']} drones, {row['charging_bays']} bays, {row['operators_per_shift']} operators/shift; utilization {row['drone_utilization_percent']:.1f}%; PHCs {row['served_phcs']}")
	print(f"Brute-force verification: {enumerated:,} assignments; MILP difference INR {mismatch:.6f}")
	print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
	main()
