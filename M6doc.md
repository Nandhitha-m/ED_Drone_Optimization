# Module 6: Simulation-Based Multi-Objective Optimization (M6)

## 1. Executive Summary & Problem Framing
Module 6 optimizes system-level operating policies and resource allocations on top of the optimal Module 4 (M4) hub-and-spoke network infrastructure. While M4 resolves static annual average capacities using MILP optimization, Module 6 evaluates operational stochasticity, queueing dynamics, weather disruptions (monsoon gusts), scheduled electrical outages, and emergency prioritization through discrete-event simulation (M5 engine).

---

## 2. Decision Variables, Objectives & Constraints

### Decision Variables ($\mathbf{x}$)
* **Vehicle Sizing Option ($v$)**: Categorical selection from candidate M1/M3 quadcopter designs (`M1_0.50kg` to `M1_2.50kg`).
* **Fleet Allocation ($n_j$)**: Non-negative integer count of drones deployed at each open hub $j \in H_{\text{open}}$.
* **Charging Infrastructure ($b_j$)**: Non-negative integer count of 1.5 kW charging bays installed at each open hub $j \in H_{\text{open}}$.
* **Spare Battery Inventory ($s_j$)**: Non-negative integer count of spare battery packs provisioned per open hub $j \in H_{\text{open}}$.
* **Emergency Reserve Ratio ($r$)**: Continuous fraction $r \in [0, 1)$ of idle drones reserved exclusively for incoming emergency orders.

### Multi-Objective Trade-off ($\mathbf{f}$)
1. **Minimize Annual System Cost (INR)**: Sum of annualized capital recovery (hubs, drones, bays, battery wear), operator salaries, grid/generator power consumption, and road fallback costs.
2. **Maximize Emergency Service Level ($S_{45}$)**: Share of emergency deliveries fulfilled within 45 minutes of order placement across all delivery modes.
3. **Minimize Road Fallback Share**: Proportion of total district deliveries defaulting to 120-minute road transport.

### Evaluated Output Metric
* **95th-Percentile Emergency Delivery Duration ($P_{95}$)**: Monitored to evaluate tail-end delays under weather grounding and queue timeouts.

---

## 3. Mathematical & Optimization Formulation

### Annual Cost Model
$$\text{Cost}_{\text{total}} = \sum_{j \in H_{\text{open}}} \left( C_{\text{hub}} + n_j C_{\text{drone}} + b_j C_{\text{bay}} + o_j C_{\text{staff}} \right) + \text{Cost}_{\text{energy}} + \text{Cost}_{\text{battery}} + \text{Cost}_{\text{road}}$$

Where capital recovery factor (CRF) is given by:
$$\text{CRF}(r, n) = \frac{r(1+r)^n}{(1+r)^n - 1}, \quad r = 0.10$$

### Objective Evaluation Procedure
* **NSGA-II Multi-Objective Optimization**: Evaluates candidate policy vectors across Common Random Numbers (CRN) using 365-day hourly arrival and weather profiles.
* **Pareto Front Identification**: Extracts non-dominated policy sets $(\text{Cost}, S_{45}, \text{Road Share})$ and identifies the knee-point solution.

---

## 4. Verification Check
* **Weighted-Sum Slice Verification**: Verify that solutions obtained via weighted-sum scalarization on a 2-objective slice lie on or behind the non-dominated Pareto front obtained by NSGA-II.
* **Surrogate Accuracy**: Ensure re-simulated Pareto front points match surrogate model predictions within stated statistical error bounds.
