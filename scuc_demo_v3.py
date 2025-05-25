import pathlib
import cvxpy as cp
import numpy as np
import pandas as pd
import incidence

basedir = pathlib.Path("data", "ne8bus")
buses = pd.read_csv(basedir / "buses.csv")
generators = pd.read_csv(basedir / "generators.csv")
lines = pd.read_csv(basedir / "lines.csv")
series = pd.read_csv(basedir / "loads.csv")
reference_bus = buses.at[0, "id"]

lines["capacity"] = lines["capacity"]

base_power = 100  # [MVA]
line_bus = incidence.line_bus(buses=buses, lines=lines)
generator_bus = incidence.generator_bus(buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)

p_min = generators["capacity"].values * 0.0
p_max = generators["capacity"].values
reactance_pu = lines["reactance"].values
line_capacity = lines["capacity"].values
ramp = generators["ramp_rate"].values
cost = generators["dispatch_cost_coef_a"].values  # assume linear
startup_cost = generators["startup_cost"].values
noload_cost = generators["noload_cost"].values

bus_ids = buses["id"].tolist()
assert set(series.columns).issuperset(bus_ids)

p = cp.Variable((len(generators), len(series)), name="p")
f = cp.Variable((len(lines), len(series)), name="f")
θ = cp.Variable((len(buses), len(series)), name="θ")

x_on = cp.Variable((len(generators), len(series)), name="x_on", boolean=True)
x_su = cp.Variable((len(generators), len(series)), name="x_su", boolean=True)
x_sd = cp.Variable((len(generators), len(series)), name="x_sd", boolean=True)


constraints = []
for t in series.index:
    bus_loads = series.loc[t, bus_ids]
    bus_injections = generator_bus.T @ p[:, t] + line_bus.T @ f[:, t]
    line_flows = cp.multiply(line_bus @ θ[:, t], base_power / reactance_pu)
    constraints.extend(
        [
            bus_injections == bus_loads,
            f[:, t] == line_flows,
            θ[reference_bus_index, t] == 0,
            # Injection bounds
            p[:, t] >= cp.multiply(x_on[:, t], p_min),
            p[:, t] <= cp.multiply(x_on[:, t], p_max),
            # Flow bounds
            f[:, t] >= -line_capacity,
            f[:, t] <= +line_capacity,
            # Relaxed binary constraints
            x_on[:, t] >= 0,
            x_on[:, t] <= 1,
            x_su[:, t] >= 0,
            x_su[:, t] <= 1,
            x_sd[:, t] >= 0,
            x_sd[:, t] <= 1,
            # Binary logic
            x_su[:, t] + x_sd[:, t] <= 1,
            # Sanity check
            cp.abs(θ[:, t]) <= np.pi,
        ]
    )

# Ramp constraints
for t in series.index[1:]:
    constraints.extend(
        [
            # Ramping constraints
            p[:, t] - p[:, t - 1] <= ramp,
            p[:, t - 1] - p[:, t] <= ramp,
            # Binary logic
            x_su[:, t] - x_sd[:, t] == x_on[:, t] - x_on[:, t - 1],
        ]
    )


objective = cp.Minimize(cp.sum([cost @ p[:, t] for t in series.index]))
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.HIGHS)

print(f"      Status: {problem.status}")
print(f"Optimal cost: ${problem.value:,.2f}")
