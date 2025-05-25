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

lines["capacity"] = lines["capacity"] + 10000000  # WARNING

base_power = 100  # [MVA]
line_bus = incidence.line_bus(buses=buses, lines=lines)
generator_bus = incidence.generator_bus(buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)

p_min = generators["capacity"].values * 0.0
p_max = generators["capacity"].values
cost = generators["dispatch_cost_coef_a"].values  # assume linear

# Dispatched/injected power [MW] for each generator
p = cp.Variable((len(generators), len(series)), name="p")
# Unit commitment flag for each generator
u = cp.Variable((len(generators), len(series)), "u")
# Power flow [MW] on each line
f = cp.Variable((len(lines), len(series)), name="f")
# Voltage angles [rad] at each bus
θ = cp.Variable((len(buses), len(series)), name="θ")

bus_ids = buses["id"].tolist()
assert set(series.columns).issuperset(bus_ids)
reactance_pu = lines["reactance"].values
line_capacity = lines["capacity"].values

total_demand = series[bus_ids].sum(axis=1)
total_generation = p_max.sum()
print("Max total generation:", total_generation)
print("    Max total demand:", total_demand.max())
print("Generators per bus:")
import textwrap


def show(x, prefix=" " * 4):
    print(textwrap.indent(str(x), prefix))


show(generators["bus_id"].value_counts())
print("Load per bus:")
show(series[bus_ids].sum(axis=0).sort_values(ascending=False))
print("Line capacities:")
show(lines[["from_bus_id", "to_bus_id", "capacity"]])

# Generation limits
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
            p[:, t] >= cp.multiply(u[:, t], p_min),
            p[:, t] <= cp.multiply(u[:, t], p_max),
            # Flow bounds
            f[:, t] >= -line_capacity,
            f[:, t] <= +line_capacity,
            # Relaxed binary constraints
            u[:, t] >= 0,
            u[:, t] <= 1,
            # Sanity check
            cp.abs(θ[:, t]) <= np.pi,
        ]
    )

objective = cp.Minimize(cp.sum([cost @ p[:, t] for t in series.index]))
print("Formulating...")
problem = cp.Problem(objective, constraints)
print("Optimizing...")
problem.solve(solver=cp.HIGHS)

print(f"      Status: {problem.status}")
print(f"Optimal cost: ${problem.value:,.2f}")
