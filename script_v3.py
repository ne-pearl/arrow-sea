# %%
import cvxpy as cp
import numpy as np

import incidence
from pjm5bus import buses, generators, lines, offers, reference_bus, base_power

# %%
line_bus = incidence.line_bus(buses=buses, lines=lines)
offer_bus = incidence.offer_bus(offers=offers, buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)
offers = offers.merge(generators, left_on="generator_id", right_on="id")

# %%
p = cp.Variable(len(offers), name="p")  # dispatched/injected power [MW]
f = cp.Variable(len(lines), name="f")  # line flows [MW]
θ = cp.Variable(len(buses), name="θ")  # bus angles [rad]

# %%
balance_constraints = [
    cp.sum([p[o] * offer_bus[o, b] for o in offers.index])
    + cp.sum([f[ell] * line_bus[ell, b] for ell in lines.index])
    == buses.at[b, "load"]
    for b in buses.index
]

# %%
flow_constraints = [
    f[ell]
    == cp.sum([line_bus[ell, b] * θ[b] for b in buses.index])
    * base_power
    / lines.at[ell, "reactance"]
    for ell in lines.index
]

# %%
objective = cp.Minimize(cp.sum([offers.at[o, "price"] * p[o] for o in offers.index]))

# %%
problem = cp.Problem(
    objective,
    [
        *balance_constraints,
        *flow_constraints,
        θ[reference_bus_index] == 0,
        f >= -lines["capacity"],
        f <= lines["capacity"],
        p >= 0,
        p <= offers["quantity"],
    ],
)
print(problem)

# %%
problem.solve(solver="ECOS")
print(f"Optimal dispatch cost: ${problem.value:.2f} / h")
# %%
offers["dispatch"] = p.value
lines["flow"] = f.value
buses["angle"] = θ.value
buses["price"] = [-c.dual_value for c in balance_constraints]

#
# Prostprocessing
#
agg = offers.groupby("generator_id").agg({"dispatch": "sum", "quantity": "sum"})
generators["dispatch"] = generators["id"].map(agg["dispatch"])
generators["capacity"] = generators["id"].map(agg["quantity"])
generators["utilization"] = generators["dispatch"] / generators["capacity"]
generators["revenue"] = generators["dispatch"] * generators["bus_id"].map(
    buses.set_index("id")["price"]
)
lines["utilization"] = lines["flow"].abs() / lines["capacity"]
offers["utilization"] = offers["dispatch"] / offers["quantity"]
offers["bus_id"] = offers["generator_id"].map(generators.set_index("id")["bus_id"])
offers["tranche"] = offers.groupby("generator_id").cumcount() + 1
offers["id"] = offers["generator_id"] + "/" + offers["tranche"].astype(str)

#
# Congestion pricing
#

# %%
offer_dispatch = offers["dispatch"]
line_flow = lines["flow"]
bus_angle = buses["angle"]
line_reactance = lines["reactance"]

# %%
assert np.allclose(
    (line_bus @ bus_angle) * base_power / line_reactance,
    line_flow,
)

# %%
bus_injections = offer_dispatch @ offer_bus - buses["load"]
free_bus_ids = [b for b in buses.index if b != reference_bus_index]
K = line_bus[:, free_bus_ids]
KtB = K.T @ np.diag(1.0 / line_reactance)
SF = np.linalg.solve(KtB @ K, -KtB).T

# %%
assert np.allclose(bus_injections, -line_flow @ line_bus)
assert np.allclose(SF @ bus_injections[free_bus_ids], line_flow)
assert np.allclose(
    (SF @ offer_bus[:, free_bus_ids].T) @ offer_dispatch
    - SF @ buses.loc[free_bus_ids, "load"],
    line_flow,
)

# %%
p = cp.Variable(len(offers), name="p")  # dispatched/injected power [MW]
f = cp.Variable(len(lines), name="f")  # line flows [MW]

# %%
balance_constraint = cp.sum(p) == sum(buses["load"])

# %%
A = SF @ offer_bus[:, free_bus_ids].T
L = SF @ buses.loc[free_bus_ids, "load"]
flow_constraints = [
    cp.sum([A[ell, o] * p[o] for o in offers.index]) - L[ell] == f[ell]
    for ell in lines.index
]

# %%
flow_lower_bounds = f >= -lines["capacity"]
flow_upper_bounds = f <= lines["capacity"]

# %%
objective = cp.Minimize(cp.sum([offers.at[o, "price"] * p[o] for o in offers.index]))

# %%
problem = cp.Problem(
    objective,
    [
        balance_constraint,
        *flow_constraints,
        flow_lower_bounds,
        flow_upper_bounds,
        p >= 0,
        p <= offers["quantity"],
    ],
)

# %%
problem.solve(solver="ECOS")

assert np.allclose(p.value, offers["dispatch"])
assert np.allclose(f.value, lines["flow"])
print(f"Optimal dispatch cost: ${problem.value:.2f} / h")

# %%
buses["lmp_energy"] = -balance_constraint.dual_value

# %%
mu_lower = -flow_lower_bounds.dual_value
mu_upper = -flow_upper_bounds.dual_value
buses["lmp_congestion"] = 0.0
buses.loc[free_bus_ids, "lmp_congestion"] = SF.T @ (mu_upper - mu_lower)

#
# Plotting
#
from matplotlib import pyplot as plt
import plotting
import verification

graph = plotting.initialize(
    buses=buses,
    generators=generators,
    lines=lines,
    offers=offers,
    base_power=base_power,
)
fig, ax = plotting.plot(graph, kscale=2.0, xscale=3.0, yscale=3.0)
plt.savefig("network.svg")
plt.show(block=False)

cycle_edges = verification.edges(graph)
cycle_aggregates = verification.edge_aggregates(cycle_edges)

bus_flows = verification.buses(graph)
bus_aggregates = bus_flows.groupby("bus").agg("sum")

print(cycle_aggregates)
print(bus_aggregates)
