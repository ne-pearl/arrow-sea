# %%
import copy
from matplotlib import pyplot as plt
import numpy as np
import sea2025
from part2_opf_helpers import clear_market, postprocess, make_graph, plot_graph

# %% [markdown]
# Worksheet 2: Optimal power flow on general networks
# ===================================================
#
# # Objectives
#
# We'll extend the LP formulation of worksheet 1 for general networks.

# %% [markdown]
# # Examine input data
#
# This is the same format as before, with a most interesting network comprised of 5 buses and 6 lines.
#
# (This particular dataset originated at PJM and is described [here](https://doi.org/10.1109/PES.2010.5589973).)
#
# As we did in the last worksheet, briefly examine the tables comprising `data`: Add a new `Code` cell for each field.

# %%
data = sea2025.data.read("data/pjm5bus")

# %% [markdown]
# # Solve the OPF problem using an existing function
#
# Let's start with an existing routine and visualize the resulting solution: We'll then walk through the underlying formulation.

# %%
result = clear_market(data)
print(f"Optimal dispatch cost: ${result.total_cost:.2f}/h")

# %% [markdown]
# We have an existing routine to visualize the solution: Optimal dispatch, line flows, and locational marginal prices.

# %%
postprocess(result)
graph = make_graph(result)
fig, ax = plot_graph(graph, kscale=1.0, xscale=1.0, yscale=1.0)
plt.show(block=False)
plt.savefig("images/pjm6bus.png")

# %% [markdown]
# ## Verify feasibility of line flows
#
# If the computed line flows are feasible, they must satisfy:
# * Balance constraint: Edge flows at each bus is zero
# * Power flow constraint: Line flows correspond to differences in (single-valued) bus voltage angles, which should sum to zero around any loop.
#
# Let's check these properties, using the solution attributes now embedded in `graph`.

# %%
bus_flows = sea2025.verification.bus_residuals(graph)
print(bus_flows)
# %% [markdown]
# Aggregating these, we get the residual in each balance constraint:

# %%
print(bus_flows.groupby("bus").agg("sum"))
# %%
cycle_edges = sea2025.verification.cycle_edges(graph)
print(cycle_edges)
# %%
print(cycle_edges.groupby("name").agg({"delta_angle": "sum"}))


# %% [markdown]
# # Check marginal costs
#
# We met marginal cost as the cost increment associated with a unit increment in load. Let's verify this property.

# %%
def cost_delta(bus_index: int) -> float:
    global data, result
    data_perturbed = copy.deepcopy(data)
    data_perturbed.buses.at[bus_index, "load"] += 1.0
    result_perturbed = clear_market(data_perturbed)
    return result_perturbed.total_cost - result.total_cost


cost_deltas = [cost_delta(i) for i in data.buses.index]
print(cost_deltas)
assert np.allclose(result.buses["price"], cost_deltas)

# %% [markdown]
# # LP formulation of OPF for general networks
#
# Having checked that out computed solution is plausible, let's try to reproduce it.
#
# The incomplete solution below is close to the "single-bus" LP formulation. Try to add the missing parts (marked `TODO`).
#
# The incidence matrices below are helpful (although you might prefer to work directly with the data tables).

# %%
buses = data.buses
generators = data.generators
lines = data.lines
offers = data.offers.copy()  # copy for update
reference_bus = data.reference_bus
base_power = data.base_power
reference_bus_index = sea2025.incidence.reference_bus(buses, reference_bus)

line_bus = sea2025.incidence.line_bus(buses=buses, lines=lines)
offer_bus = sea2025.incidence.offer_bus(offers=offers, buses=buses, generators=generators)

# %% [markdown]
# Compare each incidence matrix with the plot to understand what it refers to.

# %%
print(line_bus.shape)
line_bus  # line-bus incidence matrix

# %% [markdown]
# Note that every row ("line") has one `-1` and one `+1`, corresponding to the tail and the head of the edge.

# %%
print(offer_bus.shape)
offer_bus   # offer-bus incidence matrix

# %%
import cvxpy as cp

# Optimization decision variables
p = cp.Variable(offers.index.size, name="p")  # dispatched/injected power [MW]
f = cp.Variable(lines.index.size, name="f")  # line flows [MW]  # TODO 1: missing line flows
θ = cp.Variable(buses.index.size, name="θ")  # bus angles [rad]  # TODO 2: missing bus voltage angles

# Equality constraints on buses and lines
balance_constraints = [
    cp.sum([p[o] * offer_bus[o, b] for o in offers.index])
    + cp.sum([f[ell] * line_bus[ell, b] for ell in lines.index]) # TODO 3: Missing flow terms
    == buses.at[b, "load"]
    for b in buses.index
]
flow_constraints = [
    f[ell]
    == cp.sum([line_bus[ell, b] * θ[b] for b in buses.index]) # TODO 5: Missing angle difference
    * base_power
    / lines.at[ell, "reactance"]
    for ell in lines.index
]

objective = cp.Minimize(cp.sum([offers.at[o, "price"] * p[o] for o in offers.index]))
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
problem.solve(solver=cp.HIGHS)
assert problem.status == cp.OPTIMAL, f"Solver failed: {problem.status}"

total_cost = problem.value
dispatch = p.value
flow = f.value
angle = θ.value
marginal_price = [-c.dual_value for c in balance_constraints]

# %% [markdown]
# Verify that our solution is consistent with the earlier one that we've already checked!

# %%
assert np.allclose(dispatch, result.offers["dispatch"].values)
assert np.allclose(flow, result.lines["flow"].values)
assert np.allclose(angle, result.buses["angle"].values)
assert np.allclose(marginal_price, result.buses["price"].values)
assert np.isclose(total_cost, result.total_cost)

# %% [markdown]
# # Interpretation of the solution
#
# With your neighbour, try to find analogies between the solution plot of this 5-bus network and the offer stack of the preceding worksheet.
# * Which offer/s are marginal?
# * Which offers are fully dispatched?
# * Which LMPs correspond directly to offer prices? Which do not?
# * What is the impact of the congested line?
#
# Don't worry if you don't make progress: We'll continue with this theme on a simpler network in tne next worksheet.
#
# The solution is reproduced here for convenience.
#
# ![](images/pjm6bus-v1.png)
#
#
