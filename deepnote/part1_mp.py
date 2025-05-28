# %%
from dataclasses import fields
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
import sea2025
from part1_mp_helpers import clear_offer_stack, clear_offer_stack_fp, plot_offer_stack

# %% [markdown]
# Worksheet 1: Marginal pricing
# =============================
#
# # Objectives
#
# Revise the notions of economic dispatch and marginal price on the simplest possible network:
# * A geometric solution is available
# * We'll also consider a linear programming formulation, applicable to general networks

# %% [markdown]
# # Examine input data
#
# The format introduced in this section will be used in subsequent activities.

# %%
data = sea2025.data.read("data/fc1bus")


# %% [markdown]
# Examine the data:

# %%
fields(data)

# %%
data.buses

# %% [markdown]
# Let's store the load/demand for later use:

# %%
assert data.buses.index.size == 1
load = data.buses.at[0, "load"]  # Python indexing starts at 0
load # [MW]

# %% [markdown]
# Note the `fixed_cost` column: It is **not** used in our OPF formulation. We'll return to this later.

# %%
data.generators

# %% [markdown]
# A trivial network (one bus) has no lines:

# %%
assert data.lines.index.size == 0
data.lines  # just column headings - no data rows

# %%
data.offers

# %% [markdown]
# # Geometric solution
#
# > The **Optimal Power Flow** problem is to determine the dispatch that satisfies a specified load/demand at minimum cost.
# > As a by-product, the solution procedure furnishes the **marginal price** to use in settling the transactions (between generators and loads).
#
# We'll start with an existing routine to generate the solution (optimal dispatch and marginal price) for the network above.
#
# Take a moment to relate the data above (which tables?) to the geometric solution shown in the plot below.

# %%
result = clear_offer_stack(data, load=load)
plot_offer_stack(result.offers, load=load, marginal_price=result.marginal_price)
plt.show(block=False)
plt.savefig("images/offer-stack.png")

# %% [markdown]
# # Linear programming solution
#
# The geometric procedure (manual/automated) isn't directly applicable to networks (with multiple buses/lines).
#
# The [linear programming](https://en.wikipedia.org/wiki/Linear_programming) (LP) formulation implemented in the function above is laid out below.
#
# Let's study it and discuss the main components:
# * Decision variables `p`
# * Power balance constraint
# * Capacity constraints

# %%
import cvxpy as cp
p = cp.Variable(data.offers.index.size, name="p")
objective = cp.Minimize(cp.sum([data.offers.at[o, "price"] * p[o] for o in data.offers.index]))
balance_constraint = cp.sum([p[o] for o in data.offers.index]) == load
problem = cp.Problem(
    objective,
    [
        balance_constraint,
        p >= 0,
        p <= data.offers["quantity"],
    ],
)
problem.solve(solver=cp.HIGHS)
assert problem.status == cp.OPTIMAL
total_cost = problem.value  # [$/h]
dispatch = p.value  # [MW]
marginal_price = -balance_constraint.dual_value  # [$/MWh] the -sign is convention-dependent

# %% [markdown]
# Verify that the results match our graphical solution:
#
# ![](images/offer-stack-v1.png)

# %%
marginal_price

# %%
DataFrame({"offer": data.offers["id"].values, "dispatch": dispatch})

# %%
assert np.isclose(total_cost, sum(data.offers["price"].values * dispatch))
total_cost


# %% [markdown]
# We'll build on this LP solution when we tackle general networks (multiple buses and lines) and the Unit Commitment problem for longer planning horizons.

# %% [markdown]
# # _Marginal Costs are not Fixed!_
#
# Recall that the `fixed_cost` column of the `generators` table was **not** used in the calculations above.
#
# _What would happen if we did account for fixed costs in the OPF problem?_

# %%
def solve_fp(load: float):
    """Total and cost and increment (with respect to 1MW load increment) for dispatch problem with fixed costs."""
    global data
    result = clear_offer_stack_fp(data, load=load)
    perturbed = clear_offer_stack_fp(data, load=load + 1.0)
    price_delta = perturbed.total_cost - result.total_cost
    return result.total_cost, price_delta


# %%
loads = np.linspace(0, data.generators["capacity"].sum() - 1.0, 100)
costs = [solve_fp(load) for load in loads]
total_costs, marginal_costs = zip(*costs)

# %% [markdown]
# Skip the plotting commands below and discuss the output with your neighbour:
# * How to cost increments relate to marginal costs and to the total cost?
# * Can we explain the cost increments in light of the problem data (`generators` and `offers`)?

# %%
fig, ax_total = plt.subplots()
ax_marginal = ax_total.twinx()
ax_marginal.plot(loads, marginal_costs, label="marginal cost [$/MWh]", color="red")
ax_marginal.set_ylabel("marginal cost [$/MWh]")
ax_total.plot(loads, total_costs, label="cost [$/MWh]", color="blue")
ax_total.set_xlabel("load [MW]")
ax_total.set_ylabel("total cost [$/MWh]")
ax_total.set_title("cost increments vs load")
ax_total.grid(True)
plt.show(block=False)
plt.savefig("images/non-monotonic-prices.png")
