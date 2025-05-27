# %%
from dataclasses import fields
from matplotlib import pyplot as plt
import numpy as np
import sea2025
from part1_mp_helpers import clear_offer_stack, clear_offer_stack_fp, plot_offer_stack

# %%
data = sea2025.data.read("data/fc1bus", lines=None)
load = data.buses.at[0, "load"]
load # [MW]

# %%
fields(data)

# %%
data.buses

# %%
data.generators

# %%
data.offers

# %%
result = clear_offer_stack(data, load=load)
plot_offer_stack(result.offers, load=load, marginal_price=result.marginal_price)
plt.show(block=False)
plt.savefig("images/offer-stack.png")

# %%
def solve_fp(load: float):
    global data
    result = clear_offer_stack_fp(data, load=load)
    perturbed = clear_offer_stack_fp(data, load=load + 1.0)
    price_delta = perturbed.total_cost - result.total_cost
    return result.total_cost, price_delta


# %%
loads = np.linspace(0, data.generators["capacity"].sum() - 1.0, 100)
costs = [solve_fp(load) for load in loads]
total_costs, marginal_costs = zip(*costs)

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
