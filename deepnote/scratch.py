from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import sea2025
from part1_mp_utilities import clear_offer_stack, clear_offer_stack_fp, plot_offer_stack

data = sea2025.data.read_tuple("data/fc1bus")
max_load = data.buses["load"].sum()

data_continuous = deepcopy(data)
total_price = clear_offer_stack(
    generators=data_continuous.generators,
    offers=data_continuous.offers,
    load=max_load,
)
marginal_price = data_continuous.offers.at[0, "price"]
plot_offer_stack(data_continuous.offers, max_load, marginal_price)
plt.show(block=False)


def solve_fp(load):
    clear = clear_offer_stack_fp
    data_fp = deepcopy(data)
    total_price = clear(generators=data_fp.generators, offers=data_fp.offers, load=load)
    perturbed_total_price = clear_offer_stack_fp(
        generators=data_fp.generators, offers=data_fp.offers, load=load + 1.0
    )
    price_delta = perturbed_total_price - total_price
    print(
        f"Load {load:.1f} MW: {total_price:.2f} vs {perturbed_total_price:.2f} => {price_delta:.2f})"
    )
    return total_price, price_delta


loads = np.linspace(0, data.generators["capacity"].sum() - 1.0, 100)
costs = [solve_fp(load) for load in loads]
total_costs, marginal_costs = zip(*costs)

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
