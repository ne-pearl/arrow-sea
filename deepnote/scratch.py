from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import sea2025
from part1_mp_utilities import clear_offer_stack, clear_offer_stack_fp, plot_offer_stack

data = sea2025.data.read_tuple("data/pjm5bus")
max_load = data.buses["load"].sum()

data_continuous = deepcopy(data)
total_price, marginal_price = clear_offer_stack(
    generators=data_continuous.generators,
    offers=data_continuous.offers,
    load=max_load,
)
plot_offer_stack(data_continuous.offers, max_load, marginal_price)
plt.show(block=False)

delta_load = 1.0

def solve_fp(load):
    data_fp = deepcopy(data)
    total_price = clear_offer_stack_fp(
        generators=data_fp.generators,
        offers=data_fp.offers,
        load=load,
    )
    total_price_perturbed = clear_offer_stack_fp(
        generators=data_fp.generators,
        offers=data_fp.offers,
        load=load + delta_load,
    )
    marginal_price = (total_price_perturbed - total_price) / delta_load
    return total_price, marginal_price

pairs = [solve_fp(load) for load in np.linspace(0, max_load, 100)]
total_prices, marginal_prices = zip(*pairs)

loads = np.linspace(0, max_load - delta_load, 100)
costs = [solve_fp(load) for load in loads]
total_costs, marginal_costs = zip(*costs)

fig, ax_total = plt.subplots()
ax_marginal = ax_total.twinx()
ax_marginal.plot(loads, marginal_costs, label="marginal cost [$/MWh]", color="red")
ax_marginal.set_ylabel("marginal cost [$/MWh]")
ax_total.plot(loads, total_costs, label="cost [$/MWh]", color="blue")
ax_total.set_xlabel("load [MW]")
ax_total.set_ylabel("total cost [$/MWh]")
ax_total.set_title("Fixed Costs vs Load")
ax_total.grid(True)
plt.show(block=False)
