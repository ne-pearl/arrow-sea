import copy
from matplotlib import pyplot as plt
import numpy as np
from unibus import buses, generators, lines, offers, reference_bus, base_power
import clearance


def data():
    return dict(
        buses=copy.deepcopy(buses),
        generators=copy.deepcopy(generators),
        lines=copy.deepcopy(lines),
        offers=copy.deepcopy(offers),
        reference_bus=reference_bus,
        base_power=base_power,
        solver="HIGHS",
    )


data_opf = data()
data_fp = data()

v1 = clearance.solve1(**data_opf)
v2 = clearance.solve_fixed_costs(**data_fp)
print(f"{v1:.3f} vs {v2:.3f} (mismatch of {v1 - v2:.2e})")

load_delta = 1.0


def solve(load):
    data_fp["buses"].at[0, "load"] = load
    f = clearance.solve_fixed_costs(**data_fp)
    data_fp["buses"].at[0, "load"] = load + load_delta
    fp = clearance.solve_fixed_costs(**data_fp)
    print(f"Load {load:.1f} MW: {f:.2f} vs {fp:.2f} => {fp - f:.2f})")
    return f, fp - f


loads = np.linspace(0, generators["capacity"].sum() - load_delta, 100)
costs = [solve(load) for load in loads]
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
