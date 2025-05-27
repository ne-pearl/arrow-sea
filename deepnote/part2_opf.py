import copy
from matplotlib import pyplot as plt
import numpy as np
import sea2025
from part2_opf_helpers import clear_market, postprocess, make_graph, plot_graph

# %%
data = sea2025.data.read("data/pjm5bus")

# %%
result = clear_market(data)
print(f"Optimal dispatch cost: ${result.total_cost:.2f}/h")

# %%
postprocess(result)
graph = make_graph(result)
fig, ax = plot_graph(graph, kscale=1.0, xscale=1.0, yscale=1.0)
plt.show(block=False)

# %%
bus_residuals = sea2025.verification.bus_residuals(graph)
print(bus_residuals)
# %%
print(bus_residuals.groupby("bus").agg("sum"))
# %%
cycle_edges = sea2025.verification.cycle_edges(graph)
print(cycle_edges)
# %%
print(cycle_edges.groupby("name").agg({"delta_angle": "sum"}))


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
