from matplotlib import pyplot as plt
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
