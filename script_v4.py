import cvxpy as cp
from matplotlib import pyplot as plt

import clearance
import plotting
import verification
from pjm5bus import buses, generators, lines, offers, reference_bus, base_power

problem_value = clearance.solve2(
    buses=buses,
    generators=generators,
    lines=lines,
    offers=offers,
    reference_bus=reference_bus,
    solver=cp.GLPK,
)
print(f"Optimal dispatch cost: ${problem_value:.2f}/h")
clearance.postprocessing(buses=buses, generators=generators, lines=lines, offers=offers)

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
