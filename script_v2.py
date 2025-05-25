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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.transforms as mpt
import pygraphviz as pgv

plt.rcParams["font.family"] = "sans-serif"

nodemap = dict(
    style="filled",
    fixedsize="true",
    width=0.25,  # [inch]
    height=0.25,  # [inch]
    labelloc="c",
)
edgemap = dict(labeldistance="1.5")

graph = pgv.AGraph(
    strict=True,
    directed=True,
    name="DC-OPF",
    graph_attr=dict(
        layout="fdp",  # required to pin nodes
        start="regular",  # required to pin nodes
        mode="KK",
        overlap="false",
        sep="+0.5",
        dpi="72",
        size="6,8!",
        ratio="fill",
    ),
)

xy = np.vstack((buses["x"], buses["y"])).T
xy_min = np.min(xy, axis=0)
xy_max = np.max(xy, axis=0)
xy_mid = xy_min
xy_centered = xy - xy_mid
xy_normalized = xy_centered / np.max(np.abs(xy_centered))

bus_load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
bus_load_cmap = cm.coolwarm  # coolwarm | inferno | magma | plasma etc.
for i, bus in buses.iterrows():
    rgba = bus_load_cmap(bus_load_norm(bus["load"]))
    SCALE = 72 * 2  # 2 inches = 144 points, typical spacing
    x, y = xy_normalized[i, :] * SCALE
    graph.add_node(
        bus["id"],
        label=f"{bus['load']:.1f}MW\n${bus['price']:.2f}/MWh",
        color=mc.to_hex(rgba),
        pos=f"{x:.0f},{y:.0f}",  # truncate!
        pin=True,
        shape="square",
        **nodemap,
    )

generator_cmap = plt.get_cmap(name="tab10", lut=len(generators))
generator_color = {gid: generator_cmap(i) for i, gid in enumerate(generators["id"])}
for _, offer in offers.iterrows():
    rgba = generator_color[offer["generator_id"]]
    graph.add_node(
        offer["id"],
        label=f"≤{offer['quantity']}MW\n${offer['price']}/MWh",
        color=mc.to_hex(rgba),
        pos="0,0",
        pin=False,
        shape="circle",
        **nodemap,
    )

for n in graph.nodes():
    print(
        f"{n}: {graph.get_node(n).attr.get('pos')}: {graph.get_node(n).attr.get('pin')}"
    )


def tricolor(utilization, epsilon=1 / 100) -> str:
    assert 0.0 <= utilization <= 1.0
    if utilization < epsilon:
        return "black"
    if 1.0 - epsilon < utilization:
        return "red"
    return "blue"


edge_cmap = cm.coolwarm
for _, line in lines.iterrows():
    utilization = line["utilization"]
    rgba = edge_cmap(utilization)
    graph.add_edge(
        line["from_bus_id"],
        line["to_bus_id"],
        label=f"{line['flow']:.0f}MW\n{line['utilization']*100:.0f}%",
        color=mc.to_hex(rgba),
        fontcolor=tricolor(line["utilization"]),
        **edgemap,
    )

for _, offer in offers.iterrows():
    utilization = offer["utilization"]
    rgba = edge_cmap(utilization)
    graph.add_edge(
        offer["id"],
        offer["bus_id"],
        label=f"{offer['dispatch']*100:.0f}MW",
        color=mc.to_hex(rgba),
        fontcolor=tricolor(line["utilization"]),
        **edgemap,
    )

# total_cost = sum(offers["dispatch"] * offers["price"])
# load_payment = sum(buses["load"] * buses["price"])

# graph.add_node(
#     "__summary__",
#     label=(
#         f"    From Load: ${load_payment:.2f}/h\n"
#         f"To Generators: ${total_cost:.2f}/h\n"
#         f"   Difference: ${load_payment - total_cost:.2f}/h"
#     ),
#     rank="min",
#     shape="none",
#     # pos="7,-5",
#     # pin=True,
# )
# graph.add_edge(
#     "__summary__",
#     buses.at[0, "id"],
#     style="invis",
#     weight=0,
# )

# %%
from IPython.display import SVG, display
import warnings

filename = "network.svg"
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    graph.layout(prog="fdp")
    graph.draw(filename, format="svg", prog="fdp")
display(SVG(filename=filename))


# %%
def position(s: str) -> np.ndarray:
    x, y = s.split(",")
    return np.array([float(x), float(y)])


def nodemap(graph, field, f=lambda x: x, default=None):
    return {n: f(graph.get_node(n).attr.get(field, default)) for n in graph.nodes()}


def edgemap(graph, field, f=lambda x: x, default=None):
    return {
        (u, v): f(graph.get_edge(u, v).attr.get(field, default))
        for u, v in graph.edges()
    }


# %%
pos = nodemap(graph, "pos", position)
node_labels = nodemap(graph, "label")
edge_labels = edgemap(graph, "label")
node_colors = nodemap(graph, "color")
edge_colors = edgemap(graph, "color")
edge_fontcolors = edgemap(graph, "fontcolor")

# %%
import networkx as nx

network = nx.DiGraph()
network.add_nodes_from(graph.nodes())
network.add_edges_from(graph.edges())

import networkx as nx

font_size = 8
fig, ax = plt.subplots(figsize=(8, 6))

nx.draw_networkx_edges(
    network,
    pos,
    nodelist=edge_colors.keys(),
    edge_color=edge_colors.values(),
    ax=ax,
)
nx.draw_networkx_nodes(
    network,
    pos,
    nodelist=node_colors.keys(),
    node_color=node_colors.values(),
    # node_size=1000,
    # edgecolors="black",
    # linewidths=0.5,
    ax=ax,
)
nx.draw_networkx_labels(
    network,
    pos,
    labels={n: n for n in network.nodes()},
    font_size=font_size,
    ax=ax,
)
offset: mpt.Transform = mpt.offset_copy(
    ax.transData, fig=fig, x=0, y=-1.2 * font_size, units="points"
)
for node, (x, y) in pos.items():
    ax.text(
        x,
        y,
        node_labels[node],
        transform=offset,
        fontsize=font_size,
        color="black",
        ha="center",
        va="top",
    )
# No support for a dict of fontcolor :(
for key, label in edge_labels.items():
    nx.draw_networkx_edge_labels(
        network,
        pos,
        edge_labels={key: label},
        font_color=edge_fontcolors[key],
        font_size=font_size,
        ax=ax,
    )

ax.axis("off")
# plt.tight_layout()
plt.show(block=False)
# plt.savefig("network.png")
