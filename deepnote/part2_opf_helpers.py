import dataclasses
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.transforms as mpt
import networkx as nx
import numpy as np
from pandas import DataFrame
from sea2025.data import DataSet
from sea2025 import incidence


@dataclasses.dataclass
class Result:
    buses: DataFrame
    generators: DataFrame
    lines: DataFrame
    offers: DataFrame
    base_power: float
    total_cost: float


def clear_market(data: DataSet) -> float:
    """Solves the DC optimal power flow problem."""

    # Take references for brevity
    buses = data.buses
    generators = data.generators
    lines = data.lines
    offers = data.offers.copy()  # copy for update
    reference_bus = data.reference_bus
    base_power = data.base_power

    # Incidence matrices to simplify formulation
    line_bus = incidence.line_bus(buses=buses, lines=lines)
    offer_bus = incidence.offer_bus(offers=offers, buses=buses, generators=generators)
    reference_bus_index = incidence.reference_bus(buses, reference_bus)
    offers[:] = offers.merge(generators, left_on="generator_id", right_on="id")

    assert len(lines) > 0, "CVXPY doesn't play nicely with empty variables"

    # Optimization decision variables
    p = cp.Variable(len(offers), name="p")  # dispatched/injected power [MW]
    f = cp.Variable(len(lines), name="f")  # line flows [MW]
    θ = cp.Variable(len(buses), name="θ")  # bus angles [rad]

    # Equality constraints on buses and lines
    balance_constraints = [
        cp.sum([p[o] * offer_bus[o, b] for o in offers.index])
        + cp.sum([f[ell] * line_bus[ell, b] for ell in lines.index])
        == buses.at[b, "load"]
        for b in buses.index
    ]
    flow_constraints = [
        f[ell]
        == cp.sum([line_bus[ell, b] * θ[b] for b in buses.index])
        * base_power
        / lines.at[ell, "reactance"]
        for ell in lines.index
    ]

    # Objective function - minimize total cost
    objective = cp.Minimize(
        cp.sum([offers.at[o, "price"] * p[o] for o in offers.index])
    )

    # Add remaining constraints and solve
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

    # Copy inputs before modifying
    result = Result(
        buses=buses.copy(),
        generators=generators.copy(),
        lines=lines.copy(),
        offers=offers.copy(),
        base_power=base_power,
        total_cost=problem.value,
    )
    result.offers["dispatch"] = p.value
    result.lines["flow"] = f.value
    result.buses["angle"] = θ.value
    result.buses["price"] = [-c.dual_value for c in balance_constraints]
    return result


def postprocess(result: Result) -> None:
    """Postprocesses solution for plotting."""

    # Take references for brevity
    buses = result.buses
    generators = result.generators
    lines = result.lines
    offers = result.offers

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


def make_graph(result: Result) -> nx.DiGraph:
    """Create annotated system (NetworkX) graph."""

    # Take references for brevity
    buses = result.buses
    generators = result.generators
    lines = result.lines
    offers = result.offers
    base_power = result.base_power

    graph = nx.DiGraph(
        dispatch_cost=sum(offers["dispatch"] * offers["price"]),
        load_cost=sum(buses["load"] * buses["price"]),
        base_power=base_power,
    )

    normalize = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
    bus_load_cmap = cm.coolwarm  # coolwarm | inferno | magma | plasma etc.
    for _, bus in buses.iterrows():
        rgba = bus_load_cmap(normalize(bus["load"]))
        graph.add_node(
            bus["id"],
            label=f"{bus['load']:.1f}MW\n${bus['price']:.2f}/MWh",
            load=bus["load"],
            color=mc.to_hex(rgba),
            pos=np.array([bus["x"], bus["y"]]),
            font_color="black",
        )

    generator_cmap = plt.get_cmap(name="tab10", lut=len(generators))
    generator_color = {gid: generator_cmap(i) for i, gid in enumerate(generators["id"])}
    for _, offer in offers.iterrows():
        rgba = generator_color[offer["generator_id"]]
        graph.add_node(
            offer["id"],
            label=f"≤{offer['quantity']}MW\n${offer['price']}/MWh",
            color=mc.to_hex(rgba),
            font_color="darkgreen",
        )

    edge_cmap = cm.coolwarm
    for _, line in lines.iterrows():
        utilization = line["utilization"]
        graph.add_edge(
            line["from_bus_id"],
            line["to_bus_id"],
            label=f"{line['flow']:.0f}MW\n{utilization*100:.0f}%",
            color=edge_cmap(utilization),
            utilization=utilization,
            flow=line["flow"],
            reactance=line["reactance"],
        )

    for _, offer in offers.iterrows():
        utilization = offer["utilization"]
        graph.add_edge(
            offer["id"],
            offer["bus_id"],
            label=f"{offer['dispatch']:.0f}MW",
            color=edge_cmap(utilization),
            utilization=utilization,
            flow=offer["dispatch"],
        )

    return graph


def plot_graph(
    graph: nx.Graph,
    font_size: int = 8,
    xscale: float = 1.0,
    yscale: float = 1.0,
    kscale: float = 1.0,
    epsilon=1 / 100,
):
    edge_labels = nx.get_edge_attributes(graph, "label")
    edge_utilization = nx.get_edge_attributes(graph, "utilization")

    lower, upper = epsilon, 1.0 - epsilon
    edge_labels_unused = {
        k: v for k, v in edge_labels.items() if edge_utilization[k] < lower
    }
    edge_labels_used = {
        k: v for k, v in edge_labels.items() if lower <= edge_utilization[k] <= upper
    }
    edge_labels_saturated = {
        k: v for k, v in edge_labels.items() if upper < edge_utilization[k]
    }

    pos_unscaled = nx.get_node_attributes(graph, "pos")
    xy = np.vstack(list(pos_unscaled.values()))
    xy_shifted = xy - np.min(xy, axis=0)
    xy_scaled = xy_shifted * (np.array([xscale, yscale]) / np.max(np.abs(xy_shifted)))

    pos = nx.spring_layout(
        graph,
        pos=dict(zip(pos_unscaled.keys(), xy_scaled)),
        fixed=pos_unscaled.keys(),
        k=kscale / np.sqrt(graph.number_of_nodes()),
        iterations=10000,
    )

    fig, ax = plt.subplots(frameon=True)

    edge_colors = nx.get_edge_attributes(graph, "color")
    edge_labels = nx.get_edge_attributes(graph, "label")
    node_colors = nx.get_node_attributes(graph, "color")
    node_labels = nx.get_node_attributes(graph, "label")

    nx.draw_networkx_edges(
        graph,
        pos,
        nodelist=edge_colors.keys(),
        edge_color=edge_colors.values(),
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_colors.keys(),
        node_color=node_colors.values(),
        alpha=0.5,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: n for n in graph.nodes()},
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

    for edge_labels, font_color in [
        (edge_labels_unused, "black"),
        (edge_labels_used, "blue"),
        (edge_labels_saturated, "red"),
    ]:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_color=font_color,
            font_size=font_size,
            ax=ax,
        )

    attributes = graph.graph
    load_cost = attributes["load_cost"]
    dispatch_cost = attributes["dispatch_cost"]

    fig.suptitle(
        (
            f"    Load payment: ${load_cost:.2f}/h\n"
            f"Dispatch payment: ${dispatch_cost:.2f}/h\n"
            f"      Difference: ${load_cost - dispatch_cost:.2f}/h"
        ),
        fontsize=font_size,
    )

    return fig, ax
