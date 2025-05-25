import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.transforms as mpt
import networkx as nx
import numpy as np
from pandas import DataFrame


def initialize(
    buses: DataFrame,
    generators: DataFrame,
    lines: DataFrame,
    offers: DataFrame,
    base_power: float,
) -> nx.DiGraph:
    """Create system graph."""

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


def plot(
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
