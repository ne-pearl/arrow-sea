import networkx as nx
from pandas import DataFrame


def buses(graph: nx.Graph) -> DataFrame:
    """Balance residual at every network bus."""

    def make_rows(u: str, attributes: dict):
        """Rows for each adjacency at bus u."""

        def row(v, flow):
            return {"bus": u, "adjacency": v, "flow": flow}

        def edge(v, tail, head, orientation):
            edge = graph.edges[tail, head]
            return row(v, orientation * edge["flow"])

        yield from [edge(v, v, u, +1) for v in graph.predecessors(u)]
        yield from [edge(v, u, v, -1) for v in graph.successors(u)]
        yield row(u, -attributes["load"])

    return DataFrame(
        row
        for name, attributes in graph.nodes.data()
        if attributes.get("load") is not None
        for row in make_rows(name, attributes)
    )


def edges(graph: nx.Graph) -> DataFrame:
    """Edge attributes over every simple path."""

    base_power = graph.graph["base_power"]

    def make_row(cycle: str):
        """Next dataframe row."""
        name = "".join(cycle)
        cycle.append(cycle[0])  # close the loop
        for u, v in zip(cycle[:-1], cycle[1:]):
            sign = 1 if graph.has_edge(u, v) else -1
            if sign < 0:
                u, v = v, u
            line = graph.edges[u, v]
            yield {
                "name": name,
                "from_bus_id": u,
                "to_bus_id": v,
                "angle_difference": sign
                * line["flow"]
                * line["reactance"]
                / base_power,
                "utilization": line["utilization"],
            }

    return DataFrame(
        (
            row
            for cycle in nx.simple_cycles(graph.to_undirected())
            for row in make_row(cycle)
        ),
    )


def edge_aggregates(edges: DataFrame) -> DataFrame:
    """Aggregates edge attributes on each simple path."""
    return edges.groupby("name").agg(
        {
            "angle_difference": "sum",
            "utilization": "max",
        }
    )
