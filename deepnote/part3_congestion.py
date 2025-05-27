# %%
from matplotlib import pyplot as plt
import sea2025
from part2_opf_helpers import clear_market, postprocess, make_graph, plot_graph


def process(data):
    """Solves OPF problem and displays results."""
    result = clear_market(data)
    postprocess(result)  # add fields
    graph = make_graph(result)
    print("Cycle edges:")
    print(sea2025.verification.cycle_edges(graph))
    plot_graph(graph)
    plt.show(block=False)
    return graph


# %%
data = sea2025.data.read("data/triangle1")

# %%
print(data.buses)
# %%
print(data.generators)
# %%
print(data.lines)
# %%
print(data.offers)


# %%
graph = process(data)

# %%
data.buses.at[0, "load"] += 1.0
process(data)
