# %%
from matplotlib import pyplot as plt
import sea2025
from part2_opf_helpers import clear_market, postprocess, make_graph, plot_graph

def process(data):
    """Solves OPF problem and displays results."""
    result = clear_market(data)
    postprocess(result)  # add fields
    graph = make_graph(result)
    plot_graph(graph)
    plt.show()
    return result, graph


# %%
data = sea2025.data.read("data/triangle1")
process(data);

# %%
data.buses.at[2, "load"] += 1.0
process(data);

# %%
data = sea2025.data.read("data/triangle2")
process(data);

# %%
data.buses.at[2, "load"] += 1.0
process(data);

# %%
data = sea2025.data.read("data/triangle3")
process(data);

# %%
data.buses.at[2, "load"] += 1.0
process(data);

# %%
data = sea2025.data.read("data/triangle4")
process(data);

# %%
data.buses.at[2, "load"] += 1.0
process(data);
