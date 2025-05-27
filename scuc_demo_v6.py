import pathlib
import cvxpy as cp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import incidence

basedir = pathlib.Path("data", "ne8bus")
buses = pd.read_csv(basedir / "buses.csv")
generators = pd.read_csv(basedir / "generators.csv")
lines = pd.read_csv(basedir / "lines.csv")
series = pd.read_csv(basedir / "loads.csv")
offers = pd.read_csv(basedir / "offers.csv")
reference_bus = buses.at[0, "id"]

# Make the problem smaller for demo purposes
series = series.iloc[: series.shape[0] // 1, :]
x_on_initial = np.repeat(0, generators.index.size)

base_power = 100  # [MVA]
line_bus = incidence.line_bus(buses=buses, lines=lines)
generator_bus = incidence.generator_bus(buses=buses, generators=generators)
offer_bus = incidence.offer_bus(buses=buses, offers=offers, generators=generators)
generator_offer = incidence.generator_offer(generators=generators, offers=offers)
reference_bus_index = incidence.reference_bus(buses, reference_bus)

offer_quantity = offers["quantity"].values
offer_price = offers["price"].values

p_min = generators["capacity"].values * 0.2
p_max = generators["capacity"].values

reactance_pu = lines["reactance"].values
line_capacity = lines["capacity"].values

ramp = generators["ramp_rate"].values
startup_cost = generators["startup_cost"].values
noload_cost = generators["noload_cost"].values

# Uncomment for minimum up/down-time constraints
# min_uptime = generators["min_uptime"].values
# min_downtime = generators["min_downtime"].values

bus_ids = buses["id"].tolist()
assert set(series.columns).issuperset(bus_ids)
assert series.index[0] == 0

p = cp.Variable((len(offers), len(series)), name="p")
f = cp.Variable((len(lines), len(series)), name="f")
θ = cp.Variable((len(buses), len(series)), name="θ")

x_on = cp.Variable((len(generators), len(series)), name="x_on", boolean=True)
x_su = cp.Variable((len(generators), len(series)), name="x_su", boolean=True)
x_sd = cp.Variable((len(generators), len(series)), name="x_sd", boolean=True)

constraints = [
    # x_on[:, 0] == x_on_initial,
]

for t in series.index:
    bus_loads = series.loc[t, bus_ids]
    bus_injections = offer_bus.T @ p[:, t] + line_bus.T @ f[:, t]
    line_flows = cp.multiply(line_bus @ θ[:, t], base_power / reactance_pu)
    p_total_now = generator_offer @ p[:, t]
    constraints.extend(
        [
            bus_injections == bus_loads,
            f[:, t] == line_flows,
            θ[reference_bus_index, t] == 0,
            # Injection bounds
            p[:, t] >= 0,
            p[:, t] <= offer_quantity,
            p_total_now >= cp.multiply(x_on[:, t], p_min),
            p_total_now <= cp.multiply(x_on[:, t], p_max),
            # Flow bounds
            f[:, t] >= -line_capacity,
            f[:, t] <= +line_capacity,
            # Binary logic
            x_su[:, t] + x_sd[:, t] <= 1,
            # Sanity check
            cp.abs(θ[:, t]) <= np.pi,
        ]
    )
    if t > 0:
        p_total_prev = generator_offer @ p[:, t - 1]
        constraints.extend(
            [
                # Ramping constraints
                p_total_now - p_total_prev <= ramp + cp.multiply(p_max, x_su[:, t]),
                p_total_prev - p_total_now <= ramp + cp.multiply(p_max, x_sd[:, t]),
                # Binary logic
                x_su[:, t] - x_sd[:, t] == x_on[:, t] - x_on[:, t - 1],
            ]
        )

    # Constraints for minimum up/down-time: Implementation #1 - very expensive
    # for g in generators.index:
    #     constraints.extend(
    #         x_on[g, t + τ] >= x_su[g, t]
    #         for τ in range(min_uptime[g])
    #         if t + τ < series.hour.size
    #     )
    #     constraints.extend(
    #         x_on[g, t + τ] <= 1 - x_sd[g, t]
    #         for τ in range(min_downtime[g])
    #         if t + τ < series.hour.size
    #     )

# # Constraints for minimum up/down-time: Implementation #2 - expensive
# for g in generators.index:
#     constraints.extend(
#         cp.sum(x_on[g, t : t + min_uptime[g]]) >= min_uptime[g] * x_su[g, t]
#         for t in range(series.hour.size - min_uptime[g])
#     )
#     constraints.extend(
#         cp.sum(x_on[g, t : t + min_downtime[g]]) <= min_downtime[g] * (1 - x_sd[g, t])
#         for t in range(series.hour.size - min_downtime[g])
#     )

objective = cp.Minimize(
    cp.sum(
        [
            offer_price @ p[:, t] + noload_cost @ x_on[:, t] + startup_cost @ x_su[:, t]
            for t in series.index
        ]
    )
)

print(f"# Constraints: {len(constraints)}")
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.HIGHS, verbose=True)
print(f"       Status: {problem.status}")
print(f" Optimal cost: ${problem.value:,.2f}")

p_generator = generator_offer @ p.value
prices = offers.groupby("generator_id")["price"]
prices_min = prices.min().values
prices_max = prices.max().values

ntop = 5
stdev = np.std(p_generator, axis=1)
indices = np.sort(np.argpartition(stdev, -ntop)[-ntop:])

pmin = np.min(p_generator[indices, :])
pmax = np.max(p_generator[indices, :])

fig, axes = plt.subplots(nrows=ntop, sharex=True, constrained_layout=True)

handles = []
for i, ax_p in zip(indices, axes):
    ax_x = ax_p.twinx()

    (h1,) = ax_p.plot(p_generator[i, :], color="red", label="p [MW]", alpha=0.5)
    (h2,) = ax_x.plot(x_on.value[i, :], color="gray", label="x [0/1]", alpha=0.5)

    ax_p.set_title(
        (
            f"{generators.at[i, 'id']}: "
            f"$({prices_min[i]:.2f} - {prices_max[i]:.2f})/MWh, "
            f"Bus {generators.at[i, 'bus_id']}"
        )
    )
    ax_p.set_xlabel("hour")
    ax_p.set_xticks(range(0, p_generator.shape[1] + 1, 24))
    ax_p.set_ylim(pmin, pmax)
    ax_p.set_ylabel("p")
    ax_x.set_ylabel("x")
    ax_x.set_yticks([0, 1])

    # Remove x-axis labels for all but the bottom plot
    if ax_p != axes[-1]:
        ax_p.set_xlabel("")

    # Shade peak hours each day
    peak_start = 16  # [hour]
    peak_end = 21  # [hour]
    h3s = [
        ax_p.axvspan(
            offset + peak_start,
            offset + peak_end,
            color="orange",
            alpha=0.3,
            label=f"peak ({peak_start:02d}H-{peak_end:-02d}H)",
        )
        for offset in range(0, p_generator.shape[1], 24)
    ]

    # Collect legend handles (only once)
    if len(handles) == 0:
        handles.extend([h1, h2, h3s[1]])

fig.suptitle("dispatch & generation")
axes[-1].set_xlabel("hour")

# Add shared legend below all subplots
# fig.legend(
#     handles,
#     [h.get_label() for h in handles],
#     frameon=False,
#     loc="center",
#     ncol=1,
# )
plt.show(block=False)
