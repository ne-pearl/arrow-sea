import pathlib
import cvxpy as cp
import numpy as np
import pandas as pd
import incidence

basedir = pathlib.Path("data", "ne8bus")
buses = pd.read_csv(basedir / "buses.csv")
generators = pd.read_csv(basedir / "generators.csv")
lines = pd.read_csv(basedir / "lines.csv")
series = pd.read_csv(basedir / "loads.csv")
reference_bus = buses.at[0, "id"]

lines["capacity"] = lines["capacity"]

base_power = 100  # [MVA]
line_bus = incidence.line_bus(buses=buses, lines=lines)
generator_bus = incidence.generator_bus(buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)

p_min = generators["capacity"].values * 0.2
p_max = generators["capacity"].values
reactance_pu = lines["reactance"].values
line_capacity = lines["capacity"].values
ramp = generators["ramp_rate"].values
cost = generators["dispatch_cost_coef_a"].values  # assume linear
startup_cost = generators["startup_cost"].values
noload_cost = generators["noload_cost"].values
min_uptime = generators["min_uptime"].values
min_downtime = generators["min_downtime"].values

bus_ids = buses["id"].tolist()
assert set(series.columns).issuperset(bus_ids)
assert series.index[0] == 0

p = cp.Variable((len(generators), len(series)), name="p")
f = cp.Variable((len(lines), len(series)), name="f")
θ = cp.Variable((len(buses), len(series)), name="θ")

x_on = cp.Variable((len(generators), len(series)), name="x_on", boolean=True)
x_su = cp.Variable((len(generators), len(series)), name="x_su", boolean=True)
x_sd = cp.Variable((len(generators), len(series)), name="x_sd", boolean=True)


# def lagged(to, length):
#     return list(range(max(0, to - length + 1), to + 1))


# T = 5
# for t in range(T):
#     for n in range(2 * T):
#         r = lagged(to=t, length=n)
#         print(f"lag({t}, {n}) => {list(r)}")
#         assert len(r) == min(n, t + 1)
#         if 0 < n:
#             assert r[-1] == t
#             assert r[0] == max(0, t - n + 1)

constraints = []
for t in series.index:
    bus_loads = series.loc[t, bus_ids]
    bus_injections = generator_bus.T @ p[:, t] + line_bus.T @ f[:, t]
    line_flows = cp.multiply(line_bus @ θ[:, t], base_power / reactance_pu)
    constraints.extend(
        [
            bus_injections == bus_loads,
            f[:, t] == line_flows,
            θ[reference_bus_index, t] == 0,
            # Injection bounds
            p[:, t] >= cp.multiply(x_on[:, t], p_min),
            p[:, t] <= cp.multiply(x_on[:, t], p_max),
            # Flow bounds
            f[:, t] >= -line_capacity,
            f[:, t] <= +line_capacity,
            # Relaxed binary constraints
            x_on[:, t] >= 0,
            x_on[:, t] <= 1,
            x_su[:, t] >= 0,
            x_su[:, t] <= 1,
            x_sd[:, t] >= 0,
            x_sd[:, t] <= 1,
            # Binary logic
            x_su[:, t] + x_sd[:, t] <= 1,
            # Sanity check
            cp.abs(θ[:, t]) <= np.pi,
        ]
    )
    if t > 0:
        constraints.extend(
            [
                # Ramping constraints
                p[:, t] - p[:, t - 1] <= ramp,
                p[:, t - 1] - p[:, t] <= ramp,
                # Binary logic
                x_su[:, t] - x_sd[:, t] == x_on[:, t] - x_on[:, t - 1],
            ]
        )

    # for g in generators.index:
    #     constraints.extend(
    #         [
    #             np.sum(x_su[g, lagged(min_downtime[g], t)]) <= x_on[g, t],
    #             np.sum(x_sd[g, lagged(min_uptime[g], t)]) <= 1 - x_on[g, t],
    #         ]
    #     )


objective = cp.Minimize(cp.sum([cost @ p[:, t] for t in series.index]))
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.HIGHS)

print(f"      Status: {problem.status}")
print(f"Optimal cost: ${problem.value:,.2f}")

from matplotlib import pyplot as plt

ntop = 3
stdev = np.std(x_on.value, axis=1)
indices = np.argpartition(stdev, -ntop)[-ntop:]
pmin = np.min(p.value[indices, :])
pmax = np.max(p.value[indices, :])

fig, axes = plt.subplots(nrows=ntop, sharex=True, constrained_layout=True)
handles = []
for i, ax_p in zip(indices, axes):
    ax_x = ax_p.twinx()

    (h1,) = ax_p.plot(p.value[i, :], color="red", label="p [MW]", alpha=0.5)
    (h2,) = ax_x.plot(x_on.value[i, :], color="gray", label="x [0/1]", alpha=0.5)

    ax_p.set_title(f"Generator {i}")
    ax_p.set_xlabel("hour")
    ax_p.set_xticks(range(0, p.value.shape[1] + 1, 24))
    ax_p.set_ylim(pmin, pmax)
    ax_p.set_ylabel("p [MW]")
    ax_x.set_ylabel("x [0/1]")
    ax_x.set_yticks([0, 1])

    # Remove x-axis labels for all but the bottom plot
    if ax_p != axes[-1]:
        ax_p.set_xlabel("")

    # Shade peak hours each day
    peak_start = 8  # [hour]
    peak_end = 20  # [hour]
    h3s = [
        ax_p.axvspan(
            offset + peak_start,
            offset + peak_end,
            color="orange",
            alpha=0.3,
            label=f"peak ({peak_start:02d}H-{peak_end:-02d}H)",
        )
        for offset in range(0, p.value.shape[1], 24)
    ]

    # Collect legend handles (only once)
    if len(handles) == 0:
        handles.extend([h1, h2, h3s[1]])

fig.suptitle("commitment & dispatch")
axes[-1].set_xlabel("hour")

# Add shared legend below all subplots
fig.legend(
    handles,
    [h.get_label() for h in handles],
    loc="upper right",
    ncol=1,
    frameon=False,
)
fig.tight_layout()
plt.show()
