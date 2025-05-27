import dataclasses
from typing import Optional
import cvxpy as cp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sea2025.data import DataSet
from sea2025 import incidence


@dataclasses.dataclass
class Result:
    """Result of the optimal dispatch/pricing problem."""

    offers: pd.DataFrame
    generators: pd.DataFrame
    total_cost: float
    marginal_price: Optional[float] = None


def clear_offer_stack(data: DataSet, load: float) -> Result:
    """Optimal dispatch and marginal price at a single bus."""

    # Take references for brevity
    generators: pd.DataFrame = data.generators
    offers: pd.DataFrame = data.offers

    # generator-offer incidence matrix
    generator_offer = incidence.generator_offer(generators=generators, offers=offers)

    # Optimization decision variables and objective function
    p = cp.Variable(len(offers), name="p")  # dispatched/injected power [MW]
    objective = cp.Minimize(
        cp.sum([offers.at[o, "price"] * p[o] for o in offers.index])
    )

    # Power balance constraint
    balance_constraint = cp.sum([p[o] for o in offers.index]) == load

    # Add capacity constraints and solve
    problem = cp.Problem(
        objective,
        [
            balance_constraint,
            p >= 0,
            p <= offers["quantity"],
            generator_offer @ p <= generators["capacity"],
        ],
    )
    problem.solve(solver=cp.HIGHS)
    assert problem.status == cp.OPTIMAL, f"Solver failed: {problem.status}"

    # Copy inputs before modifying
    result = Result(
        offers=offers.copy(),
        generators=generators.copy(),
        total_cost=problem.value,
        marginal_price=-balance_constraint.dual_value,
    )
    result.offers["dispatch"] = p.value
    return result


def clear_offer_stack_fp(data: DataSet, load: float) -> float:
    """
    Optimal dispatch and marginal price at a single bus,
    accounting for fixed costs of generators.
    """

    # Take references for brevity
    generators: pd.DataFrame = data.generators
    offers: pd.DataFrame = data.offers

    # generator-offer incidence matrix
    generator_offer = incidence.generator_offer(generators=generators, offers=offers)

    # Optimization decision variables and objective function
    p = cp.Variable(len(offers), name="p")  # dispatched/injected power [MW]
    x = cp.Variable(len(generators), name="x", boolean=True)  # generator on/off status
    objective = cp.Minimize(
        cp.sum([offers.at[o, "price"] * p[o] for o in offers.index])
        + cp.sum([generators.at[g, "fixed_cost"] * x[g] for g in generators.index])
    )

    # Power balance constraint
    balance_constraint = cp.sum([p[o] for o in offers.index]) == load

    # Add capacity constraints and solve
    problem = cp.Problem(
        objective,
        [
            balance_constraint,
            p >= 0,
            p <= offers["quantity"],
            generator_offer @ p <= cp.multiply(x, generators["capacity"]),
        ],
    )

    problem.solve(solver=cp.HIGHS)
    assert problem.status == cp.OPTIMAL, f"Solver failed: {problem.status}"

    # Copy inputs before modifying
    result = Result(
        offers=offers.copy(),
        generators=generators.copy(),
        total_cost=problem.value,
        marginal_price=None,  # not available in this formulation
    )
    result.offers["dispatch"] = p.value
    result.generators["commit"] = x.value.astype(bool)
    return result


def cumsum_mid(x, start=0):
    """Interval midpoints from widths."""
    accumulated = np.concatenate(([start], np.cumsum(x)))
    return (accumulated[:-1] + accumulated[1:]) * 0.5


def plot_offer_stack(offers: pd.DataFrame, load: float, marginal_price: float):

    fig, ax = plt.subplots()
    ax.set_xlabel("Quantity (MW)")
    ax.set_ylabel("Price ($/MWh)")

    sorted_offers = offers.sort_values(by="price")
    select = sorted_offers["dispatch"] > 0
    dispatched_offers = sorted_offers[select]
    _, generator_indices = np.unique(sorted_offers["generator_id"], return_inverse=True)
    colors = plt.get_cmap("tab10")(generator_indices)

    bars = ax.bar(
        x=cumsum_mid(sorted_offers["quantity"]),
        height=sorted_offers["price"],
        width=sorted_offers["quantity"],
        color=colors,
        alpha=0.4,
    )
    ax.bar(
        x=cumsum_mid(dispatched_offers["dispatch"]),
        height=dispatched_offers["price"],
        width=dispatched_offers["dispatch"],
        color=colors[select],
        alpha=0.8,
    )
    ax.bar_label(
        bars,
        labels=sorted_offers["id"],
        label_type="center",
        padding=5,
    )
    ax.bar_label(
        bars,
        labels=sorted_offers["quantity"].map(lambda x: f"{x}MW"),
        label_type="center",
        padding=-5,
    )
    ax.bar_label(
        bars,
        labels=sorted_offers["price"].map(lambda x: f"${x:.2f}/MWh"),
        label_type="edge",
    )

    ax.axvline(x=load, color="black", label="load")
    ax.axhline(y=marginal_price, color="red", label="marginal price")
    ax.text(load, 0.1, "load", rotation=90, va="bottom", ha="center", color="black")
    ax.text(0.0, marginal_price, "marginal price", va="center", ha="left", color="red")

    return fig, ax
