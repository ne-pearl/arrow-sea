import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


def line_bus(buses: DataFrame, lines: DataFrame) -> NDArray[np.int8]:
    """Line-bus incidence matrix."""

    def incidence(ell: int, b: int) -> int:
        if buses.at[b, "id"] == lines.at[ell, "from_bus_id"]:
            return -1
        if buses.at[b, "id"] == lines.at[ell, "to_bus_id"]:
            return +1
        return 0

    result = np.array(
        [[incidence(ell, b) for b in buses.index] for ell in lines.index],
        dtype=np.int8,
    )
    assert np.all(np.sum(result, axis=1) == 0), "zero row sum"
    assert np.all(np.sum(np.abs(result), axis=1) == 2), "two nonzeros"
    return result


def offer_bus(
    offers: DataFrame,
    buses: DataFrame,
    generators: DataFrame,
) -> NDArray[np.bool_]:
    """Offer-bus incidence matrix."""

    def incidence(o: int, b: int, g: int) -> bool:
        bus_generator: bool = buses.at[b, "id"] == generators.at[g, "bus_id"]
        generator_offer: bool = generators.at[g, "id"] == offers.at[o, "generator_id"]
        return bus_generator and generator_offer

    result = np.array(
        [
            [any(incidence(o, b, g) for g in generators.index) for b in buses.index]
            for o in offers.index
        ],
        dtype=np.bool_,
    )
    assert np.all(result.sum(axis=1) == 1), "one bus per offer"
    return result


def generator_bus(
    generators: DataFrame,
    buses: DataFrame,
) -> NDArray[np.bool_]:
    """Generator-bus incidence matrix."""

    def incidence(g: int, b: int) -> bool:
        return buses.at[b, "id"] == generators.at[g, "bus_id"]

    result = np.array(
        [[incidence(g, b) for b in buses.index] for g in generators.index],
        dtype=np.bool_,
    )
    assert np.all(result.sum(axis=1) == 1), "one bus per offer"
    return result


def generator_offer(
    generators: DataFrame,
    offers: DataFrame,
) -> NDArray[np.bool_]:
    """Generator-offer incidence matrix."""

    def incidence(g: int, o: int) -> bool:
        return generators.at[g, "id"] == offers.at[o, "generator_id"]

    return np.array(
        [[incidence(g, o) for o in offers.index] for g in generators.index],
        dtype=np.bool_,
    )


def reference_bus(buses: DataFrame, id_: str) -> int:
    ids = buses["id"]
    matches = ids[ids == id_]
    assert len(matches) == 1
    return matches.index[0]
