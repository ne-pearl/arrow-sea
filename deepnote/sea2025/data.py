from collections import namedtuple
from typing import Any, Union
import pathlib
import pandas as pd


def read_dict(
    folder: Union[pathlib.Path, str],
    reference_bus_index: int = 0,
    base_power: float = 100.0,
) -> dict[str, Any]:
    """Load CSV files from case directory."""
    folder = pathlib.Path(folder)
    assert folder.is_dir(), f'"{folder.resolve()}" does not exist or is not a directory'
    buses = pd.read_csv(folder / "buses.csv")
    offers = pd.read_csv(folder / "offers.csv")
    offers["tranche"] = offers.groupby("generator_id").cumcount() + 1
    offers["id"] = offers["generator_id"] + "/" + offers["tranche"].astype(str)
    return dict(
        buses=buses,
        reference_bus=buses.at[reference_bus_index, "id"],
        generators=pd.read_csv(folder / "generators.csv"),
        lines=pd.read_csv(folder / "lines.csv"),
        offers=offers,
        base_power=base_power,
    )


DataSet = namedtuple(
    "DataSet", ["buses", "generators", "lines", "offers", "reference_bus", "base_power"]
)


def read_tuple(*args, **kwargs) -> DataSet:
    """Load CSV files from case directory."""
    d = read_dict(*args, **kwargs)
    return DataSet(
        buses=d["buses"],
        generators=d["generators"],
        lines=d["lines"],
        offers=d["offers"],
        reference_bus=d["reference_bus"],
        base_power=d["base_power"],
    )


def read(*args, **kwargs):
    """Load CSV files from case directory."""
    t = read_tuple(*args, **kwargs)
    return (
        t.buses,
        t.generators,
        t.lines,
        t.offers,
        t.reference_bus,
        t.base_power,
    )
