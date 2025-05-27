from collections import namedtuple
import dataclasses
from typing import Any, Union
import pathlib
import pandas as pd


def read_dict(
    folder: Union[pathlib.Path, str],
    reference_bus_index: int,
    base_power: float,
) -> dict[str, Any]:
    """Load CSV files from case directory."""
    folder = pathlib.Path(folder)
    assert folder.is_dir(), f'"{folder.resolve()}" does not exist or is not a directory'
    buses = pd.read_csv(folder / "buses.csv")
    offers = pd.read_csv(folder / "offers.csv")
    offers["tranche"] = offers.groupby("generator_id").cumcount() + 1
    offers["id"] = offers["generator_id"] + "/" + offers["tranche"].astype(str)
    loads_file = folder / "loads.csv"
    loads = pd.read_csv(loads_file) if loads_file.exists() else None
    if loads is not None:
        assert set(loads.columns).issuperset(buses["id"].tolist())
        assert loads.index[0] == 0
    return dict(
        buses=buses,
        generators=pd.read_csv(folder / "generators.csv"),
        lines=pd.read_csv(folder / "lines.csv"),
        offers=offers,
        loads=loads,
        reference_bus=buses.at[reference_bus_index, "id"],
        base_power=base_power,
    )


@dataclasses.dataclass
class DataSet:
    buses: pd.DataFrame
    generators: pd.DataFrame
    lines: pd.DataFrame
    offers: pd.DataFrame
    loads: pd.DataFrame | None
    reference_bus: str
    base_power: float


def read(
    folder: str,
    reference_bus_index: int = 0,
    base_power: float = 100.0,
    **overrides,
) -> DataSet:
    """Load CSV files from case directory."""
    d = read_dict(
        folder=folder,
        reference_bus_index=reference_bus_index,
        base_power=base_power,
    )
    d.update(**overrides)
    return DataSet(
        buses=d["buses"],
        generators=d["generators"],
        lines=d["lines"],
        offers=d["offers"],
        loads=d["loads"],
        reference_bus=d["reference_bus"],
        base_power=d["base_power"],
    )


# def read(folder: str, reference_bus_index: int = 0, **overrides):
#     """Load CSV files from case directory."""
#     t = read_tuple(folder, reference_bus_index=reference_bus_index, **overrides)
#     return (
#         t.buses,
#         t.generators,
#         t.lines,
#         t.offers,
#         t.reference_bus,
#         t.base_power,
#     )
