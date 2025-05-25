"""
Pre-processing of the dataset provided in:
https://bitbucket.org/kdheepak/eightbustestbedrepo/
"""

import pathlib
import cvxpy as cp
import numpy as np
import pandas as pd

input_directory = pathlib.Path("data", "ne8bus-raw")
output_directory = pathlib.Path("data", "ne8bus")
output_directory.mkdir(parents=True, exist_ok=True)


def clean_id(s: str) -> str:
    return (
        s.upper()
        .replace("/", "")
        .replace("-", "")
        .replace("NEMABOST", "NEMA")
        .replace("NEMASSBOST", "NEMA")
        .replace("SEMASS", "SEMA")
        .replace("WCMASS", "WCMA")
        .strip()
    )


generators_map = {
    "GenCo_Name": "id",
    "Zone \r\nLocation": "bus_id",
    "Capacity \r\n(MW)": "capacity",
    "Fuel Type": "fuel_type",
    "MinUp\r\nTime (hr)": "min_uptime",
    "MinDown \r\nTime (hr)": "min_downtime",
    "Ramp Rate\r\n(MW/hr)": "ramp_rate",
    "StartUp \r\nCost ($)": "startup_cost",
    "NoLoad\r\nCost ($)": "noload_cost",
    "Dispatch Cost \r\nCoefficient a \r\n($/MWh)": "dispatch_cost_coef_a",
    "Dispatch Cost \r\nCoefficient b \r\n($/MW^2h)": "dispatch_cost_coef_b",
}
generators: pd.DataFrame = pd.read_csv(
    input_directory / "genData" / "generator_data.csv",
).rename(columns=generators_map)
generators["id"] = generators["id"].str.replace(r"Generator_(\d+)", r"G\1", regex=True)
generators["bus_id"] = generators["bus_id"].map(clean_id)
generators.to_csv(output_directory / "generators.csv", index=False)

lines_map = {
    "Line": "id",
    "From Zone": "from_bus_id",
    "To Zone": "to_bus_id",
    "Distance (miles)": "distance",
    "Resistance (ohms)": "resistance_ohms",
    "Reactance (ohms)": "reactance_ohms",
    "Reactance (per unit)": "reactance",
}
lines: pd.DataFrame = pd.read_csv(
    input_directory / "TransmissionLineData" / "gridDetails.csv",
    sep=" ",
    skiprows=[0],
    names=lines_map.keys(),
).rename(columns=lines_map)

line_voltage_limit = 345  # [kV] for all lines, say
base_power = 100  # [MVA]
angle_limit = np.deg2rad(20)  # [rad]
# Synthesize approximate line capacities [MW]
lines["capacity"] = (
    (line_voltage_limit**2 / lines["reactance"]) * angle_limit / base_power
)
lines["from_bus_id"] = lines["from_bus_id"].map(clean_id)
lines["to_bus_id"] = lines["to_bus_id"].map(clean_id)
lines.to_csv(output_directory / "lines.csv", index=False)

loads_map = {
    "Hours": "hour",
    "Scenarios": "scenario",
}
all_loads_dict: dict[str, pd.DataFrame] = {
    p.stem: pd.read_csv(p).rename(columns=loads_map).set_index(list(loads_map.values()))
    for p in (input_directory / "loadData" / "CSV").glob("*.csv")
}
for name, df in all_loads_dict.items():
    assert name in df.columns
all_loads: pd.DataFrame = pd.concat(all_loads_dict.values(), axis=1).reset_index()
loads = (
    all_loads.groupby(by="hour")
    .agg(lambda x: int(np.round(np.median(x))))
    .drop(columns=["scenario"])
).reset_index()
loads.rename(
    columns=lambda x: clean_id(x) if x != "hour" else x,
    inplace=True,
)
loads.to_csv(output_directory / "loads.csv", index=False)

bus_names = loads.columns.drop("hour")
buses = pd.DataFrame(pd.Series(bus_names, name="id"))
buses.to_csv(output_directory / "buses.csv", index=False)

# offers = generators[["id", "capacity", "dispatch_cost_coef_a"]].rename(
#     columns={
#         "id": "generator_id",
#         "capacity": "quantity",
#         "dispatch_cost_coef_a": "price",
#     }
# )
# offers.to_csv(output_directory / "offers.csv", index=False)

bus_ids = set(buses["id"])
adjacencies = set(lines["from_bus_id"]) | set(lines["to_bus_id"])
assert adjacencies == bus_ids
assert set(generators["bus_id"]) <= bus_ids
