from pandas import DataFrame

base_power = 100  # [MVA]
reference_bus = "Bus1"
buses = DataFrame(
    [
        ("Bus1", 0.0, 0.0, 0.0),
    ],
    columns=["id", "load", "x", "y"],
)
generators = DataFrame(
    [
        ("A", "Bus1", 200.0, 0.0),
        ("B", "Bus1", 200.0, 6000.0),
        ("C", "Bus1", 200.0, 8000.0),
    ],
    columns=["id", "bus_id", "capacity", "fixed_cost"],
)
lines = DataFrame(
    [],
    columns=["from_bus_id", "to_bus_id", "capacity", "reactance"],
)
offers = DataFrame(
    [
        ("A", 100.0, 65.0),
        ("A", 100.0, 110.0),
        ("B", 100.0, 40.0),
        ("B", 100.0, 90.0),
        ("C", 100.0, 25.0),
        ("C", 100.0, 35.0),
    ],
    columns=["generator_id", "quantity", "price"],
)
