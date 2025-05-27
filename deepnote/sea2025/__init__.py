import pandas as pd

__all__ = [
    "important_function",
    "data",
    "incidence",
    "plots",
    "solver",
    "verification"
]

from . import data
from . import incidence
from . import plots
from . import verification

# Set display options to show more rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust width of the display
