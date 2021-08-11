from dataclasses import dataclass
from typing import Dict


@dataclass
class FitDataCollection:
    """
    Data Container for the fitted variables of SimpleImpute.
    """
    impute_col_to_fit_block_name: Dict[str, str]
    fully_set: bool
    extra_info = None  # For the KBin
