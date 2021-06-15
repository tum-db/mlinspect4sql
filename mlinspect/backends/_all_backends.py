"""
Get all available backends
"""
from typing import List

from ._backend import Backend
from ._pandas_backend import PandasBackend
from ._sklearn_backend import SklearnBackend
from ._sql_pandas_backend import SQLBackend


def get_all_backends() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [PandasBackend(), SklearnBackend()]

def get_sql_backend() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [SQLBackend()]
