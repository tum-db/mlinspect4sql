"""
Get all available backends
"""
from typing import List
from ._backend import Backend
from ._pandas_backend import PandasBackend
from ._sklearn_backend import SklearnBackend
from ._sql_backend import SQLBackend
from mlinspect.instrumentation._pipeline_executor import singleton


def get_all_backends() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [PandasBackend(), SklearnBackend()] if singleton.to_sql else [SQLBackend()]
