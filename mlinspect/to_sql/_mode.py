import dataclasses
from enum import Enum


class SQLObjRep(Enum):
    """
    The different representation of the SQL "objects".
    """
    CTE = "CTE"
    VIEW = "VIEW"


@dataclasses.dataclass
class SQLMode:
    """
    Basic information about the how to handle the SQL "objects".
    """
    mode: SQLObjRep
    materialize: bool
