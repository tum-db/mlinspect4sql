from typing import List

from mlinspect.inspections._inspection_input import OperatorType
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
import pandas
from typing import Dict, List

sql_obj_prefix = "block"


def is_operation_sql_obj(name):
    length = len(sql_obj_prefix)
    return len(name) > length and name[:length] == sql_obj_prefix


@dataclass
class ColumnTransformerLevel:
    """
    Contains all info to replicate the operation with parallelization.
    """
    column_map: dict
    from_block: set
    where_block: set
    sql_source: list  # is needed to substitute the old names resulting form the unoptimized translation.


@dataclass
class ColumnTransformerInfo:
    """
    This class will help store all necessary information to optimize sklearn.Pipelines and sklearn.ColumnTransformer.
    Args:
        self: The ColumnTransformer Object
        levels: list of the different level info objects.
        levels_map: maps each code posititon to its assigned level.
        cr_to_col_map: code_reference to target_col map
        target_obj:
        cols_to_drop:
    Note: for more details on levels, see: Bachelor-paper.
    """
    self: any
    levels: List[ColumnTransformerLevel]
    levels_map: Dict[str, int]  # code_reference to level
    cr_to_col_map: Dict[str, List[str]]  # pipeline-operation code_reference to target columns
    target_obj: any
    cols_to_drop: list


class OpTree:
    """
    This class is devoted to help optimize the query by nesting otherwise sequential operations.
    Note:
    """

    def __init__(self, op="{}", non_tracking_columns=None, tracking_columns=None, origin_table="", children=None,
                 is_const=False):
        """
        Args:
            is_const(bool):
        """
        self.op = op
        self.non_tracking_columns = [] if not non_tracking_columns else non_tracking_columns
        self.tracking_columns = [] if not tracking_columns else tracking_columns
        self.origin_table = origin_table
        self.children = children
        self.is_const = is_const

    def is_projection(self):
        return self.op == "{}"

    def resolve(self):
        return self.children is None or self.children == []


@dataclass
class TableInfo:
    """Dataclass for intern SQL-Pandas table/operation mapping.

    Parameters:
        data_object (None): Can be pandas.DataFrame or pandas.Series
        tracking_cols (list): All the columns relevant for tracking in this object#
        non_tracking_cols (list): All the columns, can differ from the ones of the pandas object f.e. with groupby.
        operation_type (OperatorType): We need this to reduce the amount of checks we need to do, when f.e. looking
            for a problem that only occurs ofter certain operations.
        origin_context(list): Is empty if the current operation created a new table, otherwise
            (f.e. binop or projection) this ordered list contains the applied operations and sources.
    """
    data_object: any
    tracking_cols: list
    non_tracking_cols: list
    operation_type: OperatorType
    origin_context: OpTree  # Will be added for Projections and binary operations

    def __hash__(self):
        return hash(self.data_object)

    def is_df(self) -> bool:
        return isinstance(self.data_object, pandas.DataFrame)

    def is_se(self) -> bool:
        return isinstance(self.data_object, pandas.Series)


class DfToStringMapping:
    """Mapping between SQL-with_table names and pandas objects(pandas.DataFrame, pandas.Series)
    To be able to trace the order in which the tables were added, we will use a list and not a dictionary for this
    mapping.
    """
    mapping = []  # contains tuples of form: (*Name*, *DataFrame*)

    def add(self, name: str, ti: TableInfo) -> None:
        """
        Note: As we don't check the variable we are assigning some pandas objects (Series, DataFrame) to, we need to
        append values at the front. This is, because we store the original pandas objects the list and not copies. So
        if these original  objects are altered, it can happen that the dataframe is already in the mapping.
        This would return us some old "with-table" from SQL. Which would be wrong!
        """
        self.mapping = [(name, ti), *self.mapping]  # Quite efficient way to add values at the front.

    def update_pandas_obj(self, old_obj, new_obj):
        _, ti = self.get_ti(old_obj)
        ti.data_object = new_obj

    def update_name(self, old_name, new_name):
        index = [x for (x, ti) in self.mapping].index(old_name)
        self.mapping[index] = new_name, self.mapping[index][1]

    def update_name_df(self, data_object, new_name):
        index = None
        for i, obj in enumerate([ti.data_object for (x, ti) in self.mapping]):
            if obj is data_object:
                index = i
        self.mapping[index] = new_name, self.mapping[index][1]

    def update_ti_df(self, data_object, new_ti):
        index = None
        for i, obj in enumerate([ti.data_object for (x, ti) in self.mapping]):
            if obj is data_object:
                index = i
        self.mapping[index] = self.mapping[index][0], new_ti

    def get_ti_from_name(self, name_to_find: str) -> TableInfo:
        return next(ti for (n, ti) in self.mapping if n == name_to_find)

    def get_name(self, df_to_find) -> str:
        """
        Args:
            df_to_find(pandas.DataFrame or pandas.Series)
        """
        return next(n for (n, ti) in self.mapping if ti.data_object is df_to_find)

    def get_name_and_ti(self, df_to_find) -> TableInfo:
        """
        Args:
            df_to_find(pandas.DataFrame or pandas.Series)
        """
        return next(x for x in self.mapping if x[1].data_object is df_to_find)

    def get_ti(self, df_to_find) -> TableInfo:
        """
        Args:
            df_to_find(pandas.DataFrame or pandas.Series)
        """
        return next(x for x in self.mapping if x[1].data_object is df_to_find)

    def contains(self, df_to_find):
        for m in self.mapping:
            if m[1].data_object is df_to_find:
                return True
        return False

    def is_projection(self, name: str) -> bool:
        ti = self.get_ti_from_name(name)
        op_tree = ti.origin_context
        return bool(op_tree) and op_tree.op == ""

    def get_origin_table(self, column_name: str, current_ctid_s: list) -> (str, str):
        """
        Returns:
             (<origin table>, <ctid>), from which the column originated. If not fround (None, None).
        """
        origin_tup = [(x, ti) for (x, ti) in self.mapping if
                      not is_operation_sql_obj(x) and ti.tracking_cols[0] in current_ctid_s]
        if len(origin_tup) == 0:
            return None, None  # in case there ist no origin -> f.e. col added
        # In case multiple ctid were present, find the origin of the passed column:
        for orig_t, orig_ti in origin_tup:
            if column_name in orig_ti.non_tracking_cols:
                return orig_t, orig_ti.tracking_cols[0]
        return None, None  # The input combination is wrong!

