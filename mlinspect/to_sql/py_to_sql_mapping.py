from typing import List

from mlinspect.inspections._inspection_input import OperatorType
from dataclasses import dataclass
import pandas


class OpTree:
    """
    This class is devoted to help optimize the query by un-nesting row-wise operations.
    """

    def __init__(self, op="", non_tracking_columns=None, tracking_columns=None, origin_table="", left=None, right=None,
                 is_const=False):
        self.op = op
        self.non_tracking_columns = [] if not non_tracking_columns else non_tracking_columns
        self.tracking_columns = [] if not tracking_columns else tracking_columns
        self.origin_table = origin_table
        self.left = left
        self.right = right
        self.is_const = is_const

    def is_projection(self):
        return self.op == ""

    def resolve(self):
        return self.left is None and self.right is None


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

    def get_columns_no_track(self, name: str) -> list:
        ti = self.get_ti_from_name(name)
        return ti.non_tracking_cols

    def get_columns_track(self, name: str) -> list:
        ti = self.get_ti_from_name(name)
        return ti.tracking_cols

    def get_ctid_of_col(self, column):
        """
        Returns:
            the name of the ctid column of the table from which the passed column came.
        """
        for orig_t, orig_ti in [(x, ti) for (x, ti) in self.mapping if "with" not in x]:
            if column in orig_ti.non_tracking_cols:
                # This is the original table form where "column" came from.
                ctid_l = orig_ti.tracking_cols
                assert len(ctid_l) == 1
                return orig_t, ctid_l[0]
        return None, None

    def is_projection(self, name: str) -> bool:
        ti = self.get_ti_from_name(name)
        op_tree = ti.origin_context
        return bool(op_tree) and op_tree.op == ""

    def get_origin_table(self, column_name: str) -> str:
        for orig_t, orig_ti in [(x, ti) for (x, ti) in self.mapping if "with" not in x]:
            table = orig_ti.data_object
            if isinstance(table, pandas.Series) and column_name == table.name:  # one column .csv
                return orig_t
            elif isinstance(table,
                            pandas.DataFrame) and column_name in table.columns.values:  # TODO: substitute by "contains_col" fucntion in TableInfo!
                return orig_t
        raise ValueError

    def get_latest_name_cols(self):
        entry = self.mapping[0]
        return entry[0], entry[1].non_tracking_cols
