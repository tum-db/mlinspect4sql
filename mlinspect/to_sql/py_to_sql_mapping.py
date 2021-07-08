from typing import List

from mlinspect.inspections._inspection_input import OperatorType
from dataclasses import dataclass
import pandas


class OpTree:
    """
    This class is devoted to simplify to handle multiple binary operations.
    Args:
        format_string(str): a String the can be formatted with Python's ".format()" function.
        origin(str): is the origin, if we are dealing with a projection. It is empty otherwise.
    """

    def __init__(self, op="", columns=[], tracking_columns=[], table="", left=None, right=None):
        self.op = op
        self.columns = columns
        self.tracking_columns = tracking_columns
        self.table = table
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


@dataclass
class TableInfo:
    """Dataclass for intern SQL-Pandas table/operation mapping.

    Parameters:
        data_object (None): Can be pandas.DataFrame or pandas.Series
        tracking_cols (list): All the columns relevant for tracking in this object
        operation_type (OperatorType): We need this to reduce the amount of checks we need to do, when f.e. looking
            for a problem that only occurs ofter certain operations.
        main_op (bool): Distinguishes between sub- and main-operations. This distinction is nice to have for SQL,
            because it further allows to skip certain checks. | df1[..] = df2 * 3 -> assign is main, * is sub
        origin_context(list): Is empty if the current operation created a new table, otherwise
            (f.e. binop or projection) this ordered list contains the applied operations and sources.
    """
    data_object: any
    tracking_cols: list
    operation_type: OperatorType
    main_op: bool
    origin_context: OpTree  # Will be added for Projections and binary operations

    def __hash__(self):
        return hash(self.data_object)

    def get_non_tracking_cols(self):
        if self.is_se():
            return self.data_object.name
        return list(set(self.data_object.columns.values) - set(self.tracking_cols))

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

    # def update_entry(self, old_entry: (str, pd.DataFrame), new_entry: (str, pd.DataFrame)):
    #     index = self.mapping.index(old_entry)
    #     self.mapping[index] = new_entry

    # def update_name_at_df(self, df, new_name):
    #     old_name = self.get_name(df)
    #     index = self.mapping.index((old_name, df))
    #     self.mapping[index] = (new_name, df)

    # def get_df(self, name_to_find: str) -> pd.DataFrame:
    #     return next(df for (n, df) in self.mapping if n == name_to_find)

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

    def get_db_output_object(self, data_object) -> pandas.DataFrame:
        name = self.get_name(data_object)

    def get_db_output_name(self, name) -> pandas.DataFrame:
        pass  # TODO
