from mlinspect.inspections._inspection_input import OperatorType
from dataclasses import dataclass
import pandas as pd


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
    """
    data_object: any
    tracking_cols: list
    operation_type: OperatorType
    main_op: bool
    optional_context: list  # Will be added for Projections and binary operations

    def __hash__(self):
        return hash(self.data_object)

    # def is_df(self) -> bool:
    #     return isinstance(self.data_object, pandas.DataFrame)


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

    def get_name(self, df_to_find: pd.DataFrame) -> str:
        return next(n for (n, ti) in self.mapping if ti.data_object is df_to_find)

    def get_name_and_ti(self, df_to_find: pd.DataFrame) -> TableInfo:
        return next(x for x in self.mapping if x[1].data_object is df_to_find)

    def get_ti(self, df_to_find: pd.DataFrame) -> TableInfo:
        return next(x for x in self.mapping if x[1].data_object is df_to_find)

    def contains(self, df_to_find):
        for m in self.mapping:
            if m[1].data_object is df_to_find:
                return True
        return False
