"""
The NoBiasIntroducedFor check for SQL
"""
import pandas
from mlinspect.monkeypatchingSQL._sql_logic import SQLLogic, mapping


def no_bias_introduced_sql_evaluate_total(sensitive_columns):
    # TO_SQL: ###############################################################################################
    # TODO: maybe remove optional rename
    print(("#" * 10) + f"NoBiasIntroducedFor ({', '.join(sensitive_columns)}):" + ("#" * 10) +
          "\n -> Files can be found under mlinspect/to_sql/generated_code\n\n")
    origin_dict = {}
    current_dict = {}
    for sc in sensitive_columns:
        origin_of_sc = ""
        current_table_sc = ""  # newest table containing the sensitive column
        for m in reversed(mapping.mapping):  # we reverse because of the adding order -> faster match
            table_name = m[0]
            table_info = m[1]
            table = table_info.data_object
            if table_name.split("_")[0] != "with":  # check that name represents an original table (f.e. '.csv')
                if isinstance(table, pandas.Series) and sc == table.name:  # one column .csv
                    origin_of_sc = table_name
                elif isinstance(table,
                                pandas.DataFrame) and sc in table.columns.values:  # TODO: substitute by "contains_col" fucntion in TableInfo!
                    origin_of_sc = table_name
            if (isinstance(table, pandas.DataFrame) and sc in table.columns.values) or \
                    (isinstance(table, pandas.Series) and sc == table.name):
                current_table_sc = table_name
        # TODO: select all relevant operations in mapping and for those, check the ratios before and after!
        # TODO: also add to select!
        assert (origin_of_sc != "")
        origin_dict[sc] = origin_of_sc
        current_dict[sc] = current_table_sc

    SQLLogic.ratio_track(origin_dict, sensitive_columns, current_dict)
    # TO_SQL DONE! ##########################################################################################
