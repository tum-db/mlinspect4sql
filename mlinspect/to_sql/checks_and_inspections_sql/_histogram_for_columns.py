"""
For the SQL run we want to make the inspections be performed on the DBMS, this code offloads the relevant operations
 and collects the resulting Info from our SQL-Queries to substitute the original "HistogramForColumns" result.
"""
import pandas
from mlinspect.backends._backend import BackendResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.to_sql.py_to_sql_mapping import DfToStringMapping
from mlinspect.to_sql.sql_query_container import SQLQueryContainer
from mlinspect.inspections._inspection_input import OperatorType


# Keep updates like this? INSPECTION_RESULTS_TO_SUBSTITUTE = [HistogramForColumns]

class SQLHistogramForColumns:
    def __init__(self, dbms_connector, mapping: DfToStringMapping, pipeline_container: SQLQueryContainer, one_run):
        """
        Args:
            dbms_connector:
            mapping:
            pipeline_container:
            one_run(bool): In case this is set, the "NoBiasIntroduced" inspection will happen once and full in SQL.
                So nothing needs to be done here.
        """
        self.current_hist = {}
        self.dbms_connector = dbms_connector
        self.mapping = mapping
        self.pipeline_container = pipeline_container
        self.one_run = one_run

    def sql_update_backend_result(self, backend_result: BackendResult,
                                  curr_sql_expr_name="",
                                  curr_sql_expr_columns=None):
        """
        Iterate all columns of the pandas object, and for each of them add the n newest/new values.
        """
        # print("\n\n" + "#" * 20)

        if self.one_run:
            return backend_result

        if curr_sql_expr_columns is None:
            curr_sql_expr_columns = []

        # print(curr_sql_expr_name)

        old_dat_node_annotations = backend_result.dag_node_annotation

        # is_unary_operator = len(curr_sql_expr_columns) == 1
        is_input_data_source = "with" not in curr_sql_expr_name
        # is_nary_operator = len(curr_sql_expr_columns) > 1

        for annotation in old_dat_node_annotations.keys():  # iterate in search for HistColumn inspections

            if not isinstance(annotation, HistogramForColumns):
                continue

            new_dict = {}
            sensitive_columns = [f"\"{x}\"" for x in annotation.sensitive_columns]

            # If a sensitive column is not in a table, this can have three reasons:
            # 1) This table has nothing to do with the others => the ctids of the original tables containing our
            #       sensitive columns are not to be found in the tracking_columns of our table
            # 1.1) This table has nothing to do with it, because its another input file.
            # 2) They were removed by a selection => compare original ctid with the ones present here.

            for sc in sensitive_columns:  # update the values based on current table.

                if sc in curr_sql_expr_columns:
                    pipe_code_addition = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                    sc_hist_result = self.dbms_connector.run(
                        self.pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                    new_dict[sc] = {float("nan") if str(x) == "None" else str(x): y for x, y in
                                    zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                    self.current_hist[sc] = new_dict[sc]
                    continue
                elif is_input_data_source:
                    # Here no tracked cols are affected:
                    new_dict[sc] = {}
                    continue

                ti = self.mapping.get_ti_from_name(curr_sql_expr_name)
                origin_table, original_ctid = self.mapping.get_origin_table(sc, ti.tracking_cols)

                if original_ctid not in ti.tracking_cols:
                    new_dict[sc] = self.current_hist[sc].copy()  # Nothing affected
                    continue

                # In the case the ctid is contained, we need to join:
                pipe_code_addition = f"SELECT tb_orig.{sc}, count(*) " \
                                     f"FROM {curr_sql_expr_name} tb_curr " \
                                     f"JOIN {origin_table} tb_orig " \
                                     f"ON tb_curr.{original_ctid}=tb_orig.{original_ctid} " \
                                     f"GROUP BY tb_orig.{sc};"

                sc_hist_result = self.dbms_connector.run(
                    self.pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                new_dict[sc] = {float("nan") if str(x) == "None" else str(x): y for x, y in
                                zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                self.current_hist[sc] = new_dict[sc]

            # Update the annotation:
            old_dat_node_annotations[annotation] = new_dict

        return backend_result


def __get_last_table(self, column_name):
    current_table_sc = ""
    for m in self.mapping.mapping:  # we reverse because of the adding order -> faster match
        table_name = m[0]
        table_info = m[1]
        table = table_info.data_object
        if (isinstance(table, pandas.DataFrame) and column_name in table.columns.values) or \
                (isinstance(table, pandas.Series) and column_name == table.name):
            current_table_sc = table_name
            break
    return current_table_sc

@staticmethod
def __have_match(list1, list2):
    return (set(list1) - set(list2)) == {}
