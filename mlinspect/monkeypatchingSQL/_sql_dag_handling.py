"""
For the SQL run we want tu use an alternative DAG only with the Info resulting from our SQL-Queries.
"""
import pandas
from mlinspect.backends._backend import AnnotatedDfObject, BackendResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.instrumentation._pipeline_executor import singleton

INSPECTION_RESULTS_TO_SUBSTITUTE = [HistogramForColumns]

class SQLHistogramUpdater:
    def __init__(self):
        self.hist = {}

    def sql_update_backend_result(self, mapping, pipeline_container, backend_result: BackendResult,
                                  curr_sql_expr_name="", curr_sql_expr_columns=None):
        """
        Iterate all columns of the pandas object, and for each of them add the n newest/new values.
        """
        if curr_sql_expr_columns is None:
            curr_sql_expr_columns = []

        # This flag is to get the result, even if the current operation is not executed in sql. f.e.: not yet supported
        search_last_table = (curr_sql_expr_name == "")

        # print("#" * 20)
        old_dat_node_annotations = backend_result.dag_node_annotation

        # is_unary_operator = list(curr_sql_expr_columns) == 1
        # is_input_data_source= not "with" in curr_sql_expr_name
        # is_nary_operator = list(curr_sql_expr_columns) > 1
        # if is_unary_operator:
        # elif is_input_data_source:
        # elif is_nary_operator:
        # else:
        for annotation in old_dat_node_annotations.keys():  # iterate in search for HistColumn inspections
            if isinstance(annotation, HistogramForColumns):  # Currently this is the only one we need to handle:
                new_dict = {}
                sensitive_columns = annotation.sensitive_columns

                # if not self.__have_match(curr_sql_expr_columns, sensitive_columns):
                #     # Here no tracked cols are affected => return old histogram
                #     old_dat_node_annotations[annotation] = self.current_hist
                #     return backend_result

                for sc in sensitive_columns:  # update the values based on current table.

                    if search_last_table:
                        curr_sql_expr_name = self.__get_last_table(sc)
                        curr_sql_expr_columns = mapping.get_columns_no_track(curr_sql_expr_name)
                    elif sc not in curr_sql_expr_columns:
                        new_dict[sc] = old_dat_node_annotations[annotation][sc] # TODO: this is the case, that leads to problems !!!
                        # to_pop.append(annotation)
                        continue

                    pipe_code_addition = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                    # print("Executing: " + pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)
                    sc_hist_result = singleton.dbms_connector.run(
                        pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                    new_dict[sc] = {str(x): y for x, y in zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                    # print(sorted(new_dict[sc].items()))
                # Update the annotation:
                old_dat_node_annotations[annotation] = new_dict
        return backend_result

    # def sql_calculate_ratio(self, backend_result: BackendResult, curr_sql_expr_name="", curr_sql_expr_columns=""):
    #     """
    #     Inserts a new node into the DAG
    #     """
    #     if curr_sql_expr_name == "" and curr_sql_expr_columns == "":
    #         return backend_result
    #
    #     old_dat_node_annotations = backend_result.dag_node_annotation
    #     for annotation in old_dat_node_annotations.keys():
    #         if isinstance(annotation, HistogramForColumns):  # Currently this is the only one we need to handle:
    #             sensitive_columns = annotation.sensitive_columns
    #             origin_dict = {}
    #             current_dict = {}
    #             for sc in sensitive_columns:
    #                 origin_of_sc = __get_origin_table(column_name=sc, mapping=mapping)
    #                 assert (origin_of_sc != "")
    #                 origin_dict[sc] = origin_of_sc
    #                 current_dict[sc] = curr_sql_expr_name
    #
    #             last_cte_names, sql_code = SQLLogic.ratio_track(origin_dict, sensitive_columns, current_dict,
    #                                                             only_code=True)  # Remember: returns two dicts.
    #             for sc in sensitive_columns:
    #                 pipe_code_addition = f"{sql_code[sc]}\nSELECT * FROM {last_cte_names[sc]};"  # select age_group, count(*) from patients_1 group by age_group ;
    #                 sc_ratio_result = singleton.dbms_connector.run(
    #                     pipeline_container.get_pipe_without_selection() + ",\n" + pipe_code_addition)
    #         else:
    #             new_histogram_for_columns = annotation
    #
    #     return backend_result

    @staticmethod
    def __get_origin_table(mapping, column_name):
        origin_of_sc = ""
        for m in reversed(mapping.mapping):  # we reverse because of the adding order -> faster match
            table_name = m[0]
            table_info = m[1]
            table = table_info.data_object
            if table_name.split("_")[0] != "with":  # check that name represents an original table (f.e. '.csv')
                if isinstance(table, pandas.Series) and column_name == table.name:  # one column .csv
                    origin_of_sc = table_name
                elif isinstance(table,
                                pandas.DataFrame) and column_name in table.columns.values:  # TODO: substitute by "contains_col" fucntion in TableInfo!
                    origin_of_sc = table_name
        return origin_of_sc

    @staticmethod
    def __get_last_table(mapping, column_name):
        current_table_sc = ""
        for m in mapping.mapping:  # we reverse because of the adding order -> faster match
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
