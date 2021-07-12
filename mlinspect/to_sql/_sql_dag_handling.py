"""
For the SQL run we want to make the inspections be performend on the DBMS, this code offloads the relevent operations
 and collect the resulting Info from our SQL-Queries.
"""
import pandas
from mlinspect.backends._backend import BackendResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns

INSPECTION_RESULTS_TO_SUBSTITUTE = [HistogramForColumns]


class SQLHistogramUpdater:
    def __init__(self, dbms_connector):
        self.current_hist = {}
        self.dbms_connector = dbms_connector

    def sql_update_backend_result(self, mapping, pipeline_container, backend_result: BackendResult,
                                  curr_sql_expr_name="",
                                  curr_sql_expr_columns=None):
        """
        Iterate all columns of the pandas object, and for each of them add the n newest/new values.
        """
        # print("\n\n" + "#" * 20)

        if curr_sql_expr_columns is None:
            curr_sql_expr_columns = []

        # print(curr_sql_expr_name)

        old_dat_node_annotations = backend_result.dag_node_annotation

        is_unary_operator = len(curr_sql_expr_columns) == 1
        is_input_data_source = "with" not in curr_sql_expr_name
        is_nary_operator = len(curr_sql_expr_columns) > 1

        for annotation in old_dat_node_annotations.keys():  # iterate in search for HistColumn inspections

            if not isinstance(annotation, HistogramForColumns):
                continue

            new_dict = {}
            sensitive_columns = annotation.sensitive_columns

            # TODO: Assert this is not necessary:
            # test = backend_result.annotated_dfobject.result_annotation.columns.values[3]
            # backend_result.annotated_dfobject.result_annotation.drop(str(HistogramForColumns(sensitive_columns)),
            #                                                          axis=1)

            # If a sensitive column is not in a table, this can have three reasons:
            # 1) This table has nothing to do with the others => the ctids of the original tables containing our
            #       sensitive columns are not to be found in the tracking_columns of our table
            # 1.1) This table has nothing to do with it, because its another input file.
            # 2) They were removed by a selection => compare original ctid with the ones present here.

            for sc in sensitive_columns:  # update the values based on current table.

                if is_input_data_source:

                    if sc in curr_sql_expr_columns:

                        pipe_code_addition = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                        sc_hist_result = self.dbms_connector.run(
                            pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                        new_dict[sc] = {float("nan") if str(x) == "None" else str(x): y for x, y in
                                        zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                        self.current_hist[sc] = new_dict[sc]
                        # print(new_dict[sc])

                    else:  # Here no tracked cols are affected:
                        new_dict[sc] = {}

                elif is_unary_operator or is_nary_operator:

                    if sc not in curr_sql_expr_columns:  # TODO: add check of ctid.
                        optional_original_table, optional_ctid = mapping.get_ctid_of_col(sc)

                        if bool(optional_ctid) and optional_ctid not in mapping.get_columns_track(curr_sql_expr_name):
                            new_dict[sc] = self.current_hist[sc].copy()  # Nothing affected
                            continue
                        else:  # The attribute still exists TODO
                            if mapping.is_projection(curr_sql_expr_name):
                                new_dict[sc] = self.current_hist[sc].copy()  # Nothing affected
                                continue
                            else:
                                pipe_code_addition = f"SELECT {sc}, count(*) " \
                                                     f"FROM {curr_sql_expr_name} tb_curr " \
                                                     f"JOIN {optional_original_table} tb_orig " \
                                                     f"ON tb_curr.{optional_ctid}=tb_orig.{optional_ctid} " \
                                                     f"GROUP BY {sc};"
                                sc_hist_result = self.dbms_connector.run(
                                    pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                                new_dict[sc] = {float("nan") if str(x) == "None" else str(x): y for x, y in
                                                zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                                self.current_hist[sc] = new_dict[sc]
                                # print(new_dict[sc])

                    else:
                        pipe_code_addition = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                        sc_hist_result = self.dbms_connector.run(
                            pipeline_container.get_pipe_without_selection() + "\n" + pipe_code_addition)[0]
                        new_dict[sc] = {float("nan") if str(x) == "None" else str(x): y for x, y in
                                        zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}
                        self.current_hist[sc] = new_dict[sc]


                else:
                    raise NotImplementedError

            # Update the annotation:
            old_dat_node_annotations[annotation] = new_dict
            # print(new_dict)

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
    #                 origin_of_sc = self.__get_origin_table(column_name=sc)
    #                 assert (origin_of_sc != "")
    #                 origin_dict[sc] = origin_of_sc
    #                 current_dict[sc] = curr_sql_expr_name
    #
    #             # last_cte_names, sql_code = self.SQLLogic.ratio_track(origin_dict, sensitive_columns, current_dict,
    #             #                                                 only_code=True)  # Remember: returns two dicts.
    #         #     for sc in sensitive_columns:
    #         #         pipe_code_addition = f"{sql_code[sc]}\nSELECT * FROM {last_cte_names[sc]};"  # select age_group, count(*) from patients_1 group by age_group ;
    #         #         sc_ratio_result = singleton.dbms_connector.run(
    #         #             pipeline_container.get_pipe_without_selection() + ",\n" + pipe_code_addition)
    #         # else:
    #         #     new_histogram_for_columns = annotation
    #
    #     return backend_result

    def __get_origin_table(self, mapping, column_name):
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

    def __get_last_table(self, mapping, column_name):
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
