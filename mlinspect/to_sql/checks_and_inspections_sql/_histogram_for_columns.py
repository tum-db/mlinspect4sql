"""
For the SQL run we want to make the inspections be performed on the DBMS, this code offloads the relevant operations
 and collects the resulting Info from our SQL-Queries to substitute the original "HistogramForColumns" result.
"""
import pandas
from mlinspect.backends._backend import BackendResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.to_sql.py_to_sql_mapping import DfToStringMapping, is_operation_sql_obj
from mlinspect.to_sql.sql_query_container import SQLQueryContainer
from mlinspect.inspections._inspection_input import OperatorType
from mlinspect.to_sql._mode import SQLObjRep


# Keep updates like this? INSPECTION_RESULTS_TO_SUBSTITUTE = [HistogramForColumns]

class SQLHistogramForColumns:
    def __init__(self, dbms_connector, mapping: DfToStringMapping, pipeline_container: SQLQueryContainer,
                 sql_obj):
        """
        Args:
            dbms_connector:
            mapping:
            pipeline_container:
        """
        self.current_hist = {}
        self.dbms_connector = dbms_connector
        self.mapping = mapping
        self.pipeline_container = pipeline_container
        self.sql_obj = sql_obj

    def sql_update_backend_result(self, result, backend_result: BackendResult,
                                  curr_sql_expr_name="",
                                  curr_sql_expr_columns=None):
        """
        Iterate all columns of the pandas object, and for each of them add the n newest/new values.

        Note:
            The sql_obj from wich we want to know the ratio, will be the ones materialized, as they are part of the
            entire query.
        """

        if curr_sql_expr_columns is None:
            curr_sql_expr_columns = []

        # print(curr_sql_expr_name)

        old_dat_node_annotations = backend_result.dag_node_annotation

        # is_unary_operator = len(curr_sql_expr_columns) == 1
        is_input_data_source = not is_operation_sql_obj(curr_sql_expr_name)
        if is_input_data_source:
            # Don't read from the first sql_obj, but from the origin:
            curr_sql_expr_name = curr_sql_expr_name.replace("_ctid", "")  # This is the data_source table.
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

            # print(f"curr_sql_expr_columns: {curr_sql_expr_columns}")

            for sc in sensitive_columns:  # update the values based on current table.
                if sc in curr_sql_expr_columns:

                    query = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                    new_dict[sc], curr_sql_expr_name = self.__get_ratio_count(query=query,
                                                                              curr_sql_expr_name=curr_sql_expr_name,
                                                                              curr_sql_expr_columns=curr_sql_expr_columns,
                                                                              init=is_input_data_source)
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
                query = f"SELECT tb_orig.{sc}, count(*) " \
                        f"FROM {curr_sql_expr_name} tb_curr " \
                        f"JOIN {origin_table} tb_orig " \
                        f"ON tb_curr.{original_ctid}=tb_orig.{original_ctid} " \
                        f"GROUP BY tb_orig.{sc};"

                new_dict[sc], curr_sql_expr_name = self.__get_ratio_count(query=query,
                                                                          curr_sql_expr_name=curr_sql_expr_name,
                                                                          curr_sql_expr_columns=curr_sql_expr_columns,
                                                                          init=is_input_data_source)
                self.current_hist[sc] = new_dict[sc]

            # Update the annotation:
            old_dat_node_annotations[annotation] = new_dict

        self.__set_optional_attribute(result, backend_result)
        return backend_result

    def __get_ratio_count(self, query, curr_sql_expr_name, curr_sql_expr_columns, init=False):
        """
        Note:
            the reason that this function returns a tuple, is that if the view is materialized, it is saved under a
            different name, this means, that we need to update the used name.
        """
        new_name = curr_sql_expr_name
        if self.sql_obj.mode == SQLObjRep.CTE:
            query = self.pipeline_container.get_pipe_without_selection() + "\n" + query
            sc_hist_result = self.dbms_connector.run(query)[0]
        else:

            if self.sql_obj.materialize and not init:
                query_update = self.pipeline_container.get_last_query_materialize(curr_sql_expr_name,
                                                                                  curr_sql_expr_columns)

                if query_update:  # assert not null => materialized query was not added yet!
                    new_view_query, new_name = query_update
                    self.mapping.update_name(curr_sql_expr_name, new_name)
                    # Add to the query to additionally access the materialized sql_obj.:
                    query = new_view_query + "\n" + query.replace(curr_sql_expr_name, new_name)

            sc_hist_result = self.dbms_connector.run(query)[0]

        # self.pipeline_container.write_to_side_query(query, f"ratio_query_{curr_sql_expr_name}")

        return {float("nan") if str(x) == "None" else
                str(x): y for x, y in zip(list(sc_hist_result[0]), list(sc_hist_result[1]))}, new_name

    @staticmethod
    def __set_optional_attribute(result, backend_result):
        # This attribute is set in the "add_dat_node" function!! Add it to our dummy object:
        if result is None:
            return
        if hasattr(backend_result.annotated_dfobject.result_data, "_mlinspect_annotation") and \
                not hasattr(result, "_mlinspect_annotation"):
            result._mlinspect_annotation = backend_result.annotated_dfobject.result_data._mlinspect_annotation
        if hasattr(backend_result.annotated_dfobject.result_data, "_mlinspect_dag_node") and \
                not hasattr(result, "_mlinspect_dag_node"):
            result._mlinspect_dag_node = backend_result.annotated_dfobject.result_data._mlinspect_dag_node

