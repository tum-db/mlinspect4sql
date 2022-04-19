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
from typing import Dict


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
        self.current_hist: Dict[str, Dict] = {}
        self.last_hist = None  # as dicts are unordered.
        self.dbms_connector = dbms_connector
        self.mapping = mapping
        self.pipeline_container = pipeline_container
        self.sql_obj = sql_obj

    def sql_update_backend_result(self, result, backend_result: BackendResult,
                                  curr_sql_expr_name="",
                                  curr_sql_expr_columns=None,
                                  keep_previous_res=False,
                                  previous_res_node=None,
                                  operation_type=None,
                                  not_materialize=False):
        """
        Iterate all columns of the pandas object, and for each of them add the n newest/new values.
        Args:
            keep_previous_res(bool): forces to keep previous results
        Note:
            The sql_obj from wich we want to know the ratio, will be the ones materialized, as they are part of the
            entire query.
        """
        if curr_sql_expr_columns is None:
            curr_sql_expr_columns = []

        old_dag_node_annotations = backend_result.dag_node_annotation
        to_check_annotations = [a for a in old_dag_node_annotations.keys() if isinstance(a, HistogramForColumns)]
        if len(to_check_annotations) == 0:
            return backend_result

        assert len(to_check_annotations) == 1
        annotation = to_check_annotations[0]

        # Only check types where histogram changes are possible:
        if operation_type in {OperatorType.TRAIN_LABELS, OperatorType.TRAIN_DATA, OperatorType.PROJECTION} and \
                previous_res_node in self.current_hist.keys():
            self.last_hist = self.current_hist[previous_res_node].copy()
            old_dag_node_annotations[annotation] = self.last_hist  # Nothing affected
            self.__set_optional_attribute(result, backend_result)
            return backend_result
        elif keep_previous_res:
            old_dag_node_annotations[annotation] = self.last_hist  # Nothing affected
            self.__set_optional_attribute(result, backend_result)
            return backend_result

        is_input_data_source = not is_operation_sql_obj(curr_sql_expr_name)
        if is_input_data_source:
            # Don't read from the first sql_obj, but from the origin:
            curr_sql_expr_name = curr_sql_expr_name.replace("_ctid", "")  # This is the data_source table.

        new_dict = {}
        sensitive_columns = [f"\"{x}\"" for x in annotation.sensitive_columns]

        # If a sensitive column is not in a table, this can have three reasons:
        # 1) This table has nothing to do with the others => the ctids of the original tables containing our
        #       sensitive columns are not to be found in the tracking_columns of our table
        # 1.1) This table has nothing to do with it, because its another input file.
        # 2) They were removed by a selection => compare original ctid with the ones present here.
        for sc in sensitive_columns:  # update the values based on current table.
            if sc in curr_sql_expr_columns:

                query = f"SELECT {sc}, count(*) FROM {curr_sql_expr_name} GROUP BY {sc};"
                new_dict[sc], curr_sql_expr_name = self.__get_ratio_count(query=query,
                                                                          curr_sql_expr_name=curr_sql_expr_name,
                                                                          curr_sql_expr_columns=curr_sql_expr_columns,
                                                                          init=is_input_data_source,
                                                                          not_materialize=not_materialize)
                continue
            elif is_input_data_source:
                # Here no tracked cols are affected:
                new_dict[sc] = {}
                continue

            ti = self.mapping.get_ti_from_name(curr_sql_expr_name)
            origin_table, original_ctid = self.mapping.get_origin_table(sc, ti.tracking_cols)

            if original_ctid not in ti.tracking_cols:
                new_dict[sc] = {}  # Nothing affected
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
                                                                      init=is_input_data_source,
                                                                      not_materialize=not_materialize)

        # Update the annotation:
        old_dag_node_annotations[annotation] = new_dict
        self.current_hist[curr_sql_expr_name] = new_dict
        self.last_hist = new_dict
        self.__set_optional_attribute(result, backend_result)
        return backend_result

    def __get_ratio_count(self, query, curr_sql_expr_name, curr_sql_expr_columns, init=False, not_materialize=False):
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

            if self.sql_obj.materialize and not init and not not_materialize:
                query_update = self.pipeline_container.get_last_query_materialize(curr_sql_expr_name,
                                                                                  curr_sql_expr_columns)

                if query_update:  # assert not null => materialized query was not added yet!
                    new_view_query, new_name = query_update
                    self.mapping.update_name(curr_sql_expr_name, new_name)
                    # Add to the query to additionally access the materialized sql_obj.:
                    query = new_view_query + "\n" + query.replace(curr_sql_expr_name, new_name)

            sc_hist_result = self.dbms_connector.run(query)[0]

        sc_hist_result_t = sc_hist_result.transpose()
        sc_hist_result_dict = list(zip(list(sc_hist_result_t[0]), list(sc_hist_result_t[1].astype(int))))
        result = dict((float("nan"), y) if str(x) == "None" else (x, y) for (x, y) in sc_hist_result_dict)
        return result, new_name

    @staticmethod
    def __set_optional_attribute(result, backend_result):
        # This attribute is set in the "add_dat_node" function!! Add it to our dummy object:

        if result is None:
            return

        try:
            if hasattr(backend_result.annotated_dfobject.result_data, "_mlinspect_annotation") and \
                    not hasattr(result, "_mlinspect_annotation"):
                result._mlinspect_annotation = backend_result.annotated_dfobject.result_data._mlinspect_annotation

            if hasattr(backend_result.annotated_dfobject.result_data, "_mlinspect_dag_node") and \
                    not hasattr(result, "_mlinspect_dag_node"):
                result._mlinspect_dag_node = backend_result.annotated_dfobject.result_data._mlinspect_dag_node
        except AttributeError:
            pass
