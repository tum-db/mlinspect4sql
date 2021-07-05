"""
Monkey patching for pandas
"""
import copy
import os

import gorilla
import pandas
import pathlib

from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.backends._pandas_backend import PandasBackend
from ._sql_logic import SQLLogic, mapping
from mlinspect.to_sql.csv_sql_handling import CreateTablesFromCSVs
from mlinspect.to_sql.py_to_sql_mapping import OpTree
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node, \
    get_dag_node_for_id, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._patch_sklearn import call_info_singleton

sql_logic = SQLLogic()


@gorilla.patches(pandas)
class PandasPatchingSQL:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('read_csv')
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args, **kwargs):
        """ Patch for ('pandas.io.parsers', 'read_csv') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(pandas, 'read_csv')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.io.parsers', 'read_csv')
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])

            # Add the restriction to only load 10 rows of the csv and add the Dataframe to the wrapper.
            kwargs["nrows"] = 10
            result = original(*args, **kwargs)

            # TO_SQL: ###############################################################################################
            sep = ","
            na_values = "?"
            header = 1
            path_to_csv = ""

            # Check relevant kwargs:
            if "filepath_or_buffer" in kwargs:  # could be: str, (path object or file-like object)
                path_to_csv = kwargs["filepath_or_buffer"]
            elif "sep" in kwargs:
                sep = kwargs["sep"]
            elif "na_values" in kwargs:
                na_values = kwargs["na_values"]
            elif "header" in kwargs:  # int that states the how many rows resemble the columns header
                header = kwargs["header"]
                if not isinstance(header, int):
                    raise NotImplementedError

            # Check args that weren't set by kwargs:
            if path_to_csv == "" and len(args) >= 1:  # Just evaluate the first arg to get the csv.
                path_to_csv = args[0]
            if len(args) >= 2:
                sep = args[1]
            if len(args) >= 3:
                raise NotImplementedError

            table_name = pathlib.Path(path_to_csv).stem + "_" + str(sql_logic.get_unique_id())

            # we need to add the ct_id columns to the original table:
            tracking_column = f"{table_name}_ctid"
            result[tracking_column] = "placeholder"

            sql_code = CreateTablesFromCSVs(path_to_csv).get_sql_code(table_name=table_name, null_symbol=na_values,
                                                                      delimiter=sep,
                                                                      header=(1 == header))

            print(sql_code + "\n")
            sql_logic.write_to_init_file(sql_code)

            # We need to instantly add the ctid to the tables:
            sql_code = f"SELECT *, ctid AS {table_name}_ctid\n" \
                       f"FROM {table_name}"

            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result,
                                      tracking_cols=[tracking_column],
                                      operation_type=OperatorType.DATA_SOURCE,
                                      main_op=False,
                                      with_block_name=table_name)
            sql_logic.write_to_pipe_query(sql_code)
            # TO_SQL DONE! ##########################################################################################

            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            description = args[0].split(os.path.sep)[-1]
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [], backend_result)
            return result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(pandas.DataFrame)
class DataFramePatchingSQL:
    """ Patches for 'pandas.core.frame' """

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'DataFrame') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'DataFrame')
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            original(self, *args, **kwargs)
            result = self
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)

            columns = list(self.columns)  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [], backend_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('dropna')
    @gorilla.settings(allow_hit=True)
    def patched_dropna(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'dropna') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'dropna')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'dropna')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            if result is None:
                raise NotImplementedError("TODO: Support inplace dropna")
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("dropna", list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__getitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__getitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', '__getitem__')
            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            if isinstance(args[0], str):  # Projection to Series
                columns = [args[0]]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails("to {}".format(columns), columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            elif isinstance(args[0], list) and isinstance(args[0][0], str):  # Projection to DF
                columns = args[0]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails("to {}".format(columns), columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            elif isinstance(args[0], pandas.Series):  # Selection
                operator_context = OperatorContext(OperatorType.SELECTION, function_info)
                columns = list(self.columns)  # pylint: disable=no-member
                if optional_source_code:
                    description = "Select by Series: {}".format(optional_source_code)
                else:
                    description = "Select by Series"
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails(description, columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            else:
                raise NotImplementedError()
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            # TO_SQL: ###############################################################################################
            tb1 = self
            tb1_name, ti = mapping.get_name_and_ti(tb1)
            source = args[0]
            columns_tracking = sql_logic.get_tracking_cols_raw(self.columns.values)
            if isinstance(source, str):  # Projection to Series
                operation_type = OperatorType.PROJECTION
                origin_context = OpTree(op="", table=tb1_name, columns=[source], tracking_columns=columns_tracking)
            elif isinstance(source, list) and isinstance(args[0][0], str):  # Projection to DF
                operation_type = OperatorType.PROJECTION
                origin_context = OpTree(op="", table=tb1_name, columns=source, tracking_columns=columns_tracking)
            elif isinstance(source, pandas.Series):  # Selection
                operation_type = OperatorType.SELECTION
                origin_context = None
            else:
                raise NotImplementedError()
            if not mapping.contains(result):  # Only create SQL-Table if it doesn't already exist.

                if isinstance(source, pandas.Series):
                    name, ti = mapping.get_name_and_ti(source)
                    tables, column, tracking_columns = sql_logic.get_origin_series(ti.origin_context)
                    if len(tables) == 1:
                        sql_code = f"SELECT * \n" \
                                   f"FROM {tables[0]} \n" \
                                   f"WHERE {column}"
                    else:
                        raise NotImplementedError  # TODO: column wise
                    # sql_code = f"SELECT tb1.{', tb1.'.join(tb1.columns.values)}\n" \
                    #            f"FROM {sql_logic.create_indexed_table(tb1_name)} as tb1,  " \
                    #            f"{sql_logic.create_indexed_table(mapping.get_name(source))} as tb2\n" \
                    #            f"WHERE tb2.{source.name} and tb1.row_number = tb2.row_number"

                else:

                    if isinstance(source, list):  # Add tracking cols, if result is pandas.DataFrame:
                        for c in sql_logic.get_tracking_cols_raw(self.columns.values):
                            result[c] = "placeholder"
                    else:
                        source = [source]
                    sql_code = f"SELECT {', '.join(source)}{sql_logic.get_tracking_cols(self.columns.values)}\n" \
                               f"FROM {tb1_name}"

            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result,
                                      tracking_cols=columns_tracking,
                                      operation_type=operation_type,
                                      main_op=False,
                                      origin_context=origin_context)
            sql_logic.write_to_pipe_query(sql_code)

            # TO_SQL DONE! ##########################################################################################
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__setitem__')
    @gorilla.settings(allow_hit=True)
    def patched__setitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__setitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__setitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.frame', '__setitem__')
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            if isinstance(args[0], str):
                input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
                input_infos = copy.deepcopy(input_infos)
                result = original(self, *args, **kwargs)
                backend_result = PandasBackend.after_call(operator_context,
                                                          input_infos,
                                                          self)
                columns = list(self.columns)  # pylint: disable=no-member
                description = "modifies {}".format([args[0]])
            else:
                raise NotImplementedError("TODO: Handling __setitem__ for key type {}".format(type(args[0])))
            # TO_SQL: ###############################################################################################
            # There are two options to handle this kind of arithmetic operations of pandas.DataFrames / pandas.series:
            # we can catch the entire operations here and create one with a single "with_statement", or create one with
            # for each operation and put these with together. The second option is preferred as is its only downside is,
            # that it is more verbose, but on the other side its simpler, more elegant and dosn't require to go over
            # the statements twice.
            tb1 = self  # Table where the new column is set, or the old one overwritten.
            new_name = args[0]
            tb2 = args[1]  # the target

            if len(args) != 2:
                raise NotImplementedError

            if not isinstance(tb2, pandas.Series) and not isinstance(tb2, pandas.DataFrame):
                # we are assigning some constant.
                sql_code = f"SELECT *, {tb2} AS {new_name}\n" \
                           f"FROM {mapping.get_name(tb1)}"
            else:
                tb2_name, tb2_ti = mapping.get_name_and_ti(tb2)
                tb1_name, tb1_ti = mapping.get_name_and_ti(tb1)
                tables, column, tracking_columns = sql_logic.get_origin_series(tb2_ti.origin_context)

                if len(tables) == 1 and tb1_name == tables[0]:
                    # We can avoid an Window function:
                    sql_code = f"SELECT *, {column} AS {new_name}\n" \
                               f"FROM {tb1_name}"
                else:
                    # TODO: Row-wise
                    raise NotImplementedError
                    # final_tracking_columns = list(set(tracking_columns) | set(tb1_ti.tracking_cols))
                    # sql_code = f"SELECT tb1.{', tb1.'.join([x for x in tb1.columns.values if x != new_name])}, " \
                    #            f"tb2.{tb2.name} AS {new_name}\n" \
                    #            f"FROM {sql_logic.create_indexed_table(mapping.get_name(tb1))} AS tb1, " \
                    #            f"{sql_logic.create_indexed_table(mapping.get_name(tb2))} AS tb2 \n" \
                    #            f"WHERE tb1.row_number = tb2.row_number"

            # Here we need to take "self", as the result of __setitem__ will be None.
            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result=self,
                                      tracking_cols=sql_logic.get_tracking_cols_raw(self.columns.values),
                                      operation_type=OperatorType.PROJECTION_MODIFY,
                                      main_op=True,
                                      origin_context=None)  # TODO
            sql_logic.write_to_pipe_query(sql_code)

            # TO_SQL DONE! ##########################################################################################
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
            assert hasattr(self, "_mlinspect_annotation")
            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('replace')
    @gorilla.settings(allow_hit=True)
    def patched_replace(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'replace') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'replace')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'replace')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            if isinstance(args[0], dict):
                raise NotImplementedError("TODO: Add support for replace with dicts")
            description = "Replace '{}' with '{}'".format(args[0], args[1])
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('merge')
    @gorilla.settings(allow_hit=True)
    def patched_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'merge') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'merge')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'merge')

            input_info_a = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            input_info_b = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info_a.annotated_dfobject,
                                                                       input_info_b.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, input_infos[1].result_data, *args[1:], **kwargs)

            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data

            # TO_SQL: ###############################################################################################

            # Attention: If two columns are merged and column names overlap between the two merge partner tables the
            # columns are renamed .._x and .._y => if this affects the ctid columns we can remove one, as they are the
            # same.
            x_ctid_columns = [x for x in result.columns.values if "ctid_x" in x]
            result = result.drop(x_ctid_columns, axis=1)
            for y in [y for y in result.columns.values if "ctid_y" in y]:
                result = result.rename(columns={y: y.split("_y")[0]})

            tb1 = input_info_a.annotated_dfobject.result_data
            tb2 = input_info_b.annotated_dfobject.result_data

            merge_type = "inner"  # default
            merge_column = ""  # default: cross-product

            if "how" in kwargs:
                merge_type = kwargs["how"]
            elif len(args) >= 2:
                merge_type = args[1]
            if "on" in kwargs:
                merge_column = kwargs["on"]
            elif len(args) >= 3:
                merge_column = args[2]

            if isinstance(merge_column, list):
                merge_column = merge_column[0]

            tb1_name = mapping.get_name(tb1)
            tb2_name = mapping.get_name(tb2)
            tb1_columns = tb1.columns.values
            tb2_columns = [x for x in tb2.columns.values if x not in tb1_columns]  # remove duplicates!
            # Attention: we need to select all columns, just using * can result in a doubled column!
            if merge_column == "":  # Cross product:
                sql_code = f"SELECT * \n" \
                           f"FROM {tb1_name}, {tb2_name}"
            else:
                sql_code = f"SELECT tb1.{', tb1.'.join(tb1_columns)}, tb2.{', tb2.'.join(tb2_columns)}\n" \
                           f"FROM {tb1_name} tb1 \n" \
                           f"{merge_type.upper()} JOIN {tb2_name} tb2" \
                           f" ON tb1.{merge_column} = tb2.{merge_column}"

            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result,
                                      tracking_cols=sql_logic.get_tracking_cols_raw(self.columns.values),
                                      operation_type=OperatorType.JOIN, main_op=True)
            sql_logic.write_to_pipe_query(sql_code)
            # TO_SQL DONE! ##########################################################################################
            description = "on '{}'".format(kwargs['on'])
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('groupby')
    @gorilla.settings(allow_hit=True)
    def patched_groupby(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'groupby') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'groupby')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'groupby')
            # We ignore groupbys, we only do something with aggs

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            result = original(self, *args, **kwargs)
            # TO_SQL: ###############################################################################################
            # Here we don't need to do anything, as the groupby alone, is irrelevent. Only together with the applied
            # operation, we need to act.
            # TO_SQL DONE! ##########################################################################################
            result._mlinspect_dag_node = input_info.dag_node.node_id  # pylint: disable=protected-access

            return result

        return execute_patched_func_no_op_id(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.groupby.generic.DataFrameGroupBy)
class DataFrameGroupByPatchingSQL:
    """ Patches for 'pandas.core.groupby.generic' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('agg')
    @gorilla.settings(allow_hit=True)
    def patched_agg(self, *args, **kwargs):
        """ Patch for ('pandas.core.groupby.generic', 'agg') """
        original = gorilla.get_original_attribute(pandas.core.groupby.generic.DataFrameGroupBy, 'agg')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.groupby.generic', 'agg')
            if not hasattr(self, '_mlinspect_dag_node'):
                raise NotImplementedError("TODO: Support agg if groupby happened in external code")
            input_dag_node = get_dag_node_for_id(self._mlinspect_dag_node)  # pylint: disable=no-member

            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)

            input_infos = PandasBackend.before_call(operator_context, [])
            result = original(self, *args, **kwargs)
            # TO_SQL: ###############################################################################################
            tb1 = self.obj
            groupby_columns = self.grouper.names
            agg_params = [x[0] for x in kwargs.values()]
            new_col_names = list(kwargs.keys())  # The name of the new column containing the aggregation
            agg_funcs = [x[1] for x in kwargs.values()]

            if not groupby_columns:  # if groupby_columns is empty we are dealing with a mapping or function.
                raise NotImplementedError

            # map pandas aggregation function to SQL (the the ones that differ):
            for i, f in enumerate(agg_funcs):
                if f == "mean":
                    agg_funcs[i] = "avg"
                elif f == "std":
                    agg_funcs[i] = "stddev"
                elif f == "var":
                    agg_funcs[i] = "variance"

            selection_string = []
            for p, n, f in zip(agg_params, new_col_names, agg_funcs):
                selection_string.append(f"{f.upper()}({p}) AS {n}")

            groupby_columns = ', '.join(groupby_columns)
            sql_code = f"SELECT {groupby_columns}, {', '.join(selection_string)} \n" \
                       f"FROM {mapping.get_name(tb1)}\n" \
                       f"GROUP BY {groupby_columns}"

            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result,
                                      tracking_cols=sql_logic.get_tracking_cols_raw(result.columns.values),
                                      operation_type=OperatorType.GROUP_BY_AGG,
                                      main_op=True)
            sql_logic.write_to_pipe_query(sql_code)
            # TO_SQL DONE! ##########################################################################################

            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            if len(args) > 0:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, args)
            else:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, kwargs)
            columns = [result.index.name] + list(result.columns)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_dag_node], backend_result)
            new_return_value = backend_result.annotated_dfobject.result_data

            return new_return_value

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.indexing._LocIndexer)  # pylint: disable=protected-access
class LocIndexerPatchingSQL:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods, too-many-locals

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(
            pandas.core.indexing._LocIndexer, '__getitem__')  # pylint: disable=protected-access

        if call_info_singleton.column_transformer_active:
            op_id = singleton.get_next_op_id()
            caller_filename = call_info_singleton.transformer_filename
            lineno = call_info_singleton.transformer_lineno
            function_info = call_info_singleton.transformer_function_info
            optional_code_reference = call_info_singleton.transformer_optional_code_reference
            optional_source_code = call_info_singleton.transformer_optional_source_code

            if isinstance(args[0], tuple) and not args[0][0].start and not args[0][0].stop \
                    and isinstance(args[0][1], list) and isinstance(args[0][1][0], str):
                # Projection to one or multiple columns, return value is df
                columns = args[0][1]
            else:
                raise NotImplementedError()

            operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
            input_info = get_input_info(self.obj, caller_filename,  # pylint: disable=no-member
                                        lineno, function_info, optional_code_reference, optional_source_code)
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(self, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("to {}".format(columns), columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            result = original(self, *args, **kwargs)

        return result


@gorilla.patches(pandas.Series)
class SeriesPatchingSQL:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(pandas.Series, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.series', 'Series')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            original(self, *args, **kwargs)
            result = self
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            if self.name:  # pylint: disable=no-member
                columns = list(self.name)  # pylint: disable=no-member
            else:
                columns = ["_1"]
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [], backend_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('isin')
    @gorilla.settings(allow_hit=True)
    def patched_isin(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'isin') """
        original = gorilla.get_original_attribute(pandas.Series, 'isin')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            values = args[0]
            if not isinstance(values, list):
                raise NotImplementedError

            result = self.isin(values)
            if isinstance(values[0], str):
                where_in_block = "\'" + "\', \'".join(values) + "\'"
            elif isinstance(values[0], list):
                where_in_block = ", ".join(values)
            else:
                raise NotImplementedError

            column = self.name
            name, ti = mapping.get_name_and_ti(self)
            new_syntax_tree = OpTree(op="IN", left=ti.origin_context, right=f"({where_in_block})")
            tables, column, tracking_columns = sql_logic.get_origin_series(new_syntax_tree)
            if len(tables) == 1 and ti.origin_context.op == "":  # TODO: add correct ENUM -> improve syntax tree.
                sql_code = f"SELECT {column}, {', '.join(tracking_columns)}\n" \
                           f"FROM {tables[0]}"
            else:
                # TODO: row wise
                raise NotImplementedError

            sql_code = sql_logic.finish_sql_call(sql_code, lineno, result,
                                      tracking_cols=ti.tracking_cols,
                                      operation_type=OperatorType.SELECTION,
                                      main_op=True,
                                      origin_context=new_syntax_tree)
            sql_logic.write_to_pipe_query(sql_code)
            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    ################
    # ARITHMETIC:
    ################
    @gorilla.name('__add__')
    @gorilla.settings(allow_hit=True)
    def patched__add__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__add__') """
        original = gorilla.get_original_attribute(pandas.Series, '__add__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("+", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__radd__')
    @gorilla.settings(allow_hit=True)
    def patched__radd__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__radd__') """
        original = gorilla.get_original_attribute(pandas.Series, '__radd__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("+", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__mul__')
    @gorilla.settings(allow_hit=True)
    def patched__mul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__mul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__mul__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("*", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rmul__')
    @gorilla.settings(allow_hit=True)
    def patched__rmul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rmul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rmul__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("*", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sum__')
    @gorilla.settings(allow_hit=True)
    def patched__sum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sum__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("+", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsum__')
    @gorilla.settings(allow_hit=True)
    def patched__rsum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsum__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("+", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sub__')
    @gorilla.settings(allow_hit=True)
    def patched__sub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sub__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("-", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsub__')
    @gorilla.settings(allow_hit=True)
    def patched__rsub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsub__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("-", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__div__')
    @gorilla.settings(allow_hit=True)
    def patched__div__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__div__') """
        original = gorilla.get_original_attribute(pandas.Series, '__div__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("/", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rdiv__')
    @gorilla.settings(allow_hit=True)
    def patched__rdiv__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rdiv__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rdiv__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("/", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    ################
    # COMPARISONS:
    ################
    @gorilla.name('__eq__')
    @gorilla.settings(allow_hit=True)
    def patched__eq__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__eq__') """
        original = gorilla.get_original_attribute(pandas.Series, '__eq__')
        left, right = SeriesPatchingSQL.__help_set_right_compare(self, args)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series("=", self.eq(right), self, right, lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__gt__')
    @gorilla.settings(allow_hit=True)
    def patched__gt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__gt__') """
        original = gorilla.get_original_attribute(pandas.Series, '__gt__')
        left, right = SeriesPatchingSQL.__help_set_right_compare(self, args)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series(">", self.gt(right), self, right, lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__ge__')
    @gorilla.settings(allow_hit=True)
    def patched__ge__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__ge__') """
        original = gorilla.get_original_attribute(pandas.Series, '__ge__')
        left, right = SeriesPatchingSQL.__help_set_right_compare(self, args)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series(">=", self.ge(right), self, right, lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__lt__')
    @gorilla.settings(allow_hit=True)
    def patched__lt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__lt__') """
        original = gorilla.get_original_attribute(pandas.Series, '__lt__')
        left, right = SeriesPatchingSQL.__help_set_right_compare(self, args)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series("<", self.lt(right), self, right, lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__le__')
    @gorilla.settings(allow_hit=True)
    def patched__le__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__le__') """
        original = gorilla.get_original_attribute(pandas.Series, '__le__')
        left, right = SeriesPatchingSQL.__help_set_right_compare(self, args)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series("<=", self.le(right), self, right, lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    ################
    # LOGICAL OPS:
    ################
    @gorilla.name('__and__')
    @gorilla.settings(allow_hit=True)
    def patched__and__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__and__') """
        original = gorilla.get_original_attribute(pandas.Series, '__and__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("AND", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__or__')
    @gorilla.settings(allow_hit=True)
    def patched__or__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__or__') """
        original = gorilla.get_original_attribute(pandas.Series, '__or__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("OR", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__not__')
    @gorilla.settings(allow_hit=True)
    def patched__not__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__not__') """
        original = gorilla.get_original_attribute(pandas.Series, '__not__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("NOT", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @staticmethod
    def __op_call_helper(op, left, args, original, rop):
        """
        To reduce code repetition.
        Args:
            op(str): operator as string as used in SQL.
            left: the "left" operand in case rop is False
            args:
            original: original function
            rop(bool): True if the non-pandas operator is on the left => we need to swap left and right, because
                the const will always be in the argument, but for f.e for a subtraction this would be wrong otherwise.
        """
        assert (len(args) == 1)
        if rop:
            right = left
            left = args[0]
            result = original(self=right, other=left)
        else:
            right = args[0]
            result = original(self=left, other=right)

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            return sql_logic.handle_operation_series(op, result, left=left, right=right, lineno=lineno)

        return execute_inspections

    @staticmethod
    def __help_set_right_compare(left, args):
        """
        Helps to transform the value to which is compared to a pandas.Series if needed.
        For example "ds1 < 3" => "ds1 < ds_of_3"
        """
        assert (len(args) == 1)
        right = args[0]
        if not isinstance(right, pandas.Series):  # => need transformation, to avoid self call
            right = pandas.Series([right]).repeat(left.size)  #
        return left, right

    # @staticmethod
    # def __fix_index_for_op(left: pandas.Series, right: pandas.Series):
    #     if isinstance(left, pandas.Series):
    #         left = left.reset_index()[left.name]
    #     if isinstance(right, pandas.Series):
    #         right = right.reset_index()[right.name]
    #     return left, right
