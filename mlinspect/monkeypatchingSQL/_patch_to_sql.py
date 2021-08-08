"""
Monkey patching for pandas
"""
import copy
import os

import gorilla
import pandas
import pathlib

from mlinspect.to_sql.py_to_sql_mapping import OpTree, PipelineLevel
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.monkeypatching._monkey_patching_utils import get_dag_node_for_id
from mlinspect.monkeypatching._patch_sklearn import call_info_singleton
from mlinspect.to_sql._mode import SQLObjRep

import gorilla
import numpy
import pandas
from sklearn import preprocessing, compose, tree, impute, linear_model, model_selection, pipeline
from tensorflow.keras.wrappers import scikit_learn as keras_sklearn_external  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers import scikit_learn as keras_sklearn_internal  # pylint: disable=no-name-in-module

from mlinspect.backends._backend import BackendResult
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, CodeReference
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._patch_numpy import MlinspectNdarray

pandas.options.mode.chained_assignment = None  # default='warn'


# Because gorillas is not able to provide the original function of comparisons e.g. (==, <, ...). It actually
# return "<method-wrapper '__eq__' of type object at 0x21b4970>" instead
# of a "<function pandas.core.arraylike.OpsMixin.__eq__(self, other)>" which is useless for our purposes, as
# it can't be called, we need to backup the original pandas comparison functions.
# more info: https://stackoverflow.com/questions/10401935/python-method-wrapper-type

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
            # Add the restriction to only load 10 rows of the csv and add the Dataframe to the wrapper.
            kwargs["nrows"] = 1
            result = original(*args, **kwargs)
            # TO_SQL: ###############################################################################################
            sep = ","
            na_values = ["?"]
            header = 1
            path_to_csv = ""
            index_col = -1

            # Check relevant kwargs:
            if "filepath_or_buffer" in kwargs:  # could be: str, (path object or file-like object)
                path_to_csv = kwargs["filepath_or_buffer"]
            elif "sep" in kwargs:
                sep = kwargs["sep"]
            elif "na_values" in kwargs:
                na_values = kwargs["na_values"]
                na_values = [na_values] if isinstance(na_values, str) else na_values
            elif "header" in kwargs:  # int that states the how many rows resemble the columns header
                header = kwargs["header"]
                if not isinstance(header, int):
                    raise NotImplementedError
            elif "index_col" in kwargs:
                index_col = kwargs["index_col"]

            # Check args that weren't set by kwargs:
            if path_to_csv == "" and len(args) >= 1:  # Just evaluate the first arg to get the csv.
                path_to_csv = args[0]
            if len(args) >= 2:
                sep = args[1]
            if len(args) >= 3:
                raise NotImplementedError

            table_name = pathlib.Path(path_to_csv).stem + f"_mlinid{op_id}"

            # we need to add the ct_id columns to the original table:
            tracking_column = f"{table_name}_ctid"
            # result[tracking_column] = "placeholder"

            col_names, sql_code = singleton.dbms_connector.add_csv(path_to_csv, table_name, null_symbols=na_values,
                                                                   delimiter=sep, header=(header == 1),
                                                                   add_mlinspect_serial=True, index_col=index_col)

            # print(sql_code + "\n")
            singleton.pipeline_container.write_to_init_file(sql_code)

            # We need to instantly add the ctid to the tables:
            sql_code = f"SELECT *, ctid AS {table_name}_ctid\n" \
                       f"FROM {table_name}"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
                                                                     tracking_cols=[tracking_column],
                                                                     operation_type=OperatorType.DATA_SOURCE,
                                                                     cte_name=table_name + "_ctid")

            # print(sql_code + "\n")
            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, cols_to_keep=col_names)

            result.reset_index(drop=True, inplace=True)
            # TO_SQL DONE! ##########################################################################################

            backend_result = PandasBackend.after_call(operator_context, [], result)
            # backend_result = BackendResult(AnnotatedDfObject(None, None), None)

            description = args[0].split(os.path.sep)[-1]
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=col_names))

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

            # TO_SQL: ###############################################################################################
            # Having a pandas.DataFrame code source, is not yet supported.

            # TO_SQL DONE! ##########################################################################################

            columns = list(self.columns)  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [],
                         singleton.update_hist.sql_update_backend_result(result, backend_result))

            return result

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            result = backend_result.annotated_dfobject.result_data

            # TO_SQL: ###############################################################################################
            # Cant use "DELETE", as not table, nee to do a selection.

            name, ti = singleton.mapping.get_name_and_ti(self)
            columns_without_tracking = ti.non_tracking_cols

            sql_code = f"SELECT *\n" \
                       f"FROM {name} \n" \
                       f"WHERE NOT ({' OR '.join([f'{x} IS NULL' for x in columns_without_tracking])})"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
                                                                     tracking_cols=ti.tracking_cols,
                                                                     operation_type=OperatorType.SELECTION)

            # print(sql_code + "\n")
            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, ti.non_tracking_cols)

            # TO_SQL DONE! ##########################################################################################

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("dropna", list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=columns_without_tracking
                                                                         ))

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            result = backend_result.annotated_dfobject.result_data
            # TO_SQL: ###############################################################################################
            tb1 = self
            tb1_name, tb1_ti = singleton.mapping.get_name_and_ti(tb1)
            source = args[0]
            columns_tracking = tb1_ti.tracking_cols
            if isinstance(source, str):  # Projection to Series
                operation_type = OperatorType.PROJECTION
                columns_without_tracking = [f"\"{source}\""]
                origin_context = OpTree(op="{}", non_tracking_columns=columns_without_tracking,
                                        tracking_columns=columns_tracking, origin_table=tb1_name)
            elif isinstance(source, list) and isinstance(args[0][0], str):  # Projection to DF
                operation_type = OperatorType.PROJECTION
                columns_without_tracking = [f"\"{x}\"" for x in source]
                origin_context = OpTree(op="{}", non_tracking_columns=columns_without_tracking,
                                        tracking_columns=columns_tracking, origin_table=tb1_name)
            elif isinstance(source, pandas.Series):  # Selection
                operation_type = OperatorType.SELECTION
                origin_context = None
                columns_without_tracking = tb1_ti.non_tracking_cols
            else:
                raise NotImplementedError()

            if isinstance(source, pandas.Series):
                name, ti = singleton.mapping.get_name_and_ti(source)
                tables, column, tracking_columns = singleton.sql_logic.resolve_to_origin(ti.origin_context)
                if len(tables) == 1:
                    sql_code = f"SELECT * \n" \
                               f"FROM {tables[0]} \n" \
                               f"WHERE {column}"
                else:
                    # row-wise
                    raise NotImplementedError

            else:
                sql_code = f"SELECT {', '.join(columns_without_tracking + columns_tracking)}\n" \
                           f"FROM {tb1_name}"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
                                                                     tracking_cols=columns_tracking,
                                                                     operation_type=operation_type,
                                                                     origin_context=origin_context)

            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, columns_without_tracking)
            # TO_SQL DONE! ##########################################################################################
            add_dag_node(dag_node, [input_info.dag_node],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=columns_without_tracking)
                         )

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
                backend_result = PandasBackend.after_call(operator_context, input_infos, self)
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
            tb1_name, tb1_ti = singleton.mapping.get_name_and_ti(tb1)

            new_name = args[0]
            tb2 = args[1]  # the target

            if len(args) != 2:
                raise NotImplementedError

            if not isinstance(tb2, pandas.Series) and not isinstance(tb2, pandas.DataFrame):
                # we are assigning some constant.
                sql_code = f"SELECT *, {tb2} AS {new_name}\n" \
                           f"FROM {singleton.mapping.get_name(tb1)}"
            else:
                tb2_name, tb2_ti = singleton.mapping.get_name_and_ti(tb2)
                tables, column, tracking_columns = singleton.sql_logic.resolve_to_origin(tb2_ti.origin_context)

                if len(tables) == 1 and tb1_name == tables[0]:
                    # We can avoid an Window function:
                    sql_code = f"SELECT *, {column} AS {new_name}\n" \
                               f"FROM {tb1_name}"
                else:
                    # TODO: Row-wise
                    print("Row-wise operations should be avoided due to performance deficits. "
                          "If this is intended \"row_wise=True\" should be passed when calling"
                          "\"execute_in_sql\"")
                    raise NotImplementedError()
                    # final_tracking_columns = list(set(tracking_columns) | set(tb1_ti.tracking_cols))
                    # sql_code = f"SELECT tb1.{', tb1.'.join([x for x in tb1.columns.values if x != new_name])}, " \
                    #            f"tb2.{tb2.name} AS {new_name}\n" \
                    #            f"FROM {singleton.sql_logic.create_indexed_table(singleton.mapping.get_name(tb1))} AS tb1, " \
                    #            f"{singleton.sql_logic.create_indexed_table(singleton.mapping.get_name(tb2))} AS tb2 \n" \
                    #            f"WHERE tb1.row_number = tb2.row_number"

            # Here we need to take "self", as the result of __setitem__ will be None.
            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result=self,
                                                                     tracking_cols=tb1_ti.tracking_cols,
                                                                     operation_type=OperatorType.PROJECTION_MODIFY,
                                                                     origin_context=None)
            # print(sql_code + "\n")
            columns_without_tracking = tb1_ti.non_tracking_cols
            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, columns_without_tracking)

            # TO_SQL DONE! ##########################################################################################
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=columns_without_tracking)
                         )

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            result = backend_result.annotated_dfobject.result_data

            # TO_SQL: ###############################################################################################
            # Here we need to replace all possible occurrences of the specific args[0] with the args[1].
            # ONLY WHOLE WORD!
            if len(args) != 2:
                raise NotImplementedError

            to_replace = args[0]  # From this
            value = args[1]  # to this
            name, ti = singleton.mapping.get_name_and_ti(self)
            string_columns = [x for x in ti.non_tracking_cols if self[x.split("\"")[1]].dtype.name == "object"]
            if len(string_columns) != 0:
                non_string_columns = ti.tracking_cols + list(set(ti.non_tracking_cols) - set(string_columns))

                select_list = []
                for s in string_columns:
                    select_list.append(f"REGEXP_REPLACE({s},\'^{to_replace}$\',\'{value}\') AS {s}")

                sql_code = f"SELECT {', '.join(non_string_columns)}{',' if len(non_string_columns) > 0 else ''} " \
                           f"{', '.join(select_list)}\n" \
                           f"FROM {name}"
                columns_without_tracking = ti.non_tracking_cols
                cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result=result,
                                                                         tracking_cols=ti.tracking_cols,
                                                                         operation_type=OperatorType.PROJECTION_MODIFY,
                                                                         origin_context=None)
                # print(sql_code + "\n")
                singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, columns_without_tracking)
            # TO_SQL DONE! ##########################################################################################

            if isinstance(args[0], dict):
                raise NotImplementedError("TODO: Add support for replace with dicts")
            description = "Replace '{}' with '{}'".format(args[0], args[1])
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=columns_without_tracking
                                                                         ))

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            result = backend_result.annotated_dfobject.result_data
            # TO_SQL: ###############################################################################################

            # Attention: If two columns are merged and column names overlap between the two merge partner tables the
            # columns are renamed .._x and .._y => if this affects the ctid columns we can remove one, as they are the
            # same.

            tb1 = self
            tb2 = args[0]

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

            tb1_name, tb1_ti = singleton.mapping.get_name_and_ti(tb1)
            tb2_name, tb2_ti = singleton.mapping.get_name_and_ti(tb2)
            tb1_columns = list(tb1_ti.non_tracking_cols) + tb1_ti.tracking_cols
            tb2_columns = [x for x in list(tb2_ti.non_tracking_cols) + tb2_ti.tracking_cols if
                           x not in tb1_columns]  # remove duplicates!
            # Attention: we need to select all columns, just using * can result in a doubled column!
            if merge_column == "":  # Cross product:
                raise NotImplementedError  # TODO -> change * to all columns in default pandas order.
                # sql_code = f"SELECT * \n" \
                #            f"FROM {tb1_name}, {tb2_name}"
            else:
                sql_code = f"SELECT tb1.{', tb1.'.join(tb1_columns)}, tb2.{', tb2.'.join(tb2_columns)}\n" \
                           f"FROM {tb1_name} tb1 \n" \
                           f"{merge_type.upper()} JOIN {tb2_name} tb2" \
                           f" ON tb1.\"{merge_column}\" = tb2.\"{merge_column}\""
            tracking_cols = list( set(tb1_ti.tracking_cols + tb2_ti.tracking_cols))
            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
                                                                     tracking_cols=tracking_cols,
                                                                     operation_type=OperatorType.JOIN)

            # print(sql_code + "\n")

            columns_without_tracking = tb1_ti.non_tracking_cols + tb2_ti.non_tracking_cols
            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, columns_without_tracking)

            # TO_SQL DONE! ##########################################################################################

            description = "on '{}'".format(kwargs['on'])
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            backend_result = singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                             curr_sql_expr_name=cte_name,
                                                                             curr_sql_expr_columns=columns_without_tracking)
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

    @staticmethod
    def __get_datatype(pandas_object, column):
        pass


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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            new_return_value = backend_result.annotated_dfobject.result_data
            # TO_SQL: ###############################################################################################
            tb1 = self.obj
            tb1_name, tb1_ti = singleton.mapping.get_name_and_ti(tb1)

            groupby_columns = self.grouper.names
            agg_params = [x[0] for x in kwargs.values()]
            new_col_names = list(kwargs.keys())  # The name of the new column containing the aggregation
            agg_funcs = [x[1] for x in kwargs.values()]

            if not groupby_columns:  # if groupby_columns is empty we are dealing with a function.
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
                selection_string.append(f"{f.upper()}(\"{p}\") AS \"{n}\"")

            if len(groupby_columns) == 1:
                groupby_string = f"\"{groupby_columns[0]}\""
                non_tracking_cols_addition = [groupby_string]
            else:
                non_tracking_cols_addition = [f"\"{x}\"" for x in groupby_columns]
                groupby_string = ', '.join(non_tracking_cols_addition)

            sql_code = f"SELECT {groupby_string}, {', '.join(selection_string)} \n" \
                       f"FROM {tb1_name}\n" \
                       f"GROUP BY {groupby_string}"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, new_return_value,
                                                                     tracking_cols=[],
                                                                     non_tracking_cols_addition=non_tracking_cols_addition,
                                                                     operation_type=OperatorType.GROUP_BY_AGG)

            # print(sql_code + "\n")

            columns_without_tracking = [f"\"{x}\"" for x in list(new_return_value.columns.values) + groupby_columns]
            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, columns_without_tracking)
            # TO_SQL DONE! ##########################################################################################
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
            add_dag_node(dag_node, [input_dag_node],
                         singleton.update_hist.sql_update_backend_result(new_return_value, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=columns_without_tracking))

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)
            result = backend_result.annotated_dfobject.result_data
            # TO_SQL: ###############################################################################################
            name, ti = singleton.mapping.get_name_and_ti(self.obj)
            # TO_SQL DONE! ##########################################################################################
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("to {}".format(columns), columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            add_dag_node(dag_node, [input_info.dag_node],
                         singleton.update_hist.sql_update_backend_result(result, backend_result,
                                                                         curr_sql_expr_name=name,
                                                                         curr_sql_expr_columns=ti.non_tracking_cols))

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
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)

            if self.name:  # pylint: disable=no-member
                columns = list(self.name)  # pylint: disable=no-member
            else:
                columns = ["_1"]
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            add_dag_node(dag_node, [],
                         singleton.update_hist.sql_update_backend_result(result, backend_result))

            return result

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

            name, ti = singleton.mapping.get_name_and_ti(self)
            new_syntax_tree = OpTree(op="{} IN {}",
                                     children=[ti.origin_context, OpTree("{}", [f"({where_in_block})"], is_const=True)])
            tables, column, tracking_columns = singleton.sql_logic.resolve_to_origin(new_syntax_tree)
            if len(tables) == 1 and ti.origin_context.is_projection():  # Not row wise
                sql_code = f"SELECT {column}, {', '.join(tracking_columns)}\n" \
                           f"FROM {tables[0]}"
            else:
                # rare row-wise case
                raise NotImplementedError

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
                                                                     tracking_cols=ti.tracking_cols,
                                                                     operation_type=OperatorType.SELECTION,
                                                                     origin_context=new_syntax_tree)

            # print(sql_code + "\n")

            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, cols_to_keep=[self.name])
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

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} {} {} {} + {} {} {} {}", self, args, original,
                                                                 rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__radd__')
    @gorilla.settings(allow_hit=True)
    def patched__radd__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__radd__') """
        original = gorilla.get_original_attribute(pandas.Series, '__radd__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} + {}", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__mul__')
    @gorilla.settings(allow_hit=True)
    def patched__mul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__mul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__mul__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} * {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rmul__')
    @gorilla.settings(allow_hit=True)
    def patched__rmul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rmul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rmul__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} * {}", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sum__')
    @gorilla.settings(allow_hit=True)
    def patched__sum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sum__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} + {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsum__')
    @gorilla.settings(allow_hit=True)
    def patched__rsum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsum__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} + {}", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sub__')
    @gorilla.settings(allow_hit=True)
    def patched__sub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sub__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} - {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsub__')
    @gorilla.settings(allow_hit=True)
    def patched__rsub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsub__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} - {}", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__div__')
    @gorilla.settings(allow_hit=True)
    def patched__div__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__div__') """
        original = gorilla.get_original_attribute(pandas.Series, '__div__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} / {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rdiv__')
    @gorilla.settings(allow_hit=True)
    def patched__rdiv__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rdiv__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rdiv__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} / {}", self, args, original, rop=True)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    ################
    # COMPARISONS:
    ################
    @gorilla.name('__ne__')
    @gorilla.settings(allow_hit=True)
    def patched__ne__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__ne__') """
        # ordering is always consistant -> <column> != <const/column>
        op = "{} != {}"
        # if args[0] == "N/A":
        #     op = "{} IS {}"
        #     args = ["NOT NULL"]
        execute_inspections = SeriesPatchingSQL.__op_call_helper(op, self, args, singleton.backup_ne, rop=False)
        return execute_patched_func(singleton.backup_ne, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__eq__')
    @gorilla.settings(allow_hit=True)
    def patched__eq__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__eq__') """
        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} == {}", self, args, singleton.backup_eq, rop=False)
        return execute_patched_func(singleton.backup_eq, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__gt__')
    @gorilla.settings(allow_hit=True)
    def patched__gt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__gt__') """
        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} > {}", self, args, singleton.backup_gt, rop=False)
        return execute_patched_func(singleton.backup_gt, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__ge__')
    @gorilla.settings(allow_hit=True)
    def patched__ge__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__ge__') """
        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} >= {}", self, args, singleton.backup_ge, rop=False)
        return execute_patched_func(singleton.backup_ge, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__lt__')
    @gorilla.settings(allow_hit=True)
    def patched__lt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__lt__') """
        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} < {}", self, args, singleton.backup_lt, rop=False)
        return execute_patched_func(singleton.backup_lt, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__le__')
    @gorilla.settings(allow_hit=True)
    def patched__le__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__le__') """
        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} <= {}", self, args, singleton.backup_le, rop=False)
        return execute_patched_func(singleton.backup_le, execute_inspections, self, *args, **kwargs)

    ################
    # LOGICAL OPS:
    ################
    @gorilla.name('__and__')
    @gorilla.settings(allow_hit=True)
    def patched__and__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__and__') """
        original = gorilla.get_original_attribute(pandas.Series, '__and__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} AND {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__or__')
    @gorilla.settings(allow_hit=True)
    def patched__or__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__or__') """
        original = gorilla.get_original_attribute(pandas.Series, '__or__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("{} OR {}", self, args, original, rop=False)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__not__')
    @gorilla.settings(allow_hit=True)
    def patched__not__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__not__') """
        original = gorilla.get_original_attribute(pandas.Series, '__not__')

        execute_inspections = SeriesPatchingSQL.__op_call_helper("NOT {}", self, args, original, rop=False)

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
            return singleton.sql_logic.handle_operation_series(op, result, left=left, right=right, line_id=op_id)

        return execute_inspections


# SKLEARN:

class SklearnCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods

    transformer_filename: str or None = None
    transformer_lineno: int or None = None
    transformer_function_info: FunctionInfo or None = None
    transformer_optional_code_reference: CodeReference or None = None
    transformer_optional_source_code: str or None = None
    column_transformer_active: bool = False


call_info_singleton = SklearnCallInfo()


@gorilla.patches(compose.ColumnTransformer)
class SklearnComposePatching:
    """ Patches for sklearn ColumnTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self,
                        transformers, *,
                        remainder='drop',
                        sparse_threshold=0.3,
                        n_jobs=None,
                        transformer_weights=None,
                        verbose=False):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=attribute-defined-outside-init
            original(self, transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                     transformer_weights=transformer_weights, verbose=verbose)

            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo('sklearn.compose._column_transformer',
                                                                     'ColumnTransformer')
        call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
        call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(compose.ColumnTransformer, 'fit_transform')

        # TO_SQL: ###############################################################################################
        # When calling original(self, *args, **kwargs) the overwritten Pipeline-functions (like SimpleImpute)
        # will be called with the relevant slice of the table.
        op_id = singleton.get_next_op_id()
        name, ti = singleton.mapping.get_name_and_ti(args[0])

        op_to_col_map = {}
        for name, op_obj, target_cols in self.transformers:
            target_cols = [f"\"{x}\"" for x in target_cols]
            if isinstance(op_obj, pipeline.Pipeline):
                for _, step in op_obj.steps:  # need to take pipeline apart
                    op_to_col_map[step.mlinspect_optional_code_reference] = target_cols
            else:
                op_to_col_map[op_obj.mlinspect_optional_code_reference] = target_cols

        # HANDLE "drop" case:
        cols_to_drop = []
        if self.remainder == "drop":
            cols_to_drop = list(set(ti.non_tracking_cols) - {x for list in op_to_col_map.values() for x in list})

        # We will need pass the input of this function to the subclass, to be able to achieve the mapping.
        global column_transformer_share
        column_transformer_share = PipelineLevel(op_to_col_map, target_obj=args[0], cols_to_drop=cols_to_drop)

        print()

        # TODO: Optimize for just running:
        result = original(self, *args, **kwargs)
        #
        # sql_code = "WITH" if singleton.sql_obj.mode == SQLObjRep.VIEW else ""
        # select_block_cols = []
        # for col in ti.non_tracking_cols:
        #     if col in column_map.keys():
        #         select_block_cols.append(column_map[col])
        #     else:
        #         select_block_cols.append(col)
        #
        # sql_code += select_block_col_trans + \
        #             f"SELECT {', '.join(select_block_cols + ti.tracking_cols)}\n" \
        #             f"FROM {name}"
        # cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, result,
        #                                                          tracking_cols=ti.tracking_cols,
        #                                                          non_tracking_cols_addition=[],
        #                                                          operation_type=OperatorType.TRANSFORMER,
        #                                                          cte_name=f"column_transformed_mlinid{op_id}")
        #
        # singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        # TO_SQL DONE! ##########################################################################################

        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name('_hstack')
    @gorilla.settings(allow_hit=True)
    def patched_hstack(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument, unused-argument, too-many-locals
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '_hstack')

        if not call_info_singleton.column_transformer_active:
            return original(self, *args, **kwargs)

        input_tuple = args[0]
        function_info = FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')
        input_infos = []
        for input_df_obj in input_tuple:
            input_info = get_input_info(input_df_obj, self.mlinspect_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
            input_infos.append(input_info)

        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        input_annotated_dfs = [input_info.annotated_dfobject for input_info in input_infos]
        backend_input_infos = SklearnBackend.before_call(operator_context, input_annotated_dfs)
        # No input_infos copy needed because it's only a selection and the rows not being removed don't change
        result = original(self, *args, **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   backend_input_infos,
                                                   result)
        result = backend_result.annotated_dfobject.result_data

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails(None, ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, backend_result)

        return result


@gorilla.patches(impute.SimpleImputer)
class SklearnSimpleImputerPatching:
    """ Patches for sklearn SimpleImputer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, missing_values=numpy.nan, strategy="mean",
                        fill_value=None, verbose=0, copy=True, add_indicator=False,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.impute._base', 'SimpleImputer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(impute.SimpleImputer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, missing_values=missing_values, strategy=strategy, fill_value=fill_value, verbose=verbose,
                     copy=copy, add_indicator=add_indicator)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, missing_values=missing_values,
                                             strategy=strategy, fill_value=fill_value, verbose=verbose, copy=copy,
                                             add_indicator=add_indicator)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'fit_transform')
        function_info = FunctionInfo('sklearn.impute._base', 'SimpleImputer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context, input_infos, result)

        new_return_value = backend_result.annotated_dfobject.result_data
        if isinstance(input_infos[0].result_data, pandas.DataFrame):
            columns = list(input_infos[0].result_data.columns)
        else:
            columns = ['array']

        op_id = singleton.get_next_op_id()
        # TO_SQL: ###############################################################################################

        name, ti, target_cols, res_for_map, cols_to_drop = find_target(self, args[0], result)
        tracking_cols = ti.tracking_cols
        non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols) - set(cols_to_drop))

        if self.strategy == "most_frequent":
            select_block = []
            count_block = ""
            for col in target_cols:
                count_table, count_code = singleton.sql_logic.column_count(name, col)

                count_block += count_code + ", \n"

                select_block.append(f"\tCOALESCE({col}, "
                                    f"(SELECT {col} "
                                    f"FROM {count_table} "
                                    f"WHERE count = (SELECT MAX(count) FROM {count_table}))) AS {col}")

            # Add the counting columns to the pipeline_container
            singleton.pipeline_container.add_statement_to_pipe(count_table, count_block, None)

            # TODO: -> in cte in view
            # if singleton.sql_obj.mode == SQLObjRep.VIEW:
            #     singleton.dbms_connector.run(count_block)
            # TODO -> return to ColTrans for optimization!
            # if col_trans:
            #     # Store everything non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols))necessary to afterwards optimize if necessary!
            #     global select_block_col_trans, column_map
            #     select_block_col_trans.append(select_block)
            #     column_map = {**column_map, **dict(zip(cols_to_impute, select_block))}

            select_block = ',\n'.join(select_block) + ",\n\t" + ", ".join(non_tracking_cols_rest + tracking_cols)

            sql_code = "WITH " + count_block[:-3] + "\n" \
                                                    f"SELECT \n{select_block} \n" \
                                                    f"FROM {name}"
        elif self.strategy == "mean":
            select_block = []
            for col in target_cols:
                select_block.append(f"\tCOALESCE({col}, (SELECT AVG({col}) FROM {name})) AS {col}")

            # TODO: -> in cte in view
            # if singleton.sql_obj.mode == SQLObjRep.VIEW:
            #     singleton.dbms_connector.run(count_block)
            # TODO -> return to ColTrans for optimization!
            # if col_trans:
            #     # Store everything non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols))necessary to afterwards optimize if necessary!
            #     global select_block_col_trans, column_map
            #     select_block_col_trans.append(select_block)
            #     column_map = {**column_map, **dict(zip(cols_to_impute, select_block))}

            select_block = ',\n'.join(select_block) + ",\n\t" + ", ".join(non_tracking_cols_rest + tracking_cols)
            sql_code = f"SELECT \n{select_block} \n" \
                       f"FROM {name}"
        else:
            raise NotImplementedError

        all_remaining_cols = non_tracking_cols_rest + target_cols

        cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
                                                                 tracking_cols=tracking_cols,
                                                                 non_tracking_cols=all_remaining_cols,
                                                                 operation_type=OperatorType.TRANSFORMER,
                                                                 cte_name=f"block_impute_mlinid{op_id}",
                                                                 update_name_in_map=not (new_return_value is res_for_map))

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        # TO_SQL DONE! ##########################################################################################
        dag_node = DagNode(op_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Simple Imputer", columns),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))

        backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=all_remaining_cols)

        add_dag_node(dag_node, [input_info.dag_node], backend_result)

        return new_return_value


@gorilla.patches(preprocessing.OneHotEncoder)
class SklearnOneHotEncoderPatching:
    """ Patches for sklearn OneHotEncoder"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, categories='auto', drop=None, sparse=True,
                        dtype=numpy.float64, handle_unknown='error',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._encoders', 'OneHotEncoder') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, categories=categories, drop=drop, sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, categories=categories, drop=drop,
                                             sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data

        op_id = singleton.get_next_op_id()
        # TO_SQL: ###############################################################################################

        name, ti, target_cols, res_for_map, cols_to_drop = find_target(self, args[0], result)
        non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols) - set(cols_to_drop))
        tracking_cols = ti.tracking_cols

        oh_block = ""
        select_block = []
        from_block = ""
        where_block = ""
        for col in target_cols:
            oh_table, oh_code = singleton.sql_logic.column_one_hot_encoding(name, col)

            oh_block += oh_code + ", \n"

            select_block.append(f"\t{col[:-1]}_one_hot\" AS {col}")
            where_block += f"\t{name}.{col} = {oh_table}.{col} AND \n"
            from_block += f"{oh_table}, "

        # Add the one hot columns to the pipeline_container
        singleton.pipeline_container.add_statement_to_pipe(oh_table, oh_block, None)

        # TODO: -> in cte in view
        # if singleton.sql_obj.mode == SQLObjRep.VIEW:
        #     singleton.dbms_connector.run(count_block)
        # TODO -> return to ColTrans for optimization!
        # if col_trans:
        #     # Store everything necessary to afterwards optimize if necessary!
        #     global select_block_col_trans, column_map
        #     select_block_col_trans.append(select_block)
        #     column_map = {**column_map, **dict(zip(cols_to_impute, select_block))}

        select_block = ",\n".join(select_block) + ",\n\t" + ", ".join(non_tracking_cols_rest + tracking_cols)

        sql_code = f"WITH {oh_block[:-3]}\n" \
                   f"SELECT \n{select_block} \n" \
                   f"FROM {name}, {from_block[:-2]}\n" \
                   f"WHERE\n {where_block[:-5]}"

        all_remaining_cols = non_tracking_cols_rest + target_cols

        cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
                                                                 tracking_cols=tracking_cols,
                                                                 non_tracking_cols=all_remaining_cols,
                                                                 operation_type=OperatorType.TRANSFORMER,
                                                                 cte_name=f"onehot_mlinid{op_id}",
                                                                 update_name_in_map=not (new_return_value is res_for_map))

        # print(sql_code + "\n")
        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        # TO_SQL DONE! ##########################################################################################

        dag_node = DagNode(op_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("One-Hot Encoder", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))

        backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=all_remaining_cols)

        add_dag_node(dag_node, [input_info.dag_node], backend_result)

        return new_return_value


@gorilla.patches(preprocessing.StandardScaler)
class SklearnStandardScalerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, copy=True, with_mean=True, with_std=True,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._data', 'StandardScaler') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, copy=copy, with_mean=with_mean, with_std=with_std)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, copy=copy, with_mean=with_mean,
                                             with_std=with_std)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.StandardScaler', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        assert isinstance(new_return_value, MlinspectNdarray)

        op_id = singleton.get_next_op_id()
        # TO_SQL: ###############################################################################################

        name, ti, target_cols, res_for_map, cols_to_drop = find_target(self, args[0], result)
        tracking_cols = ti.tracking_cols
        non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols) - set(cols_to_drop))


        if not (self.with_mean and self.with_std):
            raise NotImplementedError

        select_block = []
        for col in target_cols:
            select_block.append(f"\t(({col} - (SELECT AVG({col}) FROM {name})) / (SELECT STDDEV({col}) FROM {name}))"
                                f" AS {col}")

        # TODO: -> in cte in view
        # if singleton.sql_obj.mode == SQLObjRep.VIEW:
        #     singleton.dbms_connector.run(count_block)
        # TODO -> return to ColTrans for optimization!
        # if col_trans:
        #     # Store everything non_tracking_cols_rest = list(set(ti.non_tracking_cols) - set(target_cols))necessary to afterwards optimize if necessary!
        #     global select_block_col_trans, column_map
        #     select_block_col_trans.append(select_block)
        #     column_map = {**column_map, **dict(zip(cols_to_impute, select_block))}

        select_block = ',\n'.join(select_block) + ",\n\t" + ", ".join(non_tracking_cols_rest + tracking_cols)

        sql_code = f"SELECT \n{select_block} \n" \
                   f"FROM {name}"

        all_remaining_cols = non_tracking_cols_rest + target_cols

        cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
                                                                 tracking_cols=tracking_cols,
                                                                 non_tracking_cols=all_remaining_cols,
                                                                 operation_type=OperatorType.TRANSFORMER,
                                                                 cte_name=f"block_stdscaler_mlinid{op_id}",
                                                                 update_name_in_map=not (new_return_value is res_for_map))

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        # TO_SQL DONE! ##########################################################################################

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Standard Scaler", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))

        backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                         curr_sql_expr_name=cte_name,
                                                                         curr_sql_expr_columns=all_remaining_cols)

        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(preprocessing)
class SklearnPreprocessingPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('label_binarize')
    @gorilla.settings(allow_hit=True)
    def patched_label_binarize(*args, **kwargs):
        """ Patch for ('sklearn.preprocessing._label', 'label_binarize') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing, 'label_binarize')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.preprocessing._label', 'label_binarize')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result)
            new_return_value = backend_result.annotated_dfobject.result_data

            op_id = singleton.get_next_op_id()

            # TO_SQL: ###############################################################################################
            target_sql_obj = args[0]
            name, ti = singleton.mapping.get_name_and_ti(target_sql_obj)
            col_name = ti.non_tracking_cols

            origin_name = ti.origin_context.origin_table
            origin_ti = singleton.mapping.get_ti_from_name(origin_name)

            if "classes" in kwargs:
                classes = kwargs["classes"]
            else:
                classes = args[1]

            class_len = len(classes)
            for i in range(class_len):
                classes[i] = f"\'{classes[i]}\'" if isinstance(classes[i], str) else 0

            origin_context = OpTree(op="{}", non_tracking_columns=ti.non_tracking_cols,
                                    tracking_columns=ti.tracking_cols, origin_table=name)

            select_content = "CASE\n"
            if class_len == 2:
                select_content += f"\t\tWHEN ({col_name} = {classes[0]}) THEN 0\n" \
                                  f"\t\tWHEN ({col_name} = {classes[1]}) THEN 1\n"
            else:
                # raise Warning("Try One-Hot-Encode instead of \"sklearn.preprocessing.label_binarize\".")
                raise NotImplementedError
            select_content += "\tEND"

            sql_code = f"SELECT ({select_content}) AS {col_name},\n" \
                       f"{', '.join([x for x in origin_ti.non_tracking_cols if x != col_name] + origin_ti.tracking_cols)}\n" \
                       f"FROM {origin_name}"

            binarized_col = origin_ti.non_tracking_cols

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, new_return_value,
                                                                     tracking_cols=origin_ti.tracking_cols,
                                                                     non_tracking_cols=ti.non_tracking_cols,
                                                                     operation_type=OperatorType.TRANSFORMER,
                                                                     cte_name=f"block_binarize_mlinid{op_id}")

            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, binarized_col)
            # TO_SQL DONE! ##########################################################################################

            classes = kwargs['classes']
            description = "label_binarize, classes: {}".format(classes)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, ["array"]),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


# ################################## UTILITY ##########################################

def find_target(self, arg, target_obj):
    cols_to_drop = []
    global column_transformer_share
    if not (column_transformer_share is None):
        target_obj = column_transformer_share.target_obj  # here the correct one given trough the ColumnTransformer
        name, ti = singleton.mapping.get_name_and_ti(target_obj)
        target_cols = column_transformer_share.op_to_col_map[self.mlinspect_optional_code_reference]
        cols_to_drop = column_transformer_share.cols_to_drop
    else:
        name, ti = singleton.mapping.get_name_and_ti(arg)
        target_cols = ti.non_tracking_cols
    return name, ti, target_cols, target_obj, cols_to_drop


def level_in_column_transformer(ct, x):
    """
    Helps to find the level in the ColumnTransformer one is on.
    """
    for t in ct.transformers:
        for level, s in enumerate(t[1].steps):
            if s[1] is x:
                return level
    assert False  # not found
