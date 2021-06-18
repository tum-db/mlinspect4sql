"""
Monkey patching for pandas
"""
import copy
import os

import gorilla
import pandas
import pandas as pd
import pathlib

from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sql_backend import SQLBackend, CreateTablesFromCSVs
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node, \
    get_dag_node_for_id, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._patch_sklearn import call_info_singleton


class DfToStringMapping:
    """
    Simple data structure to track the mappings of pandas.Dataframes to SQL table names.
    """
    mapping = []  # contains tuples of form: (*Name*, *DataFrame*)

    def add(self, name: str, df: pd.DataFrame) -> None:
        self.mapping.append((name, df))

    def update_entry(self, old_entry: (str, pd.DataFrame), new_entry: (str, pd.DataFrame)):
        index = self.mapping.index(old_entry)
        self.mapping[index] = new_entry

    def update_name_at_df(self, df, new_name):
        old_name = self.get_name(df)
        index = self.mapping.index((old_name, df))
        self.mapping[index] = (new_name, df)

    def get_df(self, name_to_find: str) -> pd.DataFrame:
        return next(df for (n, df) in self.mapping if n == name_to_find)

    def get_name(self, df_to_find: pd.DataFrame) -> str:
        return next(n for (n, df) in self.mapping if df is df_to_find)

    def contains(self, df_to_find):
        for m in self.mapping:
            if m[1] is df_to_find:
                return True
        return False


# This mapping allows to keep track of the pandas.DataFrame and pandas.Series w.r.t. their SQL-table representation!
mapping = DfToStringMapping()
sql_backend = SQLBackend()


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

            # PRINT SQL: ###############################################################################################

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

            table_name = pathlib.Path(path_to_csv).stem + "_" + str(sql_backend.get_unique_id())

            sql_code = CreateTablesFromCSVs(path_to_csv).get_sql_code(table_name=table_name, null_symbol=na_values,
                                                                      delimiter=sep,
                                                                      header=(1 == header))

            print(sql_code + "\n")

            mapping.add(table_name, result)
            # PRINT SQL DONE! ##########################################################################################

            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            description = "{}".format(args[0].split(os.path.sep)[-1])
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
            # PRINT SQL: ###############################################################################################
            if not mapping.contains(result):  # Only create SQL-Table if it doesn't already exist.
                tb1 = input_info.annotated_dfobject.result_data

                select_attributes = args[0]
                if not isinstance(select_attributes, list):
                    select_attributes = [select_attributes]

                sql_code = f"SELECT {', '.join(select_attributes)} \n" \
                           f"FROM {mapping.get_name(tb1)}"

                sql_table_name, sql_code = sql_backend.wrap_in_with(sql_code, lineno)

                mapping.add(sql_table_name, result)
                print(sql_code + "\n")
            # PRINT SQL DONE! ##########################################################################################
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
            # PRINT SQL: ###############################################################################################
            # There are two options to handle this kind of arithmetic operations of pandas.DataFrames / pandas.series:
            # we can catch the entire operations here and create one with a single "with_statement", or create one with
            # for each operation and put these with together. The second option is preferred as is its only downside is,
            # that it is more verbose, but on the other side its simpler, more elegant and dosn't require to go over
            # the statements twice.
            tb1 = self
            new_name = args[0]
            tb2 = args[1]

            if len(args) != 2:
                raise NotImplementedError

            new_col = tb2.name
            old_cols = tb1.columns.values

            sql_code = f"SELECT tb1.{', tb1.'.join([x for x in old_cols if x != new_name])}, " \
                       f"tb2.{new_col} AS {new_name} \n" \
                       f"FROM {mapping.get_name(tb1)} AS tb1,  {mapping.get_name(tb2)} AS tb2"

            sql_table_name, sql_code = sql_backend.wrap_in_with(sql_code, lineno)
            mapping.add(sql_table_name, result)
            print(sql_code + "\n")

            # PRINT SQL DONE! ##########################################################################################
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

            # PRINT SQL: ###############################################################################################
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

            if merge_column == "":  # Cross product:
                sql_code = f"SELECT * \n" \
                           f"FROM {mapping.get_name(tb1)}, {mapping.get_name(tb2)}"
            else:
                sql_code = f"SELECT * \n" \
                           f"FROM {mapping.get_name(tb1)} tb1 \n" \
                           f"{merge_type.upper()} JOIN {mapping.get_name(tb2)} tb2" \
                           f" ON tb1.{merge_column} = tb2.{merge_column}"
            sql_table_name, sql_code = sql_backend.wrap_in_with(sql_code, lineno)
            mapping.add(sql_table_name, result)
            print(sql_code + "\n")
            # PRINT SQL DONE! ##########################################################################################

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
            # PRINT SQL: ###############################################################################################
            # Here we don't need to do anything, as the groupby alone, is irrelevent. Only together with the applied
            # operation, we need to act.
            # PRINT SQL DONE! ##########################################################################################
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
        groupby_columns_outer_scope = self.grouper.names  # gets the columns on which thr groupby was applied
        tb1_outer_scope = self.obj  # gets the pandas.DataFrame on which the groupby was called upon

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.groupby.generic', 'agg')
            if not hasattr(self, '_mlinspect_dag_node'):
                raise NotImplementedError("TODO: Support agg if groupby happened in external code")
            input_dag_node = get_dag_node_for_id(self._mlinspect_dag_node)  # pylint: disable=no-member

            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)

            input_infos = PandasBackend.before_call(operator_context, [])
            result = original(self, *args, **kwargs)
            # PRINT SQL: ###############################################################################################
            tb1 = tb1_outer_scope
            groupby_columns = groupby_columns_outer_scope  # could be: mapping, function, label, or list of labels
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

            sql_code = f"SELECT {', '.join(selection_string)} \n" \
                       f"FROM {mapping.get_name(tb1)} \n" \
                       f"GROUP BY {', '.join(groupby_columns)}"

            sql_table_name, sql_code = sql_backend.wrap_in_with(sql_code, lineno)
            mapping.add(sql_table_name, result)
            print(sql_code + "\n")
            # PRINT SQL DONE! ##########################################################################################

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

    ################
    # ARITHMETIC:
    ################

    @gorilla.name('__mul__')
    @gorilla.settings(allow_hit=True)
    def patched__mul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__mul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__mul__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self * args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("*", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rmul__')
    @gorilla.settings(allow_hit=True)
    def patched__rmul__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rmul__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rmul__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self * args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("*", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sum__')
    @gorilla.settings(allow_hit=True)
    def patched__sum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sum__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self + args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("+", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsum__')
    @gorilla.settings(allow_hit=True)
    def patched__rsum__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsum__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsum__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self + args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("+", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__sub__')
    @gorilla.settings(allow_hit=True)
    def patched__sub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__sub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__sub__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self - args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("-", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rsub__')
    @gorilla.settings(allow_hit=True)
    def patched__rsub__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rsub__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rsub__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self - args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("-", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__div__')
    @gorilla.settings(allow_hit=True)
    def patched__div__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__div__') """
        original = gorilla.get_original_attribute(pandas.Series, '__div__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self / args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("/", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__rdiv__')
    @gorilla.settings(allow_hit=True)
    def patched__rdiv__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__rdiv__') """
        original = gorilla.get_original_attribute(pandas.Series, '__rdiv__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            result = (self / args[0])
            assert (len(args) == 1)
            return sql_backend.handle_operation_series("/", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    ################
    # COMPARISONS:
    ################
    @gorilla.name('__eq__')
    @gorilla.settings(allow_hit=True)
    def patched__eq__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__eq__') """
        original = gorilla.get_original_attribute(pandas.Series, '__eq__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            assert (len(args) == 1)
            result = self.eq(args[0])
            return sql_backend.handle_operation_series("=", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__gt__')
    @gorilla.settings(allow_hit=True)
    def patched__gt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__gt__') """
        original = gorilla.get_original_attribute(pandas.Series, '__gt__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            assert (len(args) == 1)
            result = self.gt(args[0])
            return sql_backend.handle_operation_series(">", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__lt__')
    @gorilla.settings(allow_hit=True)
    def patched__lt__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__lt__') """
        original = gorilla.get_original_attribute(pandas.Series, '__lt__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            assert (len(args) == 1)
            result = self.lt(args[0])
            return sql_backend.handle_operation_series("<", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__le__')
    @gorilla.settings(allow_hit=True)
    def patched__le__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__le__') """
        original = gorilla.get_original_attribute(pandas.Series, '__le__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            assert (len(args) == 1)
            result = self.le(args[0])
            return sql_backend.handle_operation_series("<=", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__ge__')
    @gorilla.settings(allow_hit=True)
    def patched__ge__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__ge__') """
        original = gorilla.get_original_attribute(pandas.Series, '__ge__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            assert (len(args) == 1)
            result = self.ge(args[0])
            return sql_backend.handle_operation_series(">=", mapping, result, left=self, right=args[0], lineno=lineno)

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)
