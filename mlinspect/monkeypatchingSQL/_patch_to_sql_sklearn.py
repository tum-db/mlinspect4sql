"""
Monkey patching for pandas
"""
import copy
import os
import pathlib

import sklearn.pipeline
from mlinspect.to_sql.py_to_sql_mapping import TableInfo, OpTree, sql_obj_prefix
from mlinspect.to_sql.py_to_sql_mapping import OpTree, ColumnTransformerInfo, ColumnTransformerLevel
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.monkeypatching._monkey_patching_utils import get_dag_node_for_id
from mlinspect.monkeypatching._patch_sklearn import call_info_singleton
from mlinspect.to_sql._mode import SQLMode, SQLObjRep
from sklearn import pipeline
import gorilla
import numpy
import pandas
from sklearn import preprocessing, compose, tree, impute, linear_model, model_selection
from tensorflow.keras.wrappers import scikit_learn as keras_sklearn_external  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers import scikit_learn as keras_sklearn_internal  # pylint: disable=no-name-in-module
from mlinspect.inspections._histogram_for_columns import HistogramForColumns

from mlinspect.backends._backend import BackendResult
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, CodeReference
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._patch_numpy import MlinspectNdarray

from dataclasses import dataclass
from typing import Dict
from mlinspect.utils import store_timestamp
import time

pandas.options.mode.chained_assignment = None  # default='warn'


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


@dataclass
class FitDataCollection:
    """
    Data Container for the fitted variables of SimpleImpute.
    """
    col_to_fit_block_name: Dict[str, str]
    fully_set: bool
    extra_info = {}  # For the KBin


call_info_singleton = SklearnCallInfo()
column_transformer_share = None
just_transform_run = {}  # whether we are fitting or transforming. | Adapt behaviour!
last_name_for_concat_fit = {}


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
        # TO_SQL: ###############################################################################################
        fit_data, just_transform = differentiate_fit_transform(self, args[0], set_attributes=False)
        # TO_SQL DONE! ##########################################################################################
        if not just_transform:
            call_info_singleton.transformer_filename = self.mlinspect_filename
            call_info_singleton.transformer_lineno = self.mlinspect_lineno
            call_info_singleton.transformer_function_info = FunctionInfo('sklearn.compose._column_transformer',
                                                                         'ColumnTransformer')
            call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
            call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

            call_info_singleton.column_transformer_active = True
            original = gorilla.get_original_attribute(compose.ColumnTransformer, 'fit_transform')
            op_id = singleton.get_next_op_id()
        else:
            original = gorilla.get_original_attribute(compose.ColumnTransformer, 'transform')
            op_id = singleton.sql_logic.get_unique_id()

        # TO_SQL: ###############################################################################################
        # When calling original(self, *args, **kwargs) the overwritten Pipeline-functions (like SimpleImpute)
        # will be called with the relevant slice of the table.

        name, ti = singleton.mapping.get_name_and_ti(args[0])

        # Materialize the source if desired: -> better performance f.e. for Postgres
        materialize_query_with_name(name, ti.non_tracking_cols)

        cr_to_col_map = {}  # code_reference to column mapping
        cr_to_level_map = {}  # code_reference to level mapping
        cols_to_keep = []
        for _, (_, op_obj, target_cols) in enumerate(self.transformers):
            target_cols = [f"\"{x}\"" for x in target_cols]
            cols_to_keep += target_cols
            if isinstance(op_obj, pipeline.Pipeline):
                for level_s, (_, step) in enumerate(op_obj.steps):  # need to take pipeline apart
                    cr = step.mlinspect_optional_code_reference
                    cr_to_col_map[cr] = target_cols
                    cr_to_level_map[cr] = level_s
            else:
                cr = op_obj.mlinspect_optional_code_reference
                cr_to_col_map[cr] = target_cols
                cr_to_level_map[cr] = 0

        levels_list = [ColumnTransformerLevel({}, set(), set(), []) for _ in
                       range(max(cr_to_level_map.values()) + 1)]

        # HANDLE "drop" case:
        cols_to_drop = []
        cols_to_keep = list(dict.fromkeys(cols_to_keep))  # keeps order
        if self.remainder == "drop":
            cols_to_drop = list(set(ti.non_tracking_cols) - set(cols_to_keep))
        else:
            cols_to_keep = ti.non_tracking_cols  # here we keep all columns

        # We will need pass the input of this function to the subclass, to be able to achieve the mapping.
        global column_transformer_share
        column_transformer_share = ColumnTransformerInfo(self,
                                                         levels=levels_list,
                                                         levels_map=cr_to_level_map,
                                                         cr_to_col_map=cr_to_col_map,
                                                         target_obj=args[0], cols_to_drop=cols_to_drop)

        if len(args) == 2 and len(kwargs) == 0:
            result = original(self, args[0], **kwargs)
        else:
            result = original(self, *args, **kwargs)

        query_levels = column_transformer_share.levels
        # Query optimization for further executions:
        final_sql_code = ""
        last_sql_name = name
        for i, level in enumerate(query_levels):
            select_block = []
            column_map = level.column_map
            for col in cols_to_keep:
                if col in column_map.keys():
                    select_block.append(column_map[col])
                else:
                    select_block.append(f"\t{col}")

            select_block_s = ',\n'.join(select_block) + ",\n\t" + ", ".join(ti.tracking_cols)
            from_block_s = ", ".join(level.from_block | {last_sql_name})
            where_block_s = " AND \n".join(level.where_block)

            sql_code = f"SELECT \n{select_block_s} \n" \
                       f"FROM {from_block_s}"

            if where_block_s != "":
                sql_code += "\nWHERE\n" + where_block_s

            if i < len(query_levels) - 1:
                sql_name, sql_code = singleton.sql_logic.wrap_in_sql_obj(sql_code, op_id,
                                                                         f"lvl{i}_column_transformer",
                                                                         force_cte=True)
            else:
                # Last level reached:
                final_sql_code = final_sql_code[:-2] + "\n"
                sql_name = f"block_column_transformer_lvl{i}"

            # substitute old data sources with new ones:
            for old_s in level.sql_source:
                sql_code = sql_code.replace(old_s, last_sql_name)
            final_sql_code += sql_code + ",\n"

            last_sql_name = sql_name

        if len(query_levels) > 1:
            final_sql_code = "WITH " + final_sql_code
        final_sql_code = final_sql_code[:-2]

        cte_name, sql_code = singleton.sql_logic.finish_sql_call(final_sql_code, op_id, result,
                                                                 tracking_cols=ti.tracking_cols,
                                                                 non_tracking_cols=cols_to_keep,
                                                                 operation_type=OperatorType.TRANSFORMER,
                                                                 cte_name=last_sql_name)

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code, cols_to_keep=cols_to_keep)

        global last_name_for_concat_fit
        last_name_for_concat_fit = {}
        fit_data.fully_set = True
        # TO_SQL DONE! ##########################################################################################
        if not just_transform:
            call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(compose.ColumnTransformer, 'transform')
        return transform_logic(original, self, *args, **kwargs)  # fit_transform knows only to execute transform

    # @gorilla.name('fit')
    # @gorilla.settings(allow_hit=True)
    # def patched_fit(self, *args, **kwargs):
    #     # pylint: disable=no-method-argument
    #     original = gorilla.get_original_attribute(compose.ColumnTransformer, 'fit')
    #     return original(self, *args, **kwargs)

    @gorilla.name('_hstack')
    @gorilla.settings(allow_hit=True)
    def patched_hstack(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument, unused-argument, too-many-locals
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '_hstack')
        op_id = singleton.get_next_op_id()

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

        # Concat doesn't contain ratios: -> Empty
        old_dag_node_annotations = backend_result.dag_node_annotation
        to_check_annotations = [a for a in old_dag_node_annotations.keys() if isinstance(a, HistogramForColumns)]
        if len(to_check_annotations) > 0:
            assert len(to_check_annotations) == 1
            annotation = to_check_annotations[0]
            old_dag_node_annotations[to_check_annotations[0]] = {x: {} for x in
                                                                 [f"\"{x}\"" for x in annotation.sensitive_columns]}

        dag_node = DagNode(op_id,
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
        # TO_SQL: ###############################################################################################
        fit_data, just_transform = differentiate_fit_transform(self, args[0])
        # TO_SQL DONE! ##########################################################################################

        if not just_transform:
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
        else:
            original = gorilla.get_original_attribute(impute.SimpleImputer, 'transform')
            result = original(self, *args, **kwargs)
            op_id = singleton.sql_logic.get_unique_id()

        # TO_SQL: ###############################################################################################
        code_ref = self.mlinspect_optional_code_reference
        name, ti, target_cols, res_for_map, cols_to_drop = find_target(code_ref, args[0], result)
        tracking_cols = ti.tracking_cols
        all_cols = [x for x in ti.non_tracking_cols if not x in set(cols_to_drop)]

        selection_map = {}
        select_block = []

        # STRATEGY 1: ###
        if self.strategy == "most_frequent":
            for col in all_cols:
                if col in target_cols:

                    # Access fitted data if possible:
                    if not fit_data.fully_set:
                        fit_lookup_table, fit_lookup_code = singleton.sql_logic.column_max_count(name, col)
                        fit_data.col_to_fit_block_name[col] = fit_lookup_table
                        singleton.pipeline_container.add_statement_to_pipe(fit_lookup_table, fit_lookup_code)
                        if singleton.sql_obj.mode == SQLObjRep.VIEW:
                            singleton.dbms_connector.run(fit_lookup_code)
                    else:
                        fit_lookup_table = fit_data.col_to_fit_block_name[col]

                    select_block.append(f"\tCOALESCE({col}, (SELECT * FROM {fit_lookup_table})) AS {col}")
                    selection_map[col] = select_block[-1]
                else:
                    select_block.append(f"\t{col}")

        # STRATEGY 2: ###
        elif self.strategy == "mean":
            for col in all_cols:
                if col in target_cols:

                    # Access fitted data if possible:
                    if not fit_data.fully_set:
                        fit_lookup_table, fit_lookup_code = singleton.sql_logic.column_mean(name, col)
                        fit_data.col_to_fit_block_name[col] = fit_lookup_table
                        singleton.pipeline_container.add_statement_to_pipe(fit_lookup_table, fit_lookup_code)
                        if singleton.sql_obj.mode == SQLObjRep.VIEW:
                            singleton.dbms_connector.run(fit_lookup_code)
                    else:
                        fit_lookup_table = fit_data.col_to_fit_block_name[col]

                    select_block.append(f"\tCOALESCE({col}, (SELECT * FROM {fit_lookup_table})) AS {col}")
                    selection_map[col] = select_block[-1]
                else:
                    select_block.append(f"\t{col}")
        # STRATEGY X: ###
        else:
            raise NotImplementedError

        select_block_s = ',\n'.join(select_block) + ",\n\t" + ", ".join(tracking_cols)

        sql_code = f"SELECT \n{select_block_s} \n" \
                   f"FROM {name}"

        add_to_col_trans(selection_map=selection_map, code_ref=code_ref, sql_source=name)

        # cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
        #                                                          tracking_cols=tracking_cols,
        #                                                          non_tracking_cols=all_cols,
        #                                                          operation_type=OperatorType.TRANSFORMER,
        #                                                          cte_name=f"block_impute_mlinid{op_id}")

        cte_name, sql_code = store_current_sklearn_op(sql_code, op_id, f"block_impute_mlinid{op_id}", result,
                                                      tracking_cols, all_cols, target_cols)

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        if not just_transform:
            backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                             curr_sql_expr_name=cte_name,
                                                                             curr_sql_expr_columns=all_cols,
                                                                             keep_previous_res=False,
                                                                             not_materialize=True)

            fit_data.fully_set = True
            # TO_SQL DONE! ##########################################################################################
            dag_node = DagNode(op_id,
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Simple Imputer", columns),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value
        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'transform')
        return transform_logic(original, self, *args, **kwargs)

    # @gorilla.name('fit')
    # @gorilla.settings(allow_hit=True)
    # def patched_fit(self, *args, **kwargs):
    #     # pylint: disable=no-method-argument
    #     original = gorilla.get_original_attribute(impute.SimpleImputer, 'fit')
    #     return original(self, *args, **kwargs)


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
        # TO_SQL: ###############################################################################################
        fit_data, just_transform = differentiate_fit_transform(self, args[0])
        # TO_SQL DONE! ##########################################################################################
        if not just_transform:
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
        else:
            original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'transform')
            result = original(self, *args, **kwargs)
            op_id = singleton.sql_logic.get_unique_id()
        # TO_SQL: ###############################################################################################

        code_ref = self.mlinspect_optional_code_reference
        name, ti, target_cols, res_for_map, cols_to_drop = find_target(code_ref, args[0], result)
        tracking_cols = ti.tracking_cols
        all_cols = [x for x in ti.non_tracking_cols if not x in set(cols_to_drop)]

        selection_map = {}
        select_block = []
        from_block = []
        where_block = []

        for col in all_cols:
            if col in target_cols:

                # Access fitted data if possible:
                if not fit_data.fully_set:
                    fit_lookup_table, fit_lookup_code = singleton.sql_logic.column_one_hot_encoding(name, col)
                    fit_data.col_to_fit_block_name[col] = fit_lookup_table
                    singleton.pipeline_container.add_statement_to_pipe(fit_lookup_table, fit_lookup_code)
                    if singleton.sql_obj.mode == SQLObjRep.VIEW:
                        singleton.dbms_connector.run(fit_lookup_code)
                else:
                    fit_lookup_table = fit_data.col_to_fit_block_name[col]

                select_block.append(f"\t{col[:-1]}_one_hot\" AS {col}")
                from_block.append(f"{fit_lookup_table}")
                where_block.append(f"\t{name}.{col} = {fit_lookup_table}.{col}")

                selection_map[col] = select_block[-1]
            else:
                select_block.append(f"\t{col}")

        select_block_s = ",\n".join(select_block) + ",\n\t" + ", ".join(tracking_cols)
        from_block_s = ", ".join(from_block + [name])
        where_block_s = " AND \n".join(where_block)

        sql_code = f"SELECT \n{select_block_s} \n" \
                   f"FROM {from_block_s}\n" \
                   f"WHERE\n {where_block_s}"

        add_to_col_trans(selection_map=selection_map, from_block=from_block,
                         where_block=where_block, code_ref=code_ref, sql_source=name)

        # cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
        #                                                          tracking_cols=tracking_cols,
        #                                                          non_tracking_cols=all_cols,
        #                                                          operation_type=OperatorType.TRANSFORMER,
        #                                                          cte_name=f"block_onehot_mlinid{op_id}")

        cte_name, sql_code = store_current_sklearn_op(sql_code, op_id, f"block_onehot_mlinid{op_id}", result,
                                                      tracking_cols, all_cols, target_cols)

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        if not just_transform:
            backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                             curr_sql_expr_name=cte_name,
                                                                             curr_sql_expr_columns=all_cols,
                                                                             keep_previous_res=True,
                                                                             not_materialize=True)

            fit_data.fully_set = True
            # TO_SQL DONE! ##########################################################################################

            dag_node = DagNode(op_id,
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("One-Hot Encoder", ['array']),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code))

            add_dag_node(dag_node, [input_info.dag_node], backend_result)
            return new_return_value
        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'transform')
        return transform_logic(original, self, *args, **kwargs)

    # @gorilla.name('fit')
    # @gorilla.settings(allow_hit=True)
    # def patched_fit(self, *args, **kwargs):
    #     # pylint: disable=no-method-argument
    #     original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'fit')
    #     return original(self, *args, **kwargs)


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
        # TO_SQL: ###############################################################################################
        fit_data, just_transform = differentiate_fit_transform(self, args[0])
        # TO_SQL DONE! ##########################################################################################
        if not just_transform:
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
        else:
            original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'transform')
            result = original(self, *args, **kwargs)
            op_id = singleton.sql_logic.get_unique_id()
        # TO_SQL: ###############################################################################################
        code_ref = self.mlinspect_optional_code_reference
        name, ti, target_cols, res_for_map, cols_to_drop = find_target(code_ref, args[0], result)
        tracking_cols = ti.tracking_cols
        all_cols = [x for x in ti.non_tracking_cols if not x in set(cols_to_drop)]

        selection_map = {}
        if not (self.with_mean and self.with_std):
            raise NotImplementedError

        select_block = []
        for col in all_cols:
            if col in target_cols:

                # Access fitted data if possible:
                if not fit_data.fully_set:
                    fit_lookup_table, fit_lookup_code = singleton.sql_logic.std_scalar_values(name, col)
                    fit_data.col_to_fit_block_name[col] = fit_lookup_table
                    singleton.pipeline_container.add_statement_to_pipe(fit_lookup_table, fit_lookup_code)
                    if singleton.sql_obj.mode == SQLObjRep.VIEW:
                        singleton.dbms_connector.run(fit_lookup_code)
                else:
                    fit_lookup_table = fit_data.col_to_fit_block_name[col]

                select_block.append(
                    f"\t(({col} - (SELECT avg_col_std_scal FROM {fit_lookup_table})) / "
                    f"(SELECT std_dev_col_std_scal FROM {fit_lookup_table})) "
                    f"AS {col}")
                selection_map[col] = select_block[-1]
            else:
                select_block.append(f"\t{col}")

        select_block_s = ',\n'.join(select_block) + ",\n\t" + ", ".join(tracking_cols)

        sql_code = f"SELECT \n{select_block_s} \n" \
                   f"FROM {name}"

        add_to_col_trans(selection_map=selection_map, code_ref=code_ref, sql_source=name)

        # cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
        #                                                          tracking_cols=tracking_cols,
        #                                                          non_tracking_cols=all_cols,
        #                                                          operation_type=OperatorType.TRANSFORMER,
        #                                                          cte_name=f"block_stdscaler_mlinid{op_id}")

        cte_name, sql_code = store_current_sklearn_op(sql_code, op_id, f"block_stdscaler_mlinid{op_id}", result,
                                                      tracking_cols, all_cols, target_cols)

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        if not just_transform:
            fit_data.fully_set = True
            # TO_SQL DONE! ##########################################################################################

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Standard Scaler", ['array']),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code))
            backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                             curr_sql_expr_name=cte_name,
                                                                             curr_sql_expr_columns=all_cols,
                                                                             keep_previous_res=True,
                                                                             not_materialize=True)
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
            return new_return_value
        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'transform')
        return transform_logic(original, self, *args, **kwargs)

    # @gorilla.name('fit')
    # @gorilla.settings(allow_hit=True)
    # def patched_fit(self, *args, **kwargs):
    #     # pylint: disable=no-method-argument
    #     original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'fit')
    #     return original(self, *args, **kwargs)


@gorilla.patches(preprocessing.KBinsDiscretizer)
class SklearnKBinsDiscretizerPatching:
    """ Patches for sklearn KBinsDiscretizer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_bins=5, *, encode='onehot', strategy='quantile',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, n_bins=n_bins, encode=encode, strategy=strategy)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, n_bins=n_bins, encode=encode,
                                             strategy=strategy)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        # TO_SQL: ###############################################################################################
        fit_data, just_transform = differentiate_fit_transform(self, args[0])
        # TO_SQL DONE! ##########################################################################################
        if not just_transform:
            original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'fit_transform')
            function_info = FunctionInfo('sklearn.preprocessing._discretization', 'KBinsDiscretizer')
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
        else:
            original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'transform')
            result = original(self, *args, **kwargs)
            op_id = singleton.sql_logic.get_unique_id()

        # TO_SQL: ###############################################################################################

        code_ref = self.mlinspect_optional_code_reference
        name, ti, target_cols, res_for_map, cols_to_drop = find_target(code_ref, args[0], result)
        tracking_cols = ti.tracking_cols
        all_cols = [x for x in ti.non_tracking_cols if not x in set(cols_to_drop)]
        num_bins = self.n_bins

        if not (self.encode == "ordinal", self.strategy == "uniform"):
            raise NotImplementedError

        selection_map = {}
        select_block = []

        for col in all_cols:
            if col in target_cols:

                # Access fitted data if possible:
                if not fit_data.fully_set:
                    # create table for min:
                    min_lookup_table = f"block_kbin_fit_{singleton.sql_logic.get_unique_id()}_min"
                    min_lookup_code = f"SELECT MIN({col}) AS min_val from {name} "
                    min_lookup_table, min_lookup_code = singleton.sql_logic.wrap_in_sql_obj(min_lookup_code,
                                                                                            block_name=min_lookup_table)
                    min_lookup_table, min_lookup_code = singleton.sql_logic.materialize_if_possible(min_lookup_table,
                                                                                                    min_lookup_code)
                    singleton.pipeline_container.add_statement_to_pipe(min_lookup_table, min_lookup_code)
                    if singleton.sql_obj.mode == SQLObjRep.VIEW:
                        singleton.dbms_connector.run(min_lookup_code)
                    fit_data.extra_info[col] = min_lookup_table

                    # create table for step_sizes:
                    fit_lookup_table, fit_lookup_code = singleton.sql_logic.step_size_kbin(name, col, num_bins)
                    fit_data.col_to_fit_block_name[col] = fit_lookup_table
                    singleton.pipeline_container.add_statement_to_pipe(fit_lookup_table, fit_lookup_code)
                    if singleton.sql_obj.mode == SQLObjRep.VIEW:
                        singleton.dbms_connector.run(fit_lookup_code)
                else:
                    fit_lookup_table = fit_data.col_to_fit_block_name[col]
                    min_lookup_table = fit_data.extra_info[col]

                # select_block_sub = "\t(CASE\n" \
                #                    f"\t\tWHEN {col} < (SELECT min_val FROM {min_lookup_table}) +" \
                #                    f" (SELECT step FROM {fit_lookup_table}) THEN 0\n" \
                #                    f"\t\tWHEN {col} < (SELECT min_val FROM {min_lookup_table}) +" \
                #                    f" {num_bins - 1} * (SELECT step FROM {fit_lookup_table}) " \
                #                    f"THEN FLOOR(({col} - (SELECT min_val FROM {min_lookup_table}))/(SELECT step FROM {fit_lookup_table}))\n" \
                #                    f"\t\tELSE {num_bins - 1}\n\tEND) AS {col}"

                select_block_sub = f"(\n" \
                                   f"\tLEAST({num_bins - 1}, GREATEST(0, FLOOR(({col} - (SELECT min_val FROM {min_lookup_table}))/(SELECT step FROM {fit_lookup_table}))))\n" \
                                   f") AS {col}"

                select_block.append(select_block_sub)
                selection_map[col] = select_block[-1]
            else:
                select_block.append(f"\t{col}")

        select_block_s = ',\n'.join(select_block) + ",\n\t" + ", ".join(tracking_cols)

        sql_code = f"SELECT \n{select_block_s}\n" \
                   f"FROM {name}"

        add_to_col_trans(selection_map=selection_map, code_ref=code_ref, sql_source=name)

        # cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code, op_id, res_for_map,
        #                                                          tracking_cols=tracking_cols,
        #                                                          non_tracking_cols=all_cols,
        #                                                          operation_type=OperatorType.TRANSFORMER,
        #                                                          cte_name=f"block_kbin_mlinid{op_id}")

        cte_name, sql_code = store_current_sklearn_op(sql_code, op_id, f"block_kbin_mlinid{op_id}", result,
                                                      tracking_cols, all_cols, target_cols)

        singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

        if not just_transform:
            backend_result = singleton.update_hist.sql_update_backend_result(res_for_map, backend_result,
                                                                             curr_sql_expr_name=cte_name,
                                                                             curr_sql_expr_columns=all_cols,
                                                                             keep_previous_res=True,
                                                                             not_materialize=True)

            fit_data.fully_set = True
            # TO_SQL DONE! ##########################################################################################

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("K-Bins Discretizer", ['array']),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code))

            add_dag_node(dag_node, [input_info.dag_node], backend_result)
            return new_return_value
        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'transform')
        return transform_logic(original, self, *args, **kwargs)
    #
    # @gorilla.name('fit')
    # @gorilla.settings(allow_hit=True)
    # def patched_transform(self, *args, **kwargs):
    #     # pylint: disable=no-method-argument
    #     original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'fit')
    #     return original(self, *args, **kwargs)


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


@gorilla.patches(model_selection)
class SklearnModelSelectionPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('train_test_split')
    @gorilla.settings(allow_hit=True)
    def patched_train_test_split(*args, **kwargs):
        """ Patch for ('sklearn.model_selection._split', 'train_test_split') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(model_selection, 'train_test_split')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.model_selection._split', 'train_test_split')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.TRAIN_TEST_SPLIT, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])

            # try with dummy objects, if not possible mimic split:
            try:
                result = original(input_infos[0].result_data, *args[1:], **kwargs)
            except ValueError:
                result = [input_infos[0].result_data,
                          copy.deepcopy(input_infos[0].result_data)]  # is a list of two objects.

            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result)  # We ignore the test set for now
            train_backend_result = BackendResult(backend_result.annotated_dfobject,
                                                 backend_result.dag_node_annotation)
            test_backend_result = BackendResult(backend_result.optional_second_annotated_dfobject,
                                                backend_result.optional_second_dag_node_annotation)
            new_return_value = train_backend_result.annotated_dfobject.result_data, \
                               test_backend_result.annotated_dfobject.result_data

            op_id_2 = singleton.get_next_op_id()
            # TO_SQL: ###############################################################################################
            train_part = 0.75
            target_obj = args[0]
            if len(args) != 1:
                raise NotImplementedError

            index_mlinspect = ""
            if singleton.dbms_connector.add_mlinspect_serial:
                index_mlinspect = f"ORDER BY {singleton.dbms_connector.index_col_name} ASC"

            name, ti = singleton.mapping.get_name_and_ti(target_obj)

            # Random reordering for split:
            row_num_addition = f"WITH row_num_mlinspect_split AS(\n" \
                               f"\tSELECT *, (ROW_NUMBER() OVER({index_mlinspect})) AS row_number_mlinspect\n" \
                               f"\tFROM (SELECT * FROM {name} ORDER BY RANDOM()) {name}\n" \
                               f")\n"
            sql_code_train = f"{row_num_addition}" \
                             f"SELECT *\n" \
                             f"FROM row_num_mlinspect_split\n" \
                             f"WHERE row_number_mlinspect < {train_part} * (SELECT COUNT(*) FROM {name})"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code_train, op_id, new_return_value[0],
                                                                     tracking_cols=ti.tracking_cols,
                                                                     non_tracking_cols=ti.non_tracking_cols,
                                                                     operation_type=OperatorType.TRANSFORMER,
                                                                     cte_name=f"block_train_split_mlinid{op_id}")

            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

            train_backend_result = singleton.update_hist.sql_update_backend_result(new_return_value[0],
                                                                                   train_backend_result,
                                                                                   curr_sql_expr_name=cte_name,
                                                                                   curr_sql_expr_columns=
                                                                                   ti.non_tracking_cols)

            sql_code_test = f"{row_num_addition}" \
                            f"SELECT *\n" \
                            f"FROM row_num_mlinspect_split\n" \
                            f"WHERE row_number_mlinspect >= {train_part} * (SELECT COUNT(*) FROM {name})"

            cte_name, sql_code = singleton.sql_logic.finish_sql_call(sql_code_test, op_id, new_return_value[1],
                                                                     tracking_cols=ti.tracking_cols,
                                                                     non_tracking_cols=ti.non_tracking_cols,
                                                                     operation_type=OperatorType.TRANSFORMER,
                                                                     cte_name=f"block_test_split_mlinid{op_id}")

            singleton.pipeline_container.add_statement_to_pipe(cte_name, sql_code)

            test_backend_result = singleton.update_hist.sql_update_backend_result(new_return_value[1],
                                                                                  test_backend_result,
                                                                                  curr_sql_expr_name=cte_name,
                                                                                  curr_sql_expr_columns=
                                                                                  ti.non_tracking_cols)
            # TO_SQL DONE! ##########################################################################################

            description = "(Train Data)"
            columns = list(result[0].columns)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], train_backend_result)

            description = "(Test Data)"
            columns = list(result[1].columns)
            dag_node = DagNode(op_id_2,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], test_backend_result)

            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


# SKLEAN MODELS:
just_the_model = None


class SklearnKerasClassifierPatching:
    """ Patches for tensorflow KerasClassifier"""

    # pylint: disable=too-few-public-methods
    @gorilla.patch(keras_sklearn_internal.BaseWrapper, name='__init__', settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, build_fn=None, mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        **sk_params):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(keras_sklearn_internal.BaseWrapper, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, build_fn=build_fn, **sk_params)
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            global just_the_model
            just_the_model = copy.deepcopy(self)

        return execute_patched_func_no_op_id(original, execute_inspections, self, build_fn=build_fn, **sk_params)

    @gorilla.patch(keras_sklearn_external.KerasClassifier, name='fit', settings=gorilla.Settings(allow_hit=True))
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'fit')
        function_info = FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_data.annotated_dfobject])
        data_backend_result = SklearnBackend.after_call(operator_context,
                                                        input_infos,
                                                        args[0])

        # TO_SQL DONE! ##########################################################################################
        args_0, data_backend_result = retrieve_data_from_dbms_get_opt_backend(args[0], data_backend_result,
                                                                              OperatorType.TRAIN_DATA,
                                                                              "MODEL FIT: TRAIN LABELS")
        # TO_SQL: ###############################################################################################

        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], data_backend_result)
        # train_data_result = data_backend_result.annotated_dfobject.result_data

        # Test labels
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        input_info_train_labels = get_input_info(args[1], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                                 function_info, self.mlinspect_optional_code_reference,
                                                 self.mlinspect_optional_source_code)
        train_label_op_id = singleton.get_next_op_id()
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_labels.annotated_dfobject])
        label_backend_result = SklearnBackend.after_call(operator_context,
                                                         input_infos,
                                                         args[1])

        # TO_SQL DONE! ##########################################################################################
        args_1, label_backend_result = retrieve_data_from_dbms_get_opt_backend(args[1], label_backend_result,
                                                                               OperatorType.TRAIN_LABELS,
                                                                               "MODEL FIT: TEST LABELS")
        # TO_SQL: ###############################################################################################

        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], label_backend_result)
        # train_labels_result = label_backend_result.annotated_dfobject.result_data

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        input_infos = SklearnBackend.before_call(operator_context, input_dfs)
        # original(self, train_data_result, train_labels_result, *args[2:], **kwargs)
        estimator_backend_result = SklearnBackend.after_call(operator_context,
                                                             input_infos,
                                                             None)

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Neural Network", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)

        global just_the_model
        if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
            return just_the_model

        original(just_the_model, args_0, args_1)

        return just_the_model


@gorilla.patches(tree.DecisionTreeClassifier)
class SklearnDecisionTreePatching:
    """ Patches for sklearn DecisionTree"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=None,
                        max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None,
                        presort='deprecated', ccp_alpha=0.0, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.tree._classes', 'DecisionTreeClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, criterion=criterion, splitter=splitter, max_depth=max_depth,
                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                     random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                     class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

            global just_the_model
            just_the_model = copy.deepcopy(self)

        return execute_patched_func_no_op_id(original, execute_inspections, self, criterion=criterion,
                                             splitter=splitter, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                                             max_features=max_features,
                                             random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                             min_impurity_decrease=min_impurity_decrease,
                                             min_impurity_split=min_impurity_split,
                                             class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'fit')
        function_info = FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_data.annotated_dfobject])
        data_backend_result = SklearnBackend.after_call(operator_context,
                                                        input_infos,
                                                        args[0])
        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], data_backend_result)
        # train_data_result = data_backend_result.annotated_dfobject.result_data

        # TO_SQL DONE! ##########################################################################################
        args_0, data_backend_result = retrieve_data_from_dbms_get_opt_backend(args[0], data_backend_result,
                                                                              OperatorType.TRAIN_DATA,
                                                                              "MODEL FIT: TRAIN LABELS")
        # TO_SQL: ###############################################################################################

        # Train labels
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        input_info_train_labels = get_input_info(args[1], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                                 function_info, self.mlinspect_optional_code_reference,
                                                 self.mlinspect_optional_source_code)
        train_label_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_labels.annotated_dfobject])
        label_backend_result = SklearnBackend.after_call(operator_context,
                                                         input_infos,
                                                         args[1])
        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], label_backend_result)
        # train_labels_result = label_backend_result.annotated_dfobject.result_data

        # TO_SQL DONE! ##########################################################################################
        args_1, label_backend_result = retrieve_data_from_dbms_get_opt_backend(args[1], label_backend_result,
                                                                               OperatorType.TRAIN_LABELS,
                                                                               "MODEL FIT: TEST LABELS")
        # TO_SQL: ###############################################################################################

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        input_infos = SklearnBackend.before_call(operator_context, input_dfs)
        # original(self, train_data_result, train_labels_result, *args[2:], **kwargs)
        estimator_backend_result = SklearnBackend.after_call(operator_context,
                                                             input_infos,
                                                             None)

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Decision Tree", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)

        global just_the_model
        if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
            return just_the_model
        original(just_the_model, args_0, args_1)

        return just_the_model


@gorilla.patches(linear_model.LogisticRegression)
class SklearnLogisticRegressionPatching:
    """ Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,  # pylint: disable=invalid-name
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='lbfgs', max_iter=100,
                        multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                        l1_ratio=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.linear_model._logistic', 'LogisticRegression') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, penalty=penalty, dual=dual, tol=tol, C=C,
                     fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                     random_state=random_state, solver=solver, max_iter=max_iter,
                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                     l1_ratio=l1_ratio)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

            global just_the_model
            just_the_model = copy.deepcopy(self)

        return execute_patched_func_no_op_id(original, execute_inspections, self, penalty=penalty, dual=dual, tol=tol,
                                             C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                             class_weight=class_weight,
                                             random_state=random_state, solver=solver, max_iter=max_iter,
                                             multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                             n_jobs=n_jobs,
                                             l1_ratio=l1_ratio)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, 'fit')
        function_info = FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_data.annotated_dfobject])
        data_backend_result = SklearnBackend.after_call(operator_context,
                                                        input_infos,
                                                        args[0])
        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], data_backend_result)
        train_data_result = data_backend_result.annotated_dfobject.result_data

        # TO_SQL DONE! ##########################################################################################
        args_0, data_backend_result = retrieve_data_from_dbms_get_opt_backend(args[0], data_backend_result,
                                                                              OperatorType.TRAIN_DATA,
                                                                              "MODEL FIT: TRAIN LABELS")
        # TO_SQL: ###############################################################################################

        # Train labels
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        input_info_train_labels = get_input_info(args[1], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                                 function_info, self.mlinspect_optional_code_reference,
                                                 self.mlinspect_optional_source_code)
        train_label_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_labels.annotated_dfobject])
        label_backend_result = SklearnBackend.after_call(operator_context,
                                                         input_infos,
                                                         args[1])
        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], label_backend_result)
        train_labels_result = label_backend_result.annotated_dfobject.result_data

        # TO_SQL DONE! ##########################################################################################
        args_1, label_backend_result = retrieve_data_from_dbms_get_opt_backend(args[1], label_backend_result,
                                                                               OperatorType.TRAIN_LABELS,
                                                                               "MODEL FIT: TEST LABELS")
        # TO_SQL: ###############################################################################################

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        input_infos = SklearnBackend.before_call(operator_context, input_dfs)
        original(self, train_data_result, train_labels_result, *args[2:], **kwargs)
        estimator_backend_result = SklearnBackend.after_call(operator_context,
                                                             input_infos,
                                                             None)

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Logistic Regression", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)

        global just_the_model
        if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
            return just_the_model
        # t0 = time.time()
        original(just_the_model, args_0, args_1)
        # store_timestamp(f"TRAINING MODEL", time.time() - t0, "SQL")
        return just_the_model


@gorilla.patches(sklearn.pipeline.Pipeline)
class SklearnPipeline:
    """ Patches for tensorflow KerasClassifier"""

    @gorilla.name('score')
    @gorilla.settings(allow_hit=True)
    def patched_score(self, *args, **kwargs):
        args_0 = retrieve_data_from_dbms_get_opt_backend(self.steps[0][1].transform(args[0]),
                                                         comment="MODEL SCORE: INPUT DATA")
        args_1 = retrieve_data_from_dbms_get_opt_backend(args[1],
                                                         comment="MODEL SCORE: EXPECTED OUTPUT DATA")
        if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
            return ""
        # Here we fake using the python pipeline, and return the result obtained by working with the SQL output!
        if len(args_0) != len(args_1):
            min_len = min(len(args_0), len(args_1))
            args_0 = args_0[:min_len]
            args_1 = args_1[:min_len]
        # t0 = time.time()
        score = just_the_model.score(args_0, args_1)
        # store_timestamp(f"SCORE MODEL", time.time() - t0, "SQL")
        return score

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args, **kwargs):
        if hasattr(self, "steps"):
            args_0 = retrieve_data_from_dbms_get_opt_backend(self.steps[0][1].transform(args[0]),
                                                             comment="MODEL PREDICT: INPUT DATA")
        else:
            args_0 = retrieve_data_from_dbms_get_opt_backend(args[0])
        if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
            return ""
        # Here we fake using the python pipeline, and return the result obtained by working with the SQL output!
        return just_the_model.predict(args_0)


# ################################## UTILITY ##########################################

def find_target(code_ref, arg, target_obj):
    cols_to_drop = []
    global column_transformer_share
    if not (column_transformer_share is None):
        target_obj = column_transformer_share.target_obj  # here the correct one given trough the ColumnTransformer
        name, ti = singleton.mapping.get_name_and_ti(target_obj)
        target_cols = column_transformer_share.cr_to_col_map[code_ref]
        cols_to_drop = column_transformer_share.cols_to_drop

        # Here we need to take the original object or the one we want to build upon:
        global last_name_for_concat_fit
        intersection_cols = [col for col in target_cols if col in last_name_for_concat_fit.keys()]
        if intersection_cols:
            name = last_name_for_concat_fit[intersection_cols[0]] if last_name_for_concat_fit is not None else name
    else:
        name, ti = singleton.mapping.get_name_and_ti(arg)
        target_cols = ti.non_tracking_cols
    return name, ti, target_cols, target_obj, cols_to_drop


def add_to_col_trans(selection_map, code_ref, sql_source, from_block=[], where_block=[]):
    global column_transformer_share
    if not (column_transformer_share is None):
        level = column_transformer_share.levels_map[code_ref]
        ct_level = column_transformer_share.levels[level]
        ct_level.column_map = {**selection_map, **ct_level.column_map}
        ct_level.from_block |= set(from_block)
        ct_level.where_block |= set(where_block)
        ct_level.sql_source.append(sql_source)
        # ct_level.tracking_cols += [tc for tc in tracking_cols if tc not in ct_level.tracking_cols]


def retrieve_data_from_dbms_get_opt_backend(train_obj, backend_result=None, op_type=None, comment="",
                                            not_materialize=True):
    name, ti = singleton.mapping.get_name_and_ti(train_obj)

    cols = ti.non_tracking_cols
    if isinstance(cols, str):  # working with a pandas series.
        cols = [cols]

    sql_code = f"SELECT {', '.join(cols)} " \
               f"FROM {name} " \
               f"ORDER BY index_mlinspect"

    singleton.pipeline_container.update_pipe_head(sql_code, comment=comment)

    if hasattr(singleton.dbms_connector, "just_code") and singleton.dbms_connector.just_code:
        if not backend_result:
            return train_obj
        return train_obj, backend_result

    if singleton.sql_obj.mode == SQLObjRep.CTE:
        train_data = \
            singleton.dbms_connector.run(singleton.pipeline_container.get_pipe_without_selection() + "\n" + sql_code)[0]
    else:
        train_data = singleton.dbms_connector.run(sql_code)[0]

    train_data = mimic_implicit_dim_count_mlinspect(ti, train_data)
    if not backend_result:
        return train_data
    return train_data, singleton.update_hist.sql_update_backend_result(train_data,
                                                                       backend_result,
                                                                       curr_sql_expr_name=name,
                                                                       curr_sql_expr_columns=ti.non_tracking_cols,
                                                                       operation_type=op_type,
                                                                       previous_res_node=name,
                                                                       not_materialize=not_materialize)


def mimic_implicit_dim_count_mlinspect(ti, output):
    if isinstance(ti.data_object, pandas.Series) and ti.data_object.dtype == "bool":
        return pandas.Series(output.squeeze(), dtype=bool)
    global just_the_model
    return output


def set_mlinspect_attributes(args_0):
    """
    Sets mlinspect related attributes required for DAG creation and resolution.
    Can be set even if not used, besides redundancy not downside.
    """
    global column_transformer_share
    input = column_transformer_share.target_obj
    args_0._mlinspect_dag_node = input._mlinspect_dag_node
    args_0._mlinspect_annotation = input._mlinspect_annotation


def differentiate_fit_transform(self, args_0, set_attributes=True):
    """
    To decide whether the current call is a fit_transform vs transform call!
    """
    if hasattr(self, "_fit_data"):
        fit_data = self._fit_data
    else:
        self._fit_data = FitDataCollection({}, fully_set=False)
        fit_data = self._fit_data
    just_transform = fit_data.fully_set  # if already fitted => only transform => no mlinspect functions
    if set_attributes and not just_transform:
        set_mlinspect_attributes(args_0)
    return fit_data, just_transform  # split this way for better readability


def transform_logic(original, self, *args, **kwargs):
    """
    This function calls the correct function based on the intend of fit_transform vs transform.
    """
    # pylint: disable=no-method-argument
    if self._fit_data.fully_set:
        return self.fit_transform(*args, **kwargs)
    # only in the first fit_transform call:
    return original(self, *args, **kwargs)


def materialize_query_with_name(name, cols_to_keep):
    if singleton.sql_obj.materialize:
        query_update = singleton.pipeline_container.get_last_query_materialize(name, cols_to_keep)
        if query_update:  # assert not null => materialized query was not added yet!
            new_view_query, new_name = query_update
            singleton.mapping.update_name(name, new_name)
            singleton.dbms_connector.run(new_view_query)


def store_current_sklearn_op(sql_code, op_id, cte_name, result, tracking_cols, non_tracking_cols, target_cols):
    cte_name, sql_code = singleton.sql_logic.wrap_in_sql_obj(sql_code, op_id, cte_name)
    new_ti_result = TableInfo(data_object=result,
                              tracking_cols=tracking_cols,
                              non_tracking_cols=non_tracking_cols,
                              operation_type=OperatorType.TRANSFORMER,
                              origin_context=None)
    singleton.mapping.add(cte_name, new_ti_result)
    if singleton.sql_obj.mode == SQLObjRep.VIEW:
        singleton.dbms_connector.run(sql_code)  # Create the view.

    # Store info of where the newest column should be taken from:
    global last_name_for_concat_fit
    for col in target_cols:
        last_name_for_concat_fit[col] = cte_name

    return cte_name, sql_code
