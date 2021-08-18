"""
Instrument and executes the pipeline
"""
import ast
import pathlib
from typing import Iterable, List

import gorilla
import pandas
import nbformat
import networkx
from astmonkey.transformers import ParentChildNodeTransformer
import astor  # from astmonkey import visitors -> visitors unfortunately is buggy.
from nbconvert import PythonExporter

from .. import monkeypatchingSQL
from .. import monkeypatching
from ._call_capture_transformer import CallCaptureTransformer
from .._inspector_result import InspectorResult
from ..checks._check import Check
from ..inspections import InspectionResult
from ..inspections._inspection import Inspection

# to_sql related imports:
from mlinspect.to_sql._mode import SQLMode, SQLObjRep
from mlinspect.to_sql.dbms_connectors.dbms_connector import Connector
from mlinspect.to_sql._sql_logic import SQLLogic
from mlinspect.to_sql.sql_query_container import SQLQueryContainer
from mlinspect.to_sql.checks_and_inspections_sql._histogram_for_columns import SQLHistogramForColumns
from mlinspect.to_sql.py_to_sql_mapping import DfToStringMapping
from mlinspect.checks._no_bias_introduced_for import NoBiasIntroducedFor
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    # pylint: disable=too-many-instance-attributes

    source_code_path = None
    source_code = None
    script_scope = {}
    lineno_next_call_or_subscript = -1
    col_offset_next_call_or_subscript = -1
    end_lineno_next_call_or_subscript = -1
    end_col_offset_next_call_or_subscript = -1
    next_op_id = 0
    next_missing_op_id = -1
    track_code_references = True
    op_id_to_dag_node = dict()
    inspection_results = InspectionResult(networkx.DiGraph(), dict())
    inspections = []
    custom_monkey_patching = []
    # to SQL related attributes:

    # user input flags:
    to_sql = False
    dbms_connector = None

    # for intern use:
    pipeline_result = None
    mapping = None
    pipeline_container = None
    update_hist = None
    sql_logic = None
    sql_obj = None
    root_dir_to_sql = pathlib.Path(__file__).resolve().parent.parent / "to_sql/generated_code"

    backup_eq = None
    backup_ne = None
    backup_lt = None
    backup_le = None
    backup_gt = None
    backup_ge = None

    def run(self, *,
            notebook_path: str or None = None,
            python_path: str or None = None,
            python_code: str or None = None,
            inspections: Iterable[Inspection] or None = None,
            checks: Iterable[Check] or None = None,
            reset_state: bool = True,
            track_code_references: bool = True,
            custom_monkey_patching: List[any] = None,
            to_sql: bool = False,
            dbms_connector: Connector = None,
            mode: str = "",
            materialize: bool = False,
            row_wise: bool = False
            ) -> InspectorResult:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        # pylint: disable=too-many-arguments

        # Add all SQL related attributes:
        self.to_sql = to_sql
        if self.to_sql:

            if mode not in [r.value for r in SQLObjRep]:
                raise ValueError("The attribute mode can either be \"CTE\" or \"VIEW\".")
            if mode == "CTE" and materialize:
                raise ValueError("Materializing is only available for mode \"VIEW\".")

            self.sql_obj = SQLMode(SQLObjRep.CTE if mode == "CTE" else SQLObjRep.VIEW, materialize)

            self.dbms_connector = dbms_connector
            if not self.dbms_connector:  # is None
                print("\nJust translation to SQL is performed! "
                      "\n-> SQL-Code placed at: mlinspect/to_sql/generated_code.sql\n")
                self.dbms_connector = PostgresqlConnector(just_code=True, add_mlinspect_serial=row_wise)
                checks = None
                inspections = []

            # Empty the "to_sql_output" folder if necessary:
            [f.unlink() for f in self.root_dir_to_sql.glob("*.sql") if f.is_file()]

            # This mapping allows to keep track of the pandas.DataFrame and pandas.Series w.r.t. to SQL-table repr.!
            self.mapping = DfToStringMapping()
            self.pipeline_container = SQLQueryContainer(self.root_dir_to_sql, sql_obj=self.sql_obj)
            self.update_hist = SQLHistogramForColumns(self.dbms_connector, self.mapping, self.pipeline_container,
                                                       sql_obj=self.sql_obj)
            sql_logic_id = 1
            if self.sql_logic:  # Necessary to avoid duplicate names, when running multiple inspections in a row!
                sql_logic_id = self.sql_logic.id
            self.sql_logic = SQLLogic(mapping=self.mapping, pipeline_container=self.pipeline_container,
                                      dbms_connector=self.dbms_connector, sql_obj=self.sql_obj, id=sql_logic_id)

            # Fix the problem gorilla has with restoring the comparison operators:
            self.backup_eq = pandas.Series.__eq__
            self.backup_ne = pandas.Series.__ne__
            self.backup_lt = pandas.Series.__lt__
            self.backup_le = pandas.Series.__le__
            self.backup_gt = pandas.Series.__gt__
            self.backup_ge = pandas.Series.__ge__

        if reset_state:
            # reset_state=False should only be used internally for performance experiments etc!
            # It does not ensure the same inspections are still used as args etc.
            self.reset()

        if inspections is None:
            inspections = []
        if checks is None:
            checks = []
        if custom_monkey_patching is None:
            custom_monkey_patching = []

        check_inspections = set()

        for check in checks:
            check_inspections.update(check.required_inspections)
            if isinstance(check, NoBiasIntroducedFor) and self.to_sql:
                check._to_sql = self.to_sql
                check.mapping = self.mapping
                check.pipeline_container = self.pipeline_container
                check.dbms_connector = self.dbms_connector
                check.sql_obj = self.sql_obj

        all_inspections = list(set(inspections).union(check_inspections))
        self.inspections = all_inspections
        self.track_code_references = track_code_references
        self.custom_monkey_patching = custom_monkey_patching

        # Here the modified code is created and run:
        self.run_inspections(notebook_path, python_code, python_path)
        check_to_results = dict(
            (check, check.evaluate(self.inspection_results)) for check in checks)
        return InspectorResult(self.inspection_results.dag, self.inspection_results.dag_node_to_inspection_results,
                               check_to_results, to_sql_pipe_result=self.pipeline_result)

    def run_inspections(self, notebook_path, python_code, python_path):
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use, too-many-locals
        self.source_code, self.source_code_path = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(self.source_code)

        parsed_modified_ast = self.instrument_pipeline(parsed_ast, self.track_code_references)

        modified_code = astor.to_source(parsed_modified_ast)
        # Do the monkey patching and the inspection:
        exec(compile(modified_code, filename=self.source_code_path, mode="exec"), PipelineExecutor.script_scope)
        return

    def get_next_op_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_op_id = self.next_op_id
        self.next_op_id += 1
        return current_op_id

    def get_next_missing_op_id(self):
        """
        Each unknown operator in the DAG gets a consecutive unique negative id
        """
        current_missing_op_id = self.next_missing_op_id
        self.next_missing_op_id -= 1
        return current_missing_op_id

    def reset(self):
        """
        Reset all attributes in the singleton object. This can be used when there are multiple repeated calls to mlinspect
        """
        self.source_code_path = None
        self.source_code = None
        self.script_scope = {}
        self.lineno_next_call_or_subscript = -1
        self.col_offset_next_call_or_subscript = -1
        self.end_lineno_next_call_or_subscript = -1
        self.end_col_offset_next_call_or_subscript = -1
        self.next_op_id = 0
        self.next_missing_op_id = -1
        self.track_code_references = True
        self.op_id_to_dag_node = dict()
        self.inspection_results = InspectionResult(networkx.DiGraph(), dict())
        self.inspections = []
        self.custom_monkey_patching = []

        # to_sql related reset:
        if self.to_sql:
            [f.unlink() for f in self.root_dir_to_sql.glob("*.sql") if f.is_file()]
            self.mapping = DfToStringMapping()
            self.pipeline_container = SQLQueryContainer(self.root_dir_to_sql, sql_obj=self.sql_obj)
            self.update_hist = SQLHistogramForColumns(self.dbms_connector, self.mapping, self.pipeline_container,
                                                       sql_obj=self.sql_obj)
            sql_logic_id = 1
            if self.sql_logic:  # Necessary to avoid duplicate names, when running multiple inspections in a row!
                sql_logic_id = self.sql_logic.id
            self.sql_logic = SQLLogic(mapping=self.mapping, pipeline_container=self.pipeline_container,
                                      dbms_connector=self.dbms_connector, sql_obj=self.sql_obj, id=sql_logic_id)

    @staticmethod
    def instrument_pipeline(parsed_ast, track_code_references):
        """
        Instrument the pipeline AST to instrument function calls
        """
        # insert set_code_reference calls
        if track_code_references:
            #  Needed to get the parent assign node for subscript assigns.
            #  Without this, "pandas_df['baz'] = baz + 1" would only be "pandas_df['baz']"
            parent_child_transformer = ParentChildNodeTransformer()
            parsed_ast = parent_child_transformer.visit(parsed_ast)
            call_capture_transformer = CallCaptureTransformer()
            parsed_ast = call_capture_transformer.visit(parsed_ast)
            parsed_ast = ast.fix_missing_locations(parsed_ast)

        # from mlinspect2._pipeline_executor import set_code_reference, monkey_patch
        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation._pipeline_executor',
                                          names=[ast.alias(name='set_code_reference_call', asname=None),
                                                 ast.alias(name='set_code_reference_subscript', asname=None),
                                                 ast.alias(name='monkey_patch', asname=None),
                                                 ast.alias(name='undo_monkey_patch', asname=None)],
                                          level=0)
        parsed_ast.body.insert(0, func_import_node)

        # monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.insert(1, inspect_import_node)
        # undo_monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='undo_monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.append(inspect_import_node)

        parsed_ast = ast.fix_missing_locations(parsed_ast)

        return parsed_ast

    @staticmethod
    def load_source_code(notebook_path, python_path, python_code):
        """
        Load the pipeline source code from the specified source
        """
        sources = [notebook_path, python_path, python_code]
        assert sum(source is not None for source in sources) == 1
        if python_path is not None:
            with open(python_path) as file:
                source_code = file.read()
            source_code_path = python_path
        elif notebook_path is not None:
            with open(notebook_path) as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)
            source_code_path = notebook_path
        elif python_code is not None:
            source_code = python_code
            source_code_path = "<string-source>"
        else:
            assert False
        return source_code, source_code_path


# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def set_code_reference_call(lineno, col_offset, end_lineno, end_col_offset, **kwargs):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return kwargs


def set_code_reference_subscript(lineno, col_offset, end_lineno, end_col_offset, arg):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return arg


def monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.apply(patch)


def undo_monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.revert(patch)

    # Fix the problem gorilla has with restoring the comparison operators:
    if singleton.to_sql:
        pandas.Series.__eq__ = singleton.backup_eq
        pandas.Series.__ne__ = singleton.backup_ne
        pandas.Series.__lt__ = singleton.backup_lt
        pandas.Series.__le__ = singleton.backup_le
        pandas.Series.__gt__ = singleton.backup_gt
        pandas.Series.__ge__ = singleton.backup_ge


def get_monkey_patching_patch_sources():
    """
    Get monkey patches provided by mlinspect and custom patches provided by the user
    """
    patch_sources = [monkeypatching]
    if singleton.to_sql:
        patch_sources = [monkeypatchingSQL]
    patch_sources.extend(singleton.custom_monkey_patching)
    return patch_sources
