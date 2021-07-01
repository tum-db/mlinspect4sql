"""
For the SQL run we want tu use an alternative DAG only with the Info we need.
"""
import ast
import dataclasses
import sys
from typing import List
import warnings

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from mlinspect.backends._backend import AnnotatedDfObject, BackendResult
from mlinspect.backends._pandas_backend import execute_inspection_visits_data_source
from mlinspect.inspections._inspection_input import OperatorContext, OperatorType
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.instrumentation._pipeline_executor import singleton


@dataclasses.dataclass(frozen=True)
class SQLInfo:
    """ Infos required to do the inspections and checks after running the pipeline """
    dag_node: DagNode
    annotated_dfobject: AnnotatedDfObject





def add_dag_node(dag_node: DagNode, dag_node_parents: List[DagNode], sql_info: SQLInfo):
    """
    Inserts a new node into the DAG
    """
    # pylint: disable=protected-access
    # print("")
    # print("{}:{}: {}".format(dag_node.caller_filename, dag_node.lineno, dag_node.module))

    # print("source code: {}".format(dag_node.optional_source_code))
    # annotated_df = backend_result.annotated_dfobject
    #
    # if annotated_df.result_data is not None:
    #     annotated_df.result_data._mlinspect_dag_node = dag_node.node_id
    #     if annotated_df.result_annotation is not None:
    #         # TODO: Remove this branching once we support all operators with DAG node mapping
    #         warnings.simplefilter(action='ignore', category=UserWarning)
    #         annotated_df.result_data._mlinspect_annotation = annotated_df.result_annotation
    # if dag_node_parents:
    #     for parent in dag_node_parents:
    #         singleton.inspection_results.dag.add_edge(parent, dag_node)
    # else:
    #     singleton.inspection_results.dag.add_node(dag_node)
    # singleton.op_id_to_dag_node[dag_node.node_id] = dag_node
    # if annotated_df is not None:
    #     singleton.inspection_results.dag_node_to_inspection_results[dag_node] = backend_result.dag_node_annotation
    pass

