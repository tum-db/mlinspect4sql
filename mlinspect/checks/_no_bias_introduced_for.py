"""
The NoBiasIntroducedFor check
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Dict
import collections

import pandas
from matplotlib import pyplot
from pandas import DataFrame

from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import OperatorType, FunctionInfo
from mlinspect.instrumentation._dag_node import DagNode
from mlinspect.inspections._inspection_result import InspectionResult
from mlinspect.monkeypatchingSQL._sql_logic import SQLBackend, mapping


@dataclasses.dataclass(eq=False, frozen=True)
class BiasDistributionChange:
    """
    Did the histogram change too much for one given operation?
    """
    dag_node: DagNode
    acceptable_change: bool
    min_relative_ratio_change: float
    before_and_after_df: DataFrame

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.dag_node == other.dag_node and
                self.acceptable_change == other.acceptable_change and
                self.min_relative_ratio_change == other.min_relative_ratio_change and
                self.before_and_after_df.equals(other.before_and_after_df))


@dataclasses.dataclass
class NoBiasIntroducedForResult(CheckResult):
    """
    Did the histogram change too much for some operations?
    """
    bias_distribution_change: Dict[DagNode, Dict[str, BiasDistributionChange]]


class NoBiasIntroducedFor(Check):
    """
    Does the user pipeline introduce bias because of operators like joins and selects?
    """

    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns, min_allowed_relative_ratio_change=-0.3):
        self.sensitive_columns = sensitive_columns
        self.min_allowed_relative_ratio_change = min_allowed_relative_ratio_change

    @property
    def check_id(self):
        """The id of the Check"""
        return tuple(self.sensitive_columns), self.min_allowed_relative_ratio_change

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [HistogramForColumns(self.sensitive_columns)]

    def evaluate(self, inspection_result: InspectionResult, to_sql: bool) -> CheckResult:
        """Evaluate the check"""
        # pylint: disable=too-many-locals

        # TO_SQL: ###############################################################################################
        if to_sql:
            print("/*" + ("#" * 10) + f"NoBiasIntroducedFor ({', '.join(self.sensitive_columns)}):" + ("#" * 10) + "*/")
            origin_dict = {}
            current_dict = {}
            for sc in self.sensitive_columns:
                origin_of_sc = ""
                current_table_sc = ""  # newest table containing the sensitive column
                for m in reversed(mapping.mapping):  # we reverse because of the adding order -> faster match
                    table_name = m[0]
                    table_info = m[1]
                    table = table_info.data_object
                    if table_name.split("_")[0] != "with":  # check that name represents an original table (f.e. '.csv')
                        if isinstance(table, pandas.Series) and sc == table.name:  # one column .csv
                            origin_of_sc = table_name
                        elif isinstance(table,
                                        pandas.DataFrame) and sc in table.columns.values:  # TODO: substitute by "contains_col" fucntion in TableInfo!
                            origin_of_sc = table_name
                    if (isinstance(table, pandas.DataFrame) and sc in table.columns.values) or \
                            (isinstance(table, pandas.Series) and sc == table.name):
                        current_table_sc = table_name
                # TODO: select all relevant operations in mapping and for those, check the ratios before and after!
                # TODO: also add to select!
                assert (origin_of_sc != "")
                origin_dict[sc] = origin_of_sc
                current_dict[sc] = current_table_sc

            SQLBackend.ratio_track(origin_dict, self.sensitive_columns, current_dict)
            print("/*" + ("#" * 10) + f"NoBiasIntroducedFor DONE" + ("#" * 10) + "*/")
        # TO_SQL DONE! ##########################################################################################

        dag = inspection_result.dag
        histograms = {}
        for dag_node, inspection_results in inspection_result.dag_node_to_inspection_results.items():
            histograms[dag_node] = inspection_results[HistogramForColumns(self.sensitive_columns)]
        relevant_nodes = [node for node in dag.nodes if node.operator_info.operator in {OperatorType.JOIN,
                                                                                        OperatorType.SELECTION} or
                          (node.operator_info.function_info == FunctionInfo('sklearn.impute._base', 'SimpleImputer')
                           and set(node.details.columns).intersection(self.sensitive_columns))]
        check_status = CheckStatus.SUCCESS
        bias_distribution_change = collections.OrderedDict()
        issue_list = []
        for node in relevant_nodes:
            parents = list(dag.predecessors(node))
            column_results = collections.OrderedDict()
            for column in self.sensitive_columns:
                column_result = self.get_histograms_for_node_and_column(column, histograms, node, parents)
                column_results[column] = column_result
                if not column_result.acceptable_change:
                    issue = "A {} causes a min_relative_ratio_change of '{}' by {}, a value below the " \
                            "configured minimum threshold {}!" \
                        .format(node.operator_info.operator.value, column, column_result.min_relative_ratio_change,
                                self.min_allowed_relative_ratio_change)
                    issue_list.append(issue)
                    check_status = CheckStatus.FAILURE

            bias_distribution_change[node] = column_results
        if issue_list:
            description = " ".join(issue_list)
        else:
            description = None
        return NoBiasIntroducedForResult(self, check_status, description, bias_distribution_change)

    def get_histograms_for_node_and_column(self, column, histograms, node, parents):
        """
        Compute histograms for a dag node like a join and a concrete sensitive column like race
        """
        # pylint: disable=too-many-locals, too-many-arguments
        after_map = histograms[node][column]
        after_df = DataFrame(after_map.items(), columns=["sensitive_column_value", "count_after"])

        before_map = {}
        for parent in parents:
            parent_histogram = histograms[parent][column]
            before_map = {**before_map, **parent_histogram}
        before_df = DataFrame(before_map.items(), columns=["sensitive_column_value", "count_before"])

        joined_df = before_df.merge(after_df, on="sensitive_column_value", how="outer")
        joined_df = joined_df.sort_values(by=['sensitive_column_value']).reset_index(drop=True)
        joined_df["count_before"] = joined_df["count_before"].fillna(0, downcast='infer')
        joined_df["count_after"] = joined_df["count_after"].fillna(0, downcast='infer')

        # TODO: What information is useful/what is confusing?
        # joined_df["absolute_change"] = joined_df["count_after"] - joined_df["count_before"]
        # joined_df["relative_change"] = joined_df["absolute_change"] / joined_df["count_before"]
        joined_df["ratio_before"] = joined_df["count_before"] / joined_df["count_before"].sum()
        joined_df["ratio_after"] = joined_df["count_after"] / joined_df["count_after"].sum()
        # joined_df["absolute_ratio_change"] = joined_df["ratio_after"] - joined_df["ratio_before"]
        absolute_ratio_change = joined_df["ratio_after"] - joined_df["ratio_before"]
        joined_df["relative_ratio_change"] = absolute_ratio_change / joined_df["ratio_before"]

        # Dropping nan values (e.g., missing value imputation) is a distribution change we consider okay
        not_nan = joined_df["sensitive_column_value"].notnull()
        min_relative_ratio_change = joined_df[not_nan]["relative_ratio_change"].min()

        all_changes_acceptable = min_relative_ratio_change >= self.min_allowed_relative_ratio_change
        return BiasDistributionChange(node, all_changes_acceptable, min_relative_ratio_change, joined_df)

    @staticmethod
    def plot_distribution_change_histograms(distribution_change: BiasDistributionChange, filename=None,
                                            save_to_file=False):
        """
        Plot before and after histograms visualising a DistributionChange
        """
        pyplot.subplot(1, 2, 1)
        keys = distribution_change.before_and_after_df["sensitive_column_value"]
        keys = [str(key) for key in keys]  # Necessary because of null values
        before_values = distribution_change.before_and_after_df["count_before"]
        after_values = distribution_change.before_and_after_df["count_after"]

        pyplot.bar(keys, before_values)
        pyplot.gca().set_title("before")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        pyplot.subplot(1, 2, 2)

        pyplot.bar(keys, after_values)
        pyplot.gca().set_title("after")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        fig = pyplot.gcf()
        fig.set_size_inches(12, 4)

        if save_to_file:
            fig.savefig(filename + '.svg', bbox_inches='tight')
            fig.savefig(filename + '.png', bbox_inches='tight', dpi=800)

        pyplot.show()
        pyplot.close()

    @staticmethod
    def get_distribution_changes_overview_as_df(no_bias_check_result: NoBiasIntroducedForResult) -> DataFrame:
        """
        Get a pandas DataFrame with an overview of all DistributionChanges
        """
        operator_types = []
        code_references = []
        function_infos = []
        code_snippets = []
        descriptions = []
        assert isinstance(no_bias_check_result.check, NoBiasIntroducedFor)
        sensitive_column_names = no_bias_check_result.check.sensitive_columns
        sensitive_column_names = ["'{}' distribution change below the configured minimum test threshold".format(name)
                                  for name in sensitive_column_names]
        sensitive_columns = []
        for _ in range(len(sensitive_column_names)):
            sensitive_columns.append([])
        for dag_node, distribution_change in no_bias_check_result.bias_distribution_change.items():
            operator_types.append(dag_node.operator_info.operator)
            if dag_node.optional_code_info is not None:
                code_references.append(dag_node.optional_code_info.code_reference)
                code_snippets.append(dag_node.optional_code_info.source_code)
            else:
                code_references.append(dag_node.code_location.lineno)
                code_snippets.append("You can enable code reference tracking for more details.")
            function_infos.append(dag_node.operator_info.function_info)

            descriptions.append(dag_node.details.description or "")
            for index, change_info in enumerate(distribution_change.values()):
                sensitive_columns[index].append(not change_info.acceptable_change)
        return DataFrame(zip(operator_types, descriptions, code_references, code_snippets, function_infos,
                             *sensitive_columns),
                         columns=[
                             "operator_type",
                             "description",
                             "code_reference",
                             "source_code",
                             "module",
                             *sensitive_column_names])
