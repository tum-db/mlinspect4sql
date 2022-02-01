"""
Tests whether the inspection results from the CTE, VIEW and MATERIALIZED VIEW are the same.
"""
import unittest
import os
from mlinspect.utils import get_project_root
from mlinspect import PipelineInspector
from mlinspect.inspections import RowLineage, MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector


# pylint: disable=protected-access
# !/usr/bin/env python -W ignore::DeprecationWarning
class TestInspectionOutput(unittest.TestCase):
    HEALTHCARE_FILE_PY_R = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                        "healthcare", "healthcare_res.py")
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                      "healthcare", "healthcare.py")
    COMPAS_FILE_PY = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests", "compas",
                                  "compas.py")
    COMPAS_FILE_PY_R = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests", "compas",
                                  "compas_res.py")
    ADULT_SIMPLE_FILE_PY = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                        "adult_simple", "adult_simple.py")
    ADULT_SIMPLE_FILE_PY_R = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                        "adult_simple", "adult_simple_res.py")
    ADULT_COMPLEX_FILE_PY = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                         "adult_complex", "adult_complex.py")
    ADULT_COMPLEX_FILE_PY_R = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                         "adult_complex", "adult_complex_res.py")


    HEALTHCARE_BIAS = ["age_group", "race"]
    COMPAS_BIAS = ["sex", "race"]
    ADULT_COMPLEX_BIAS = ["race"]
    ADULT_SIMPLE_BIAS = ["race"]

    umbra_path = r"/home/maximilian/TUM/umbra"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")

    @staticmethod
    def run_inspection(file_location, bias, to_sql, dbms_connector=None, mode=None, materialize=None):
        result = ""
        inspector_result = PipelineInspector \
            .on_pipeline_from_py_file(file_location) \
            .add_custom_monkey_patching_module(custom_monkeypatching) \
            .add_check(NoBiasIntroducedFor(bias)) \
            .add_check(NoIllegalFeatures()) \
            .add_check(NoMissingEmbeddings()) \
            .add_required_inspection(RowLineage(5)) \
            .add_required_inspection(MaterializeFirstOutputRows(5))

        if to_sql:
            inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector,
                                                               mode=mode, materialize=materialize, row_wise=False)
            if dbms_connector is None:
                return
        else:
            inspector_result = inspector_result.execute()

        check_results = inspector_result.check_to_check_results
        no_bias_check_result = check_results[NoBiasIntroducedFor(bias)]

        distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
            no_bias_check_result)
        result += distribution_changes_overview_df.to_markdown()

        for i in list(no_bias_check_result.bias_distribution_change.items())[:-1]:
            _, join_distribution_changes = i
            for column, distribution_change in join_distribution_changes.items():
                result += "\n"
                result += f"\033[1m Column '{column}'\033[0m"
                result += distribution_change.before_and_after_df.to_markdown()

        return result

    @staticmethod
    def run_for_pandas(file_location, bias):
        res = TestInspectionOutput.run_inspection(file_location=file_location, bias=bias, to_sql=False)
        return res

    @staticmethod
    def run_for_post(file_location, bias, mode="", materialize=False):
        res = TestInspectionOutput.run_inspection(file_location=file_location, bias=bias, to_sql=True,
                                                  dbms_connector=TestInspectionOutput.dbms_connector_p, mode=mode,
                                                  materialize=materialize)
        return res

    @staticmethod
    def run_for_umbra(file_location, bias, mode=""):
        res = TestInspectionOutput.run_inspection(file_location=file_location, bias=bias, to_sql=True,
                                                  dbms_connector=TestInspectionOutput.dbms_connector_u, mode=mode,
                                                  materialize=False)
        return res

    @staticmethod
    def full_equality(file_location, bias):
        res1 = TestInspectionOutput.run_for_pandas(file_location, bias=bias)

        res2 = TestInspectionOutput.run_for_post(file_location, bias=bias, mode="CTE")
        res3 = TestInspectionOutput.run_for_umbra(file_location, bias=bias, mode="CTE")

        res4 = TestInspectionOutput.run_for_post(file_location, bias=bias, mode="VIEW")
        res5 = TestInspectionOutput.run_for_umbra(file_location, bias=bias, mode="VIEW")

        res6 = TestInspectionOutput.run_for_post(file_location, bias=bias, mode="VIEW", materialize=True)

        return res1 == res2 == res3 == res4 == res5 == res6

    # ATTENTION: Only use one test at a time: executing multiple will trigger a bug in gorillas, and some functions
    # won't be caught correctly any more.

    # def test_CTE_inspection_output_equality_healthcare(self):
    #     """
    #     Tests that all inspection results are equal for HEALTHCARE_FILE_PY -> for non_random part!
    #     """
    #     assert self.full_equality(self.HEALTHCARE_FILE_PY_R, bias=self.HEALTHCARE_BIAS)

    # def test_CTE_inspection_output_equality_compas(self):
    #     """
    #     Tests that all inspection results are equal for COMPAS_FILE_PY -> for non_random part!
    #     """
    #     assert self.full_equality(self.COMPAS_FILE_PY_R, bias=self.COMPAS_BIAS)

    # def test_CTE_inspection_output_equality_adult_simple(self):
    #     """
    #     Tests that all inspection results are equal for ADULT_SIMPLE_FILE_PY -> for non_random part!
    #     """
    #     assert self.full_equality(self.ADULT_SIMPLE_FILE_PY_R, bias=self.ADULT_SIMPLE_BIAS)

    def test_CTE_inspection_output_equality_adult_complex(self):
        """
        Tests that all inspection results are equal for ADULT_COMPLEX_FILE_PY -> for non_random part!
        """
        assert self.full_equality(self.ADULT_COMPLEX_FILE_PY_R, bias=self.ADULT_COMPLEX_BIAS)


if __name__ == '__main__':
    unittest.main()
