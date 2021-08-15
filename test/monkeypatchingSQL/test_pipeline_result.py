"""
Tests whether the pipeline output from the CTE, VIEW and MATERIALIZED VIEW are the same.
"""
import unittest
from mlinspect.utils import get_project_root
from mlinspect import PipelineInspector
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
import pathlib
from inspect import cleandoc
import numpy as np
from example_to_sql._code_as_string import get_healthcare_pipe_code, get_sql_query


# !/usr/bin/env python -W ignore::DeprecationWarning
class TestPipelineOutput(unittest.TestCase):
    real_res = None

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")

    umbra_path = r"/home/luca/Documents/Bachelorarbeit/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    @staticmethod
    def real_result():
        setup_code, test_code = get_healthcare_pipe_code(only_pandas=False, include_training=False)
        pipeline_code = setup_code + "\n" + test_code
        script_scope = {}
        exec(compile(pipeline_code, filename="xxx", mode="exec"), script_scope)
        return script_scope["result"]

    def full_equality(self, mode, materialized):
        setup_code, test_code = get_healthcare_pipe_code(only_pandas=False, include_training=False)
        pipeline_code = setup_code + "\n" + test_code

        if not self.real_res:
            self.real_res = self.real_result()

        setup_code, test_code = get_sql_query(pipeline_code, mode, materialized)
        pipeline_code_sql = setup_code + "\n" + test_code
        sql_result_p = TestPipelineOutput.dbms_connector_p.run(pipeline_code_sql)[0].astype(float)
        if not materialized:
            sql_result_u = TestPipelineOutput.dbms_connector_u.run(pipeline_code_sql)[0].astype(float)
        else:
            sql_result_u = sql_result_p
        # sql_result[sql_result == ""] = "nan"

        # Row-Order is irrelevant!
        return np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(self.real_res.flat, axis=0)) and \
               np.allclose(np.sort(sql_result_u.flat, axis=0), np.sort(self.real_res.flat, axis=0))

    def test_pipeline_result_for_equality(self):
        """
        Tests that all inspection results are equal for HEALTHCARE_FILE_PY!
        """
        assert self.full_equality(mode="VIEW", materialized=False)

    def test_pipeline_result_for_equality_cte(self):
        """
        Tests that all inspection results are equal for HEALTHCARE_FILE_PY!
        """
        assert self.full_equality(mode="CTE", materialized=False)

    def test_pipeline_result_for_equality_view_mat(self):
        """
        Tests that all inspection results are equal for HEALTHCARE_FILE_PY!
        """
        assert self.full_equality(mode="VIEW", materialized=True)
