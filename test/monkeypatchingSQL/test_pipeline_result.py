
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


# !/usr/bin/env python -W ignore::DeprecationWarning
class TestPipelineOutput(unittest.TestCase):
    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")

    umbra_path = r"/home/luca/Documents/Bachelorarbeit/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    pipeline_code = cleandoc("""
        import os
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from mlinspect.utils import get_project_root
        from sklearn.preprocessing import KBinsDiscretizer

        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"),
            na_values='')
        histories = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"),
            na_values='')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

        impute_and_one_hot_encode = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
            ('numeric', StandardScaler(), ['num_children']),
            ('kbin', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'), ['income'])
        ], remainder='drop')
        data = featurisation.fit_transform(data)
        """)

    @staticmethod
    def pipeline_real_result():
        """
        Test pipeline containing all functions.
        """
        import os
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from mlinspect.utils import get_project_root
        from sklearn.preprocessing import KBinsDiscretizer

        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"),
            na_values='')
        histories = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"),
            na_values='')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

        impute_and_one_hot_encode = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
            ('numeric', StandardScaler(), ['num_children']),
            ('kbin', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'), ['income'])
        ], remainder='drop')
        return featurisation.fit_transform(data)

    @staticmethod
    def get_sql_query(pipeline_code, mode, materialize):
        PipelineInspector \
            .on_pipeline_from_string(pipeline_code) \
            .execute_in_sql(dbms_connector=None, mode=mode, materialize=materialize)

        setup_file = \
            pathlib.Path(get_project_root() / r"mlinspect/to_sql/generated_code/create_table.sql")
        test_file = \
            pathlib.Path(get_project_root() / r"mlinspect/to_sql/generated_code/pipeline.sql")

        with setup_file.open("r") as file:
            setup_code = file.read()

        with test_file.open("r") as file:
            test_code = file.read()

        return setup_code, test_code

    @staticmethod
    def full_equality(mode, materialized):
        real_result = np.array(TestPipelineOutput.pipeline_real_result()).astype(float)

        setup_code, test_code = TestPipelineOutput.get_sql_query(TestPipelineOutput.pipeline_code,
                                                                   mode=mode, materialize=materialized)
        pipeline_code = setup_code + "\n" + test_code

        sql_result_p = TestPipelineOutput.dbms_connector_p.run(pipeline_code)[0].astype(float)
        if not materialized:
            sql_result_u = TestPipelineOutput.dbms_connector_u.run(pipeline_code)[0].astype(float)
        else:
            sql_result_u = sql_result_p
        # sql_result[sql_result == ""] = "nan"

        # Row-Order is irrelevant!
        return np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(real_result.flat, axis=0)) and \
               np.allclose(np.sort(sql_result_u.flat, axis=0), np.sort(real_result.flat, axis=0))

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

