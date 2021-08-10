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
import pathlib
from inspect import cleandoc


# !/usr/bin/env python -W ignore::DeprecationWarning
class TestInspectionOutput(unittest.TestCase):
    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")

    pipeline_code = cleandoc("""
        import os
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from mlinspect.utils import get_project_root

        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv(
            os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"),
            na_values='?')
        histories = pd.read_csv(
            os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"),
            na_values='?')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
        impute_and_one_hot_encode = Pipeline([
            ('impute',
                SimpleImputer(strategy='most_frequent')),
            ('encode',
                OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode,
                ['smoker', 'county', 'race']),
            ('numeric', StandardScaler(), ['num_children', 'income']),
        ], remainder='drop')
        featurisation.fit_transform(data)
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

        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"),
            na_values='?')
        histories = pd.read_csv(
            os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"),
            na_values='?')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
        impute_and_one_hot_encode = Pipeline([
            ('impute',
             SimpleImputer(strategy='most_frequent')),
            ('encode',
             OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
            ('numeric', StandardScaler(), ['num_children', 'income']),
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
    def full_equality():
        setup_code, test_code = TestInspectionOutput.get_sql_query(TestInspectionOutput.pipeline_code,
                                                                   mode="VIEW", materialize=False)
        pipeline_code = setup_code + "\n" + test_code

        sql_result = TestInspectionOutput.dbms_connector_p.run(pipeline_code)
        real_result = TestInspectionOutput.pipeline_real_result()
        print()
        return sql_result == real_result

    def test_pipeline_result_for_equality(self):
        """
        Tests that all inspection results are equal for HEALTHCARE_FILE_PY!
        """
        assert self.full_equality()


if __name__ == '__main__':
    # unittest.main()
    test = TestInspectionOutput
    test.test_pipeline_result_for_equality()