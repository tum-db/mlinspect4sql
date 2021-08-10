# import os
#
# import numpy as np
#
# from mlinspect.utils import get_project_root
# import pandas as pd
# from mlinspect import PipelineInspector, OperatorType
# from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
# from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
# from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
# from inspect import cleandoc
# from example_pipelines.healthcare import custom_monkeypatching
# from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
# import time
# from IPython.display import display
# from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
#
# # import tensorflow as tf
# # tf.logging.set_verbosity(tf.logging.ERROR)
# HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
# COMPAS_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
# ADULT_SIMPLE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")
# ADULT_COMPLEX_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")
# ROW_WISE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "row_wise", "row_wise.py")
#
# HEALTHCARE_BIAS = ["age_group", "race"]
# COMPAS_BIAS = ["sex", "race"]
# ADULT_COMPLEX_BIAS = ["race"]
# ADULT_SIMPLE_BIAS = ["race"]
#
#
# def run_inspection(file_location, bias, to_sql, dbms_connector=None, mode=None, materialize=None):
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     from mlinspect.visualisation import save_fig_to_path
#
#     inspector_res = PipelineInspector\
#         .on_pipeline_from_py_file(file_location) \
#         .add_custom_monkey_patching_module(custom_monkeypatching) \
#         .add_check(NoBiasIntroducedFor(bias)) \
#         .add_check(NoIllegalFeatures()) \
#         .add_check(NoMissingEmbeddings()) \
#         .add_required_inspection(RowLineage(5)) \
#         .add_required_inspection(MaterializeFirstOutputRows(5))
#
#     if to_sql:
#         # assert (dbms_connector)
#         inspector_result = inspector_res.execute_in_sql(dbms_connector=dbms_connector,
#                                                            mode=mode, materialize=materialize, row_wise=False)
#         if dbms_connector is None:
#             return
#     else:
#         inspector_result = inspector_res.execute()
#
#
#     extracted_dag = inspector_result.dag
#     filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
#     save_fig_to_path(extracted_dag, filename)
#     im = Image.open(filename)
#     plt.imshow(im)
#
#     check_results = inspector_result.check_to_check_results
#     no_bias_check_result = check_results[NoBiasIntroducedFor(bias)]
#
#     distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
#         no_bias_check_result)
#     print(distribution_changes_overview_df.to_markdown())
#
#     for i in list(no_bias_check_result.bias_distribution_change.items()):
#         _, join_distribution_changes = i
#         for column, distribution_change in join_distribution_changes.items():
#             print("")
#             print(f"\033[1m Column '{column}'\033[0m")
#             print(distribution_change.before_and_after_df.to_markdown())
#
#
# def run_for_all(file_location, bias, mode="", materialize=None):
#
#     # t0 = time.time()
#     # run_inspection(file_location=file_location, bias=bias,  to_sql=False)
#     # t1 = time.time()
#     # print("\nTime spend with original: " + str(t1 - t0))
#
#
#     # t0 = time.time()
#     # run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_p, mode=mode, materialize=materialize)
#     # t1 = time.time()
#     # print("\nTime spend with modified SQL inspections: " + str(t1 - t0))
#
#     t0 = time.time()
#     run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_u, mode=mode, materialize=False)
#     t1 = time.time()
#     print("\nTime spend with modified SQL inspections: " + str(t1 - t0))
#
#
#
# umbra_path = r"/home/luca/Documents/Bachelorarbeit/umbra-students"
# dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
#                                   umbra_dir=umbra_path)
#
# dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
#                                        port=5432,
#                                        host="localhost")
#
# if __name__ == "__main__":
#     run_for_all(file_location=HEALTHCARE_FILE_PY,       bias=HEALTHCARE_BIAS,       mode="VIEW", materialize=True)
#     # run_for_all(file_location=COMPAS_FILE_PY,           bias=COMPAS_BIAS,           mode="VIEW", materialize=True)
#     # run_for_all(file_location=ADULT_SIMPLE_FILE_PY,     bias=ADULT_SIMPLE_BIAS,     mode="VIEW", materialize=True)
#     # run_for_all(file_location=ADULT_COMPLEX_FILE_PY,    bias=ADULT_COMPLEX_BIAS,    mode="VIEW", materialize=True)
#

"""
Tests whether the inspection results from the CTE, VIEW and MATERIALIZED VIEW are the same.
"""
# import unittest
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
import pandas
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


# !/usr/bin/env python -W ignore::DeprecationWarning
class TestInspectionOutput():

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
            os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"),
            na_values='')
        histories = pd.read_csv(
            os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"),
            na_values='')

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
            ('numeric', StandardScaler(), ['num_children']),
            ('kbin', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'), ['income'])
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
        #  columns in the end: 'smoker', 'county', 'num_children', 'race', 'income'
        data = featurisation.fit_transform(data)
        # return np.hstack((data[["income"]], data_k))
        return data

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

        # ['smoker', 'county', 'num_children', 'race', 'income']
        real_result = np.array(TestInspectionOutput.pipeline_real_result()).astype(float)


        setup_code, test_code = TestInspectionOutput.get_sql_query(TestInspectionOutput.pipeline_code,
                                                                   mode="VIEW", materialize=False)
        pipeline_code = setup_code + "\n" + test_code

        # ['smoker', 'income', 'race', 'num_children', 'county']
        sql_result = TestInspectionOutput.dbms_connector_p.run(pipeline_code)[0].astype(float)
        # sql_result = TestInspectionOutput.dbms_connector_u.run(pipeline_code)[0].astype(float)
        # sql_result[sql_result == ""] = "nan"



        print()

        # incomes = real_result.transpose()[5].astype(float)
        # incomes_sql = sql_result.transpose()[5].astype(float)

        return np.allclose(np.sort(sql_result.flat, axis=0), np.sort(real_result.flat, axis=0))  # Order is irrelevant!

    def pipeline_result_for_equality(self):
        """
        Tests that all inspection results are equal for HEALTHCARE_FILE_PY!
        """
        assert self.full_equality()


if __name__ == '__main__':
    # unittest.main()
    test = TestInspectionOutput()
    test.pipeline_result_for_equality()