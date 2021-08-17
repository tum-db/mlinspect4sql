import os

import numpy as np
import pathlib
from mlinspect.utils import get_project_root
import pandas as pd
from mlinspect import PipelineInspector, OperatorType
from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from inspect import cleandoc
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
import time
from IPython.display import display
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from example_to_sql._code_as_string import get_healthcare_pipe_code

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
COMPAS_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
ADULT_SIMPLE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")
ADULT_COMPLEX_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")
ROW_WISE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "row_wise", "row_wise.py")

HEALTHCARE_BIAS = ["age_group", "race"]
COMPAS_BIAS = ["sex", "race"]
ADULT_COMPLEX_BIAS = ["race"]
ADULT_SIMPLE_BIAS = ["race"]


def run_inspection(file_location, bias, to_sql, dbms_connector=None, mode=None, materialize=None):
    from PIL import Image
    import matplotlib.pyplot as plt
    from mlinspect.visualisation import save_fig_to_path

    inspector_res = PipelineInspector \
        .on_pipeline_from_py_file(file_location) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(bias)) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        # assert (dbms_connector)
        inspector_result = inspector_res.execute_in_sql(dbms_connector=dbms_connector,
                                                        mode=mode, materialize=materialize, row_wise=False)
        if dbms_connector is None:
            return
    else:
        inspector_result = inspector_res.execute()

    extracted_dag = inspector_result.dag
    filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
    save_fig_to_path(extracted_dag, filename)
    im = Image.open(filename)
    plt.imshow(im)

    check_results = inspector_result.check_to_check_results
    no_bias_check_result = check_results[NoBiasIntroducedFor(bias)]

    distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
        no_bias_check_result)
    print(distribution_changes_overview_df.to_markdown())

    for i in list(no_bias_check_result.bias_distribution_change.items()):
        _, join_distribution_changes = i
        for column, distribution_change in join_distribution_changes.items():
            print("")
            print(f"\033[1m Column '{column}'\033[0m")
            print(distribution_change.before_and_after_df.to_markdown())


def run_for_all(file_location, bias, mode="", materialize=None):
    # t0 = time.time()
    # run_inspection(file_location=file_location, bias=bias, to_sql=False)
    # t1 = time.time()
    # print("\nTime spend with original: " + str(t1 - t0))

    t0 = time.time()
    run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_p, mode=mode,
                   materialize=materialize)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # t0 = time.time()
    # run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_u, mode=mode,
    #                materialize=False)
    # t1 = time.time()
    # print("\nTime spend with modified SQL inspections: " + str(t1 - t0))


umbra_path = r"/home/luca/Documents/Bachelorarbeit/umbra-students"
dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                  umbra_dir=umbra_path)

dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                       port=5432,
                                       host="localhost")


def pipeline_real_result(pipeline_location):
    with pathlib.Path(pipeline_location).open("r") as file:
        pipeline_code = file.read()
    variables = {}
    exec(compile(pipeline_code, mode="exec"), variables)
    return variables["result"]


def get_sql_query(pipeline_code, mode, materialize):
    # with pathlib.Path(pipeline_location).open("r") as file:
    #     pipeline_code = file.read()

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


def full_equality(pipeline_location, mode, materialized):
    real_result = np.array(pipeline_real_result()).astype(float)

    setup_code, test_code = get_sql_query(pipeline_location, mode=mode, materialize=materialized)
    pipeline_code = setup_code + "\n" + test_code

    sql_result_p = dbms_connector_p.run(pipeline_code)[0].astype(float)
    if not materialized:
        sql_result_u = dbms_connector_u.run(pipeline_code)[0].astype(float)
    else:
        sql_result_u = sql_result_p

    # Row-Order is irrelevant!
    result = np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(real_result.flat, axis=0)) and \
             np.allclose(np.sort(sql_result_u.flat, axis=0), np.sort(real_result.flat, axis=0))

    return result


if __name__ == "__main__":
    # h = r"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_histories_generated_1000.csv"
    # p = r"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_patients_generated_1000.csv"
    # h = r"/home/luca/Documents/Bachelorarbeit/mlinspect/test/monkeypatchingSQL/pipelines_for_tests/healthcare/histories.csv"
    # p = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv")
    # setup_code, test_code = get_healthcare_pipe_code(path_patients=p, path_histories=h, only_pandas=False,
    #                                                  include_training=False)
    # pipeline_code = setup_code + "\n" + test_code
    # script_scope = {}
    # exec(compile(pipeline_code, filename="xxx", mode="exec"), script_scope)
    # real_result = script_scope["result"]
    # print()
    #
    # setup_code, test_code = get_sql_query(pipeline_code, mode="VIEW", materialize=False)
    # pipeline_code_sql = setup_code + "\n" + test_code
    # sql_result_p = dbms_connector_p.run(pipeline_code_sql)[0].astype(float)
    # sql_result_u = dbms_connector_u.run(pipeline_code_sql)[0].astype(float)
    # print()
    #
    # result = np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(real_result.flat, axis=0)) and \
    #          np.allclose(np.sort(sql_result_u.flat, axis=0), np.sort(real_result.flat, axis=0))
    #
    # print(result)
    #
    # setup_code, test_code = get_sql_query(pipeline_code, mode="CTE", materialize=False)
    # pipeline_code_sql = setup_code + "\n" + test_code
    # sql_result_p = dbms_connector_p.run(pipeline_code_sql)[0].astype(float)
    # sql_result_u = dbms_connector_u.run(pipeline_code_sql)[0].astype(float)
    # print()
    #
    # result = np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(real_result.flat, axis=0)) and \
    #          np.allclose(np.sort(sql_result_u.flat, axis=0), np.sort(real_result.flat, axis=0))
    #
    # print(result)
    #
    # setup_code, test_code = get_sql_query(pipeline_code, mode="VIEW", materialize=True)
    # pipeline_code_sql = setup_code + "\n" + test_code
    # sql_result_p = dbms_connector_p.run(pipeline_code_sql)[0].astype(float)
    # print()
    #
    # result = np.allclose(np.sort(sql_result_p.flat, axis=0), np.sort(real_result.flat, axis=0))
    #
    # print(result)
    #
    # PipelineInspector \
    #     .on_pipeline_from_string(pipeline_code) \
    #     .execute_in_sql(dbms_connector=None, mode="VIEW", materialize=True)
    #
    run_for_all(file_location=HEALTHCARE_FILE_PY,       bias=HEALTHCARE_BIAS,       mode="VIEW", materialize=True)
    # run_for_all(file_location=HEALTHCARE_FILE_PY, bias=HEALTHCARE_BIAS, mode="VIEW", materialize=False)
    # run_for_all(file_location=COMPAS_FILE_PY,           bias=COMPAS_BIAS,           mode="VIEW", materialize=False)
    # run_for_all(file_location=ADULT_SIMPLE_FILE_PY,     bias=ADULT_SIMPLE_BIAS,     mode="VIEW", materialize=True)
    # run_for_all(file_location=ADULT_COMPLEX_FILE_PY,    bias=ADULT_COMPLEX_BIAS,    mode="VIEW", materialize=True)

    # full_equality(mode="VIEW", materialized=False)
