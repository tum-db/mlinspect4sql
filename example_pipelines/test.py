import os

import numpy as np

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

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
COMPAS_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
ADULT_SIMPLE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")
ADULT_COMPLEX_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex",
                                     "adult_complex.py")
ROW_WISE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "row_wise",
                                "row_wise.py")

HEALTHCARE_BIAS = ["age_group", "race"]
COMPAS_BIAS = ["sex", "race"]


def run_inspection(file_location, bias, to_sql, sql_one_run=False, dbms_connector=None, mode=None, materialize=None):
    from PIL import Image
    import matplotlib.pyplot as plt
    from mlinspect.visualisation import save_fig_to_path

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(file_location) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(bias)) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        assert (dbms_connector)
        inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector, sql_one_run=sql_one_run,
                                                           mode=mode, materialize=materialize, row_wise=False)
    else:
        inspector_result = inspector_result.execute()

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


def run_for_all(file_location, bias, one_pass=False, mode="", materialize=None):
    # t0 = time.time()
    # run_inspection(to_sql=False)
    # t1 = time.time()
    # print("\nTime spend with original: " + str(t1 - t0))

    t0 = time.time()
    run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_u, sql_one_run=one_pass, mode=mode, materialize=materialize)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # t0 = time.time()
    # run_inspection(file_location=file_location, bias=bias, to_sql=True, dbms_connector=dbms_connector_p, sql_one_run=one_pass, mode=mode, materialize=materialize)
    # t1 = time.time()
    # print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

umbra_path = r"/home/luca/Documents/Bachelorarbeit/umbra-students"
dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                  umbra_dir=umbra_path)

dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                       port=5432,
                                       host="localhost")

if __name__ == "__main__":
    # run_for_all(file_location=HEALTHCARE_FILE_PY, bias=HEALTHCARE_BIAS, one_pass=False, mode="CTE", materialize=False)
    run_for_all(file_location=COMPAS_FILE_PY, bias=COMPAS_BIAS, one_pass=False, mode="VIEW", materialize=False)
    # full_row_wise(one_pass=False, mode="CTE", materialize=False)
    # full_compas(one_pass=False, mode="VIEW", materialize=False)
    # full_adult_simple(one_pass=False, mode="VIEW")
    # full_adult_complex(one_pass=False, mode="CTE")
