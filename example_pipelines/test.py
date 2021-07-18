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

def example_one(to_sql, despite=True, sql_one_run=False, dbms_connector=None, reset=False):
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_FILE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(["age_group", "race"])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        assert (dbms_connector)
        inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector, reset_state=reset,
                                                           sql_one_run=sql_one_run)
    else:
        inspector_result = inspector_result.execute(reset_state=reset)

    if despite:
        extracted_dag = inspector_result.dag

        filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
        from mlinspect.visualisation import save_fig_to_path
        from PIL.Image import Image
        save_fig_to_path(extracted_dag, filename)

        from PIL import Image
        import matplotlib.pyplot as plt
        im = Image.open(filename)
        plt.imshow(im)
        # plt.waitforbuttonpress()

        dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
        check_results = inspector_result.check_to_check_results
        no_bias_check_result = check_results[NoBiasIntroducedFor(["age_group", "race"])]

        distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
            no_bias_check_result)
        print(distribution_changes_overview_df.to_markdown())

        for i in list(no_bias_check_result.bias_distribution_change.items()):
            _, join_distribution_changes = i
            for column, distribution_change in join_distribution_changes.items():
                print("")
                print(f"\033[1m Column '{column}'\033[0m")
                print(distribution_change.before_and_after_df.to_markdown())


def example_compas(to_sql, despite=True, sql_one_run=True, dbms_connector=None, reset=False):
    COMPAS_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(COMPAS_FILE_PY) \
        .add_check(NoBiasIntroducedFor(['sex', 'race'])) \
        .add_check(NoIllegalFeatures()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        assert (dbms_connector)
        inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector, reset_state=reset,
                                                           sql_one_run=sql_one_run)
    else:
        inspector_result = inspector_result.execute(reset_state=reset)

    if despite:
        extracted_dag = inspector_result.dag

        filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
        from mlinspect.visualisation import save_fig_to_path
        from PIL.Image import Image
        save_fig_to_path(extracted_dag, filename)

        from PIL import Image
        import matplotlib.pyplot as plt
        im = Image.open(filename)
        plt.imshow(im)
        # plt.waitforbuttonpress()

        dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
        check_results = inspector_result.check_to_check_results
        no_bias_check_result = check_results[NoBiasIntroducedFor(['sex', 'race'])]

        distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
            no_bias_check_result)
        print(distribution_changes_overview_df.to_markdown())

        for i in list(no_bias_check_result.bias_distribution_change.items()):
            _, join_distribution_changes = i
            for column, distribution_change in join_distribution_changes.items():
                print("")
                print(f"\033[1m Column '{column}'\033[0m")
                print(distribution_change.before_and_after_df.to_markdown())


def example_adult_simple(to_sql, despite=True, sql_one_run=True, dbms_connector=None, reset=False):
    ADULT_SIMPLE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(ADULT_SIMPLE_FILE_PY) \
        .add_check(NoBiasIntroducedFor(["race"])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        assert (dbms_connector)
        inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector, reset_state=reset,
                                                           sql_one_run=sql_one_run)
    else:
        inspector_result = inspector_result.execute(reset_state=reset)

    if despite:
        extracted_dag = inspector_result.dag

        filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
        from mlinspect.visualisation import save_fig_to_path
        from PIL.Image import Image
        save_fig_to_path(extracted_dag, filename)

        from PIL import Image
        import matplotlib.pyplot as plt
        im = Image.open(filename)
        plt.imshow(im)
        # plt.waitforbuttonpress()

        dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
        check_results = inspector_result.check_to_check_results
        no_bias_check_result = check_results[NoBiasIntroducedFor(["race"])]

        distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
            no_bias_check_result)
        print(distribution_changes_overview_df.to_markdown())

        for i in list(no_bias_check_result.bias_distribution_change.items()):
            _, join_distribution_changes = i
            for column, distribution_change in join_distribution_changes.items():
                print("")
                print(f"\033[1m Column '{column}'\033[0m")
                print(distribution_change.before_and_after_df.to_markdown())


def example_adult_complex(to_sql, despite=True, sql_one_run=True, dbms_connector=None, reset=False):
    ADULT_COMPLEX_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(ADULT_COMPLEX_FILE_PY) \
        .add_check(NoBiasIntroducedFor(["race"])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    if to_sql:
        assert (dbms_connector)
        inspector_result = inspector_result.execute_in_sql(dbms_connector=dbms_connector, reset_state=reset,
                                                           sql_one_run=sql_one_run)
    else:
        inspector_result = inspector_result.execute(reset_state=reset)

    if despite:
        extracted_dag = inspector_result.dag

        filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.png")
        from mlinspect.visualisation import save_fig_to_path
        from PIL.Image import Image
        save_fig_to_path(extracted_dag, filename)

        from PIL import Image
        import matplotlib.pyplot as plt
        im = Image.open(filename)
        plt.imshow(im)
        # plt.waitforbuttonpress()

        dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
        check_results = inspector_result.check_to_check_results
        no_bias_check_result = check_results[NoBiasIntroducedFor(["race"])]

        distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
            no_bias_check_result)
        print(distribution_changes_overview_df.to_markdown())

        for i in list(no_bias_check_result.bias_distribution_change.items()):
            _, join_distribution_changes = i
            for column, distribution_change in join_distribution_changes.items():
                print("")
                print(f"\033[1m Column '{column}'\033[0m")
                print(distribution_change.before_and_after_df.to_markdown())


def full_healthcare(one_pass=False):
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                           host="localhost")
    t0 = time.time()
    example_one(to_sql=False, reset=True)
    t1 = time.time()
    print("\nTime spend with original: " + str(t1 - t0))

    t0 = time.time()
    example_one(to_sql=True, dbms_connector=dbms_connector_u, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    t0 = time.time()
    example_one(to_sql=True, dbms_connector=dbms_connector_p, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # print("\n\n" + "#" * 20 + "NOW WITH BIGGER SIZES:" + "#" * 20 + "\n\n")


def full_compas(one_pass=False):
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                           host="localhost")
    t0 = time.time()
    example_compas(to_sql=False, reset=True)
    t1 = time.time()
    print("\nTime spend with original: " + str(t1 - t0))

    t0 = time.time()
    example_compas(to_sql=True, dbms_connector=dbms_connector_u, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    t0 = time.time()
    example_compas(to_sql=True, dbms_connector=dbms_connector_p, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # print("\n\n" + "#" * 20 + "NOW WITH BIGGER SIZES:" + "#" * 20 + "\n\n")


def full_adult_simple(one_pass=False):
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")
    t0 = time.time()
    example_adult_simple(to_sql=False, reset=True)
    t1 = time.time()
    print("\nTime spend with original: " + str(t1 - t0))

    t0 = time.time()
    example_adult_simple(to_sql=True, dbms_connector=dbms_connector_u, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    t0 = time.time()
    example_adult_simple(to_sql=True, dbms_connector=dbms_connector_p, reset=True, sql_one_run=one_pass)
    t1 = time.time()
    print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # print("\n\n" + "#" * 20 + "NOW WITH BIGGER SIZES:" + "#" * 20 + "\n\n")


def full_adult_complex(one_pass=False):
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    dbms_connector_u = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/",
                                      umbra_dir=umbra_path)

    dbms_connector_p = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password",
                                           port=5432,
                                           host="localhost")
    t0 = time.time()
    example_adult_complex(to_sql=False, reset=True)
    t1 = time.time()
    print("\nTime spend with original: " + str(t1 - t0))

    # t0 = time.time()
    # example_adult_complex(to_sql=True, dbms_connector=dbms_connector_u, reset=True, sql_one_run=one_pass)
    # t1 = time.time()
    # print("\nTime spend with modified SQL inspections: " + str(t1 - t0))
    #
    # t0 = time.time()
    # example_adult_complex(to_sql=True, dbms_connector=dbms_connector_p, reset=True, sql_one_run=one_pass)
    # t1 = time.time()
    # print("\nTime spend with modified SQL inspections: " + str(t1 - t0))

    # print("\n\n" + "#" * 20 + "NOW WITH BIGGER SIZES:" + "#" * 20 + "\n\n")


if __name__ == "__main__":
    # full_healthcare(one_pass=False)
    # full_compas(one_pass=False)
    full_adult_simple(one_pass=False)
    # full_adult_complex(one_pass=False)
