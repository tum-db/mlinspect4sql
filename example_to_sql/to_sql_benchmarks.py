"""
Performance showcase of novel "to_sql" functionality in mlinspect

Here the performance of the proposed inspection using sql will be compared to the original one in pandas. Part of
the "healthcare" and "compas" pipeline will be used.

parameters are defined in _benchmark_utility.py
for examples, writes log to example_to_sql/plots/aRunLog.csv
"""

import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import timeit
from inspect import cleandoc
from mlinspect.utils import get_project_root
from _code_as_string import get_healthcare_pipe_code, get_compas_pipe_code, get_adult_simple_pipe_code, \
    get_adult_complex_pipe_code, get_sql_query_for_pipeline, get_compas_pipe_code_with_timestamps
from _benchmark_utility import plot_compare, PLOT_DIR, write_to_log, write_brake_to_log, DO_CLEANUP, SIZES, BENCH_REP, \
    MLINSPECT_ROOT_DIR, UMBRA_USER, UMBRA_PW, UMBRA_DB, UMBRA_PORT, UMBRA_HOST, POSTGRES_USER, POSTGRES_PW, \
    POSTGRES_DB, POSTGRES_PORT, POSTGRES_HOST
from data_generation.compas_data_generation import generate_compas_dataset
from data_generation.healthcare_data_generation import generate_healthcare_dataset
from data_generation.adult_data_generation import generate_adult_dataset

from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from pandas_connector import PandasConnector

# Data GenerationCTE
# To be able to benchmark and compare the different approaches, some datasets
# will need to be generated before. The datasets are just and expansion of the
# original ones.
# We only generate the files, that are not already existing:

COMPAS_DATA_PATHS = "c", generate_compas_dataset(SIZES)
HEALTHCARE_DATA_PATHS = "h", generate_healthcare_dataset(SIZES)
ADULT_SIMPLE_DATA_PATHS = "as", generate_adult_dataset(SIZES)
ADULT_COMPLEX_DATA_PATHS = "ac", generate_adult_dataset(SIZES)

POSTGRES_CONNECTOR = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                         host="localhost")
PANDAS_CONNECTOR = PandasConnector()


def get_inspection_code(pipeline_code, to_sql, dbms_connector, no_bias_list, mode, materialize=False):
    setup_code = cleandoc(f"""
from inspect import cleandoc
from mlinspect.utils import get_project_root
from mlinspect import PipelineInspector
from mlinspect.inspections import RowLineage, MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from inspect import cleandoc
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from mlinspect import PipelineInspector

dbms_connector_u = UmbraConnector(dbname=\'{UMBRA_DB}\', user=\'{UMBRA_USER}\', password=\'{UMBRA_PW}\',
    port={UMBRA_PORT}, host=\'{UMBRA_HOST}\')
dbms_connector_p = PostgresqlConnector(dbname=\'{POSTGRES_DB}\', user=\'{POSTGRES_USER}\',
    password=\'{POSTGRES_PW}\', port={POSTGRES_PORT}, host=\'{POSTGRES_HOST}\')

pipeline_code = cleandoc(f\"\"\"{pipeline_code}\"\"\")

pipeline_inspector = PipelineInspector.on_pipeline_from_string(pipeline_code) \\
    .add_custom_monkey_patching_module(custom_monkeypatching) \\
    .add_check(NoBiasIntroducedFor({no_bias_list})) \\
    .add_check(NoIllegalFeatures()) \\
    .add_check(NoMissingEmbeddings()) \\
    .add_required_inspection(RowLineage(5)) \\
    .add_required_inspection(MaterializeFirstOutputRows(5))
    """) + "\n"
    if to_sql:
        return setup_code, f"pipeline_inspector.execute_in_sql(dbms_connector={dbms_connector}, " \
                           f"mode=\'{mode}\', materialize={materialize})"

    return setup_code, f"pipeline_inspector.execute()"


def run(setup_code, test_code, to_sql=False, dbms_connector=None, no_bias=None, mode=None, materialize=False,
        inspection=True):
    pipeline_code = setup_code + test_code

    # print(pipeline_code)

    if not inspection:
        if not to_sql:
            return PANDAS_CONNECTOR.benchmark_run(pandas_code=test_code, setup_code=setup_code)
        elif dbms_connector == 'dbms_connector_p':
            dbms_connector_engine = POSTGRES_CONNECTOR
        elif dbms_connector == 'dbms_connector_u':
            dbms_connector_engine = UmbraConnector(dbname="", user="postgres", password=" ", port=UMBRA_PORT, host="/tmp/")
        else:
            raise NotImplementedError
        t0 = time.time()
        setup_code, test_code = get_sql_query_for_pipeline(pipeline_code, mode=mode, materialize=materialize)
        t1 = time.time()
        transpilationtime = (t1 - t0)*1000
        print("Transpilation time (w/o inspection) " + str(t1 - t0))
        dbms_connector_engine.run(setup_code)
        (avg, var, std) = dbms_connector_engine.benchmark_run(test_code, repetitions=BENCH_REP)
        return avg+transpilationtime, var+transpilationtime, std+transpilationtime

    setup_code, test_code = get_inspection_code(pipeline_code, to_sql, dbms_connector, no_bias,
                                                mode, materialize)
    """
    As desired for the paper review the variance and std of the executed code is collected, even though not suggested 
    by the documentation (https://docs.python.org/3/library/timeit.html): 
    "Note: It’s tempting to calculate mean and standard deviation from the result vector and report these. 
    However, this is not very useful. In a typical case, the lowest value gives a lower bound for how fast your 
    machine can run the given code snippet; higher values in the result vector are typically not caused by variability 
    in Python’s speed, but by other processes interfering with your timing accuracy. So the min() of the result is 
    probably the only number you should be interested in. After that, you should look at the entire vector and apply 
    common sense rather than statistics."
    """
    result = []
    if to_sql:
        for i in range(BENCH_REP):
            print(f"SQL run {i + 1} of {BENCH_REP} ...")
            # This special case is necessary to deduct the time for dropping the existing tables and views and to
            # get var and std!
            result.append(timeit.timeit(test_code, setup=setup_code, number=1) * 1000)  # in ms
        return sum(result) / BENCH_REP, np.var(result), np.std(result)

    for i in range(BENCH_REP):
        print(f"Default run {i + 1} of {BENCH_REP} ...")
        result.append(timeit.timeit(test_code, setup=setup_code, number=1) * 1000)  # in ms
    return sum(result) / BENCH_REP, np.var(result), np.std(result)


def pipeline_benchmark(data_paths, mode, no_bias=None, only_pandas=False, with_train=False, inspection=True):
    pandas_times = []
    postgres_times = []
    umbra_times = []

    target, data_paths = data_paths

    write_brake_to_log()

    for i, (path_to_csv_1, path_to_csv_2) in enumerate(data_paths):
        if i >= len(SIZES):
            continue

        if "c" == target:
            setup_code, test_code = get_compas_pipe_code(path_to_csv_1, path_to_csv_2, only_pandas=only_pandas,
                                                         include_training=with_train)
            pipeline_name = "COMPAS"

        elif "c_t" == target:
            setup_code, test_code = get_compas_pipe_code_with_timestamps(path_to_csv_1, path_to_csv_2,
                                                                         only_pandas=only_pandas,
                                                                         include_training=with_train,
                                                                         engine_name="PostgreSQL/Umbra")
                                                                         # engine_name="Original")
            pipeline_name = "COMPAS_TIMED"

        elif "h" == target:
            setup_code, test_code = get_healthcare_pipe_code(path_to_csv_1, path_to_csv_2, only_pandas=only_pandas,
                                                             include_training=with_train)
            pipeline_name = "HEALTHCARE"

        elif "as" == target:
            setup_code, test_code = get_adult_simple_pipe_code(path_to_csv_1, only_pandas=only_pandas,
                                                               include_training=with_train)
            pipeline_name = "ADULT_SIMPLE"

        elif "ac" == target:
            setup_code, test_code = get_adult_complex_pipe_code(path_to_csv_1, path_to_csv_2, only_pandas=only_pandas,
                                                                include_training=with_train)
            pipeline_name = "ADULT_COMPLEX"
        else:
            assert False

        print(f"##### Running ...  -- size {SIZES[i]} ######")
        ################################################################################################################
        # time Pandas/Original:

        pandas_times.append(run(setup_code, test_code, to_sql=False, dbms_connector=None, no_bias=no_bias,
                                inspection=inspection))
        write_to_log(pipeline_name, only_pandas=only_pandas, inspection=inspection, size=SIZES[i], mode=mode,
                     materialize=False, engine="Pandas/Original", time=pandas_times[-1],
                     csv_file_paths=[path_to_csv_1, path_to_csv_2], with_train=with_train)
        write_brake_to_log()
        ################################################################################################################
        # time Postgres:
        if mode == "CTE" or only_pandas:
            postgres_times.append(run(setup_code, test_code, to_sql=True, dbms_connector="dbms_connector_p",
                                      no_bias=no_bias, mode=mode, materialize=False, inspection=inspection))
            write_to_log(pipeline_name, only_pandas=only_pandas, inspection=inspection, size=SIZES[i], mode=mode,
                         materialize=False, engine="PostgreSQL", time=postgres_times[-1],
                         csv_file_paths=[path_to_csv_1, path_to_csv_2], with_train=with_train)

        if mode == "VIEW" and not only_pandas:
            # time Postgres materialized:
            postgres_times.append(run(setup_code, test_code, to_sql=True, dbms_connector="dbms_connector_p",
                                      no_bias=no_bias, mode=mode, materialize=True, inspection=inspection))
            write_to_log(pipeline_name, only_pandas=only_pandas, inspection=inspection, size=SIZES[i], mode=mode,
                         materialize=True, engine="PostgreSQL", time=postgres_times[-1],
                         csv_file_paths=[path_to_csv_1, path_to_csv_2], with_train=with_train)
            write_brake_to_log()
            postgres_times.append(run(setup_code, test_code, to_sql=True, dbms_connector="dbms_connector_p",
                                      no_bias=no_bias, mode=mode, materialize=False, inspection=inspection))
            write_to_log(pipeline_name, only_pandas=only_pandas, inspection=inspection, size=SIZES[i], mode=mode,
                         materialize=False, engine="PostgreSQL", time=postgres_times[-1],
                         csv_file_paths=[path_to_csv_1, path_to_csv_2], with_train=with_train)
        write_brake_to_log()
        ################################################################################################################
        # time Umbra:
        umbra_times.append(run(setup_code, test_code, to_sql=True, dbms_connector="dbms_connector_u", no_bias=no_bias,
                               mode=mode, materialize=False, inspection=inspection))
        write_to_log(pipeline_name, only_pandas=only_pandas, inspection=inspection, size=SIZES[i], mode=mode,
                     materialize=False, engine="Umbra", time=umbra_times[-1],
                     csv_file_paths=[path_to_csv_1, path_to_csv_2], with_train=with_train)
        ################################################################################################################

    # names = ["Pandas", f"Postgresql", f"Umbra - Not Materialized"]
    # table = [pandas_times, postgres_times, umbra_times]
    # if not materialize:  # remove non-existing umbra values
    #     names = names[:-1]
    #     table = table[:-1]
    # plot_compare(title, SIZES, all_y=table, all_y_names=names, save=True)


if __name__ == "__main__":
    healthcare_no_bias = "[\'age_group\', \'race\']"
    compas_no_bias = "[\'sex\', \'race\']"
    adult_no_bias = "[\'race\']"

    # # Benchmark of default inspection using MATERIALIZED VIEWs:
    # While doing the default inspection, parts of the generated code get executed multiple times. This is the reason
    # the option to MATERIALIZE Views, called more that once, got added. This way NO part of the pipeline is executed
    # more that ONCE.
    # In our case only postgres supports this option.

    # BENCHMARK OF THE PURE PIPELINE: - ONLY PANDAS PART: ##############################################################
    print("#### healthcare")
    pipeline_benchmark(HEALTHCARE_DATA_PATHS, mode="CTE", only_pandas=True, inspection=False)
    pipeline_benchmark(HEALTHCARE_DATA_PATHS, mode="VIEW", only_pandas=True, inspection=False)

    # Only selected parts: -> manually changed in the code providing function: "get_compas_pipe_code" what was changed
    # is described in the paper.
    print("#### compas")
    pipeline_benchmark(COMPAS_DATA_PATHS, mode="CTE", only_pandas=True, inspection=False)
    pipeline_benchmark(COMPAS_DATA_PATHS, mode="VIEW", only_pandas=True, inspection=False)

    # No relevant pandas share: -> so not covered. Still is available and functioning, if of interest.
    print("#### adult simple")
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, mode="CTE", only_pandas=True, inspection=False)
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, mode="VIEW", only_pandas=True, inspection=False)

    print("#### adult complex")
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, mode="CTE", only_pandas=True, inspection=False)
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, mode="VIEW", only_pandas=True, inspection=False)

    ####################################################################################################################

    # BENCHMARK OF THE PURE PIPELINE: - FULL: ##########################################################################
    print("#### adult simple + sk")
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, mode="CTE", only_pandas=False, inspection=False)
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, mode="VIEW", only_pandas=False, inspection=False)

    print("#### adult complex + sk")
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, mode="CTE", only_pandas=False, inspection=False)
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, mode="VIEW", only_pandas=False, inspection=False)

    print("#### healthcare + sk") 
    #pipeline_benchmark(HEALTHCARE_DATA_PATHS, mode="CTE", only_pandas=False, inspection=False)
    pipeline_benchmark(HEALTHCARE_DATA_PATHS, mode="VIEW", only_pandas=False, inspection=False)

    print("#### compas + sk")
    pipeline_benchmark(COMPAS_DATA_PATHS, mode="CTE", only_pandas=False, inspection=False)
    pipeline_benchmark(COMPAS_DATA_PATHS, mode="VIEW", only_pandas=False, inspection=False)
    ####################################################################################################################

    # INSPECTION OF THE PURE PIPELINE: - FULL: #########################################################################
    print("#### healthcare inspection")
    pipeline_benchmark(HEALTHCARE_DATA_PATHS, no_bias=healthcare_no_bias, mode="CTE", inspection=True)
    pipeline_benchmark(HEALTHCARE_DATA_PATHS, no_bias=healthcare_no_bias, mode="VIEW", inspection=True)

    print("#### adult simple inspection")
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, no_bias=adult_no_bias, mode="CTE", inspection=True)
    pipeline_benchmark(ADULT_SIMPLE_DATA_PATHS, no_bias=adult_no_bias, mode="VIEW", inspection=True)

    print("#### compas inspection")
    pipeline_benchmark(COMPAS_DATA_PATHS, no_bias=compas_no_bias, mode="CTE", inspection=True)
    pipeline_benchmark(COMPAS_DATA_PATHS, no_bias=compas_no_bias, mode="VIEW", inspection=True)

    print("#### adult complex inspection")
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, no_bias=adult_no_bias, mode="CTE", inspection=True)
    pipeline_benchmark(ADULT_COMPLEX_DATA_PATHS, no_bias=adult_no_bias, mode="VIEW", inspection=True)

    ####################################################################################################################

    # END-TO-END incl. TRAINING OF THE PURE PIPELINE: - FULL: ##########################################################

    orig_health = "h", [
        (
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/healthcare/histories.csv"),
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/healthcare/patients.csv")
        )
    ]

    orig_compas = "c", [
        (
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/compas/compas_train.csv"),
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/compas/compas_test.csv")
        )
    ]

    orig_adult_paths = [
        (
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/adult_complex/adult_train.csv"),
            os.path.join(str(get_project_root()), r"test/monkeypatchingSQL/pipelines_for_tests/adult_complex/adult_test.csv")
        )
    ]


    ####################################################################################################################

    # END-TO-END incl. TIMING OF TRAINING OF THE PURE PIPELINE: - FULL: ################################################
    # COMPAS_DATA_PATHS = "c_t", COMPAS_DATA_PATHS[1]
    # COMPAS_DATA_PATHS = "c_t", orig_compas[1]
    # print(COMPAS_DATA_PATHS)
    # pipeline_benchmark(COMPAS_DATA_PATHS, no_bias=compas_no_bias, mode="VIEW", inspection=True, with_train=True)

    # pipeline_benchmark(orig_compas, no_bias=compas_no_bias, mode="VIEW", inspection=True, with_train=True)
    ####################################################################################################################

    # # Main memory usage consideration:
    # Despite the improved runtime, also the main memory usage can profit from relying
    # on a DBMS. This can be also seen with Grizzly: https://edbt2021proceedings.github.io/docs/p174.pdf
    #
    # This code for example requires more than 32GB in main memory to run:

    # size = (10 ** 6)
    # path_to_csv_his, path_to_csv_pat = generate_healthcare_dataset([size])[0]
    # setup_code, test_code = get_healthcare_pipe_code(path_to_csv_his, path_to_csv_pat, add_impute_and_onehot=True)
    #
    # test_code += "\n" + "test = impute_and_one_hot_encode.fit_transform(data)"
    #
    # pipe_code = setup_code + test_code
    # Requires trough swiftly:
    # print(f"Running postgres...  -- size {size}")
    # print("postgres_time: " + str(run(pipe_code, True, "dbms_connector_p", no_bias=healthcare_no_bias, mode="CTE",
    #                                   materialize=False)))

    # print("Postgres run successful!")

    # Requires lots of memory (600+ GiB):
    # print(f"Running pandas...  -- size {size}")
    # print("pandas_times: " + str(run(pipe_code, to_sql=False, dbms_connector=None, no_bias=healthcare_no_bias)))
    # print("Pandas run done!")

    # # Clean_up:
    # if DO_CLEANUP:
    #     [f.unlink() for f in PLOT_DIR.glob("*_.png") if f.is_file()]
