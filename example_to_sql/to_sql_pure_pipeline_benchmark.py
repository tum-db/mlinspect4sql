"""
Benchmark pure pipeline.
Runtime comparison, of the translated pipelines.

## Required packages:
#See: requirements/requirements.txt and requirements/requirements.dev.txt
"""

import pathlib

from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from mlinspect import PipelineInspector
from mlinspect.utils import get_project_root
from _code_as_string import get_healthcare_pipe_code, get_healthcare_sql_str
from pandas_connector import PandasConnector
from _benchmark_utility import plot_compare, PLOT_DIR, write_to_log, DO_CLEANUP, SIZES, BENCH_REP, \
    MLINSPECT_ROOT_DIR, UMBRA_DIR, UMBRA_USER, UMBRA_PW, UMBRA_DB, UMBRA_PORT, UMBRA_HOST, POSTGRES_USER, POSTGRES_PW, \
    POSTGRES_DB, POSTGRES_PORT, POSTGRES_HOST
from data_generation.compas_data_generation import generate_compas_dataset
from data_generation.healthcare_data_generation import generate_healthcare_dataset
from time import time
# Data Generation:
# To be able to benchmark and compare the different approaches, some datasets
# will need to be generated before. The datasets are just and expansion of the
# original ones.
# We only generate the files, that are not already existing:

COMPAS_DATA_PATHS = generate_compas_dataset(SIZES)
HEALTHCARE_DATA_PATHS = generate_healthcare_dataset(SIZES)


def pure_pipeline_benchmark(mode, materialize, only_pandas=False, title="HealthcarePurePipeComparison"):
    umbra_times = []
    postgres_times = []
    pandas_times = []

    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                   host="localhost")
    pandas = PandasConnector()

    for i, (path_to_csv_1, path_to_csv_2) in enumerate(HEALTHCARE_DATA_PATHS):
        print(f"ITERATION: {i} - for table size of: {SIZES[i]}")

        setup_code_orig, test_code_orig = get_healthcare_pipe_code(path_to_csv_1, path_to_csv_2, only_pandas,
                                                                   include_training=False)

        setup_code, test_code = get_healthcare_sql_str(setup_code_orig + "\n" + test_code_orig, mode=mode,
                                                       materialize=materialize)

        ################################################################################################################
        # time Umbra:
        if not materialize:
            umbra = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/", umbra_dir=UMBRA_DIR)
            umbra.run(setup_code)
            umbra_times.append(umbra.benchmark_run(test_code, repetitions=BENCH_REP))
            write_to_log("HEALTHCARE", SIZES[i], mode, materialize, csv_file_paths=[path_to_csv_1, path_to_csv_2],
                         engine="Umbra", time=umbra_times[-1])

        ################################################################################################################
        # time Pandas:
        t0 =time()
        pandas_times.append(pandas.benchmark_run(pandas_code=test_code_orig, setup_code=setup_code_orig,
                                                 repetitions=BENCH_REP))
        write_to_log("HEALTHCARE", SIZES[i], mode, materialize, csv_file_paths=[path_to_csv_1, path_to_csv_2],
                     engine="Pandas", time=pandas_times[-1])
        t1 = time()
        print("runtime: " + str(t1-t0))
        ################################################################################################################
        # time Postgres:
        t0 =time()
        postgres.run(setup_code)
        postgres_times.append(postgres.benchmark_run(test_code, repetitions=BENCH_REP))
        write_to_log("HEALTHCARE", SIZES[i], mode, materialize, csv_file_paths=[path_to_csv_1, path_to_csv_2],
                     engine="Postgresql", time=postgres_times[-1])
        t1 = time()
        print("runtime: " + str(t1 - t0))
        ################################################################################################################

    print(f"Plotting..")
    names = ["Pandas", "Postgresql", "Umbra"]
    table = [pandas_times, postgres_times, umbra_times]
    if materialize:
        names = names[:-1]
        table = table[:-1]
    plot_compare(title, SIZES, all_y=table, all_y_names=names, save=True)


if __name__ == "__main__":
    # Just the pandas part:
    # pure_pipeline_benchmark(mode="CTE", materialize=False, only_pandas=True,
    #                         title="HealthcarePurePipeComparisonOnlyPandasCTE")
    #
    # pure_pipeline_benchmark(mode="VIEW", materialize=False, only_pandas=True,
    #                         title="HealthcarePurePipeComparisonOnlyPandasVIEW")
    #
    # pure_pipeline_benchmark(mode="VIEW", materialize=True, only_pandas=True,
    #                         title="HealthcarePurePipeComparisonOnlyPandasVIEWMAT")

    # Attention: materialize without inspection makes no sense -> only one execution anyway!
    pure_pipeline_benchmark(mode="VIEW", materialize=True, only_pandas=False,
                            title="HealthcarePurePipeComparisonFullVIEWMAT")

    # pure_pipeline_benchmark(mode="CTE", materialize=False, only_pandas=False,
    #                         title="HealthcarePurePipeComparisonFullCTE")
    #
    # pure_pipeline_benchmark(mode="VIEW", materialize=False, only_pandas=False,
    #                         title="HealthcarePurePipeComparisonFullVIEW")


    # With OneHotEncoding and SimpleImputer:
    # pure_pipeline_benchmark(add_impute_and_onehot=True, title="HealthcarePurePipeComparisonSimpImpOneHot")

    # Clean_up:
    # if DO_CLEANUP:
    #     [f.unlink() for f in PLOT_DIR.glob("*_.png") if f.is_file()]
