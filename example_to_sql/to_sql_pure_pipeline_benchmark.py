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
from _benchmark_utility import plot_compare, PLOT_DIR, write_to_log, SIZES, DO_CLEANUP, SIZES, BENCH_REP, \
    MLINSPECT_ROOT_DIR, UMBRA_DIR, UMBRA_USER, UMBRA_PW, UMBRA_DB, UMBRA_PORT, UMBRA_HOST, POSTGRES_USER, POSTGRES_PW, \
    POSTGRES_DB, POSTGRES_PORT, POSTGRES_HOST
from data_generation.compas_data_generation import generate_compas_dataset
from data_generation.healthcare_data_generation import generate_healthcare_dataset

# Data Generation:
# To be able to benchmark and compare the different approaches, some datasets
# will need to be generated before. The datasets are just and expansion of the
# original ones.
# We only generate the files, that are not already existing:

COMPAS_DATA_PATHS = generate_compas_dataset(SIZES)
HEALTHCARE_DATA_PATHS = generate_healthcare_dataset(SIZES)


def pure_pipeline_benchmark(mode, materialize, add_impute_and_onehot=False, title="HealthcarePurePipeComparison"):
    umbra_times = []
    postgres_times = []
    pandas_times = []

    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                   host="localhost")
    pandas = PandasConnector()

    for i, (path_to_csv_his, path_to_csv_pat) in enumerate(HEALTHCARE_DATA_PATHS):
        print(f"ITERATION: {i} - for table size of: {SIZES[i]}")

        setup_code_orig, test_code_orig = get_healthcare_pipe_code(path_to_csv_his, path_to_csv_pat,
                                                                   add_impute_and_onehot=add_impute_and_onehot)

        setup_code, test_code = get_healthcare_sql_str(setup_code_orig + "\n" + test_code_orig, mode=mode,
                                                       materialize=materialize)

        ################################################################################################################
        # time Umbra:
        umbra = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/", umbra_dir=UMBRA_DIR)
        umbra.run(setup_code)
        umbra_times.append(umbra.benchmark_run(test_code, repetitions=BENCH_REP))

        ################################################################################################################
        # time Postgres:
        postgres.run(setup_code)
        postgres_times.append(postgres.benchmark_run(test_code, repetitions=BENCH_REP))

        ################################################################################################################
        # time Pandas:
        pandas_times.append(pandas.benchmark_run(pandas_code=test_code_orig, setup_code=setup_code_orig,
                                                 repetitions=BENCH_REP))
        ################################################################################################################

    print(f"Plotting..")
    names = ["Umbra", "Postgresql", "Pandas"]
    table = [umbra_times, postgres_times, pandas_times]
    plot_compare(title, SIZES, all_y=table, all_y_names=names, save=True)


if __name__ == "__main__":
    # Just the pandas part:
    pure_pipeline_benchmark(mode="CTE", materialize=False, add_impute_and_onehot=False,
                            title="HealthcarePurePipeComparison")

    pure_pipeline_benchmark(mode="VIEW", materialize=False, add_impute_and_onehot=False,
                            title="HealthcarePurePipeComparisonVIEW")

    # With OneHotEncoding and SimpleImputer:
    # pure_pipeline_benchmark(add_impute_and_onehot=True, title="HealthcarePurePipeComparisonSimpImpOneHot")

    # Clean_up:
    if DO_CLEANUP:
        [f.unlink() for f in PLOT_DIR.glob("*_.png") if f.is_file()]
