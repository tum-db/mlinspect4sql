"""
Benchmark simple sql operations.

This is thought to show possible performance difference for simple operations performed in DBMS and Python/pandas.
This is not an exhaustive benchmark, more test are necessary for a general conclusion.
"""

from data_generation.healthcare_data_generation import generate_healthcare_dataset
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from pandas_connector import PandasConnector
from _benchmark_utility import plot_compare, PLOT_DIR, SIZES, DO_CLEANUP, SIZES, BENCH_REP, \
    MLINSPECT_ROOT_DIR, UMBRA_USER, UMBRA_PW, UMBRA_DB, UMBRA_PORT, UMBRA_HOST, POSTGRES_USER, POSTGRES_PW, \
    POSTGRES_DB, POSTGRES_PORT, POSTGRES_HOST
from _code_as_string import Join, GroupBy, Selection, Projection

# Data Generation
# To be able to benchmark and compare the different approaches, some datasets
# will need to be generated before. The datasets are just and expansion of the
# original ones.
# We only generate the files, that are not already existing:

HEALTHCARE_DATA_PATHS = generate_healthcare_dataset(SIZES)


# Based on mlinspect benchmarks.

def simple_op_benchmark():
    t1_name = "histories"
    t2_name = "patients"

    operations = ["Join", "Select", "Project", "GroupBy"]

    umbra_times = [[] for _ in operations]
    postgres_times = [[] for _ in operations]
    pandas_times = [[] for _ in operations]

    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                   host="localhost")
    pandas = PandasConnector()
    repetitions = 10
    for i, (table1, table2) in enumerate(HEALTHCARE_DATA_PATHS):
        umbra = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/")

        umbra.add_csv(table_name=t2_name, path_to_csv=table2, null_symbols=["?"], delimiter=",", header=True)
        umbra.add_csv(table_name=t1_name, path_to_csv=table1, null_symbols=["?"], delimiter=",", header=True)

        postgres.add_csv(table_name=t2_name, path_to_csv=table2, null_symbols=["?"], delimiter=",", header=True)
        postgres.add_csv(table_name=t1_name, path_to_csv=table1, null_symbols=["?"], delimiter=",", header=True)

        print(f"ITERATION: {i} - for table size of: {SIZES[i]}")

        input_join = t1_name, t2_name, "ssn"
        umbra_times[0].append(umbra.benchmark_run(Join.get_sql_code(*input_join), repetitions))
        postgres_times[0].append(postgres.benchmark_run(Join.get_sql_code(*input_join), repetitions))
        pandas_times[0].append(
            pandas.benchmark_run(Join.get_pandas_code(table1, table2, "ssn"), repetitions=repetitions))

        input_sel = t1_name, "complications", ">", "5"
        umbra_times[1].append(umbra.benchmark_run(Selection.get_sql_code(*input_sel), repetitions))
        postgres_times[1].append(postgres.benchmark_run(Selection.get_sql_code(*input_sel), repetitions))
        pandas_times[1].append(
            pandas.benchmark_run(Selection.get_pandas_code(table1, "complications", ">", "5"),
                                 repetitions=repetitions))

        input_project = t1_name, "smoker"
        umbra_times[2].append(umbra.benchmark_run(Projection.get_sql_code(*input_project), repetitions))
        postgres_times[2].append(postgres.benchmark_run(Projection.get_sql_code(*input_project), repetitions))
        pandas_times[2].append(
            pandas.benchmark_run(Projection.get_pandas_code(table1, "smoker"), repetitions=repetitions))

        input_project = t1_name, "smoker", "complications", "AVG"
        umbra_times[3].append(umbra.benchmark_run(GroupBy.get_sql_code(*input_project), repetitions))
        postgres_times[3].append(postgres.benchmark_run(GroupBy.get_sql_code(*input_project), repetitions))
        pandas_times[3].append(
            pandas.benchmark_run(GroupBy.get_pandas_code(table1, "smoker", "complications", "mean"),
                                 repetitions=repetitions))
        # in the end we have 3 lists == [[*joins*][*selections*][*projections*]]

    names = ["Umbra", "Postgresql", "Pandas"]
    for i, title in enumerate(operations):
        table = [umbra_times[i], postgres_times[i], pandas_times[i]]
        plot_compare(title, SIZES, all_y=table, all_y_names=names, save=True)


if __name__ == "__main__":
    simple_op_benchmark()

    # Clean_up:
    if DO_CLEANUP:
        [f.unlink() for f in PLOT_DIR.glob("*_.png") if f.is_file()]
