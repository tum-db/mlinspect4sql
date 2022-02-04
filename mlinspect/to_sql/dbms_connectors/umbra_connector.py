from mlinspect.to_sql.data_source_sql_handling import CreateTablesFromDataSource
from .dbms_connector import Connector
from .connector_utility import results_to_np_array
from mlinspect.utils import store_timestamp
import psycopg2
import subprocess
import time
import fcntl
import os
import pandas
import tempfile
import csv
import re


class UmbraConnector(Connector):
    def __init__(self, dbname, user, password, port, host, just_code=False, add_mlinspect_serial=True):
        """
        Note: For Umbra:
            1) clone the Umbra repo.
            2) build everything: "mkdir build && cd build && cmake .. && make"
            3) Optional: Create User
            3) create a db file with: "./bin/sql -createdb <dbname>"
            4) Start server (with data base): "./build/server /path/to/<dbname> -port=5433 -address=localhost"
                Start server (with new base): "./build/server "" -port=5433 -address=localhost"
            5) Confirm it is running: "sudo netstat -lntup | grep '5433\\|5432'"
            6) Connect with arguments below: - In terminal: "psql -h /tmp -p 5433 -U postgres"
                db_name: "healthcare_benchmark"
                user: "postgres"
                password:"password" // not used
                port:"5432"
                host:"localhost"
        Not implemented yet:
        DROP, ALTER, DELETE, CREATE MATERIALIZED VIEW

        ATTENTION: The added table CAN be forced to contain a index column, called: "index_mlinspect" +
            create an index on it: "CREATE UNIQUE INDEX id_mlinspect ON <table_name> (index_mlinspect);"
            this can be done trough setting add_mlinspect_serial to True! -> Allows row-wise ops
        """
        self.add_mlinspect_serial = add_mlinspect_serial
        self.just_code = just_code
        if just_code:
            return
        super().__init__(dbname, user, password, port, host)
        self.db_settings = {"dbname": dbname, "user": user, "password": password, "port": port, "host": host}
        self.connection = psycopg2.connect(**self.db_settings)
        self.cur = self.connection.cursor()

    def __del__(self):
        if not self.just_code:
            # print(self.connection)
            self.connection.close()

    def run(self, sql_query):
        results = []
        if self.just_code:
            return []
        for q in super()._prepare_query(sql_query):
            #print(q)  # Very helpful for debugging
            q = self.fix_avg_overflow(q)
            self.cur.execute(q)
            try:
                # t0 = time.time()
                query_output = self.cur.fetchall()
                column_names = [c.name for c in self.cur.description]
                results.append((column_names, query_output))
                # if ('ORDER BY index_mlinspect' in q):
                #     store_timestamp(f"(DATA MOVE/TANSFORMATION COST) LOAD RESULT TRAIN/TEST", time.time() - t0, "Umbra")
            except psycopg2.ProgrammingError:  # Catch the case no result is available (f.e. create Table)
                continue
        # t0 = time.time()
        results = results_to_np_array(results)
        # if ('ORDER BY index_mlinspect' in q):
        #     store_timestamp(f"(DATA MOVE/TANSFORMATION COST) TRANSFORM RESULT TRAIN/TEST", time.time() - t0, "Umbra")
        return results

    def benchmark_run(self, sql_query, repetitions=1, verbose=True):
        """
        Returns time in ms.
        """
        print("Executing Query in Umbra...") if verbose else 0
        sql_queries = super()._prepare_query(sql_query)
        assert len(sql_queries) != 0
        if len(sql_queries) > 1:
            for q in sql_queries[:-1]:
                q = self.fix_avg_overflow(q)
                self.cur.execute(q)
            sql_query = sql_queries[-1]

        new_output = []
        # Get old output out of the way:
        for _ in iter(lambda: self.server.stdout.readline(), b''):
            continue

        for _ in range(repetitions):  # Execute the Query multiple times:
            # print(sql_query)
            sql_query = self.fix_avg_overflow(sql_query)
            self.cur.execute(sql_query)
            new_output.append(self.server.stdout.readline().decode("utf-8"))

        assert (len(new_output) == repetitions)
        result_exec_times_sum = 0
        for output in new_output:
            try:
                result_exec_times_sum += float(output.split("execution")[0].split(" ")[-3])
            except ValueError:
                continue  # No execution time found here..
        bench_time = result_exec_times_sum / repetitions
        print(f"Done in {bench_time * 1000}ms!") if verbose else 0
        return bench_time * 1000

    def add_csv(self, path_to_csv: str, table_name: str, null_symbols: list, delimiter: str, header: bool, *args,
                **kwargs):
        """ See parent. """
        # create the index column:
        try:
            index_col = -1
            if self.add_mlinspect_serial:
                if "index_col" in kwargs and kwargs["index_col"] != -1:
                    index_col = kwargs["index_col"]  # This will be used as serial
                else:
                    _, path_to_tmp = tempfile.mkstemp(prefix=table_name, suffix=".csv")

                    with open(path_to_csv, 'r') as csvinput:
                        with open(path_to_tmp, 'w') as csvoutput:
                            writer = csv.writer(csvoutput)
                            csv_reader = csv.reader(csvinput)
                            if header:
                                writer.writerow(next(csv_reader) + ["index_mlinspect"])
                            for i, row in enumerate(csv_reader):
                                writer.writerow(row + [str(i)])
                    path_to_csv = path_to_tmp
            col_names, sql_code = CreateTablesFromDataSource.get_sql_code_csv(path_to_csv, table_name=table_name,
                                                                              null_symbols=null_symbols,
                                                                              delimiter=delimiter,
                                                                              header=header,
                                                                              add_mlinspect_serial=index_col != -1,
                                                                              index_col=index_col)
            self.run(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            self.run(sql_code)

            if self.add_mlinspect_serial:
                create_index = f"CREATE UNIQUE INDEX id_mlinspect_{table_name} ON {table_name} (index_mlinspect);"
                self.run(create_index)
                sql_code += "\n" + create_index

        finally:
            if self.add_mlinspect_serial and not "index_col" in kwargs:
                os.remove(path_to_tmp)  # do cleanup
        return col_names, sql_code

    def add_dataframe(self, data_frame: pandas.DataFrame, table_name: str, *args, **kwargs) -> (list, str):
        col_names, sql_code = CreateTablesFromDataSource.get_sql_code_data_frame(data_frame, table_name=table_name,
                                                                                 add_mlinspect_serial=False)

        self.run(sql_code)
        return col_names, sql_code

    @staticmethod
    def fix_avg_overflow(string):
        """
        Improvised fix for overflows in AVG in Umbra.
        """
        p = re.compile("AVG\(((\S)+)\)")
        for m in p.findall(string):
            m = m[0]
            string = string.replace(f"AVG({m}) ", f"(SUM(0.00001 * {m}) / COUNT(*)) * 100000")
        return string
