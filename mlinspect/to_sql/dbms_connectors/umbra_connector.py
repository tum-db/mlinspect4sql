from abc import ABC
from mlinspect.to_sql.data_source_sql_handling import CreateTablesFromDataSource
from .dbms_connector import Connector
import psycopg2
import subprocess
import time
import fcntl
import os
import pandas
import tempfile
import csv


class UmbraConnector(Connector):
    def __init__(self, dbname, user, password, port, host, umbra_dir=None, add_mlinspect_serial=False):
        """
        Starts a new empty Umbra server is 'DROP' is not implemented yet.

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
        super().__init__(dbname, user, password, port, host)
        self.umbra_dir = umbra_dir
        self.add_mlinspect_serial = add_mlinspect_serial

        # check if already running:
        result = subprocess.run("pgrep serve", stdout=subprocess.PIPE, shell=True)
        if result.returncode == 0:  # here we check if the process is already running
            subprocess.run(f"kill -9 {result.stdout.decode('utf-8').strip()}", stdout=subprocess.DEVNULL, shell=True)
        command = f"./build/server \"\" -port=5433 -address=localhost"
        self.server = subprocess.Popen(command, cwd=self.umbra_dir, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # Set output handle to non-blocking (essential to read all that is available and not wait for process term.):
        fcntl.fcntl(self.server.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        time.sleep(0.1)  # wait for the server to start
        self.connection = psycopg2.connect(dbname=dbname, user=user, password=password, port=port, host=host)
        self.cur = self.connection.cursor()

    # Destructor:
    def __del__(self):
        self.server.kill()

    def run(self, sql_query):
        results = []
        for q in super()._prepare_query(sql_query):
            # print(q + "\n")
            self.cur.execute(q)
            try:
                results.append(self.cur.fetchall())
            except psycopg2.ProgrammingError:  # Catch the case no result is available (f.e. create Table)
                continue
        return [pandas.DataFrame(r) for r in results]

    def benchmark_run(self, sql_query, repetitions=1, verbose=True):
        print("Executing Query in Umbra...") if verbose else 0
        sql_query = super()._prepare_query(sql_query)
        if len(sql_query) != 1:
            raise ValueError("Can only benchmark ONE query!")
        sql_query = sql_query[0]

        new_output = []

        # Get old output out of the way:
        for _ in iter(lambda: self.server.stdout.readline(), b''):
            continue

        for _ in range(repetitions):  # Execute the Query multiple times:
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
        print(f"Done in {bench_time}!") if verbose else 0
        return bench_time

    def add_csv(self, path_to_csv: str, table_name: str, null_symbols: list, delimiter: str, header: bool, *args,
                **kwargs):
        """ See parent. """
        # create the index column:
        try:
            if self.add_mlinspect_serial:
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
                                                                              add_mlinspect_serial=False)
            self.run(sql_code)

            if self.add_mlinspect_serial:
                create_index = f"CREATE UNIQUE INDEX id_mlinspect_{table_name} ON {table_name} (index_mlinspect);"
                self.run(create_index)
                sql_code += "\n" + create_index

        finally:
            if self.add_mlinspect_serial:
                os.remove(path_to_tmp)  # do cleanup
        return col_names, sql_code

    def add_dataframe(self, data_frame: pandas.DataFrame, table_name: str, *args, **kwargs) -> (list, str):
        col_names, sql_code = CreateTablesFromDataSource.get_sql_code_data_frame(data_frame, table_name=table_name,
                                                                                 add_mlinspect_serial=False)

        self.run(sql_code)
        return col_names, sql_code
