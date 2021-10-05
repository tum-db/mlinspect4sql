from mlinspect.to_sql.data_source_sql_handling import CreateTablesFromDataSource
from mlinspect.to_sql.dbms_connectors.dbms_connector import Connector
from .connector_utility import results_to_np_array
import psycopg2
import pandas
from mlinspect.utils import store_timestamp
import time


class PostgresqlConnector(Connector):
    def __init__(self, dbname="", user="", password="", port="", host="", just_code=False, add_mlinspect_serial=True):
        """
        Note: For Postgresql:
            1) install Postgresql and start the server
            2) assert it is running: "sudo netstat -lntup | grep '5433\\|5432'"

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
            # print(q)  # Very helpful for debugging
            self.cur.execute(q)
            # print("DONE")
            try:
                # t0 = time.time()
                query_output = self.cur.fetchall()
                column_names = [c.name for c in self.cur.description]
                results.append((column_names, query_output))
                # if ('ORDER BY index_mlinspect' in q):
                #     store_timestamp(f"(DATA MOVE/TANSFORMATION COST) LOAD RESULT TRAIN/TEST", time.time() - t0, "PostgreSQL")
            except psycopg2.ProgrammingError:  # Catch the case no result is available (f.e. create Table)
                continue
        # t0 = time.time()
        results = results_to_np_array(results)
        # if ('ORDER BY index_mlinspect' in q):
        #     store_timestamp(f"(DATA MOVE/TANSFORMATION COST) TRANSFORM RESULT TRAIN/TEST", time.time() - t0, "PostgreSQL")
        return results

    def benchmark_run(self, sql_query, repetitions=1, verbose=True):
        exe_times = []
        print("Executing Query in Postgres...") if verbose else 0

        if "MATERIALIZED" in "".join(sql_query):
            print("MATERIALIZED")
        if "VIEW" in "".join(sql_query):
            print("VIEW")
        else:
            print("CTE")

        sql_queries = super()._prepare_query(sql_query)
        assert len(sql_queries) != 0

        time_for_materialization = 0
        if len(sql_queries) > 1:
            for q in sql_queries[:-1]:
                if "MATERIALIZED" in q:
                    t0 = time.time()
                    self.cur.execute(q[:-1])
                    t1 = time.time()
                    time_for_materialization += t1-t0
                else:
                    self.cur.execute(q)
        sql_query = sql_queries[-1]
        for _ in range(repetitions):
            # self.cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) (\n" + sql_query[:-1] + "\n);")
            # result = self.cur.fetchall()
            # exe_times.append(result[0][0][0]['Execution Time'] + time_for_materialization)

            # This seems more reliable: (but includes time for server to return output)
            t0 = time.time()
            self.cur.execute(sql_query)
            t1 = time.time()
            exe_times.append((t1-t0) * 1000) # for ms

        t = sum(exe_times) / repetitions
        print(f"Done in {t}ms!") if verbose else 0
        return t

    def add_csv(self, path_to_csv: str, table_name: str, null_symbols: list, delimiter: str, header: bool, *args,
                **kwargs):
        """ See parent. """
        index_col = -1
        if "index_col" in kwargs:
            index_col = kwargs["index_col"]  # This will be used as serial

        col_names, sql_code = CreateTablesFromDataSource.get_sql_code_csv(path_to_csv, table_name=table_name,
                                                                          null_symbols=null_symbols,
                                                                          delimiter=delimiter,
                                                                          header=header,
                                                                          add_mlinspect_serial=self.add_mlinspect_serial,
                                                                          index_col=index_col)

        self.run(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        self.run(sql_code)

        create_index = ""
        if self.add_mlinspect_serial:
            col_names.append(self.index_col_name)
            # create_index = f"CREATE UNIQUE INDEX id_mlinspect_{table_name} ON {table_name} (index_mlinspect);"
            # self.run(create_index)

        return col_names, sql_code + "\n" + create_index

    def add_dataframe(self, data_frame: pandas.DataFrame, table_name: str, *args, **kwargs) -> (list, str):

        col_names, sql_code = CreateTablesFromDataSource.get_sql_code_csv(data_frame, table_name=table_name,
                                                                          add_mlinspect_serial=False)

        self.run(sql_code)
        return col_names, sql_code
