from mlinspect.to_sql.csv_sql_handling import CreateTablesFromCSVs
from dbms_connector import Connector
import psycopg2


class PostgresqlConnector(Connector):
    def __init__(self, dbname, user, password, port, host):
        """
        Note: For Postgresql:
            1) install Postgresql and start the server
            2) assert it is running: "sudo netstat -lntup | grep '5433\\|5432'"
        """
        super().__init__(dbname, user, password, port, host)
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, port=port, host=host)
        self.cur = self.conn.cursor()

    def __del__(self):
        print(self.conn)
        self.conn.close()

    def run(self, sql_query):
        results = []
        for q in super()._prepare_query(sql_query):
            try:
                self.cur.execute(q)
                results.append(self.cur.fetchall())
            except psycopg2.ProgrammingError:  # Catch the case no result is available (f.e. create Table)
                continue

        return results

    def benchmark_run(self, sql_query, repetitions=1, verbose=True):
        exe_times = []
        print("Executing Query in Postgres...") if verbose else 0

        sql_query = super()._prepare_query(sql_query)
        if len(sql_query) != 1:
            raise ValueError("Can only benchmark ONE query!")
        sql_query = sql_query[0]

        for _ in range(repetitions):
            self.cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) (\n" + sql_query[:-1] + "\n);")
            result = self.cur.fetchall()
            exe_times.append(result[0][0][0]['Execution Time'])
        time = sum(exe_times) / repetitions
        print(f"Done in {time}!") if verbose else 0
        return time

    def add_csv(self, path_to_csv: str, table_name: str, null_symbol: str, delimiter: str, header: bool, *args,
                **kwargs):
        """ See parent. """
        col_names, sql_code = CreateTablesFromCSVs(path_to_csv).get_sql_code(table_name=table_name,
                                                                             null_symbol=null_symbol,
                                                                             delimiter=delimiter,
                                                                             header=header)
        self.run(sql_code)
        return col_names, sql_code


if __name__ == "__main__":
    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                   host="localhost")
    with open(
            r"/home/luca/Documents/Bachelorarbeit/BA_code_mlinspect_fork/mlinspect_fork/mlinspect/mlinspect/to_sql/"
            r"generated_code/create_table.sql") as file:
        content = file.read()
    drop_p = f"DROP TABLE IF EXISTS patients_1;"
    drop_h = f"DROP TABLE IF EXISTS histories_2;"
    postgres.run(drop_p)
    postgres.run(drop_h)
    res = postgres.run(content)

    # ATTENTION: FOR SOME REASON, CLOSING THE CONNECTION BEFORE RUNNING ANALYSE YIELDS SHORTER EXEC TIME!!
    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432,
                                   host="localhost")

    with open(
            r"/home/luca/Documents/Bachelorarbeit/BA_code_mlinspect_fork/mlinspect_fork/mlinspect/mlinspect/to_sql/"
            r"generated_code/pipeline.sql") as file:
        pipe = file.read()

    postgres.benchmark_run(pipe, repetitions=100)
