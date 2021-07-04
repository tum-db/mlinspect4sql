from dbms_connector import Connector
import psycopg2
import subprocess
import time


class UmbraConnector(Connector):
    def __init__(self, dbname, user, password, port, host, umbra_dir=None):
        """
        Starts a new empty Umbra server is 'DROP' is not implemented yet.

        Note: For Umbra:
            1) clone the Umbra repo.
            2) build everything: "mkdir build && cd build && cmake .. && make"
            3) Optional: Create User
            3) create a db file with: "./bin/sql -createdb <dbname>"
            4) Start server (with data base): "./build/server /path/to/<dbname> -port=5433 -address=localhost"
            5) Confirm it is running: "sudo netstat -lntup | grep '5433\|5432'"
            6) Connect with arguments below: - In terminal: "psql -h /tmp -p 5433 -U postgres"
                db_name: "healthcare_benchmark"
                user: "postgres"
                password:"password" // not used
                port:"5432"
                host:"localhost"
        Not implemented yet:
        DROP, ALTER, DELETE, CREATE MATERIALIZED VIEW
        """
        super().__init__(dbname, user, password, port, host)
        self.umbra_dir = umbra_dir

        # check if already running:
        result = subprocess.run("pgrep serve", stdout=subprocess.PIPE, shell=True)
        if result.returncode == 0:  # here we check if the process is already running
            subprocess.run(f"kill -9 {result.stdout.decode('utf-8').strip()}", stdout=subprocess.PIPE, shell=True)
        command = f"./build/server \"\" -port=5433 -address=localhost"
        self.server = subprocess.Popen(command, cwd=self.umbra_dir, shell=True, stdout=subprocess.DEVNULL)
        time.sleep(0.1)  # wait for the server to start
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, port=port, host=host)
        self.cur = conn.cursor()

    def run(self, sql_query):
        sql_query = sql_query.strip()
        sql_queries = sql_query.split(";")
        results = []
        for q in sql_queries:
            if q == "":
                continue
            self.cur.execute(q + ";")
            try:
                results.append(self.cur.fetchall())
            except psycopg2.ProgrammingError:  # Catch the case no result is available (f.e. create Table)
                continue
        return results

    def close(self):
        self.server.kill()


if __name__ == "__main__":
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    umbra = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/", umbra_dir=umbra_path)
    with open(
            r"/home/luca/Documents/Bachelorarbeit/BA_code_mlinspect_fork/mlinspect_fork/mlinspect/mlinspect/to_sql/generated_code/create_table.sql") as file:
        content = file.read()
    result = umbra.run(content)

    with open(
            r"/home/luca/Documents/Bachelorarbeit/BA_code_mlinspect_fork/mlinspect_fork/mlinspect/mlinspect/to_sql/generated_code/pipeline.sql") as file:
        content = file.read()
    result = umbra.run(content)
    print(result)
    umbra.close()
