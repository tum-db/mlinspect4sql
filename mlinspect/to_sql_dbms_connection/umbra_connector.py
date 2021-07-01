from dbms_connector import Connector
import psycopg2


class UmbraConnector(Connector):
    def __init__(self, dbname, user, password, port, host):
        """
        Note: For Umbra:
            1) clone the Umbra repo.
            2) build everything: "cd umbra && mkdir build && cd build && cmake .. && make"
            3) Optional: Create User
            3) create a db file with: "./bin/sql -createdb <dbname>"
            4) Start server (with data base): "./build/server /path/to/<dbname> -port=5433 -address=localhost"
            5) Confirm it is running: "sudo netstat -lntup | grep '5433\|5432'"
            6) Connect with arguments below:
                db_name: "healthcare_benchmark"
                user: "postgres"
                password:"password" // not used
                port:"5432"
                host:"localhost"
        """
        super().__init__(dbname, user, password, port, host)
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, port=port, host=host)
        self.cur = conn.cursor()

    def run(self, sql_query):
        sql_query = sql_query.strip()
        if sql_query[-1] != ";":
            sql_query = sql_query + ";"
        self.cur.execute(sql_query)
        return self.cur

    def get_result(self):
        return self.cur.fetchall()


if __name__ == "__main__":
    umbra = UmbraConnector(dbname="healthcare_benchmark", user="postgres", password=" ", port=5433, host="/tmp/")
    cur = umbra.run("SELECT 1;")
    print(cur.fetchall())
