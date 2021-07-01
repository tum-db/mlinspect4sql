from dbms_connector import Connector
import psycopg2


class PostgresqlConnector(Connector):
    def __init__(self, dbname, user, password, port, host):
        """
        Note: For Postgresql:
            1) install Postgresql and start the server
            2) assert it is running: "sudo netstat -lntup | grep '5433\|5432'"
            3) Create a new DB: "CREATE DATABASE healthcare_benchmark;"
            4) Connect with arguments below:
                db_name: "healthcare_benchmark"
                user: "postgres"
                password:"password" // not used
                port:"5433"
                host:"/tmp/"
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
    postgres = PostgresqlConnector(dbname="healthcare_benchmark", user="luca", password="password", port=5432, host="localhost")
    cur = postgres.run("SELECT 1;")
    print(cur.fetchall())