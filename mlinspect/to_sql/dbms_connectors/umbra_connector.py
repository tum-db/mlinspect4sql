import pathlib
import random

from dbms_connector import Connector
import psycopg2
from random import randrange
import subprocess
import time


class UmbraConnector(Connector):
    def __init__(self, dbname, user, password, port, host, umbra_dir=None):
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
        self.umbra_dir = umbra_dir
        # if dbname == "":  # Create new DB
        #     self.dbname = self.__new_database()
        # else:
        #     self.dbname = dbname

        # check if already running:
        result = subprocess.run("pgrep serve", stdout=subprocess.PIPE, shell=True)
        if result.returncode == 0:  # here we check if the process is already running
            subprocess.run(f"kill -9 {result.stdout.decode('utf-8').strip()}", stdout=subprocess.PIPE, shell=True)
        command = f"./build/server \"\" -port=5433 -address=localhost"
        self.server = subprocess.Popen(command, cwd=self.umbra_dir, shell=True, stdout=subprocess.DEVNULL)
        time.sleep(0.05)  # wait for the server to start
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

    def __new_database(self):
        new_db_name = f"healthcare_benchmark_{randrange(10000)}"
        command = f"./bin/sql -createdb {new_db_name}"
        subprocess.Popen("exec " + command, cwd=self.umbra_dir, shell=True)
        return new_db_name

    def close(self):
        # [x.unlink() for x in pathlib.Path(self.umbra_dir).glob(f"{self.dbname}*")]
        self.server.kill()
        return


if __name__ == "__main__":
    umbra_path = r"/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students"
    umbra = UmbraConnector(dbname="", user="postgres", password=" ", port=5433, host="/tmp/", umbra_dir=umbra_path)
    cur = umbra.run("SELECT 1;")
    print(cur.fetchall())
    umbra.close()