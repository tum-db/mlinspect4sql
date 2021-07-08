import abc


class Connector(abc.ABC):

    def __init__(self, dbname, user, password, port, host):
        pass

    def run(self, sql_query):
        """
        Args:
            sql_query(str): The query
        Returns:
            List of all the query results.
        """
        raise NotImplementedError()

    def benchmark_run(self, sql_query, repetitions):
        """
        Args:
            sql_query(str): Single query
            repetitions(int): number of repetition
        Returns:
            List of all the query results.
        """
        raise NotImplementedError()

    @staticmethod
    def _prepare_query(sql_query):
        """
        Strips the query and returns a list of the sub-queries.
        """
        sql_query = sql_query.strip()
        return [x + ";" for x in sql_query.split(";") if x != ""]
