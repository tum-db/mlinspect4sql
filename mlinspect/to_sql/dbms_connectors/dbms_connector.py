import abc


class Connector(abc.ABC):

    def __init__(self, dbname, user, password, port, host):
        pass

    def run(self, sql_query):
        """
        This function needs to be able to execute a string of one or multiple SQL-Statements divided by a ";". The
        results have to ge returned.
        Args:
            sql_query(str): The query
        Returns:
            List of all the query results.
        """
        raise NotImplementedError()

    def get_result(self):
        """
        """
        raise NotImplementedError()
