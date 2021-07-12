import abc
import pandas

class Connector(abc.ABC):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __del__(self):
        """Does the cleanup"""
        pass

    @abc.abstractmethod
    def add_csv(self, path_to_csv: str, table_name: str, null_symbols: list, delimiter: str, header: bool, *args,
                **kwargs) -> (list, str):
        """ Has to correctly add the passed csv to the DBMS under the passed Name. Implement DROP of old if necessary.
        Args:
            path_to_csv(str):
            table_name(str): the name under which the table will be accessed.
            null_symbols(list of strings):
            delimiter(str):
            header(str):
        Returns:
            Tuple(column_names, sql_code)
                column_names: All column names as list.
                sql_code: The SQL code to create the table, if none is available just return "".
        Note:
            If more attributes are required, add them through args and kwargs.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, sql_query) -> pandas.DataFrame:
        """
        Args:
            sql_query(str): The query
        Returns:
            pandas.DataFrame containing the query results.
        """
        raise NotImplementedError()

    def benchmark_run(self, sql_query, repetitions):
        """
        Args:
            sql_query(str): Single query
            repetitions(int): number of repetition
        Returns:
            Time im ms.
        Note:
            Not abstract, as optional.
        """
        raise NotImplementedError()

    @staticmethod
    def _prepare_query(sql_query):
        """
        Strips the query and returns a list of the sub-queries.
        """
        sql_query = sql_query.strip()
        return [x + ";" for x in sql_query.split(";") if x != ""]
