import abc
import pandas


class Connector(abc.ABC):
    """
    This Class represents an interface to the user to provide a DBMS.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """startup"""
        pass

    @abc.abstractmethod
    def __del__(self):
        """cleanup"""
        pass

    @abc.abstractmethod
    def add_csv(self, path_to_csv: str, table_name: str, null_symbols: list, delimiter: str, header: bool, *args,
                **kwargs) -> (list, str):
        """ Has to correctly add the passed csv to the DBMS under the given name. Implements DROP of old if necessary.
        Args:
            path_to_csv(str): path to the csv.
            table_name(str): the name under which the table will be accessed.
            null_symbols(list): list of symbols representing [null] in the DBMS.
            delimiter(str): the separator char for the csv.
            header(bool): states if the header can be taken from the csv or names need to be generated.
        Returns:
            Tuple(column_names, sql_code)
                column_names: All column names as list.
                sql_code: The SQL code to create the table, if none is available just return "".
        Note:
            If more attributes are required, add them through args and kwargs.

        ATTENTION: The added table NEEDS to contain a index column, called: "index_mlinspect" +
            create an index on it: "CREATE UNIQUE INDEX id_mlinspect ON <table_name> (index_mlinspect);"
        """
        raise NotImplementedError()

    def add_dataframe(self, data_frame: pandas.DataFrame, table_name: str, *args, **kwargs) -> (list, str):
        """ Has to correctly add the passed csv to the DBMS under the given name. Implements DROP of old if necessary.
        Args:
            data_frame(pandas.DataFrame): the pandas.DataFrame object.
            table_name(str): the name under which the table will be accessed.
        Returns:
            Tuple(column_names, sql_code)
                column_names: All column names as list.
                sql_code: The SQL code to create the table, if none is available just return "".
        Note:
            If more attributes are required, add them through args and kwargs.

        ATTENTION: The added table NEEDS to contain a index column, called: "index_mlinspect" +
            create an index on it: "CREATE UNIQUE INDEX id_mlinspect ON <table_name> (index_mlinspect);"
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, sql_query: str) -> pandas.DataFrame:
        """
        Args:
            sql_query(str): query as string.
        Returns:
            pandas.DataFrame containing the query results.
        """
        raise NotImplementedError()

    def benchmark_run(self, sql_query: str, repetitions: int):
        """
        Args:
            sql_query(str): query as string.
            repetitions(int): number of repetition
        Returns:
            Time im ms.
        Note:
            Not abstract, it is optional.
        """
        raise NotImplementedError()

    @staticmethod
    def _prepare_query(sql_query):
        """
        Strips the query and returns a list of the sub-queries.
        """
        sql_query = sql_query.strip()
        return [x + ";" for x in sql_query.split(";") if x != ""]
