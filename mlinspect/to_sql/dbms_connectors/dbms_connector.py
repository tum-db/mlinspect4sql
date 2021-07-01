import abc


class Connector(abc.ABC):

    def __init__(self, dbname, user, password, port, host):
        pass

    def run(self, sql_query):
        """
        """
        raise NotImplementedError()#

    def get_result(self):
        """
        """
        raise NotImplementedError()
