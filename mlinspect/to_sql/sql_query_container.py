import os


class SQLQueryContainer:
    """ Container class that holds the entire pipeline up until current code.
    """

    def __init__(self, root_dir_to_sql, write_to_file=True):
        self.pipeline_query = []
        self.write_to_file = write_to_file

        # Filenames of the files that will work as container for the pipeline translation.
        self.file_path_pipe = root_dir_to_sql / "pipeline.sql"
        self.file_path_create_table = root_dir_to_sql / "create_table.sql"
        self.root_dir_to_sql = root_dir_to_sql

    def get_pipe_with_changed_selection(self, selection_statement):
        pass

    def add_statement_to_pipe(self, last_cte_name, sql_code, cols_to_keep):
        """
        Stores and add statements of the pipeline internally and at "mlinspect/to_sql/generated_code/pipeline.sql"
        Note:
            The code blocks in the list need to have three proprieties:
                - The fist block if its a CTE starts with "WITH"
                - Every list element, besides the last ends with a comma.
                - The last list element is a "SELECT ... ;" statement.
        """
        self.pipeline_query = self.pipeline_query[:-1]
        self.pipeline_query.append(sql_code)
        select_line = self.__add_select_line(self.file_path_pipe, last_cte_name, cols_to_keep)
        self.pipeline_query.append(select_line)

    def write_to_init_file(self, sql_code):
        """
        Stores and writes the code for the table creation to "mlinspect/to_sql/generated_code/create_table.sql"
        """
        with self.file_path_create_table.open(mode="a", ) as file:
            file.write(sql_code + "\n\n")

    def write_to_side_query(self, last_cte_name, sql_code, file_name):
        """
        Stores and add side statements (like ratio measurements) at "mlinspect/to_sql/generated_code/ratio_<column>.sql"
        """
        if len(file_name.split(".")) == 1:
            file_name = file_name + ".sql"
        path = self.root_dir_to_sql / file_name
        SQLQueryContainer.__del_select_line(path)
        with path.open(mode="a")as file:
            file.write(sql_code)
        SQLQueryContainer.__add_select_line(path, last_cte_name)

    def __write_to_pipe_query(self, last_cte_name, sql_code, cols_to_keep):
        SQLQueryContainer.__del_select_line(self.file_path_pipe)
        with self.file_path_pipe.open(mode="a") as file:
            file.write(sql_code)
        SQLQueryContainer.__add_select_line(self.file_path_pipe, last_cte_name, cols_to_keep)

    @staticmethod
    def __del_select_line(path, add_comma=True):
        """
        Delestes the last line and add a comma (",")
        Note:
            Partly taken from: https://stackoverflow.com/a/10289740/9621080
        """
        with path.open("a+", encoding="utf-8") as file:

            # Move the pointer (similar to a cursor in a text editor) to the end of the file
            if file.seek(0, os.SEEK_END) == 0:  # File is empty
                return

            # This code means the following code skips the very last character in the file -
            # i.e. in the case the last line is null we delete the last line
            # and the penultimate one
            pos = file.tell() - 1

            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                file.seek(pos, os.SEEK_SET)

            # So long as we're not at the start of the file, delete all the characters ahead
            # of this position
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()

            # Add comma if desired:
            file.write(",\n") if add_comma else 0

    @staticmethod
    def __add_select_line(path, last_cte_name, cols_to_keep=None):
        """
        Args:
            cols_to_keep(list)
        """
        if cols_to_keep is None or cols_to_keep == []:
            selection = "*"
        else:
            selection = ", ".join(cols_to_keep)
        line = f"\nSELECT {selection} FROM {last_cte_name};"
        with path.open(mode="a") as f:
            f.write(line)
        return line
