import os
from mlinspect.to_sql._mode import SQLObjRep, SQLMode


class SQLQueryContainer:
    """ Container class that holds the entire pipeline up until current code.
    """

    def __init__(self, root_dir_to_sql, sql_obj: SQLMode, write_to_file=True):
        self.pipeline_query = []
        self.write_to_file = write_to_file

        self.sql_obj = sql_obj

        # Filenames of the files that will work as container for the pipeline translation.
        self.file_path_pipe = root_dir_to_sql / "pipeline.sql"
        self.file_path_create_table = root_dir_to_sql / "create_table.sql"
        self.root_dir_to_sql = root_dir_to_sql

    def get_pipe_without_selection(self):
        assert self.sql_obj.mode == SQLObjRep.CTE  # only then this op makes sense
        return f",\n".join(self.pipeline_query[:-1]).strip()

    def add_statement_to_pipe(self, last_cte_name, sql_code, cols_to_keep=None):
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
        select_line = self.__write_to_pipe_query(last_cte_name, sql_code, cols_to_keep)
        self.pipeline_query.append(select_line)

    def update_pipe_head(self, head, comment=""):
        self.pipeline_query = self.pipeline_query[:-1]
        # Uncomment last pipeline head:
        if not comment == "":
            with self.file_path_pipe.open(mode="r") as f:
                head = f"\n/*ATTENTION: {comment}\n" \
                       f"{head};\n*/\n\n" + head

        SQLQueryContainer.__del_select_line(self.file_path_pipe, False)

        with self.file_path_pipe.open(mode="a") as f:
            f.write(head)
        self.pipeline_query.append(head)
        return

    def write_to_init_file(self, sql_code):
        """
        Stores and writes the code for the table creation to "mlinspect/to_sql/generated_code/create_table.sql"
        """
        with self.file_path_create_table.open(mode="a", ) as file:
            file.write(sql_code + "\n\n")

    def write_to_side_query(self, full_sql_code, file_name):
        """
        Stores and add side statements (like ratio measurements) at "mlinspect/to_sql/generated_code/ratio_<column>.sql"
        """
        if len(file_name.split(".")) == 1:
            file_name = file_name + ".sql"
        path = self.root_dir_to_sql / file_name
        with path.open(mode="a") as file:
            file.write(full_sql_code)

    def __write_to_pipe_query(self, last_cte_name, sql_code, cols_to_keep=None):
        SQLQueryContainer.__del_select_line(self.file_path_pipe, self.sql_obj.mode == SQLObjRep.CTE)
        with self.file_path_pipe.open(mode="a") as file:
            file.write(sql_code)
        return SQLQueryContainer.__add_select_line(self.file_path_pipe, last_cte_name, cols_to_keep)

    def get_last_query_materialize(self, sql_obj_to_materialize, cols_to_keep=None):
        """
        This function return the code and sql_obj name to materialize the passed view, by creating a materialized one
        and changing the name in the mapping to the new one.
        """
        assert (self.sql_obj.mode == SQLObjRep.VIEW and self.sql_obj.materialize)  # only then this op makes sense

        for i, query in enumerate(reversed(self.pipeline_query[:-1])):
            if sql_obj_to_materialize in query.split("AS")[0]:  # This is the statements, that created the view.
                if "MATERIALIZED" in query:
                    return None
                new_name = sql_obj_to_materialize + "_materialized"
                query = query.split(f"CREATE VIEW {sql_obj_to_materialize}")[1]
                new_view_query = f"CREATE MATERIALIZED VIEW {new_name}" + query
                self.pipeline_query[len(self.pipeline_query) - i - 2] = new_view_query

                self.__write_to_pipe_query(new_name, sql_code=new_view_query, cols_to_keep=cols_to_keep)

                return new_view_query, new_name
        assert False

    @staticmethod
    def __del_select_line(path, add_comma):
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
