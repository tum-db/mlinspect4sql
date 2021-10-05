import pathlib
from mlinspect.utils import get_project_root
import pandas as pd


class CreateTablesFromDataSource:
    """Infer a table schema from a CSV."""

    @staticmethod
    def _check_file(file):
        root_dir = get_project_root()
        file = pathlib.Path(file)
        if file.is_file():
            file = str(file)
        elif (root_dir / file).is_file():
            file = str(root_dir / file)
        else:
            raise FileNotFoundError
        return file

    @staticmethod
    def _get_data(path_to_csv):
        """Load data from CSV"""
        return pd.read_csv(path_to_csv, header=0, encoding='utf-8', nrows=100)

    @staticmethod
    def _get_schema_from_data_frame(data_frame):
        """Infers schema from CSV."""
        # Also the column names need to be quoted to avoid getting a keywortd as "end"
        col_names = [f"\"{x}\"" for x in data_frame.columns.values if x != "index_mlinspect"]

        if "index_mlinspect" in data_frame.columns.values:
            col_names.append("index_mlinspect")

        types = list(data_frame.dtypes)
        # return types as str -> except 'string' is replaced by varchar(100) to comply with umbra
        for i, value in enumerate(types):
            if value.name == 'object':
                types[i] = 'VARCHAR(100)'
            elif value.name == 'int64':
                types[i] = 'INT'
            elif value.name == 'float64':
                types[i] = 'FLOAT'
            else:
                raise NotImplementedError
        return col_names, types

    @staticmethod
    def _create_table_code(data_frame, table_name, add_mlinspect_serial, index_col):

        names, data_types = CreateTablesFromDataSource._get_schema_from_data_frame(data_frame)

        drop_old_table = f"DROP TABLE IF EXISTS {table_name};"
        if add_mlinspect_serial:
            if index_col != -1:
                names[index_col] = "index_mlinspect"

        create_table = f"CREATE TABLE {table_name} (\n\t" + ",\n\t".join(
            [i + " " + j for i, j in zip(names, data_types)])

        if add_mlinspect_serial and index_col == -1:
            create_table += ",\n\tindex_mlinspect SERIAL PRIMARY KEY"
        create_table += "\n)"

        return names, drop_old_table, create_table

    @staticmethod
    def get_sql_code_csv(path_to_csv, table_name, null_symbols=None, delimiter=",", header=True,
                         drop_old=False, add_mlinspect_serial=False, index_col=-1):
        """
        Args:
            add_mlinspect_serial(bool): add serial index -> take index_col if not -1
            index_col(int):index of index column if present -> -1 can be used for no
                index column. !Attention: see pandas "index_col"!
        """
        if null_symbols is None:
            null_symbols = ["?"]

        data_frame = CreateTablesFromDataSource._get_data(path_to_csv)

        names, drop_old_table, create_table = CreateTablesFromDataSource._create_table_code(data_frame, table_name,
                                                                                            add_mlinspect_serial,
                                                                                            index_col=index_col)

        if len(null_symbols) != 1:
            raise NotImplementedError("Currently only ONE null symbol supported!")

        add_data = f"COPY {table_name}({', '.join([i for i in list(names)])}) " \
                   f"FROM '{path_to_csv}' WITH (" \
                   f"DELIMITER '{delimiter}', NULL '{null_symbols[0]}', FORMAT CSV, HEADER {'TRUE' if header else 'FALSE'})"

        if drop_old:
            return f"{drop_old_table};\n\n{create_table};\n\n{add_data};"
        return names, f"{create_table};\n\n{add_data};"

    @staticmethod
    def get_sql_code_data_frame(data_frame, table_name, drop_old=False, add_mlinspect_serial=False):

        # Still open TODO
        names, drop_old_table, create_table = CreateTablesFromDataSource._create_table_code(data_frame, table_name,
                                                                                            add_mlinspect_serial)

        add_data = f"INSERT INTO table_name \n" \
                   f"VALUES"

        for index, row in data_frame.iterrows():
            add_data += f"(\n" \
                        f"{', '.join(row)}" \
                        f"\n),\n"

        add_data = add_data[:-2] + ";\n"

        if drop_old:
            return f"{drop_old_table};\n\n{create_table};\n\n{add_data};"
        return names, f"{create_table};\n\n{add_data};"
