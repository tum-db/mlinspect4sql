from tableschema import Table
import pathlib
from mlinspect.utils import get_project_root
import pandas as pd


class CreateTablesFromCSVs:
    """Infer a table schema from a CSV."""

    def __init__(self, file):
        root_dir = get_project_root()
        file = pathlib.Path(file)
        if file.is_file():
            self.file = str(file)
        elif (root_dir / file).is_file():
            self.file = str(root_dir / file)
        else:
            raise FileNotFoundError

    def _get_data(self):
        """Load data from CSV"""
        return pd.read_csv(self.file, header=0, encoding='utf-8')

    def _get_schema_from_csv(self):
        """Infers schema from CSV."""
        table = Table(self.file)
        table.infer(limit=500, confidence=0.55)
        schema = table.schema.descriptor

        if schema["missingValues"][0] != "":
            raise ValueError(f"Unfortunately not all columns could be parsed -> {schema['missingValues'][0]}")

        # Also the column names need to be quoted to avoid getting a keywortd as "end"
        col_names = [f"\"{x['name']}\"" for x in schema["fields"]]

        # return types as str -> except 'string' is replaced by varchar(100) to comply with umbra
        types = []
        for i, value in enumerate(schema['fields']):
            if value['type'] == 'number':
                types.append('float')
            elif value['type'] == 'string':
                types.append("varchar(100)")
            else:
                types.append(value['type'])
        return col_names, types

    def get_sql_code(self, table_name, null_symbols=None, delimiter=",", header=True, drop_old=False,
                     add_mlinspect_serial=False):
        if null_symbols is None:
            null_symbols = ["?"]
        names, data_types = self._get_schema_from_csv()

        drop_old_table = f"DROP TABLE IF EXISTS {table_name};"

        create_table = f"CREATE TABLE {table_name} (\n\t" + ",\n\t".join(
            [i + " " + j for i, j in zip(names, data_types)])
        if add_mlinspect_serial:
            create_table += ",\nindex_mlinspect SERIAL PRIMARY KEY\n)"
        else:
            create_table += "\n)"

        if len(null_symbols) != 1:
            raise NotImplementedError("Currently only ONE null symbol supported!")

        add_data = f"COPY {table_name}({', '.join([i for i in list(names)])}) " \
                   f"FROM '{self.file}' WITH (" \
                   f"DELIMITER '{delimiter}', NULL '{null_symbols[0]}', FORMAT CSV, HEADER {'TRUE' if header else 'FALSE'})"

        if drop_old:
            return f"{drop_old_table};\n\n{create_table};\n\n{add_data};"
        return names, f"{create_table};\n\n{add_data};"


class CreateTableRaw:
    def get_sql_code(self, table_name, column_names, column_types, column_values, drop_old=False):
        drop_old_table = f"DROP TABLE IF EXISTS {table_name};"

        create_table = f"CREATE TABLE {table_name} (\n\t" + ",\n\t".join(
            [i + " " + j for i, j in zip(column_names, column_types)]) + "\n)"

        add_data = f"COPY {table_name}({', '.join([i for i in list(column_names)])}) " \
                   f"FROM VALUES"

        if drop_old:
            create_table = f"{drop_old_table};\n\n{create_table}"
        return f"{create_table};\n\n{add_data};"
