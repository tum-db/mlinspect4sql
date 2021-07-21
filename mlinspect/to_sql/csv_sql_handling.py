from tableschema import Table
import pathlib
from mlinspect.utils import get_project_root
import pandas as pd


class CreateTablesFromCSVs:
    """Infer a table schema from a CSV."""

    def __init__(self, file, ):
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
        return pd.read_csv(self.file, header=0, encoding='utf-8', nrows=100)

    def _get_schema_from_csv(self):
        """Infers schema from CSV."""
        csv = self._get_data()

        # Also the column names need to be quoted to avoid getting a keywortd as "end"
        col_names = [f"\"{x}\"" for x in csv.columns.values]
        types = list(csv.dtypes)
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

    def get_sql_code(self, table_name, null_symbols=None, delimiter=",", header=True, drop_old=False,
                     add_mlinspect_serial=False):
        if null_symbols is None:
            null_symbols = ["?"]
        names, data_types = self._get_schema_from_csv()

        drop_old_table = f"DROP TABLE IF EXISTS {table_name};"

        create_table = f"CREATE TABLE {table_name} (\n\t" + ",\n\t".join(
            [i + " " + j for i, j in zip(names, data_types)])
        if add_mlinspect_serial:
            create_table += ",\n\tindex_mlinspect SERIAL PRIMARY KEY\n)"
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
