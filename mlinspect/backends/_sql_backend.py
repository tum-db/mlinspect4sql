import pandas
import pandas as pd
from tableschema import Table
import sqlalchemy
from sqlalchemy.types import Integer, Text, BOOLEAN
import pathlib
from mlinspect.utils import get_project_root

ROOT_DIR = get_project_root()


class SQLBackend:
    first_with = True
    id = 1

    def wrap_in_with(self, sql_code, lineno):
        with_block_name = f"with_{lineno}_{self.get_unique_id()}"
        sql_code = sql_code.replace('\n', '\n\t')  # for nice formatting
        sql_code = f"{with_block_name} AS (\n\t{sql_code}\n),"
        if self.first_with:
            sql_code = "WITH " + sql_code
            self.first_with = False
        return with_block_name, sql_code

    def get_unique_id(self):
        self.id += 1
        return self.id - 1

    def handle_operation_series(self, operator, mapping, result, left, right, lineno):
        left_column = left
        right_column = right
        left_origin = ""
        right_origin = ""
        if isinstance(left, pandas.Series):
            left_column = "l." + left_column.name
            left_origin = f"{mapping.get_name(left)} AS l, "
        if isinstance(right, pandas.Series):
            right_column = "r." + right_column.name
            right_origin = f"{mapping.get_name(right)} AS r"
        assert left_origin + right_origin != ""  # at least one needs to be set!

        rename = ""
        if operator in [">", "<", ">=", "=", "<="]:  # a rename gets necessary, as otherwise the name will be "None"
            rename = f"AS compare_{self.get_unique_id()}"
            result.name = rename.split(" ")[1]

        sql_code = f"SELECT {left_column} {operator} {right_column} {rename}\n" \
                   f"FROM {left_origin}{right_origin}"

        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno)
        mapping.add(sql_table_name, result)
        print(sql_code + "\n")
        return result

    def handle_operation_dataframe(self, operator, mapping, result, left, right, lineno):

        # TODO: handle operations over pandas.DataFrames
        operator = "*"
        multiplicand = left
        multiplier = right
        affected_columns = self.name
        if not isinstance(affected_columns, list):
            affected_columns = [affected_columns]

        sql_code = f"SELECT {f' {operator} {multiplier} , '.join(affected_columns)} {operator} {multiplier} \n" \
                   f"FROM {mapping.get_name(multiplicand)}"

        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno)
        mapping.add(sql_table_name, result)
        print(sql_code + "\n")
        return result


class CreateTablesFromCSVs:
    """Infer a table schema from a CSV."""

    def __init__(self, file):
        file = pathlib.Path(file)
        if file.is_file():
            self.file = str(file)
        elif (ROOT_DIR / file).is_file():
            self.file = str(ROOT_DIR / file)
        else:
            raise FileNotFoundError

    def _get_data(self):
        """Load data from CSV"""
        return pd.read_csv(self.file, header=0, encoding='utf-8')

    def _get_schema_from_csv(self, types_as_string=False):
        """Infers schema from CSV."""
        table = Table(self.file)
        table.infer(limit=500, confidence=0.55)
        schema = table.schema.descriptor

        if schema["missingValues"][0] != "":
            raise ValueError(f"Unfortunately not all columns could be parsed -> {schema['missingValues'][0]}")

        col_names = []
        for i, dic in enumerate(schema['fields']):
            col_names.append(dic['name'])

        if types_as_string:  # return types as str -> except 'string' is replaced by varchar(100) to comply with umbra
            return col_names, [("varchar(100)" if i['type'] == "string" else i['type']) for i in schema['fields']]
        types = []
        for i, value in enumerate(schema['fields']):
            if value['type'] == 'integer':
                types.append(Integer)
            elif value['type'] == 'string':
                types.append(Text)
            elif value['type'] == 'boolean':
                types.append(BOOLEAN)
            else:
                raise NotImplementedError

        return col_names, types

    @staticmethod
    def _get_table_def(col_names, types, table_name):
        """Create new sqlalchemy table from CSV and generated schema."""
        meta = sqlalchemy.MetaData()
        columns = []
        for name, data_t in zip(col_names, types):
            columns.append(sqlalchemy.Column(name, data_t))
        return sqlalchemy.Table(table_name, meta, *columns)

    def add_to_database(self, engine, table_name):
        """
        Notes:
           - For a big table see: https://stackoverflow.com/a/34523707/9621080 -> To make if fast!
        """
        names, data_types = self._get_schema_from_csv()
        table_definition = self._get_table_def(names, data_types, table_name)

        # drop the table if it exists:
        engine.execute(f"DROP TABLE IF EXISTS {table_definition.fullname};")

        # create the table, with the passed code
        table_definition.create(engine)

        table_name = str(table_definition.fullname)
        table_columns = ", ".join([str(i.name) for i in list(table_definition.columns)])
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cmd = f"COPY {table_name}({table_columns}) FROM STDIN WITH (DELIMITER ',', NULL '?', FORMAT CSV, HEADER TRUE)"
        with open(self.file) as csv:
            cursor.copy_expert(cmd, csv)
        conn.commit()

    def get_sql_code(self, table_name, null_symbol="?", delimiter=",", header=True, drop_old=False):
        names, data_types = self._get_schema_from_csv(types_as_string=True)

        drop_old_table = f"DROP TABLE IF EXISTS {table_name};"

        create_table = f"CREATE TABLE {table_name} (\n\t" + ",\n\t".join(
            [i + " " + j for i, j in zip(names, data_types)]) + "\n)"

        add_data = f"COPY {table_name}({', '.join([i for i in list(names)])}) " \
                   f"FROM '{self.file}' WITH (" \
                   f"DELIMITER '{delimiter}', NULL '{null_symbol}', FORMAT CSV, HEADER {'TRUE' if header else 'FALSE'})"

        if drop_old:
            return f"{drop_old_table};\n\n{create_table};\n\n{add_data};"
        return f"{create_table};\n\n{add_data};"
