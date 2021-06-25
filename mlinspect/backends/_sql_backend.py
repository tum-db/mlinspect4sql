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
        """
        Wrappes the passed sql code in a WITH... AS block. Takes into account, that WITH only needs to be used once.
        """
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

        # a rename gets necessary, as otherwise in binary ops the name will be "None"
        rename = f"op_{self.get_unique_id()}"
        result.name = rename  # don't forget to set pandas object name!
        where_block = ""
        from_block = ""
        if isinstance(left, pandas.Series) and isinstance(right, pandas.Series):
            select_block = f"(l.{left.name} {operator} r.{right.name}) AS {rename}"
            from_block = f"{self.create_indexed_table(mapping.get_name(left))} l, " \
                         f"{self.create_indexed_table(mapping.get_name(right))} r"
            where_block = f"\nWHERE l.row_number = r.row_number"
        elif isinstance(left, pandas.Series):
            select_block = f"({left.name} {operator} {right}) AS {rename}"
            from_block = f"{mapping.get_name(left)}"
        elif isinstance(right, pandas.Series):
            select_block = f"({left} {operator} {right.name}) AS {rename}"
            from_block = f"{mapping.get_name(right)}"

        sql_code = f"SELECT {select_block}\n" \
                   f"FROM {from_block}" \
                   f"{where_block}"

        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno)
        mapping.add(sql_table_name, result)
        print(sql_code + "\n")
        return result

    def __handle_operation_dataframe(self, operator, mapping, result, left, right, lineno):

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

    @staticmethod
    def __column_ratio_original(table_orig, column_name):
        print(f"original_ratio_{column_name} as (\n"
              f"\tselect i.{column_name}, (count(*) * 1.0 / (select count(*) from {table_orig})) as ratio\n"
              f"\tfrom {table_orig} i\n"
              f"\tgroup by i.{column_name}\n"
              "),")

    @staticmethod
    def __column_ratio_current(table_orig, table_new, column_name):
        """
        Here the query for the new/current ratio of the values inside the passed column is provided.
        """
        print(f"current_ratio_{column_name} as (\n"
              f"\tselect orig.{column_name}, (\n"
              f"\t\t(select count(temp.original_label)\n"
              f"\t\tfrom (\n"
              f"\t\t\tselect lookup.original_label\n"
              f"\t\t\tfrom lookup_{column_name}_{table_new} lookup, {table_new} curr\n"
              f"\t\t\twhere lookup.current_label = curr.{column_name} and "
              f"orig.{column_name} = lookup.original_label and lookup.current_label is not null) temp\n"
              f"\t\t) * 1.0 / (select count(*) from {table_new})) as ratio\n"
              f"\tfrom {table_orig} orig,  {table_new} curr, lookup_{column_name}_{table_new} lookup\n"
              f"\twhere curr.{table_orig}_ctid = orig.ctid\n"
              f"\tgroup by orig.{column_name}\n"
              f"),")

    @staticmethod
    def __lookup_table(table_orig, table_new, column_name):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        print(f"lookup_{column_name}_{table_new} as (\n"
              f"\tselect distinct orig.{column_name} as original_label, curr.{column_name} as current_label\n"
              f"\tfrom {table_orig} orig,  {table_new} curr\n"
              f"\twhere curr.{table_orig}_ctid = orig.ctid\n"
              f"),")

    @staticmethod
    def ratio_track(origin_dict, column_names, table_name):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        :param origin_dict: Dictionary with all the origin tables of the single attributes.
        :param column_names: The column names of which we want to have the ratio comparison
        :param table_name: the name of the table of which we need the 'current' ratio.
        :return: None -> see stdout
        """
        for i in column_names:
            table_orig = origin_dict[i]
            SQLBackend.__lookup_table(table_orig, table_new=table_name, column_name=i)
        print()
        for i in column_names:
            table_orig = origin_dict[i]
            SQLBackend.__column_ratio_original(table_orig, column_name=i)
        print()
        for i in column_names:
            table_orig = origin_dict[i]
            SQLBackend.__column_ratio_current(table_orig, table_new=table_name, column_name=i)

    @staticmethod
    def create_indexed_table(table_name):
        return f"(SELECT *, ROW_NUMBER() OVER(ORDER BY NULL) FROM {table_name})"


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

        col_names = [x["name"] for x in schema["fields"]]

        # return types as str -> except 'string' is replaced by varchar(100) to comply with umbra
        types = []
        for i, value in enumerate(schema['fields']):
            if types_as_string:
                if value['type'] == 'number':
                    types.append('float')
                elif value['type'] == 'string':
                    types.append("varchar(100)")
                else:
                    types.append(value['type'])
            else:
                if value['type'] == 'integer' or value['type'] == 'number':
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
