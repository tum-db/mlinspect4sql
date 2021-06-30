import pandas
import pandas as pd
from tableschema import Table
import sqlalchemy
from sqlalchemy.types import Integer, Text, BOOLEAN
import pathlib
from mlinspect.utils import get_project_root


class SQLBackend:
    first_with = True
    id = 1

    def wrap_in_with(self, sql_code, lineno, with_block_name=""):
        """
        Wrappes the passed sql code in a WITH... AS block. Takes into account, that WITH only needs to be used once.
        """
        if with_block_name == "":
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
        columns_t = []
        if isinstance(left, pandas.Series) and isinstance(right, pandas.Series):
            name_l = mapping.get_name(left)
            name_r = mapping.get_name(right)

            select_block = f"(l.{left.name} {operator} r.{right.name}) AS {rename}, "
            select_addition = f"l.{', l.'.join(set(series_to_col_map[name_l]) - set(series_to_col_map[name_r]))}"
            if select_addition != "l.":  # only add if non empty
                select_block = select_block + select_addition
            select_block = select_block + f"r.{', r.'.join(set(series_to_col_map[name_r]))}"

            from_block = f"{self.create_indexed_table(name_l)} l, " \
                         f"{self.create_indexed_table(name_r)} r"
            where_block = f"\nWHERE l.row_number = r.row_number"
            columns_t = list(set(series_to_col_map[name_l]) | set(series_to_col_map[name_r]))
        elif isinstance(left, pandas.Series):
            name_l = mapping.get_name(left)
            select_block = f"({left.name} {operator} {right}) AS {rename}, {', '.join(series_to_col_map[name_l])}"
            from_block = name_l
            columns_t = [left.name] + series_to_col_map[name_l]
        elif isinstance(right, pandas.Series):
            name_r = mapping.get_name(right)
            select_block = f"({left} {operator} {right.name}) AS {rename}, {', '.join(series_to_col_map[name_r])}"
            from_block = name_r
            columns_t = [right.name] + series_to_col_map[name_r]

        sql_code = f"SELECT {select_block}\n" \
                   f"FROM {from_block}" \
                   f"{where_block}"

        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno)

        # Add mapping to be able to select the tracking columns in sql:
        series_to_col_map[sql_table_name] = self.get_tracking_cols_raw(columns_t)

        mapping.add(sql_table_name, result)
        print(sql_code + "\n")
        self.write_to_pipe_query(sql_code)
        return result

    @staticmethod
    def write_to_table_init(sql_code, file_name=""):
        if file_name == "":
            file_name = "create_table.sql"
        with (ROOT_DIR_TO_SQL / file_name).open(mode="a") as file:
            file.write(sql_code)

    @staticmethod
    def write_to_pipe_query(sql_code, file_name=""):
        if file_name == "":
            file_name = "pipeline.sql"
        with (ROOT_DIR_TO_SQL / file_name).open(mode="a") as file:
            file.write(sql_code + "\n")

    @staticmethod
    def write_to_side_query(sql_code, name):
        if len(name.split(".")) == 1:
            name = name + ".sql"
        with (ROOT_DIR_TO_SQL / name).open(mode="a")as file:
            file.write(sql_code)

    @staticmethod
    def get_tracking_cols(columns, table_name=""):
        addition = ""
        if table_name != "":
            addition = table_name + "."
        columns_track = SQLBackend.get_tracking_cols_raw(columns)
        if columns_track:
            return ", " + addition + f', {addition}'.join(SQLBackend.get_tracking_cols_raw(columns))
        return ""

    @staticmethod
    def get_tracking_cols_raw(columns):
        return [x for x in columns if "ctid" in x]

    @staticmethod
    def __column_ratio_original(table_orig, column_name):
        return f"original_ratio_{column_name} as (\n" \
               f"\tselect i.{column_name}, (count(*) * 1.0 / (select count(*) from {table_orig})) as ratio\n" \
               f"\tfrom {table_orig} i\n" \
               f"\tgroup by i.{column_name}\n" \
               "),"

    @staticmethod
    def __column_ratio_current(table_orig, table_new, column_name):
        """
        Here the query for the new/current ratio of the values inside the passed column is provided.
        """
        return f"current_ratio_{column_name} AS (\n" \
               f"\tSELECT orig.{column_name}, (\n" \
               f"\t\t(SELECT count(*)\n" \
               f"\t\tFROM (\n" \
               f"\t\t\tSELECT lookup.original_label\n" \
               f"\t\t\tFROM lookup_{column_name}_{table_new} lookup, {table_new} curr\n" \
               f"\t\t\tWHERE lookup.current_label = curr.{column_name} and " \
               f"orig.{column_name} = lookup.original_label or" \
               f"(lookup.current_label is NULL and orig.{column_name} is NULL and curr.{column_name} is NULL)) temp\n" \
               f"\t\t) * 1.0 / (select count(*) FROM {table_new})) AS ratio\n" \
               f"\tFROM {table_orig} orig,  {table_new} curr, lookup_{column_name}_{table_new} lookup\n" \
               f"\tWHERE curr.{table_orig}_ctid = orig.{table_orig}_ctid\n" \
               f"\tGROUP BY orig.{column_name}\n" \
               f"),"

    @staticmethod
    def __lookup_table(table_orig, table_new, column_name):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        return f"lookup_{column_name}_{table_new} AS (\n" \
               f"\tSELECT distinct orig.{column_name} AS original_label, curr.{column_name} AS current_label\n" \
               f"\tFROM {table_orig} orig,  {table_new} curr\n" \
               f"\tWHERE curr.{table_orig}_ctid = orig.{table_orig}_ctid\n" \
               f"),"

    @staticmethod
    def __overview_table(table_new, column_name):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        return f"overview_{column_name}_{table_new} as (\n" \
               f"\tSELECT n.* , o.ratio AS ratio_original \n" \
               f"\tFROM current_ratio_{column_name} n right JOIN original_ratio_{column_name} o " \
               f"ON o.{column_name} = n.{column_name} or (o.{column_name} is NULL and n.{column_name} is NULL)\n" \
               f"),"

    @staticmethod
    def ratio_track(origin_dict, column_names, current_dict):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        :param origin_dict: Dictionary with all the origin tables of the single attributes.
        :param column_names: The column names of which we want to have the ratio comparison
        :param current_dict: Dictionary that maps the names of the sensitive columns to the current table with the
            new ratio we want to check.
        :return: None -> see stdout
        """
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLBackend.__lookup_table(table_orig, table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLBackend.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLBackend.__column_ratio_original(table_orig, column_name=i) + "\n"
            print(sql_code)
            SQLBackend.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLBackend.__column_ratio_current(table_orig, table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLBackend.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            sql_code = SQLBackend.__overview_table(table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLBackend.write_to_side_query(sql_code, f"ratio_{i}")

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


class DfToStringMapping:
    """
    Simple data structure to track the mappings of pandas.Dataframes to SQL table names.
    """
    mapping = []  # contains tuples of form: (*Name*, *DataFrame*)

    def add(self, name: str, df: pd.DataFrame) -> None:
        """
        Note: As we don't check the variable we are assigning some pandas objects (Series, DataFrame) to, we need to append values at the
        front. This is, because we store the original pandas objects the list and not copies. So if these original objects are altered,
        it can happen that the dataframe is already in the mapping. This would return us some old with table from SQL.
        Which would be wrong!
        """
        self.mapping = [(name, df), *self.mapping]  # Quite efficient way to append values at the front.

    def update_entry(self, old_entry: (str, pd.DataFrame), new_entry: (str, pd.DataFrame)):
        index = self.mapping.index(old_entry)
        self.mapping[index] = new_entry

    def update_name_at_df(self, df, new_name):
        old_name = self.get_name(df)
        index = self.mapping.index((old_name, df))
        self.mapping[index] = (new_name, df)

    def get_df(self, name_to_find: str) -> pd.DataFrame:
        return next(df for (n, df) in self.mapping if n == name_to_find)

    def get_name(self, df_to_find: pd.DataFrame) -> str:
        return next(n for (n, df) in self.mapping if df is df_to_find)

    def contains(self, df_to_find):
        for m in self.mapping:
            if m[1] is df_to_find:
                return True
        return False


# This mapping allows to keep track of the pandas.DataFrame and pandas.Series w.r.t. their SQL-table representation!
mapping = DfToStringMapping()  # TODO: substitute by: from typing import Dict

# This mapping is needed to be able to handle the tracking columns when working on pandas.Series
series_to_col_map = {}

ROOT_DIR = get_project_root()
ROOT_DIR_TO_SQL = ROOT_DIR / "to_sql_output"

# Empty the "to_sql_output" folder if necessary:
[f.unlink() for f in ROOT_DIR_TO_SQL.glob("*") if f.is_file()]
