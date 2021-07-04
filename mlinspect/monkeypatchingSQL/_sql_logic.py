import pandas
from mlinspect.utils import get_project_root
from mlinspect.inspections._inspection_input import OperatorType
from mlinspect.to_sql.py_to_sql_mapping import TableInfo, DfToStringMapping

ROOT_DIR = get_project_root()
ROOT_DIR_TO_SQL = ROOT_DIR / "mlinspect" / "to_sql" / "generated_code"

# Empty the "to_sql_output" folder if necessary:
[f.unlink() for f in ROOT_DIR_TO_SQL.glob("*") if f.is_file()]

# This mapping allows to keep track of the pandas.DataFrame and pandas.Series w.r.t. their SQL-table representation!
mapping = DfToStringMapping()


class SQLLogic:
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
            name_l, ti_l = mapping.get_n_ti(left)
            name_r, ti_r = mapping.get_n_ti(right)

            select_block = f"(l.{left.name} {operator} r.{right.name}) AS {rename}, "
            select_addition = f"l.{', l.'.join(set(ti_l.tracking_cols) - set(ti_r.tracking_cols))}"
            if select_addition != "l.":  # only add if non empty
                select_block = select_block + select_addition
            select_block = select_block + f"r.{', r.'.join(set(ti_r.tracking_cols))}"

            from_block = f"{self.create_indexed_table(name_l)} l, " \
                         f"{self.create_indexed_table(name_r)} r"
            where_block = f"\nWHERE l.row_number = r.row_number"
            columns_t = list(set(ti_l.tracking_cols) | set(ti_r.tracking_cols))
        elif isinstance(left, pandas.Series):
            name_l, ti_l = mapping.get_n_ti(left)
            select_block = f"({left.name} {operator} {right}) AS {rename}, {', '.join(ti_l.tracking_cols)}"
            from_block = name_l
            columns_t = [left.name] + ti_l.tracking_cols
        elif isinstance(right, pandas.Series):
            name_r, ti_r = mapping.get_n_ti(right)
            select_block = f"({left} {operator} {right.name}) AS {rename}, {', '.join(ti_r.tracking_cols)}"
            from_block = name_r
            columns_t = [right.name] + ti_r.tracking_cols

        sql_code = f"SELECT {select_block}\n" \
                   f"FROM {from_block}" \
                   f"{where_block}"

        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno)

        mapping_result = TableInfo(data_object=result,
                                   tracking_cols=self.get_tracking_cols_raw(columns_t),
                                   operation_type=OperatorType.SELECTION,
                                   main_op=True,
                                   optional_context=[])

        mapping.add(sql_table_name, mapping_result)
        print(sql_code + "\n")
        self.write_to_pipe_query(sql_code)
        return result

    def finish_sql_call(self, sql_code, lineno, result, tracking_cols, operation_type, main_op, optional_context=[],
                        with_block_name=""):
        sql_table_name, sql_code = self.wrap_in_with(sql_code, lineno, with_block_name=with_block_name)
        mapping_result = TableInfo(data_object=result,
                                   tracking_cols=tracking_cols,
                                   operation_type=operation_type,
                                   main_op=main_op,
                                   optional_context=optional_context)
        mapping.add(sql_table_name, mapping_result)
        print(sql_code + "\n")

    @staticmethod
    def write_to_init_file(sql_code, file_name=""):
        if file_name == "":
            file_name = "create_table.sql"
        with (ROOT_DIR_TO_SQL / file_name).open(mode="a", ) as file:
            file.write(sql_code + "\n\n")

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
            file.write(sql_code + "\n")

    @staticmethod
    def get_tracking_cols(columns, table_name=""):
        addition = ""
        if table_name != "":
            addition = table_name + "."
        columns_track = SQLLogic.get_tracking_cols_raw(columns)
        if columns_track:
            return ", " + addition + f', {addition}'.join(SQLLogic.get_tracking_cols_raw(columns))
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

        Note:
        supports column renaming -> in case the dict contains one.
        """
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLLogic.__lookup_table(table_orig, table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLLogic.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLLogic.__column_ratio_original(table_orig, column_name=i) + "\n"
            print(sql_code)
            SQLLogic.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            table_orig = origin_dict[i]
            sql_code = SQLLogic.__column_ratio_current(table_orig, table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLLogic.write_to_side_query(sql_code, f"ratio_{i}")
        for i in column_names:
            sql_code = SQLLogic.__overview_table(table_new=current_dict[i], column_name=i) + "\n"
            print(sql_code)
            SQLLogic.write_to_side_query(sql_code, f"ratio_{i}")

    @staticmethod
    def create_indexed_table(table_name):
        return f"(SELECT *, ROW_NUMBER() OVER(ORDER BY NULL) FROM {table_name})"
