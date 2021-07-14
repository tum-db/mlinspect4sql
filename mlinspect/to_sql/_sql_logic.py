import pandas
from mlinspect.utils import get_project_root
from mlinspect.inspections._inspection_input import OperatorType
from mlinspect.to_sql.py_to_sql_mapping import TableInfo, DfToStringMapping, OpTree
from mlinspect.to_sql.sql_query_container import SQLQueryContainer


class SQLLogic:
    first_with = True
    id = 1

    def __init__(self, mapping, pipeline_container):
        self.mapping = mapping
        self.pipeline_container = pipeline_container

    def wrap_in_with(self, sql_code, lineno, with_block_name=""):
        """
        Wrappes the passed sql code in a WITH... AS block. Takes into account, that WITH only needs to be used once.
        """
        if with_block_name == "":
            with_block_name = f"with_{lineno}_{self.get_unique_id()}"
        sql_code = sql_code.replace('\n', '\n\t')  # for nice formatting
        sql_code = f"{with_block_name} AS (\n\t{sql_code}\n)"
        if self.first_with:
            sql_code = "WITH " + sql_code
            self.first_with = False
        return with_block_name, sql_code

    def get_unique_id(self):
        self.id += 1
        return self.id - 1

    def handle_operation_series(self, operator, result, left, right, lineno):
        """
        Args:
        """
        # a rename gets necessary, as otherwise in binary ops the name will be "None"
        rename = f"op_{self.get_unique_id()}"
        result.name = rename  # don't forget to set pandas object name!
        where_block = ""
        from_block = ""
        columns_t = []
        origin_context = []
        if isinstance(left, pandas.Series) and isinstance(right, pandas.Series):
            name_l, ti_l = self.mapping.get_name_and_ti(left)
            name_r, ti_r = self.mapping.get_name_and_ti(right)

            origin_context = OpTree(op=operator, left=ti_l.origin_context, right=ti_r.origin_context)
            tables, column, tracking_columns = self.get_origin_series(origin_context)

            if len(tables) == 1:
                select_block = f"({column}) AS {rename}, {', '.join(tracking_columns)}"
                from_block = tables[0]
            else:
                from_block = f"{self.create_indexed_table(name_l)} l, " \
                             f"{self.create_indexed_table(name_r)} r"
                where_block = f"\nWHERE l.row_number = r.row_number"
                select_block = f"(l.{left.name} {operator} r.{right.name}) AS {rename}, "
                select_addition = f"l.{', l.'.join(set(ti_l.tracking_cols) - set(ti_r.tracking_cols))}"
                if select_addition != "l.":  # only add if non empty
                    select_block = select_block + select_addition
                select_block = select_block + f"r.{', r.'.join(set(ti_r.tracking_cols))}"

        elif isinstance(left, pandas.Series):
            name_l, ti_l = self.mapping.get_name_and_ti(left)
            origin_context = OpTree(op=operator, left=ti_l.origin_context, right=right)
            tables, column, tracking_columns = self.get_origin_series(origin_context)
            select_block = f"{column} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]
            if len(tables) > 1:
                raise NotImplementedError  # TODO Row-wise

        else:
            assert (isinstance(right, pandas.Series))
            name_r, ti_r = self.mapping.get_name_and_ti(right)
            origin_context = OpTree(op=operator, left=left, right=ti_r.origin_context)
            tables, column, tracking_columns = self.get_origin_series(origin_context)
            select_block = f"{column} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]
            if len(tables) > 1:
                raise NotImplementedError  # TODO Row-wise

        sql_code = f"SELECT {select_block}\n" \
                   f"FROM {from_block}" \
                   f"{where_block}"

        cte_name, sql_code = self.finish_sql_call(sql_code, lineno, result,
                                                  tracking_cols=self.get_tracking_cols_raw(tracking_columns),
                                                  operation_type=OperatorType.BIN_OP,
                                                  main_op=True,
                                                  origin_context=origin_context)
        self.pipeline_container.add_statement_to_pipe(cte_name, sql_code, [rename])
        return result

    def get_origin_series(self, origin_context):
        """
        Gets the correct origin table and column for series.
        Returns:
            Tuple, where the first element is the table, and the second is the column name.

        Note: Currently only supporting a single origin! -> TODO: has to be solved with row-wise operations see: create_indexed_table.
        """
        if not isinstance(origin_context, OpTree):
            return [], origin_context, []
        op = origin_context.op
        if op == "":
            # We are dealing with a projection:
            table = origin_context.table  # format: [(table1, [col1, ...]), ...]
            columns = origin_context.columns
            tracking_columns = origin_context.tracking_columns
            if not len(columns) == 1:
                raise NotImplementedError  # Only Series supported as of now.
            column = columns[0]
            return [table], column, tracking_columns

        table_r, content_r, tracking_columns_r = self.get_origin_series(origin_context.right)
        table_l, content_l, tracking_columns_l = self.get_origin_series(origin_context.left)

        tables = list(set(table_r + table_l))
        tracking_columns = list(set(tracking_columns_l + tracking_columns_r))
        if len(tables) == 1:
            new_table = tables
            new_tracking_columns = tracking_columns
        else:
            # TODO adding naming etc. + row-wise
            raise NotImplementedError
            # table = "TODO"

        new_content = f"({content_l} {op} {content_r})"
        return new_table, new_content, new_tracking_columns

    def finish_sql_call(self, sql_code, lineno, result, tracking_cols, operation_type, main_op, origin_context=None,
                        cte_name=""):
        final_cte_name, sql_code = self.wrap_in_with(sql_code, lineno, with_block_name=cte_name)
        mapping_result = TableInfo(data_object=result,
                                   tracking_cols=tracking_cols,
                                   operation_type=operation_type,
                                   main_op=main_op,
                                   origin_context=origin_context)
        self.mapping.add(final_cte_name, mapping_result)
        # print(sql_code + "\n")
        return final_cte_name, sql_code

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
    def get_non_tracking_cols_raw(columns):
        return [x for x in columns if "ctid" not in x]

    @staticmethod
    def __column_ratio_original(table_orig, column_name):
        return f"original_ratio_{column_name} AS (\n" \
               f"\tSELECT i.{column_name}, (count(*) * 1.0 / (select count(*) FROM {table_orig})) AS ratio\n" \
               f"\tFROM {table_orig} i\n" \
               f"\tGROUP BY i.{column_name}\n" \
               "),\n"

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
               f"),\n"

    @staticmethod
    def __lookup_table(table_orig, table_new, column_name, ctid_col=None):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        table_new_done = table_new
        if ctid_col:
            table_new_done = f"(SELECT {column_name}, {ctid_col} " \
                         f"FROM {table_new} tb_curr " \
                         f"JOIN {table_orig} tb_orig " \
                         f"ON tb_curr.{ctid_col}=tb_orig.{ctid_col})"

        return f"lookup_{column_name}_{table_new} AS (\n" \
               f"\tSELECT distinct orig.{column_name} AS original_label, curr.{column_name} AS current_label\n" \
               f"\tFROM {table_orig} orig,  {table_new_done} curr\n" \
               f"\tWHERE curr.{table_orig}_ctid = orig.{table_orig}_ctid\n" \
               f"),\n"

    @staticmethod
    def __overview_table(table_new, column_name):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        cte_name = f"overview_{column_name}_{table_new}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT n.* , o.ratio AS ratio_original \n" \
                         f"\tFROM current_ratio_{column_name} n right JOIN original_ratio_{column_name} o " \
                         f"ON o.{column_name} = n.{column_name} or (o.{column_name} is NULL and n.{column_name} " \
                         f"is NULL)\n )"

    @staticmethod
    def ratio_track(origin_dict, column_names, current_dict, join_dict):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        Args:
            origin_dict: Dictionary with all the origin tables of the single attributes.
            column_names: The column names of which we want to have the ratio comparison
            current_dict: Dictionary that maps the names of the sensitive columns to the current table with the
                new ratio we want to check.
            join_dict: Dict for the columns not present in the corresponding table, for which we will need to join.
        Return:
             None -> see stdout

        Note:
        supports column renaming -> in case the dict contains one.
        """
        sql_code = {cn: "" for cn in column_names}
        last_cte_names = {cn: "" for cn in column_names}

        for i in column_names:
            ctid_col = None
            table_orig = origin_dict[i]
            if i in current_dict.keys():
                table_curr = current_dict[i]
            else:
                table_curr, ctid_col = join_dict[i]
            sql_code[i] += SQLLogic.__lookup_table(table_orig, table_new=table_curr, column_name=i, ctid_col=ctid_col)
            sql_code[i] += SQLLogic.__column_ratio_original(table_orig, column_name=i)
            sql_code[i] += SQLLogic.__column_ratio_current(table_orig, table_new=table_curr, column_name=i)
            cte_name, sql_code_addition = SQLLogic.__overview_table(table_new=table_curr, column_name=i)
            last_cte_names[i] = cte_name
            sql_code[i] += sql_code_addition

        # # Write the code for each column of interest to the corresponding file:
        # for i in column_names:
        #     self.pipeline_container.write_to_side_query(last_cte_names[i], sql_code[i], f"ratio_{i}")
        return last_cte_names, sql_code

    @staticmethod
    def create_indexed_table(table_name):
        return f"(SELECT *, ROW_NUMBER() OVER() FROM {table_name})"
