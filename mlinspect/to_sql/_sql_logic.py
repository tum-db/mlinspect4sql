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
            origin_context = OpTree(op=operator, left=ti_l.origin_context, right=OpTree(op=str(right), is_const=True))
            tables, column, tracking_columns = self.get_origin_series(origin_context)
            select_block = f"{column} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]
            if len(tables) > 1:
                raise NotImplementedError  # TODO Row-wise

        else:
            assert (isinstance(right, pandas.Series))
            name_r, ti_r = self.mapping.get_name_and_ti(right)
            origin_context = OpTree(op=operator, left=OpTree(op=str(left), is_const=True), right=ti_r.origin_context)
            tables, column, tracking_columns = self.get_origin_series(origin_context)
            select_block = f"{column} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]
            if len(tables) > 1:
                raise NotImplementedError  # TODO Row-wise

        sql_code = f"SELECT {select_block}\n" \
                   f"FROM {from_block}" \
                   f"{where_block}"

        cte_name, sql_code = self.finish_sql_call(sql_code, lineno, result,
                                                  tracking_cols=tracking_columns,
                                                  non_tracking_cols_addition=[rename],
                                                  operation_type=OperatorType.BIN_OP,
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
        if origin_context.is_const:
            return [], origin_context.op, []
        op = origin_context.op
        if op == "":
            # We are dealing with a projection:
            table = origin_context.origin_table  # format: [(table1, [col1, ...]), ...]
            columns = origin_context.non_tracking_columns
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

    def finish_sql_call(self, sql_code, lineno, result, tracking_cols, non_tracking_cols_addition, operation_type, origin_context=None,
                        cte_name=""):
        final_cte_name, sql_code = self.wrap_in_with(sql_code, lineno, with_block_name=cte_name)
        if isinstance(result, pandas.Series):
            non_tracking_cols = f"\"{result.name}\""
        else:
            non_tracking_cols = [f"\"{x}\"" for x in result.columns.values] + non_tracking_cols_addition
        mapping_result = TableInfo(data_object=result,
                                   tracking_cols=tracking_cols,
                                   non_tracking_cols=non_tracking_cols,
                                   operation_type=operation_type,
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
        raise NotImplementedError
        return [x for x in columns if "ctid" in x]

    @staticmethod
    def get_non_tracking_cols_raw(columns):
        raise NotImplementedError
        return [x for x in columns if "ctid" not in x]

    @staticmethod
    def __column_ratio(table, column_name, prefix=""):
        return f"{prefix}_ratio_{column_name} AS (\n" \
               f"\tSELECT {column_name}, (count(*) * 1.0 / (select count(*) FROM {table})) AS ratio\n" \
               f"\tFROM {table} \n" \
               f"\tGROUP BY {column_name}\n" \
               "),\n"

    @staticmethod
    def __column_ratio_current(table_orig, table_new, column_name, prefix, ctid_col=None):
        """
        Here the query for the new/current ratio of the values inside the passed column is provided.
        """
        if ctid_col:
            return f"{prefix}_ratio_{column_name} AS (\n" \
                   f"\tSELECT tb_orig.{column_name}, (count(*) * 1.0 / (select count(*) FROM {table_new})) AS ratio\n" \
                   f"\tFROM {table_new} tb_curr " \
                   f"JOIN {table_orig} tb_orig " \
                   f"ON tb_curr.{ctid_col}=tb_orig.{ctid_col}\n" \
                   f"\tGROUP BY tb_orig.{column_name}\n" \
                   "),\n"
        return SQLLogic.__column_ratio(table_new, column_name, prefix)

    @staticmethod
    def __overview_table(table_new, column_name, prefix_original="original", prefix_current="current"):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        cte_name = f"overview_{column_name}_{table_new}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT n.{column_name}, n.ratio AS ratio_new, o.ratio AS ratio_original \n" \
                         f"\tFROM {prefix_current}_ratio_{column_name} n " \
                         f"RIGHT JOIN {prefix_original}_ratio_{column_name} o " \
                         f"ON o.{column_name} = n.{column_name})"

    @staticmethod
    def __no_bias(table_new, column_name, threshold, prefix_original="original", prefix_current="current"):
        """
        Creates the lookup_table to cope with possible projections.
        (Attention: Does not respect renaming of columns - as mlinspect doesn't)
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        cte_name = f"overview_{column_name}_{table_new}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT SUM(CASE WHEN ABS(n.ratio - o.ratio) < ABS({threshold}) THEN 1 ELSE 0 END) " \
                         f"= count(*) AS " \
                         f"no_bias_introduced_flag\n" \
                         f"\tFROM {prefix_current}_ratio_{column_name} n " \
                         f"RIGHT JOIN {prefix_original}_ratio_{column_name} o " \
                         f"ON o.{column_name} = n.{column_name}\n),\n"

    @staticmethod
    def ratio_track(origin_dict, column_names, current_dict, join_dict, threshold, only_passed=True):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        Args:
            origin_dict: Dictionary with all the origin tables of the single attributes.
            column_names: The column names of which we want to have the ratio comparison
            current_dict: Dictionary that maps the names of the sensitive columns to the current table with the
                new ratio we want to check.
            join_dict: Dict for the columns not present in the corresponding table, for which we will need to join.
            threshold: Threshold for which the bias is considered not a problem.
        Return:
             None -> see stdout

        Note:
        supports column renaming -> in case the dict contains one.
        """
        sql_code = ""
        last_cte_names = []

        for i in column_names:
            ctid_col = None
            table_orig = origin_dict[i]
            if i in current_dict.keys():
                table_curr = current_dict[i]
            else:
                table_curr, ctid_col = join_dict[i]
            sql_code += SQLLogic.__column_ratio(table_orig, column_name=i, prefix="original")
            sql_code += SQLLogic.__column_ratio_current(table_orig, table_new=table_curr, column_name=i,
                                                        prefix="current", ctid_col=ctid_col)
            cte_name, sql_code_addition = SQLLogic.__no_bias(table_new=table_curr, column_name=i, threshold=threshold)

            sql_code += sql_code_addition
            last_cte_names.append(cte_name)

        sql_code = sql_code[:-2] # remove the last comma!
        # # Write the code for each column of interest to the corresponding file:
        # for i in column_names:
        #     self.pipeline_container.write_to_side_query(last_cte_names[i], sql_code[i], f"ratio_{i}")

        if only_passed:
            sql_code += "\nSELECT "
            from_block = ""
            for n in last_cte_names:
                sql_code += f"{n}.no_bias_introduced_flag, "
                from_block += f"{n}, "
            sql_code = sql_code[:-2]
            from_block = from_block[:-2]
            sql_code += f"\nFROM {from_block};"
        else:
            raise NotImplementedError

        return sql_code

    @staticmethod
    def create_indexed_table(table_name):
        return f"(SELECT *, ROW_NUMBER() OVER() FROM {table_name})"
