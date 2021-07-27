import pandas
from mlinspect.inspections._inspection_input import OperatorType
from mlinspect.to_sql.py_to_sql_mapping import TableInfo, OpTree, sql_obj_prefix
from mlinspect.to_sql._mode import SQLObjRep
from mlinspect.monkeypatching._patch_numpy import MlinspectNdarray


class SQLLogic:
    first_with = True

    def __init__(self, mapping, pipeline_container, dbms_connector, sql_obj, id=1):
        self.mapping = mapping
        self.pipeline_container = pipeline_container
        self.sql_obj = sql_obj
        self.dbms_connector = dbms_connector
        self.id = id

    def wrap_in_sql_obj(self, sql_code, position_id=-1, block_name=""):
        """
        Wraps the passed sql code in a WITH... AS or VIEW block. Takes into account, that WITH only needs to be
        used once.
        """
        if block_name == "":
            block_name = f"{sql_obj_prefix}_mlinid{position_id}_{self.get_unique_id()}"
        sql_code = sql_code.replace('\n', '\n\t')  # for nice formatting
        if self.sql_obj.mode == SQLObjRep.CTE:
            sql_code = f"{block_name} AS (\n\t{sql_code}\n)"
            if self.first_with:
                sql_code = "WITH " + sql_code
                self.first_with = False
        elif self.sql_obj.mode == SQLObjRep.VIEW:
            sql_code = f"CREATE VIEW {block_name} AS (\n\t{sql_code}\n);\n"

        return block_name, sql_code

    def get_unique_id(self):
        self.id += 1
        return self.id - 1

    def handle_operation_series(self, operator, result, left, right, line_id):
        """
        Args:
        """
        # a rename gets necessary, as otherwise in binary ops the name will be "None"
        rename = f"op_{self.get_unique_id()}"
        result.name = rename  # don't forget to set pandas object name!
        where_block = ""

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
            if isinstance(right, str):
                right = f"\'{right}\'"
            origin_context = OpTree(op=operator, left=ti_l.origin_context, right=OpTree(op=str(right), is_const=True))
            tables, column, tracking_columns = self.get_origin_series(origin_context)
            select_block = f"{column} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]
            if len(tables) > 1:
                raise NotImplementedError  # TODO Row-wise

        else:
            assert (isinstance(right, pandas.Series))
            if isinstance(left, str):
                left = f"\'{left}\'"
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

        cte_name, sql_code = self.finish_sql_call(sql_code, line_id, result,
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

        new_content = f"({content_l} {op} {content_r})"
        return new_table, new_content, new_tracking_columns

    def finish_sql_call(self, sql_code, line_id, result, tracking_cols, non_tracking_cols_addition, operation_type,
                        origin_context=None, cte_name=""):
        final_cte_name, sql_code = self.wrap_in_sql_obj(sql_code, line_id, block_name=cte_name)
        if isinstance(result, pandas.Series):
            non_tracking_cols = f"\"{result.name}\""
        elif isinstance(result, pandas.DataFrame):
            non_tracking_cols = [f"\"{x}\"" for x in result.columns.values] + non_tracking_cols_addition
        elif isinstance(result, MlinspectNdarray):
            non_tracking_cols = non_tracking_cols_addition
        else:
            raise NotImplementedError
        mapping_result = TableInfo(data_object=result,
                                   tracking_cols=tracking_cols,
                                   non_tracking_cols=non_tracking_cols,
                                   operation_type=operation_type,
                                   origin_context=origin_context)
        self.mapping.add(final_cte_name, mapping_result)

        if self.sql_obj.mode == SQLObjRep.VIEW:
            self.dbms_connector.run(sql_code)  # Create the view.

        # print(sql_code + "\n")
        return final_cte_name, sql_code

    @staticmethod
    def __column_ratio(table, column_name, prefix=""):
        """
        Here the query for the original ratio of the values inside the passed column is provided.
        """
        column_name = column_name.replace("\"", "")
        table_name = f"{prefix}_ratio_{column_name}"
        return table_name, f"{table_name} AS (\n" \
                           f"\tSELECT {column_name}, (COUNT(*) * 1.0 / (SELECT COUNT(*) FROM {table})) AS ratio\n" \
                           f"\tFROM {table} \n" \
                           f"\tGROUP BY {column_name}\n" \
                           "),\n"

    @staticmethod
    def __column_ratio_with_join(table_orig, table_new, column_name, prefix, join_ctid):
        """
        Here the query for the new/current ratio of the values inside the passed column is provided.
        """
        column_name_title = column_name.replace("\"", "")
        table_name = f"{prefix}_ratio_{column_name_title}"
        return table_name, f"{table_name} AS (\n" \
                           f"\tSELECT o.{column_name}, (COUNT(*) * 1.0 / (SELECT count(*) " \
                           f"FROM {table_new})) AS ratio\n" \
                           f"\tFROM {table_new} n " \
                           f"JOIN {table_orig} o " \
                           f"ON n.{join_ctid}=o.{join_ctid}\n" \
                           f"\tGROUP BY o.{column_name}\n" \
                           "),\n"

    @staticmethod
    def __overview_ratio(table_new, column_name, ratio_table_new, ratio_table_old):
        """
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        column_name = column_name.replace("\"", "")
        cte_name = f"overview_{column_name}_{table_new}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT n.{column_name}, n.ratio AS ratio_new, o.ratio AS ratio_original \n" \
                         f"\tFROM {ratio_table_new} n " \
                         f"RIGHT JOIN {ratio_table_old} o " \
                         f"ON o.{column_name} = n.{column_name})"

    @staticmethod
    def __overview_bias(new_ratio, origin_ratio, column_name, threshold, suffix=""):
        """
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        column_name_title = column_name.replace("\"", "")
        cte_name = f"overview_{suffix}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT SUM(CASE WHEN ABS((n.ratio - o.ratio)/o.ratio) " \
                         f"< ABS({threshold}) THEN 1 ELSE 0 END) " \
                         f"= count(*) AS " \
                         f"no_bias_introduced_flag\n" \
                         f"\tFROM {new_ratio} n " \
                         f"RIGHT JOIN {origin_ratio} o " \
                         f"ON o.{column_name} = n.{column_name} OR " \
                         f"(o.{column_name} IS NULL AND n.{column_name} IS NULL)\n),\n"

    @staticmethod
    def ratio_track_original_ref(origin_table: str, column_name: str):
        """
        Creates the query for the original table.

        Args:
            origin_table:
            column_name:
        Return:
            (<sql_code>, <new_object_name>) the generated code, as well as
        """

        return SQLLogic.__column_ratio(origin_table, column_name=column_name, prefix="original")

    @staticmethod
    def ratio_track_curr(origin_sql_obj: str, current_table: str, column_name: str, threshold: float,
                         origin_ratio_table: str, join_ctid: str = None):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        Args:
            origin_sql_obj(str): original table reference name of the data source
            origin_ratio_table(str): original ration table (returned by SQLLogic.ratio_track_original_ref)
            current_table:
            column_name:
            join_ctid: If a join needs to be performed to make the comparison.
            threshold: Threshold for which the bias is considered not a problem.
        Return:
            (<sql_code>, <new_object_name>) the generated code, as well as
        Note:
        supports column renaming -> in case the dict contains one.
        """
        sql_code = ""
        prefix = "current_" + current_table
        if join_ctid:

            current_table_ratio, sql_code_addition = SQLLogic.__column_ratio_with_join(origin_sql_obj,
                                                                                       table_new=current_table,
                                                                                       column_name=column_name,
                                                                                       prefix=prefix,
                                                                                       join_ctid=join_ctid)

            sql_code += sql_code_addition
        else:
            current_table_ratio, sql_code_addition = SQLLogic.__column_ratio(current_table, column_name=column_name,
                                                                             prefix=prefix)
            sql_code += sql_code_addition

        overview_suffix = column_name.replace('\"', '') + "_" + current_table
        cte_name, sql_code_addition = SQLLogic.__overview_bias(new_ratio=current_table_ratio,
                                                               origin_ratio=origin_ratio_table,
                                                               column_name=column_name,
                                                               threshold=threshold, suffix=overview_suffix)

        return cte_name, sql_code + sql_code_addition

    @staticmethod
    def ratio_track_final_selection(sql_ratio_obj_names: list):
        """
        Despite the sql code result looking scary, each table only contains EXACTLY 1 value.
        """
        sql_code = "\nSELECT "
        from_block = ""
        for n in sql_ratio_obj_names:
            sql_code += f"{n}.no_bias_introduced_flag, "
            from_block += f"{n}, "
        sql_code = sql_code[:-2]
        from_block = from_block[:-2]
        sql_code += f"\nFROM {from_block};\n"
        return sql_code

    @staticmethod
    def create_indexed_table(table_name):
        return f"(SELECT *, ROW_NUMBER() OVER() FROM {table_name})"

    # ################################################ SKLEARN #########################################################

    def column_count(self, table, column_name):
        table_name = f"{table}_{self.get_unique_id()}_count"
        sql_code = f"SELECT {column_name}, COUNT(*) AS count\n" \
                   f"FROM {table} \n" \
                   f"GROUP BY {column_name}\n"
        return self.wrap_in_sql_obj(sql_code, block_name=table_name)

    def column_one_hot_encoding(self, table, col):
        table_name = f"{table}_{self.get_unique_id()}_onehot"
        sql_code = f"select {col}, \n" \
                   f"(array_fill(0, ARRAY[\"rank\" - 1]) || 1 ) || " \
                   f"array_fill(0, ARRAY[ cast((select count(distinct({col})) from {table}) as int) - " \
                   f"(\"rank\")]) as {col[:-1]}_one_hot\" \n" \
                   f"\tfrom (\n" \
                   f"\tselect {col}, CAST(ROW_NUMBER() OVER() AS int) AS \"rank\" \n" \
                   f"\tfrom (select distinct({col}) from {table}) oh\n" \
                   f") one_hot_help"
        return self.wrap_in_sql_obj(sql_code, block_name=table_name)
