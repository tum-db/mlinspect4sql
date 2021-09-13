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

    def wrap_in_sql_obj(self, sql_code, position_id=-1, block_name="", force_cte=False, force_name=False):
        """
        Wraps the passed sql code in a WITH... AS or VIEW block. Takes into account, that WITH only needs to be
        used once.

        Args:
            force_cte(bool): allows to force the use of CTE, but no "WITH" in the beginning will be used.
        """
        if block_name == "":
            block_name = f"{sql_obj_prefix}_mlinid{position_id}_{self.get_unique_id()}"
        else:
            block_name += f'_{self.get_unique_id()}' if not force_name else ""
        sql_code = sql_code.replace('\n', '\n\t')  # for nice formatting
        if self.sql_obj.mode == SQLObjRep.CTE or force_cte:
            sql_code = f"{block_name} AS (\n\t{sql_code}\n)"
            if self.first_with and not force_cte:
                sql_code = "WITH " + sql_code
                self.first_with = False
        elif self.sql_obj.mode == SQLObjRep.VIEW:
            sql_code = f"CREATE VIEW {block_name} AS (\n\t{sql_code}\n);\n"
        else:
            raise NotImplementedError

        return block_name, sql_code

    def get_unique_id(self):
        self.id += 1
        return self.id - 1

    def handle_operation_series(self, operator, result, left, right, line_id):
        """
        Handles the nesting of binary operations on single columns.
        """
        # a rename gets necessary, as otherwise in binary ops the name will be "None"
        rename = f"op_{self.get_unique_id()}"
        result.name = rename  # don't forget to set pandas object name!
        where_block = ""

        if isinstance(left, pandas.Series) and isinstance(right, pandas.Series):
            name_l, ti_l = self.mapping.get_name_and_ti(left)
            name_r, ti_r = self.mapping.get_name_and_ti(right)

            origin_context = OpTree(op=operator, children=[ti_l.origin_context, ti_r.origin_context])
            tables, content, tracking_columns = self.resolve_to_origin(origin_context)

            if len(tables) == 1:
                select_block = f"({content}) AS {rename}, {', '.join(tracking_columns)}"
                from_block = tables[0]
            else:
                # row-wise, f.e. like:
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
            origin_context = OpTree(op=operator,
                                    children=[ti_l.origin_context, OpTree("{}", [str(right)], is_const=True)])
            tables, content, tracking_columns = self.resolve_to_origin(origin_context)
            select_block = f"{content} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]

        else:
            assert (isinstance(right, pandas.Series))
            if isinstance(left, str):
                left = f"\'{left}\'"
            name_r, ti_r = self.mapping.get_name_and_ti(right)
            origin_context = OpTree(op=operator,
                                    children=[OpTree("{}", [str(left)], is_const=True), ti_r.origin_context])
            tables, content, tracking_columns = self.resolve_to_origin(origin_context)
            select_block = f"{content} AS {rename}, {', '.join(tracking_columns)}"
            from_block = tables[0]

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

    def resolve_to_origin(self, origin_context):
        """
        Resolves the OPTree behind origin_context to its new select_statement (incl. tracking and name).
        """
        op = origin_context.op
        if origin_context.is_const:
            return [], op.format(*origin_context.non_tracking_columns), []

        if origin_context.is_projection():
            table = origin_context.origin_table  # format: [(table1, [col1, ...]), ...]
            columns = origin_context.non_tracking_columns
            tracking_columns = origin_context.tracking_columns
            if not len(columns) == 1:
                raise NotImplementedError  # Only Series supported as of now.
            column = columns[0]
            return [table], column, tracking_columns

        children_resolved = [self.resolve_to_origin(child) for child in origin_context.children]

        tables = list(set([n for name, _, _ in children_resolved for n in name]))
        contents = [content for _, content, _ in children_resolved]
        tracking_columns = list(set([tc for _, _, track_col in children_resolved for tc in track_col]))

        if len(tables) == 1:
            new_table = tables
            new_tracking_columns = tracking_columns
        else:
            # row-wise
            raise NotImplementedError

        new_content = op.format(*contents)
        return new_table, new_content, new_tracking_columns

    def finish_sql_call(self, sql_code, line_id, result, tracking_cols, operation_type, non_tracking_cols_addition=[],
                        origin_context=None, cte_name="", update_name_in_map=False, non_tracking_cols=None,
                        no_wrap=False, force_name=False):
        """
        Helper that: wraps the code and stores it in the mapping.
        """
        if not no_wrap:
            final_cte_name, sql_code = self.wrap_in_sql_obj(sql_code, line_id, cte_name, force_name=force_name)
        else:
            assert cte_name != ""
            final_cte_name = cte_name
        if not non_tracking_cols:
            if isinstance(result, pandas.Series):
                non_tracking_cols = f"\"{result.name}\""
            elif isinstance(result, pandas.DataFrame):
                non_tracking_cols = [f"\"{x}\"" for x in result.columns.values] + non_tracking_cols_addition
            # elif isinstance(result, MlinspectNdarray):
            #     non_tracking_cols = non_tracking_cols_addition
            else:
                raise NotImplementedError

        new_ti_result = TableInfo(data_object=result,
                                  tracking_cols=tracking_cols,
                                  non_tracking_cols=non_tracking_cols,
                                  operation_type=operation_type,
                                  origin_context=origin_context)

        if update_name_in_map:
            self.mapping.update_name_df(result, final_cte_name)
            self.mapping.update_ti_df(result, new_ti_result)
        else:
            self.mapping.add(final_cte_name, new_ti_result)

        if self.sql_obj.mode == SQLObjRep.VIEW:
            self.dbms_connector.run(sql_code)  # Create the view.

        return final_cte_name, sql_code

    @staticmethod
    def __column_ratio(table, column, ctid, prefix=""):
        """
        Here the query for the original ratio of the values inside the passed column is provided.
        """
        clean_column_name = column.replace('\"', '')
        table_name = f"{prefix}_ratio_{clean_column_name}"
        return table_name, f"{table_name} AS (\n" \
                           f"\tSELECT {ctid}, (COUNT(*) * 1.0 / (SELECT COUNT(*) FROM {table})) AS ratio\n" \
                           f"\tFROM {table} \n" \
                           f"\tGROUP BY {ctid}\n" \
                           "),\n"

    @staticmethod
    def __column_ratio_with_join(table_orig, table_new, column, ctid, prefix, join_ctid):
        """
        Here the query for the new/current ratio of the values inside the passed column is provided.
        """
        clean_column_name = column.replace('\"', '')
        table_name = f"{prefix}_ratio_{clean_column_name}"
        return table_name, f"{table_name} AS (\n" \
                           f"\tSELECT o.{ctid}, (COUNT(*) * 1.0 / (SELECT count(*) " \
                           f"FROM {table_new})) AS ratio\n" \
                           f"\tFROM {table_new} n " \
                           f"JOIN {table_orig} o " \
                           f"ON n.{join_ctid}=o.{join_ctid}\n" \
                           f"\tGROUP BY o.{ctid}\n" \
                           "),\n"

    @staticmethod
    def __overview_ratio(table_new, column, ctid, ratio_table_new, ratio_table_old):
        """
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        clean_column_name = column.replace('\"', '')
        cte_name = f"overview_{clean_column_name}_{table_new}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT n.{ctid}, n.ratio AS ratio_new, o.ratio AS ratio_original \n" \
                         f"\tFROM {ratio_table_new} n " \
                         f"RIGHT JOIN {ratio_table_old} o " \
                         f"ON o.{ctid} = n.{ctid})"

    @staticmethod
    def __overview_bias(new_ratio, origin_ratio, ctid, threshold, suffix):
        """
        Note: Naming convention: the ctid of the original table that gets tracked is called '*original_table_name*_ctid'
        """
        cte_name = f"overview_{suffix}"
        return cte_name, f"{cte_name} AS (\n" \
                         f"\tSELECT SUM(CASE WHEN ABS((n.ratio - o.ratio)/o.ratio) " \
                         f"< ABS({threshold}) THEN 1 ELSE 0 END) " \
                         f"= count(*) AS " \
                         f"no_bias_introduced_flag\n" \
                         f"\tFROM {new_ratio} n " \
                         f"RIGHT JOIN {origin_ratio} o " \
                         f"ON o.{ctid} = n.{ctid}\n),\n"
        # f"OR " \
        # f"(o.{ctid} IS NULL AND n.{ctid} IS NULL)\n),\n"

    @staticmethod
    def ratio_track_original_ref(origin_table: str, column: str, ctid: str):
        """
        Creates the query for the original table.

        Args:
            origin_table:
            column_name:
        Return:
            (<sql_code>, <new_object_name>) the generated code, as well as
        """

        return SQLLogic.__column_ratio(origin_table, column, ctid=ctid, prefix="original")

    @staticmethod
    def ratio_track_curr(origin_sql_obj: str, current_table: str, column: str, ctid: str, threshold: float,
                         origin_ratio_table: str, join_ctid: str = None):
        """
        Creates the full query for the overview of the change in ratio of a certain attribute.

        Args:
            origin_sql_obj(str): original table reference name of the data source
            origin_ratio_table(str): original ration table (returned by SQLLogic.ratio_track_original_ref)
            current_table:
            ctid:
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
                                                                                       column=column,
                                                                                       ctid=ctid,
                                                                                       prefix=prefix,
                                                                                       join_ctid=join_ctid)

            sql_code += sql_code_addition
        else:
            current_table_ratio, sql_code_addition = SQLLogic.__column_ratio(current_table, ctid=ctid, column=column,
                                                                             prefix=prefix)
            sql_code += sql_code_addition

        overview_suffix = column.replace('\"', '') + "_" + current_table
        cte_name, sql_code_addition = SQLLogic.__overview_bias(new_ratio=current_table_ratio,
                                                               origin_ratio=origin_ratio_table,
                                                               ctid=ctid,
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

    def column_max_count(self, table, column_name):
        table_name = f"block_impute_fit_{self.get_unique_id()}_most_frequent"
        sql_code = f"WITH counts_help AS (\n" \
                   f"\tSELECT {column_name}, COUNT(*) AS count\n" \
                   f"\tFROM {table} \n" \
                   f"\tGROUP BY {column_name}\n" \
                   f")\n" \
                   f"SELECT {column_name} AS most_frequent \n" \
                   f"FROM counts_help\n" \
                   f"WHERE counts_help.count = (SELECT MAX(count) FROM counts_help)\n" \
                   f"LIMIT 1"
        block_name, sql_code = self.wrap_in_sql_obj(sql_code, block_name=table_name)
        return self.materialize_if_possible(block_name, sql_code)

    def column_mean(self, table, column_name):
        table_name = f"block_impute_fit_{self.get_unique_id()}_mean"
        sql_code = f"SELECT (SELECT AVG({column_name}) FROM {table}) AS {column_name}"
        block_name, sql_code = self.wrap_in_sql_obj(sql_code, block_name=table_name)
        return self.materialize_if_possible(block_name, sql_code)

    def column_one_hot_encoding(self, table, col):
        table_name = f"block_one_hot_fit_{self.get_unique_id()}"
        sql_code = f"SELECT {col}, \n" \
                   f"(array_fill(0, ARRAY[\"rank\" - 1]) || 1 ) || " \
                   f"array_fill(0, ARRAY[ CAST((select COUNT(distinct({col})) FROM {table}) AS int) - " \
                   f"(\"rank\")]) AS {col[:-1]}_one_hot\" \n" \
                   f"\tFROM (\n" \
                   f"\tSELECT {col}, CAST(ROW_NUMBER() OVER() AS int) AS \"rank\" \n" \
                   f"\tFROM (SELECT distinct({col}) FROM {table}) oh\n" \
                   f") one_hot_help"
        block_name, sql_code = self.wrap_in_sql_obj(sql_code, block_name=table_name)
        return self.materialize_if_possible(block_name, sql_code)

    def step_size_kbin(self, table, column_name, n_bins):
        """
        Helper for transpiling the KBinsDiscretizer.
        """
        table_name = f"block_kbin_fit_{self.get_unique_id()}_step_size"
        sql_code = f"SELECT (" \
                   f"(SELECT MAX({column_name}) FROM {table}) - (SELECT MIN({column_name}) FROM {table})) / {n_bins} " \
                   f"AS step"
        block_name, sql_code = self.wrap_in_sql_obj(sql_code, block_name=table_name)
        return self.materialize_if_possible(block_name, sql_code)

    def std_scalar_values(self, table, column_name):
        """
        Helper for transpiling the KBinsDiscretizer.
        """
        table_name = f"block_std_scalar_fit_{self.get_unique_id()}_std_avg"
        sql_code = f"SELECT " \
                   f"(SELECT AVG({column_name}) FROM {table}) AS avg_col_std_scal," \
                   f"(SELECT STDDEV_POP({column_name}) FROM {table}) AS std_dev_col_std_scal"
        block_name, sql_code = self.wrap_in_sql_obj(sql_code, block_name=table_name)
        return self.materialize_if_possible(block_name, sql_code)

    def materialize_if_possible(self, block_name, sql_code):
        if self.sql_obj.materialize:
            new_block_name = block_name + "_materialized"
            return new_block_name, sql_code.replace(f"CREATE VIEW {block_name}",
                                                    f"CREATE MATERIALIZED VIEW {new_block_name}")
        return block_name, sql_code
