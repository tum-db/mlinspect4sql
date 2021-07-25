"""
For the SQL run we want to make the inspections be performed on the DBMS, this code offloads the relevant operations
 and collects the resulting Info from our SQL-Queries to substitute the original "NoBiasIntroducedFor" result.
"""
import pandas
from mlinspect.to_sql._sql_logic import SQLLogic
from mlinspect.to_sql.py_to_sql_mapping import DfToStringMapping
from mlinspect.to_sql.sql_query_container import SQLQueryContainer


class SQLNoBiasIntroducedFor:
    def __init__(self, dbms_connector, mapping: DfToStringMapping, pipeline_container: SQLQueryContainer):
        """
        Args:
            dbms_connector:
            mapping:
            pipeline_container:
            one_run(bool): In case this is set, the "NoBiasIntroduced" inspection will happen once and full in SQL.
                So nothing needs to be done here.
        """
        self.current_hist = {}
        self.dbms_connector = dbms_connector
        self.mapping = mapping
        self.pipeline_container = pipeline_container

    def no_bias_introduced_sql_evaluate_total(self, sensitive_columns, threshold,
                                              relevant_sql_objs):  # TODO: only_passed also allow to return the query result!!
        # TO_SQL: ###############################################################################################
        # print(("#" * 10) + f"NoBiasIntroducedFor ({', '.join(sensitive_columns)}):" + ("#" * 10) +
        #       "\n -> Files can be found under mlinspect/to_sql/generated_code\n\n")

        sensitive_columns = [f"\"{x}\"" for x in sensitive_columns]
        sql_code_final = ""
        sql_obj_final = []  # The sql objects (cte, view) containing the bias result.

        origin_to_ratio_mapping = {}  # This mapping tracks the data source tables.

        for sc in sensitive_columns:
            """
            For each sc we need to find the most "recent" SQL object(CTE or VIEW), that contains either the column, or
            the ctid that allows to measure the column ratio!  
            
            Process:
            1) If not existing: crete table with original ratio for each sensitive column.
            2) Create a table comparing it all the relevant current tables. 
            """

            for (name, ti) in relevant_sql_objs:

                origin_table, origin_ctid = self.mapping.get_origin_table(sc, ti.tracking_cols)

                if not (origin_table, sc) in origin_to_ratio_mapping:
                    sql_obj_name, sql_code = SQLLogic.ratio_track_original_ref(origin_table, sc)
                    origin_to_ratio_mapping[(origin_table, sc)] = sql_obj_name
                    sql_code_final += sql_code

                if sc in ti.non_tracking_cols:
                    # check ratio of "normal" columns:

                    sql_obj_name, sql_code = SQLLogic.ratio_track_curr(origin_table, name,
                                                                       sc, threshold=threshold,
                                                                       origin_ratio_table=origin_to_ratio_mapping[
                                                                           (origin_table, sc)])
                    sql_obj_final.append(sql_obj_name)
                    sql_code_final += sql_code
                else:
                    # check if ratio over the ctid could be calculated:

                    if bool(origin_ctid):
                        sql_obj_name, sql_code = SQLLogic.ratio_track_curr(origin_table, name,
                                                                           sc, threshold=threshold,
                                                                           join_ctid=origin_ctid,
                                                                           origin_ratio_table=origin_to_ratio_mapping[
                                                                               (origin_table, sc)])
                        sql_obj_final.append(sql_obj_name)
                        sql_code_final += sql_code

        return sql_code_final[:-2] + "\n" + SQLLogic.ratio_track_final_selection(sql_obj_final)

        # TO_SQL DONE! ##########################################################################################
