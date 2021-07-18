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
                                              only_passed=True):  # TODO: only_passed also allow to return the query result!!
        # TO_SQL: ###############################################################################################
        # print(("#" * 10) + f"NoBiasIntroducedFor ({', '.join(sensitive_columns)}):" + ("#" * 10) +
        #       "\n -> Files can be found under mlinspect/to_sql/generated_code\n\n")

        sensitive_columns = [f"\"{x}\"" for x in sensitive_columns]
        sql_code_final = ""
        for sc in sensitive_columns:
            """
            For each sc we need to find the most "recent" SQL object(CTE or VIEW), that contains either the column, or
            the ctid that allows to measure the column ratio!  
            """
            origin_dict = {}
            current_dict = {}
            to_join_dict = {}

            for (name, ti) in self.mapping.mapping:

                if sc in ti.non_tracking_cols:  # check if part of "normal" columns
                    origin_dict[sc] = self.mapping.get_origin_table(sc, ti.tracking_cols)
                    current_dict[sc] = name
                    sql_code_final += SQLLogic.ratio_track(origin_dict, [sc], current_dict, to_join_dict,
                                                           threshold, only_passed=only_passed)
                else:
                    # check if a possible ratio over the ctid could be calculated:
                    _, optional_ctid = self.mapping.get_ctid_of_col(sc, ti.tracking_cols)
                    if bool(optional_ctid) and optional_ctid in ti.tracking_cols:
                        to_join_dict[sc] = (name, optional_ctid)
                        sql_code_final += SQLLogic.ratio_track(origin_dict, [sc], current_dict,
                                                               to_join_dict,
                                                               threshold, only_passed=only_passed)

        return sql_code_final

        # TO_SQL DONE! ##########################################################################################
