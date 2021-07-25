from inspect import cleandoc


# ########################################### FOR THE OP BENCHMARK #####################################################
class Join:
    """
    Code for simple join.
    """

    @staticmethod
    def get_name():
        return "Join"

    @staticmethod
    def get_pandas_code(path_1, path_2, join_attr, na_val="?"):
        pandas_code = cleandoc(f"""
        df1 = pandas.read_csv(r\"{path_1}\", na_values=[\"{na_val}\"])
        df2 = pandas.read_csv(r\"{path_2}\", na_values=[\"{na_val}\"])
        result = df1.merge(df2, on=['{join_attr}'])
        """)
        return pandas_code

    @staticmethod
    def get_sql_code(ds_1, ds_2, join_attr):
        sql_code = cleandoc(f"""
        SELECT * 
        FROM {ds_1}
        INNER JOIN {ds_2} ON {ds_1}.{join_attr} = {ds_2}.{join_attr}
        """)
        return sql_code


class Projection:
    """
    Won't be used!!
    """

    @staticmethod
    def get_name():
        return "Projection"

    @staticmethod
    def get_pandas_code(path, attr, na_val="?"):
        pandas_code = cleandoc(f"""
        df = pandas.read_csv(r\"{path}\", na_values=[\"{na_val}\"])
        result = df[[\"{attr}\"]]
        """)
        return pandas_code

    @staticmethod
    def get_sql_code(ds, attr):
        sql_code = cleandoc(f"""
        SELECT count({attr})
        FROM {ds}
        """)
        return sql_code


class Selection:
    """
    Code for selection benchmark, with aggregation.
    """

    @staticmethod
    def get_name():
        return "Selection"

    @staticmethod
    def get_pandas_code(path, attr, cond="", value="", na_val="?"):
        pandas_code = cleandoc(f"""
        df = pandas.read_csv(r\"{path}\", na_values=[\"{na_val}\"])
        result = df.loc[lambda df: (df['{attr}'] {cond} {value}), :]
        """)
        return pandas_code

    @staticmethod
    def get_sql_code(ds, attr, cond="", value=""):
        sql_code = cleandoc(f"""
        SELECT *
        FROM {ds}
        WHERE {attr} {cond} {value}
        """)
        return sql_code


class GroupBy:
    """
    Code for group by benchmark.
    """

    @staticmethod
    def get_name():
        return "GroupBy"

    @staticmethod
    def get_pandas_code(path, attr1, attr2, op, na_val="?"):
        pandas_code = cleandoc(f"""
        df = pandas.read_csv(r\"{path}\", na_values=[\"{na_val}\"])
        complications = df.groupby('{attr1}').agg(mean_complications=('{attr2}', '{op}'))
        """)
        return pandas_code

    @staticmethod
    def get_sql_code(ds, attr1, attr2, op):
        # Here we don't need a count, as smoker is boolean and we will only need to print 3 values: null, True, False
        sql_code = cleandoc(f"""
        SELECT {attr1}, {op}({attr2})
        FROM {ds}
        GROUP BY {attr1}
        """)
        return sql_code


# ######################################################################################################################

# ########################################### FOR THE PURE PIPELINE BENCHMARK ##########################################
def get_healthcare_pipe_code(path_patients, path_histories):
    setup_code = cleandoc("""
        import os
        import pandas as pd
        from mlinspect.utils import get_project_root
        """)

    test_code = cleandoc(f"""
        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv('{path_patients}', na_values='?')
        histories = pd.read_csv('{path_histories}', na_values='?')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group') \
            .agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
        """)

    return setup_code, test_code

# ######################################################################################################################
