from inspect import cleandoc
import pathlib
from mlinspect import PipelineInspector
from mlinspect.utils import get_project_root


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
def get_healthcare_pipe_code(path_patients, path_histories, add_impute_and_onehot=False):
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
    if add_impute_and_onehot:
        setup_code += "\n" + cleandoc(f"""
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
        """)

        test_code += "\n" + cleandoc(f"""
        impute_and_one_hot_encode = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        """)

    return setup_code + "\n", test_code


def get_compas_pipe_code(path_train, path_test):
    setup_code = cleandoc("""
        import os
        import pandas as pd
        """)

    test_code = cleandoc(f"""
        train_file = os.path.join('{path_train}')
        train_data = pd.read_csv(train_file, na_values='', index_col=0)
        
        test_file = os.path.join('{path_test}')
        test_data = pd.read_csv(test_file, na_values='', index_col=0)
        
        train_data = train_data[
            ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
             'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        test_data = test_data[
            ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
             'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        
        train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data[
            'days_b_screening_arrest'] >= -30)]
        train_data = train_data[train_data['is_recid'] != -1]
        train_data = train_data[train_data['c_charge_degree'] != "O"]
        train_data = train_data[train_data['score_text'] != 'N/A']
        
        train_data = train_data.replace('Medium', "Low")
        test_data = test_data.replace('Medium', "Low")
        """)

    return setup_code + "\n", test_code


def get_healthcare_sql_str(pipeline_code, mode, materialize):
    PipelineInspector \
        .on_pipeline_from_string(pipeline_code) \
        .execute_in_sql(dbms_connector=None, mode=mode, materialize=materialize)

    setup_file = \
        pathlib.Path(get_project_root() / r"mlinspect/to_sql/generated_code/create_table.sql")
    test_file = \
        pathlib.Path(get_project_root() / r"mlinspect/to_sql/generated_code/pipeline.sql")

    with setup_file.open("r") as file:
        setup_code = file.read()

    with test_file.open("r") as file:
        test_code = file.read()

    return setup_code, test_code


# ######################################################################################################################

def print_generated_code():
    generated_files = (get_project_root() / r"mlinspect/to_sql/generated_code").glob("*.sql")
    for file in generated_files:
        with file.open() as f:
            print(f.read())
