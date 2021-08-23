from inspect import cleandoc
import pathlib
from mlinspect import PipelineInspector
from mlinspect.utils import get_project_root
import os
from mlinspect.utils import store_timestamp
import time

# ########################################### FOR THE SIMPLE OP BENCHMARK ##############################################
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
def get_healthcare_pipe_code(path_patients=None, path_histories=None, only_pandas=False, include_training=True):
    if path_patients is None or path_histories is None:
        path_patients = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv")
        path_histories = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "histories.csv")

    setup_code = cleandoc("""
        import warnings
        import os
        import pandas as pd
        from mlinspect.utils import get_project_root
        """)

    test_code = cleandoc(f"""
        COUNTIES_OF_INTEREST = ['county2', 'county3']

        patients = pd.read_csv('{path_patients}', na_values='')
        histories = pd.read_csv('{path_histories}', na_values='')

        data = patients.merge(histories, on=['ssn'])
        complications = data.groupby('age_group') \
            .agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
        """)
    if not only_pandas:
        setup_code += "\n" + cleandoc("""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, MyKerasClassifier, create_model
        """)

        training_part = cleandoc("""
                    neural_net = MyKerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0)
                    pipeline = Pipeline([
                        ('features', featurisation),
                        ('learner', neural_net)
                    ])
                    train_data, test_data = train_test_split(data)
                    model = pipeline.fit(train_data, train_data['label'])
                    print("Mean accuracy: " + str(model.score(test_data, test_data['label'])) )
                    """)

        test_code += "\n" + cleandoc("""
        impute_and_one_hot_encode = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
            # ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
            ('numeric', StandardScaler(), ['num_children', 'income']),
        ], remainder='drop')
        """) + "\n" + training_part
        if not include_training:
            test_code = test_code.replace(training_part, "result = featurisation.fit_transform(data)")

    return setup_code + "\n", test_code


def get_compas_pipe_code(compas_train=None, compas_test=None, only_pandas=False, include_training=True):
    if compas_train is None or compas_test is None:
        compas_train = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                    "compas", "compas_train.csv")
        compas_test = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                   "compas", "compas_test.csv")

    setup_code = cleandoc("""
        import os
        import pandas as pd
        from mlinspect.utils import get_project_root
        """)

    test_code = cleandoc(f"""
        train_data = pd.read_csv(r\"{compas_train}\", na_values='', index_col=0)
        test_data = pd.read_csv(r\"{compas_test}\", na_values='', index_col=0)
        train_data = train_data[
            ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
             'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        test_data = test_data[
           ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
            'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        
        train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data['days_b_screening_arrest'] >= -30)]
        train_data = train_data[train_data['is_recid'] != -1]
        train_data = train_data[train_data['c_charge_degree'] != "O"]
        train_data = train_data[train_data['score_text'] != 'N/A']
        
        train_data = train_data.replace('Medium', "Low")
        test_data = test_data.replace('Medium', "Low")
        """)

    if not only_pandas:
        setup_code += "\n" + cleandoc("""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
        """)

        training_part = cleandoc("""
        pipeline = Pipeline([
            ('features', featurizer),
            ('classifier', LogisticRegression())
        ])
        pipeline.fit(train_data, train_labels.ravel())
        print(pipeline.score(test_data, test_labels.ravel()))
        """)

        test_code += "\n" + cleandoc("""
        
        train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low'])
        test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low'])
        
        impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])                       
        impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])             
        featurizer = ColumnTransformer(transformers=[
            ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
            ('impute2_and_bin', impute2_and_bin, ['age'])
        ])
        """) + "\n" + training_part

        if not include_training:
            test_code = test_code.replace(training_part, "result = featurizer.fit_transform(train_data)")

    return setup_code + "\n", test_code


def get_compas_pipe_code_with_timestamps(compas_train=None, compas_test=None, only_pandas=False, include_training=True,
                                         engine_name=""):
    engine_name = f"\"{engine_name}\""

    if compas_train is None or compas_test is None:
        compas_train = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                    "compas", "compas_train.csv")
        compas_test = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                                   "compas", "compas_test.csv")

    setup_code = cleandoc("""
        import os
        import pandas as pd
        from mlinspect.utils import get_project_root, store_timestamp
        import time
        """)

    test_code = cleandoc(f"""
        t0 = time.time()
        train_data = pd.read_csv(r\"{compas_train}\", na_values='', index_col=0)
        store_timestamp("LOAD CSV 1", time.time() - t0, {engine_name})
        
        t0 = time.time()
        test_data = pd.read_csv(r\"{compas_test}\", na_values='', index_col=0)
        store_timestamp("LOAD CSV 2", time.time() - t0, {engine_name})
        
        t0 = time.time()
        train_data = train_data[
            ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
             'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        store_timestamp("PROJECT TRAIN", time.time() - t0, {engine_name})
        
        t0 = time.time()
        test_data = test_data[
           ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
            'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        store_timestamp("PROJECT TEST", time.time() - t0, {engine_name})
        
        t0 = time.time()
        train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data['days_b_screening_arrest'] >= -30)]
        store_timestamp("SELECT TRAIN 1", time.time() - t0, {engine_name})
        
        t0 = time.time()
        train_data = train_data[train_data['is_recid'] != -1]
        store_timestamp("SELECT TRAIN 2", time.time() - t0, {engine_name})
                
        t0 = time.time()
        train_data = train_data[train_data['c_charge_degree'] != "O"]
        store_timestamp("SELECT TRAIN 3", time.time() - t0, {engine_name})
                
        t0 = time.time()
        train_data = train_data[train_data['score_text'] != 'N/A']
        store_timestamp("SELECT TRAIN 4", time.time() - t0, {engine_name})
                
        t0 = time.time()

        train_data = train_data.replace('Medium', "Low")
        store_timestamp("REPLACE TRAIN", time.time() - t0, {engine_name})
                
        t0 = time.time()
        test_data = test_data.replace('Medium', "Low")
        store_timestamp("REPLACE TEST", time.time() - t0, {engine_name})
                
        """)

    if not only_pandas:
        setup_code += "\n" + cleandoc("""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
        """)

        training_part = cleandoc(f"""
        t0 = time.time()
        pipeline = Pipeline([
            ('features', featurizer),
            ('classifier', LogisticRegression())
        ])
        pipeline.fit(train_data, train_labels.ravel())
        store_timestamp("PIPELINE+MODEL FIT", time.time() - t0, {engine_name})
        t0 = time.time()
        print(pipeline.score(test_data, test_labels.ravel()))
        store_timestamp("PIPELINE+MODEL SCORE", time.time() - t0, {engine_name})
        """)

        test_code += "\n" + cleandoc(f"""
        
        t0 = time.time()
        train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low'])
        store_timestamp("BINARIZE TRAIN", time.time() - t0, {engine_name})
        
        t0 = time.time()
        test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low'])
        store_timestamp("BINARIZE TEST", time.time() - t0, {engine_name})

        impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])                       
        impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])             
        featurizer = ColumnTransformer(transformers=[
            ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
            ('impute2_and_bin', impute2_and_bin, ['age'])
        ])
        """) + "\n" + training_part

        if not include_training:
            test_code = test_code.replace(training_part, "result = featurizer.fit_transform(train_data)")

    return setup_code + "\n", test_code


def get_adult_simple_pipe_code(train=None, only_pandas=False, include_training=True):
    if train is None:
        train = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                             "adult_complex", "adult_train.csv")

    setup_code = cleandoc("""
        import os
        import pandas as pd
        """)

    test_code = cleandoc(f"""
        train_file = r\"{train}\"
        raw_data = pd.read_csv(train_file, na_values='', index_col=0)
        data = raw_data.dropna()

        """)

    if not only_pandas:
        setup_code += "\n" + cleandoc(""" 
        from sklearn import compose, tree, pipeline
        from sklearn import preprocessing
        """)

        training_part = cleandoc("""
        income_pipeline = pipeline.Pipeline([
            ('features', feature_transformation),
            ('classifier', tree.DecisionTreeClassifier())
        ])
        income_pipeline.fit(data, labels)
        """)

        test_code += "\n" + cleandoc("""
        labels = preprocessing.label_binarize(data['income-per-year'], classes=['>50K', '<=50K'])
                
        feature_transformation = compose.ColumnTransformer(transformers=[
            ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass']),
            ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])
        ])
        """) + "\n" + training_part

        if not include_training:
            test_code = test_code.replace(training_part, "result = feature_transformation.fit_transform(data)")

    return setup_code + "\n", test_code


def get_adult_complex_pipe_code(train=None, test=None, only_pandas=False, include_training=True):
    if train is None or test is None:
        train = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                             "adult_complex", "adult_train.csv")
        test = os.path.join(str(get_project_root()), "test", "monkeypatchingSQL", "pipelines_for_tests",
                            "adult_complex", "adult_test.csv")

    setup_code = cleandoc("""
        import os
        import pandas as pd
        from sklearn import preprocessing
        """)

    test_code = cleandoc(f"""
        train_file =  r\"{train}\"
        train_data = pd.read_csv(train_file, na_values='', index_col=0)
        test_file =  r\"{test}\"
        test_data = pd.read_csv(test_file, na_values='', index_col=0)
        train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])
        test_labels = preprocessing.label_binarize(test_data['income-per-year'], classes=['>50K', '<=50K'])
        """)

    if not only_pandas:
        setup_code += "\n" + cleandoc(""" 
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from mlinspect.utils import get_project_root
        """)

        training_part = cleandoc("""
        nested_income_pipeline = Pipeline([
            ('features', nested_feature_transformation),
            ('classifier', DecisionTreeClassifier())])
        
        nested_income_pipeline.fit(train_data, train_labels)
        
        print(nested_income_pipeline.score(test_data, test_labels))
        """)

        test_code += "\n" + cleandoc("""
        nested_categorical_feature_transformation = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ])
        nested_feature_transformation = ColumnTransformer(transformers=[
            ('categorical', nested_categorical_feature_transformation, ['education', 'workclass']),
            ('numeric', StandardScaler(), ['age', 'hours-per-week'])
        ])
        result = nested_feature_transformation.fit_transform(train_data, train_labels)
        """) + "\n" + training_part

        if not include_training:
            test_code = test_code.replace(training_part, "")

    return setup_code + "\n", test_code


def get_sql_query_for_pipeline(pipeline_code, mode, materialize):
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
