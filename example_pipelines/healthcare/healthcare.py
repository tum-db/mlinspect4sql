
"""Predicting which patients are at a higher risk of complications"""
import warnings
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, MyKerasClassifier, \
    create_model
from mlinspect.utils import get_project_root, store_timestamp
import time

# FutureWarning: Sklearn 0.24 made a change that breaks remainder='drop', that change will be fixed
#  in an upcoming version: https://github.com/scikit-learn/scikit-learn/pull/19263
warnings.filterwarnings('ignore')

COUNTIES_OF_INTEREST = ['county2', 'county3']
t0 = time.time()
patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                    "patients.csv"), na_values='')
histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                     "histories.csv"), na_values='')
store_timestamp("patients = pd.read_csv .. & histories = pd.read_csv ..", time.time()-t0)

# t0 = time.time()
# patients = pd.read_csv(r"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_patients_generated_100000.csv",
#     na_values='?')
# histories = pd.read_csv(r"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_histories_generated_100000.csv",
#     na_values='?')
# store_timestamp("patients = pd.read_csv .. & histories = pd.read_csv ..", time.time()-t0)


t0 = time.time()
data = patients.merge(histories, on=['ssn'])
store_timestamp("data = patients.merge .. ", time.time()-t0)

t0 = time.time()
complications = data.groupby('age_group') \
    .agg(mean_complications=('complications', 'mean'))
store_timestamp("complications = data.groupby .. ", time.time()-t0)

t0 = time.time()
data = data.merge(complications, on=['age_group'])
store_timestamp("data = data.merge(complications .. ", time.time()-t0)

t0 = time.time()
data['label'] = data['complications'] > 1.2 * data['mean_complications']
store_timestamp("data['label'] = data['complications'] > 1.2 * data['mean_complications']", time.time()-t0)

t0 = time.time()
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']][data['county'].isin(COUNTIES_OF_INTEREST)]
store_timestamp("data = data[['smoker', 'las .. ", time.time()-t0)

impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    # ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')
neural_net = MyKerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0)
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net)])

t0 = time.time()
train_data, test_data = train_test_split(data)
store_timestamp("train_test_split .. ", time.time()-t0)

t0 = time.time()
model = pipeline.fit(train_data, train_data['label'])
store_timestamp("train_test_split .. ", time.time()-t0)

t0 = time.time()
print("Mean accuracy: {}".format(model.score(test_data, test_data['label'])))
store_timestamp("train_test_split .. ", time.time()-t0)
