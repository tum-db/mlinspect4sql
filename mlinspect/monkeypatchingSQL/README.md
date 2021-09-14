# Support for different libraries and API functions

* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('pandas.io.parsers', 'read_csv')`      | Data Source | 
| `('pandas.core.frame', '__getitem__')`, arg type: strings | Projection|
| `('pandas.core.frame', '__getitem__')`, arg type: series | Selection |
| `('pandas.core.frame', 'dropna')` | Selection      |
| `('pandas.core.frame', 'replace')` | Projection (Mod)      |
| `('pandas.core.frame', '__setitem__')` | Projection (Mod)      |
| `('pandas.core.frame', 'merge')` | Join      |
| `('pandas.core.frame', 'groupby')` | Nothing (until a following agg call)     |
| `('pandas.core.groupbygeneric', 'agg')` | Groupby/Agg      |

## Sklearn 
* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('sklearn.compose._column_transformer', 'ColumnTransformer')`, column selection      | Projection | 
| `('sklearn.compose._column_transformer', 'ColumnTransformer')`, concatenation      | Concatenation      | 
| `('sklearn.preprocessing._encoders', 'OneHotEncoder')`, arg type: strings | Transformer |
| `('sklearn.preprocessing._data', 'StandardScaler')` | Transformer      |
| `('sklearn.impute._base’, 'SimpleImputer')` | Transformer      |
| `('sklearn.preprocessing._discretization', 'KBinsDiscretizer')` | Transformer      |
| `('sklearn.tree._classes', 'DecisionTreeClassifier')` | Estimator      |
| `('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')` | Estimator      |
| `('sklearn.linear_model._logistic', 'LogisticRegression')` | Estimator      |
| `('sklearn.model_selection._split', 'train_test_split')` | Split (Train/Test)      |
| `('sklearn.preprocessing._label', 'label_binarize')` | Projection (Mod)      |
