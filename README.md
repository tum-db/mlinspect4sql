mlinspect-SQL
================================

## Run mlinspect locally

Prerequisite: Python 3.8

1. Clone this repository
2. Set up the environment

	`cd mlinspect` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `pip install -e .[dev]` <br>

5. To ensure everything works, you can run the tests (without graphviz, the visualisation test will fail)

    `python setup.py test` <br>
    
## How to use mlinspect
mlinspect makes it easy to analyze your pipeline and automatically check for common issues.
```python
from mlinspect import PipelineInspector
from mlinspect.inspections import MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor

IPYNB_PATH = ...

inspector_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(IPYNB_PATH)\
        .add_required_inspection(MaterializeFirstOutputRows(5))\
        .add_check(NoBiasIntroducedFor(['race']))\
        .execute()

extracted_dag = inspector_result.dag
dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
check_to_check_results = inspector_result.check_to_check_results
```

With execution outsourced to a Database Management System (DBMS):

```python
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect import PipelineInspector
from mlinspect.inspections import MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor

dbms_connector = PostgresqlConnector(...)

IPYNB_PATH = ...

inspector_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(IPYNB_PATH)\
        .add_required_inspection(MaterializeFirstOutputRows(5))\
        .add_check(NoBiasIntroducedFor(['race']))\
        .execute_in_sql(dbms_connector=dbms_connector, mode="VIEW", materialize=True)

extracted_dag = inspector_result.dag
dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
check_to_check_results = inspector_result.check_to_check_results
```

## Detailed Example
We prepared a [demo notebook](demo/feature_overview/feature_overview.ipynb) to showcase mlinspect and its features.

## Supported libraries and API functions
mlinspect already supports a selection of API functions from `pandas` and `scikit-learn`. 
Extending mlinspect to support more and more API functions and libraries will be an ongoing effort.
However, mlinspect won't just crash when it encounters functions it doesn't recognize yet. 
For more information, please see [here](mlinspect/monkeypatching/README.md).

For more information on the functions supported w.r.t execution outsourced to
DBMS, please see [here](mlinspect/monkeypatchingSQL/README.md).

## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))

## Original Publications
* [Stefan Grafberger, Shubha Guha, Julia Stoyanovich, Sebastian Schelter (2021). mlinspect: a Data Distribution Debugger for Machine Learning Pipelines. ACM SIGMOD (demo).](https://stefan-grafberger.com/publications/mlinspect-a-data-distribution-debugger-for-machine-learning-pipelines/mlinspect-demo.pdf)
* [Stefan Grafberger, Julia Stoyanovich, Sebastian Schelter (2020). Lightweight Inspection of Data Preprocessing in Native Machine Learning Pipelines. Conference on Innovative Data Systems Research (CIDR).](https://stefan-grafberger.com/publications/lightweight-inspection-of-data-preprocessing-in-native-machine-learning-pipelines/mlinspect-cidr.pdf)

## BA Thesis:#
* [Thesis](example_to_sql/main.pdf)