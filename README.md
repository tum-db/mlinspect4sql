mlinspect-SQL
================================
This is an SQL extension to the [mlinspect framework](https://github.com/stefan-grafberger/mlinspect) to transpile Python library functions to SQL for execution within a database system.

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
    

## How to use the SQL backend
We prepared two examples, the [first](notebooks/example_to_sql/to_sql_demo_pure_pipeline.ipynb) is to demonstrate execution of machine learning pipelines only, the [second](example_to_sql/to_sql_demo_inspection.ipynb) demonstrate a full end-to-end machine learning pipeline that compares the performance of different backends.

In order to run the latter one, you need a PostgreSQL database system running (at port 5432) in the background with an user `luca` with password `password` that is allowed to copy from CSV files and has access to the respective database.

	create user luca;
	alter role luca with password 'password';
	grant pg_read_server_files to luca;
	create database healthcare_benchmark;
	grant all privileges on database healthcare_benchmark to luca;

To also run the benchmarks in Umbra, you need an Umbra server running at port 5433.

For more information on the functions supported w.r.t execution outsourced to DBMS, please see [here](mlinspect/monkeypatchingSQL/README.md).

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
