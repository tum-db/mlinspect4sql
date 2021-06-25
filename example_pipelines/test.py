import os
from mlinspect.utils import get_project_root
import pandas as pd
from mlinspect import PipelineInspector, OperatorType
from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from inspect import cleandoc
from example_pipelines.healthcare import custom_monkeypatching
import time


def example_one():
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
    t0 = time.time()
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_FILE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(["age_group", "race"])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .execute(to_sql=True)
    t1 = time.time()

    print(t1 - t0)


def example_compas():
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
    t0 = time.time()
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_FILE_PY) \
        .execute(to_sql=True)
    t1 = time.time()

    print(t1 - t0)


def example_adult_simple():
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")
    t0 = time.time()
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_FILE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(["age_group", "race"])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .execute(to_sql=True)
    t1 = time.time()

    print(t1 - t0)


def example_adult_complex():
    HEALTHCARE_FILE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")
    t0 = time.time()
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_FILE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(["age_group", "race"])) \
        .execute(to_sql=True)
    t1 = time.time()

    print(t1 - t0)


def example_two():
    na_values = "?"
    path_to_patient_csv = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                       "healthcare_patients.csv")
    path_to_history_csv = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                       "healthcare_histories.csv")
    code = cleandoc(f"""   
    import pandas as pd
    import random 
    patients = pd.read_csv(r"{path_to_patient_csv}", na_values="{na_values}")
    histories = pd.read_csv(r"{path_to_history_csv}", na_values="{na_values}")
    histories["blub"] = patients["num_children"] > 2
    """)
    #     data = patients.merge(histories, on=['ssn'])
    inspector_result = PipelineInspector.on_pipeline_from_string(code) \
        .add_check(NoBiasIntroducedFor(["num_children"])) \
        .execute()
    extracted_dag = inspector_result.dag
    inspection_results = inspector_result.inspection_to_annotations
    check_results = inspector_result.check_to_check_results
    print(check_results)


# def test():
#     patients = pd.read_csv(os.path.join(
#         str(get_project_root(**set_code_reference_call(20, 40, 20, 58)), **set_code_reference_call(20, 36, 20, 59)),
#         'example_pipelines', 'healthcare',
#         'patients.csv', **set_code_reference_call(20, 23, 21, 51)),
#         **set_code_reference_call(20, 11, 21, 67, na_values='?'))


if __name__ == "__main__":
    example_one()
    # example_compas()
    # path_to_patient_csv = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
    #                                    "healthcare_patients.csv")
    # test = pd.read_csv(path_to_patient_csv, nrows=10, header=0)
    # path_to_patient_csv = pandas_to_sql.wrap_df(test, "blub")
    # sol = path_to_patient_csv.get_sql_string()
    # print()
