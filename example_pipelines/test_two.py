from inspect import cleandoc
from mlinspect.utils import get_project_root
from mlinspect import PipelineInspector
from mlinspect.inspections import RowLineage, MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from inspect import cleandoc
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect.to_sql.dbms_connectors.postgresql_connector import PostgresqlConnector
from mlinspect.to_sql.dbms_connectors.umbra_connector import UmbraConnector
from mlinspect import PipelineInspector
import warnings
import os
import pandas as pd
from mlinspect.utils import get_project_root

def get_healthcare_csv_paths():
    files = []
    for i in [(10 ** i) for i in range(2, 6, 1)]:
        path_to_csv_his = fr"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_histories_generated_{i}.csv"
        path_to_csv_pat = fr"/home/luca/Documents/Bachelorarbeit/BA_code/data_generation/generated_csv/healthcare_patients_generated_{i}.csv"
        files.append((path_to_csv_his, path_to_csv_pat))
    return files

def pipeline_code(path_patients, path_histories):
    setup_code = cleandoc("""
import warnings
import os
import pandas as pd
from mlinspect.utils import get_project_root
            """)

    test_code = f"warnings.filterwarnings('ignore')\n" \
                f"COUNTIES_OF_INTEREST = ['county2', 'county3']\n" \
                f"patients = pd.read_csv(r'{path_patients}', na_values='?')\n" \
                f"histories = pd.read_csv(r'{path_histories}', na_values='?')\n" \
                f"data = patients.merge(histories, on=['ssn'])\n" \
                f"complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))\n" \
                f"data = data.merge(complications, on=['age_group'])\n" \
                f"data['label'] = data['complications'] > 1.2 * data['mean_complications']\n" \
                f"data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]\n" \
                f"data = data[data['county'].isin(COUNTIES_OF_INTEREST)]\n"

    return setup_code + "\n" + test_code

patients = os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "patients.csv")
histories = os.path.join( str(get_project_root()), "example_pipelines", "healthcare", "histories.csv")






umbra_path = r'/home/luca/Documents/Bachelorarbeit/Umbra/umbra-students'
dbms_connector_u = UmbraConnector(dbname='', user='postgres', password=' ', port=5433, host='/tmp/',
    umbra_dir=umbra_path)
dbms_connector_p = PostgresqlConnector(dbname='healthcare_benchmark', user='luca',
    password='password', port=5432, host='localhost')
# pipeline_code = cleandoc(f"""{pipeline_code}""")


for histories, patients in get_healthcare_csv_paths():
    pc = pipeline_code(patients, histories)

    pipeline_inspector = PipelineInspector.on_pipeline_from_string(pc) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_check(NoBiasIntroducedFor(['age_group', 'race'])) \
        .add_check(NoIllegalFeatures()) \
        .add_check(NoMissingEmbeddings()) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5))

    inspector_result = pipeline_inspector.execute_in_sql(dbms_connector=dbms_connector_p, reset_state=True)

    dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
    check_results = inspector_result.check_to_check_results
    no_bias_check_result = check_results[NoBiasIntroducedFor(["age_group", "race"])]

    distribution_changes_overview_df = NoBiasIntroducedFor.get_distribution_changes_overview_as_df(
        no_bias_check_result)
    print(distribution_changes_overview_df.to_markdown())

    for i in list(no_bias_check_result.bias_distribution_change.items()):
        _, join_distribution_changes = i
        for column, distribution_change in join_distribution_changes.items():
            print("")
            print(f"\033[1m Column '{column}'\033[0m")
            print(distribution_change.before_and_after_df.to_markdown())