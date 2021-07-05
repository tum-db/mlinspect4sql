"""
Tests whether the monkey patching works for all patched pandas methods
"""
import importlib
import math
from inspect import cleandoc

import networkx
import pandas
from pandas import DataFrame
from testfixtures import compare, StringComparison

from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.inspections._lineage import RowLineage, LineageId
import unittest


# !/usr/bin/env python -W ignore::DeprecationWarning
class TestStringMethods(unittest.TestCase):

    def test_bin_ops(self):
        """
        """
        test_code = cleandoc("""
            import pandas as pd
            df = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [7, 5, 4, 2, 1]})
            df['label'] = (1 - df['A']) > (1.2 * df['B'] + 10)
            """)
        try:
            _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                      inspections=[RowLineage(2)], to_sql=True)
        except Exception:
            self.fail()

    def test_bin_ops_complex_1(self):
        """
        """
        test_code = cleandoc("""
            import warnings
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root
            
            COUNTIES_OF_INTEREST = ['county2', 'county3']
            
            patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                "patients.csv"), na_values='')
            histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                 "histories.csv"), na_values='')
            
            data = patients.merge(histories, on=['ssn'])
            complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
            data = data.merge(complications, on=['age_group'])
            data['label'] = (1 - data['complications']) > (1.2 * data['mean_complications'] + 10)
            """)
        try:
            _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                      inspections=[RowLineage(2)], to_sql=True)
        except Exception:
            self.fail()

    def test_bin_ops_complex_2(self):
        """
        """
        test_code = cleandoc("""
            import warnings
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            COUNTIES_OF_INTEREST = ['county2', 'county3']

            patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                "patients.csv"), na_values='')
            histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                 "histories.csv"), na_values='')

            data = patients.merge(histories, on=['ssn'])
            complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
            data = data.merge(complications, on=['age_group'])
            data['label'] = ((1 - data['complications']) > (1.2 * complications['mean_complications'] + 10))
            """)
        try:
            _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                      inspections=[RowLineage(2)], to_sql=True)
        except Exception:
            self.fail()

    def test_bin_ops_complex_3(self):
        """
        """
        test_code = cleandoc("""
            import warnings
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            COUNTIES_OF_INTEREST = ['county2', 'county3']

            patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                "patients.csv"), na_values='')
            histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                 "histories.csv"), na_values='')

            data = patients.merge(histories, on=['ssn'])
            complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
            data = data.merge(complications, on=['age_group'])
            data['label'] = 2
            """)
        try:
            _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                      inspections=[RowLineage(2)], to_sql=True)
        except Exception:
            self.fail()

    def test_bin_ops_complex_4(self):
        """
        """
        test_code = cleandoc("""
            import warnings
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            COUNTIES_OF_INTEREST = ['county2', 'county3']

            patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                "patients.csv"), na_values='')
            histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                                 "histories.csv"), na_values='')

            data = patients.merge(histories, on=['ssn'])
            complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
            data = data.merge(complications, on=['age_group'])
            data['label'] = data['complications'] > 1.2 * data['mean_complications']
            data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
            data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
            """)
        try:
            _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                      inspections=[RowLineage(2)], to_sql=True)
        except Exception:
            self.fail()


if __name__ == '__main__':
    unittest.main()
