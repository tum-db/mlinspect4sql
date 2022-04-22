import timeit
import numpy as np


class PandasConnector:
    @staticmethod
    def benchmark_run(pandas_code, setup_code="import pandas", repetitions=1, verbose=True):
        """Runs and times pandas code."""
        result = []
        for i in range(repetitions):
            print(f"Default run {i + 1} of {repetitions} ...") if verbose else 0
            result.append(timeit.timeit(pandas_code, setup=setup_code, number=1) * 1000)  # in ms
            print(f"Done in {result[-1]}!")
        return sum(result) / repetitions, np.var(result), np.std(result)
