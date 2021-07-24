import timeit


class PandasConnector:

    @staticmethod
    def benchmark_run(pandas_code, setup_code="import pandas", repetitions=1, verbose=True):
        """Runs and times pandas code."""
        print("Executing Query in Pandas...") if verbose else 0
        time = (timeit.timeit(pandas_code, setup=setup_code, number=repetitions) / repetitions) * 1000  # in ms
        print(f"Done in {time}!") if verbose else 0
        return time
