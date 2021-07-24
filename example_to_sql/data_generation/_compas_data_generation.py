import pandas
from .._benchmark_utility import ROOT_DIR, REPETITIONS


def generate_compas_dataset():
    """
    As the compas pipeline does not use any joins, the data will just be augmented, by replicating the existing one.
    """
    train_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/compas_train.csv")
    test_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/compas_test.csv")

    paths = []
    for i in REPETITIONS:
        target_paths = (ROOT_DIR / r"data_generation" / f"generated_csv/compas_train_generated_{i}.csv",
                        ROOT_DIR / r"data_generation" / f"./generated_csv/compas_test_generated_{i}.csv")
        new_train = pandas.concat([train_src] * ((i // len(train_src)) + 1), ignore_index=True)[:i]
        new_train.to_csv(target_paths[0], index=False)

        new_train = pandas.concat([test_src] * ((i // len(test_src)) + 1), ignore_index=True)[:i]
        new_train.to_csv(target_paths[1], index=False)

        paths.append(target_paths)
        print(f"Data generated for: size = {i} -- compas")
