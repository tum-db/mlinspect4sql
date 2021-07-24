import pandas
from example_to_sql._benchmark_utility import ROOT_DIR


def generate_compas_dataset(sizes):
    """
    As the compas pipeline does not use any joins, the data will just be augmented, by replicating the existing one.
    """
    train_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/compas_train.csv")
    test_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/compas_test.csv")

    paths = []
    for i in sizes:
        target_paths = (ROOT_DIR / r"data_generation" / f"generated_csv/compas_train_generated_{i}.csv",
                        ROOT_DIR / r"data_generation" / f"generated_csv/compas_test_generated_{i}.csv")

        if not target_paths[0].exists():
            new_train = pandas.concat([train_src] * ((i // len(train_src)) + 1), ignore_index=True)[:i]
            new_train.to_csv(target_paths[0], index=False)

        if not target_paths[1].exists():
            new_train = pandas.concat([test_src] * ((i // len(test_src)) + 1), ignore_index=True)[:i]
            new_train.to_csv(target_paths[1], index=False)

        paths.append(target_paths)
        print(f"Data generated or found for: size = {i} -- compas")

    return paths