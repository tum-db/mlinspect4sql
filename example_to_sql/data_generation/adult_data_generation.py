import pandas
from example_to_sql._benchmark_utility import ROOT_DIR


def generate_adult_dataset(sizes):
    """
    As the adult pipelines dont not use any joins, the data will just be augmented, by replicating the existing one.
    """
    train_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/adult_train.csv", index_col=0)
    test_src = pandas.read_csv(ROOT_DIR / r"data_generation/original_csv/adult_test.csv", index_col=0)

    paths = []
    for i in sizes:
        target_paths = (ROOT_DIR / r"data_generation" / f"generated_csv/adult_train_generated_{i}.csv",
                        ROOT_DIR / r"data_generation" / f"generated_csv/adult_test_generated_{i}.csv")

        if not target_paths[0].exists():
            new_train = pandas.concat([train_src] * ((i // len(train_src)) + 1), ignore_index=True)[:i]
            new_train.to_csv(target_paths[0], index=True, index_label="")

        if not target_paths[1].exists():
            new_train = pandas.concat([test_src] * ((i // len(test_src)) + 1), ignore_index=True)[:i]
            new_train.to_csv(target_paths[1], index=True, index_label="")

        paths.append(target_paths)
        print(f"Data generated or found for: size = {i} -- adult")

    return paths


if __name__ == "__main__":
    generate_adult_dataset([100])
