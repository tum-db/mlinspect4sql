import pandas
from faker import Faker
import random
from example_to_sql._benchmark_utility import ROOT_DIR


def set_some_null(input_list, null_percentage, null_symbol=""):
    """
    Generates the data, but randomly sets 'null' values while generating the others.
    """
    for i in range(len(input_list)):
        if random.random() < null_percentage:
            input_list[i] = null_symbol
    return input_list


def similar_value(callable_gen, other_values, *args, **kwargs):
    """
    Guarantees matching values for most of ones pass in other_values.
    """
    result = []
    for i in range(len(other_values)):
        if random.random() < 0.8:
            result.append(other_values[i])
        else:
            result.append(callable_gen(*args, **kwargs))
    return result


def create_fake_histories_dataset(target_path, target_lines):
    Faker.seed(0)
    faker = Faker()
    history_ssn = [faker.ssn() for _ in range(target_lines)]
    history_smoker = set_some_null([faker.pybool() for _ in range(target_lines)], 0.05, "")
    history_complications = [faker.pyint(min_value=0, max_value=10, step=1) for i in range(target_lines)]

    new_file = pandas.DataFrame({'smoker': history_smoker, 'complications': history_complications, 'ssn': history_ssn})
    new_file.to_csv(target_path, index=False)

    return history_ssn


def create_fake_patients_dataset(target_path, target_lines, other_ssn):
    """
        - id is never null.
        - first_name is never null
        - last_name is never null
        - income is never null
        - num_children is never null
        - race is null sometimes -> 0.109 of times
        - county is null sometimes -> 0.206 of times
        - age_group is never null
        - ssn is null sometimes -> 0.113 0.109 of times

        id,first_name,last_name,race,county,num_children,income,age_group,ssn
    """
    Faker.seed(0)
    faker = Faker()
    id = [i for i in range(target_lines)]
    first_name = [faker.first_name() for _ in range(target_lines)]
    last_name = [faker.last_name() for _ in range(target_lines)]
    race = set_some_null(["race" + str(random.randint(1, 3)) for _ in range(target_lines)], 0.109, "")
    county = set_some_null(["county" + str(random.randint(1, 3)) for _ in range(target_lines)], 0.206, "")
    num_children = [str(random.randint(1, 5)) for _ in range(target_lines)]
    income = [str(random.randint(10000, 300000)) for _ in range(target_lines)]
    age_group = ["group" + str(random.randint(1, 3)) for _ in range(target_lines)]
    ssn = set_some_null(similar_value(faker.ssn, other_ssn), 0.113, "")

    new_file = pandas.DataFrame(
        {'id': id,
         'first_name': first_name,
         'last_name': last_name,
         'race': race,
         'county': county,
         'num_children': num_children,
         'income': income,
         'age_group': age_group,
         'ssn': ssn})

    new_file.to_csv(target_path, index=False)


def generate_healthcare_dataset(sizes):
    """
    As the compas pipeline does not use any joins, the data will just be augmented, by replicating the existing one.
    """
    paths = []
    for i in sizes:
        target_paths = (ROOT_DIR / r"data_generation" / f"generated_csv/healthcare_histories_generated_{i}.csv",
                        ROOT_DIR / r"data_generation" / f"generated_csv/healthcare_patients_generated_{i}.csv")

        if not target_paths[0].exists() or not target_paths[1].exists():
            ssn_list = create_fake_histories_dataset(target_paths[0], i)
            create_fake_patients_dataset(target_paths[1], i, ssn_list)

        paths.append(target_paths)
        print(f"Data generated or found for: size = {i} -- healthcare")

    return paths
