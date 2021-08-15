import pandas as pd
from numpy.random import randint
import random


def simple_fake_data(target_path, data_frame_rows):
    """
    Code adapted from: https://github.com/stefan-grafberger/mlinspect/blob/19ca0d6ae8672249891835190c9e2d9d3c14f28f/experiments/performance/_benchmark_utils.py
    """

    a = randint(0, 100, size=([data_frame_rows]))
    b = randint(0, 100, size=([data_frame_rows]))
    c = randint(0, 100, size=([data_frame_rows]))
    d = randint(0, 100, size=([data_frame_rows]))
    categories = ['cat_a', 'cat_b', 'cat_c']
    group_col_1 = pd.Series(random.choices(categories, k=data_frame_rows))
    group_col_2 = pd.Series(random.choices(categories, k=data_frame_rows))
    group_col_3 = pd.Series(random.choices(categories, k=data_frame_rows))
    df = pd.DataFrame(zip(a, b, c, d, group_col_1, group_col_2, group_col_3), columns=['a', 'b', 'c', 'd',
                                                                                       'group_col_1', 'group_col_2',
                                                                                       'group_col_3'])
    df.to_csv(target_path, index_label="id")
    return


if __name__ == '__main__':
    for k in range(2, 4, 1):
        i = pow(10, k)
        # while i <= pow(10, 7):
        print(f"data generated for: size = {i}")
        simple_fake_data(f"./generated_csv/simple_fake_data_second{i}.csv", i)
        simple_fake_data(f"./generated_csv/simple_fake_data{i}.csv", i)
