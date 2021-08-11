import numpy as np
import ast


def results_to_np_array(results: list):
    # Result to DataFrame:
    to_return = []
    for i, (col_names, res) in enumerate(results):
        # print(str(col_names) + "__" + "#" * 10)
        columns = np.transpose(np.array(res))

        if len(columns.shape) == 1 or columns.shape[1] == 0:
            to_return.append(np.array(res))  # r is empty
            continue

        columns_as_np_arrays = []
        for name, col in zip(col_names, columns):
            if "_ctid" in name:
                continue  # skip the tracking columns.
            elif isinstance(col[0], list):
                columns_as_np_arrays.append(np.asarray(list(col)))  # Col to np.array
                continue
            elif isinstance(col[0], str) and "{" in col[0] and "}" in col[0]:  # likely also a list:
                if isinstance(ast.literal_eval(col[0]), set):  # 100% list
                    # required to keep ordering:
                    col = [ast.literal_eval(x.replace("}", "]").replace("{", "[")) for x in col]
                    columns_as_np_arrays.append(np.asarray(col))
                    continue
            columns_as_np_arrays.append(np.expand_dims(np.asarray(col), axis=1))
        res = np.hstack(columns_as_np_arrays)
        try:  # try to convert to float if possible
            res = res.astype(np.float64)
        except ValueError:
            pass
        to_return.append(res)
    return to_return
