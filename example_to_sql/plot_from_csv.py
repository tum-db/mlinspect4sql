import pandas

from _benchmark_utility import plot_compare, bar_plot_compare
import csv
import pathlib
import pandas as pd


def plot_bench_file(path):
    sizes = [(10 ** i) for i in range(2, 8, 1)]

    pandas = []
    umbra_cte = []
    umbra_view = []
    postgres_cte = []
    postgres_view = []
    postgres_view_mat = []

    with pathlib.Path(path).open("r") as f:

        for i, line in enumerate(f.readlines()):
            line_parts = line.split(", ")

            # print(line_parts[-1])
            if len(line_parts) <= 1 or line_parts[-1] == "#\n":
                continue

            pipeline_name = line_parts[0]
            pipeline_part = line_parts[1]
            exec_detail = line_parts[2]

            mode = line_parts[4]

            materialized = line_parts[5]
            time = float(line_parts[7])
            if line_parts[6] == "Pandas":
                pandas.append(time)

            elif line_parts[6] == "PostgreSQL":
                if mode == "CTE":
                    postgres_cte.append(time)
                elif mode == "VIEW":
                    if materialized == "MATERIALIZED":
                        postgres_view_mat.append(time)
                    elif materialized == "NON-MATERIALIZED":
                        postgres_view.append(time)
                    else:
                        assert False
                else:
                    assert False

            elif line_parts[6] == "Umbra":
                if mode == "CTE":
                    umbra_cte.append(time)
                elif mode == "VIEW":
                    umbra_view.append(time)
                else:
                    assert False
            else:
                assert line_parts[6] == "Pandas"

    # PLOT:
    names = ["Original", f"PostgreSQL - CTE", f"PostgreSQL - VIEW", f"PostgreSQL - VIEW MAT.", f"Umbra - CTE",
             f"Umbra - VIEW"]
    table = [pandas, postgres_cte, postgres_view, postgres_view_mat, umbra_cte, umbra_view]
    i = 0
    for x in table:
        if not x:
            names = names[:i] + names[i + 1:]
            i -= 1
        i += 1

    table = [t for t in table if t != []]
    title = f"{pipeline_name}_{pipeline_part}_{exec_detail}"

    plot_compare(title, sizes, all_y=table, all_y_names=names, save=True)


def plot_bench_file_precise(path, title):
    sizes = [(10 ** i) for i in range(2, 8, 1)]

    original = []
    postgres_view_mat = []
    db_overhead_cost_sum = 0

    with pathlib.Path(path).open("r") as f:

        for i, line in enumerate(f.readlines()):
            line_parts = line.split(", ")

            # print(line_parts[-1])
            if len(line_parts) <= 1 or line_parts[-1] == "#\n":
                continue

            op_name = line_parts[0]
            try:
                time = float(line_parts[1].strip())
            except ValueError:
                time = float(line_parts[7])
                engine = line_parts[6]
                if engine == "Pandas/Original":
                    # original.append(("Original", "FULL RUNTIME", time))
                    full_time_original = time
                elif engine == "PostgreSQL":
                    # postgres_view_mat.append((engine, "FULL RUNTIME", time))
                    full_time_post = time

            total_time_so_far = line_parts[2]
            engine = line_parts[3].strip()

            if engine == "Original":
                if "DATA MOVE/TANSFORMATION COST" in op_name:
                    assert False
                else:
                    original.append((engine, op_name, time))


            elif engine == "PostgreSQL" or engine == "SQL":
                if engine == "PostgreSQL":
                    if "DATA MOVE/TANSFORMATION COST" in op_name:
                        db_overhead_cost_sum += time
                    else:
                        postgres_view_mat.append((engine, op_name, time))
                else:
                    pass
            else:
                pass

    original = [(x[0], x[1].lower(), x[2]) for x in original]
    postgres_view_mat = [(x[0], x[1].lower(), x[2]) for x in postgres_view_mat]

    columns = ["engine", "operation", "runtime (ms)"]
    data = pandas.DataFrame(original + postgres_view_mat)
    data.columns = columns

    data_2 = pandas.DataFrame(index=data.columns.values)
    total_mlinspect_cost_original = full_time_original - sum([x[2] for x in original if x[1] != "FULL RUNTIME"])
    total_mlinspect_cost_postgres = full_time_post - sum([x[2] for x in postgres_view_mat if x[1] != "FULL RUNTIME"])

    size = int(title.split("_")[-1])

    data_2 = data_2.append(
        {columns[0]: f"Original", columns[1]: "mlinspect cost", columns[2]: total_mlinspect_cost_original},
        ignore_index=True)
    data_2 = data_2.append(
        {columns[0]: f"PostgreSQL", columns[1]: "mlinspect cost", columns[2]: total_mlinspect_cost_postgres},
        ignore_index=True)
    data = data.append(
        {columns[0]: f"PostgreSQL", columns[1]: "data move/transf. cost", columns[2]: db_overhead_cost_sum},
        ignore_index=True)

    print("db_overhead_cost_sum: " + str(db_overhead_cost_sum))
    # print(data.to_markdown())

    bar_plot_compare(title, data=data, save=True, shape=(12., 4.))

    return data_2


def plot_simple_bar(path, title):

    pandas = []
    umbra_view = []
    postgres_view = []
    postgres_view_mat = []
    postgres_cte = []

    with pathlib.Path(path).open("r") as f:

        for i, line in enumerate(f.readlines()):
            line_parts = line.split(", ")

            # print(line_parts[-1])
            if len(line_parts) <= 1 or line_parts[-1] == "#\n":
                continue

            pipeline_name = line_parts[0]
            pipeline_part = line_parts[1]
            exec_detail = line_parts[2]

            mode = line_parts[4]

            materialized = line_parts[5]
            time = float(line_parts[7])
            if line_parts[6] == "Pandas/Original":
                pandas.append(time)

            elif line_parts[6] == "PostgreSQL":
                if mode == "CTE":
                    postgres_cte.append(time)
                elif mode == "VIEW":
                    if materialized == "MATERIALIZED":
                        postgres_view_mat.append(time)
                    elif materialized == "NON-MATERIALIZED":
                        postgres_view.append(time)
                    else:
                        assert False
                else:
                    assert False

            elif line_parts[6] == "Umbra":
                if mode == "VIEW":
                    umbra_view.append(time)
                else:
                    assert False
            else:
                assert line_parts[6] == "Pandas"

    # PLOT:
    names = ["Original", f"PostgreSQL - VIEW", f"PostgreSQL - VIEW MAT.", f"Umbra - VIEW"]
    table = [pandas, postgres_view, postgres_view_mat, umbra_view]
    i = 0
    for x in table:
        if not x:
            names = names[:i] + names[i + 1:]
            i -= 1
        i += 1

    table = [t[0] for t in table if t != []]
    title = f"{pipeline_name}_{pipeline_part}_{exec_detail}"

    data = pd.DataFrame({"operation": ["cost" for t in table], "runtime (ms)": table, "engine": names})

    bar_plot_compare(title, data=data, save=True, shape=(8., 8.))

if __name__ == "__main__":
    # PLOT FOR ONLY PANDAS RUN:
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/only_pandas/compas_simple_only_pandas_pure_pipe.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/only_pandas/healthcare_only_pandas_pure_pipe.csv")

    # PLOT FOR FULL RUN:
    # plot_bench_file(
    #     r"/example_to_sql/plots/full_inspection/a_healthcare_full_pure_pipe.csv")
    # plot_bench_file(
    #     r"/example_to_sql/plots/full_inspection/b_compas_full_pure_pipe.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/full/c_adult_simple_full_inspection.csv")
    # plot_bench_file(
    #     r"/example_to_sql/plots/full/d_adult_complex_full_pure_pipe.csv")

    # PLOT FOR FULL INSPECTION:
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/full_inspection/a_healthcare_full_inspection.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/final_plots/b_compas_full_inspection.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/final_plots/c_adult_simple_full_inspection.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/final_plots/d_adult_complex_full_inspection.csv")

    # PLOT FOR END-TO-END:
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/a_healthcare_end_to_end.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/b_compas_end_to_end.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/c_adult_simple_end_to_end.csv")
    # plot_bench_file(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/d_adult_complex_end_to_end.csv")

    # PLOT PRECISE END-TO-END:
    # data_info_1 = plot_bench_file_precise(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/precise/compas_end_to_end_precise.csv",
    #     "PRECISE_COMPARE_COMPAS_10000")
    # data_info_2 = plot_bench_file_precise(
    #     r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/end_to_end/precise/compas_end_to_end_precise_1000000.csv",
    #     "PRECISE_COMPARE_COMPAS_1000000")
    # data_info_2.columns = ["engine", "operation", "runtime (ms)"]
    #
    # bar_plot_compare("PRECISE_COMPARE_COMPAS_1000000_INFO", data=data_info_1.append(data_info_2), save=True,
    #                  shape=(4., 4.), y_axis_ticks=range(0, 2000, 200))

    plot_simple_bar(r"example_to_sql/plots/end_to_end/presi_endtoend_h.csv",
                    "end-to-end_h")
    plot_simple_bar(r"example_to_sql/plots/end_to_end/presi_endtoend_c.csv",
                    "end-to-end_c")


