from _benchmark_utility import plot_compare
import csv
import pathlib


def plot_bench_file(path):
    sizes = []

    pandas = []
    umbra_cte = []
    umbra_view = []
    postgres_cte = []
    postgres_view = []
    postgres_view_mat = []

    with pathlib.Path(path).open("r") as f:

        for i, line in enumerate(f.readlines()):
            line_parts = line.split(", ")

            if len(line_parts) <= 1:
                continue

            pipeline_name = line_parts[0]
            pipeline_part = line_parts[1]
            exec_detail = line_parts[2]

            mode = line_parts[4]

            materialized = line_parts[5]
            time = float(line_parts[7])
            sizes.append(line_parts[3]) if mode == "VIEW" else 0
            if line_parts[6] == "Pandas" and mode == "VIEW":
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
    names = ["Pandas", f"Postgresql - CTE", f"Postgresql - VIEW", f"Postgresql - VIEW MATERIALIZED", f"Umbra - CTE",
             f"Umbra - VIEW"]
    table = [pandas, postgres_cte, postgres_view, postgres_view_mat, umbra_cte, umbra_view]
    for i, x in enumerate(table):
        if not x:
            names = names[:i] + names[i + 1:]

    table = [t for t in table if t != []]
    title = f"{pipeline_name}_{pipeline_part}_{exec_detail}"

    plot_compare(title, sizes, all_y=table, all_y_names=names, save=True)


if __name__ == "__main__":
    plot_bench_file(
        r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/final_plots/compas_simple_only_pandas_pure_pipe.csv")
    plot_bench_file(
        r"/home/luca/Documents/Bachelorarbeit/mlinspect/example_to_sql/plots/final_plots/healthcare_only_pandas_pure_pipe.csv")
