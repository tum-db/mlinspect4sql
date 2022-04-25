import pathlib
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mlinspect.utils import get_project_root
from mlinspect.utils import store_timestamp
import time

# Some parameters you might want check:
DO_CLEANUP = True
SIZES = [(10 ** i) for i in range(2, 6, 1)]
BENCH_REP = 1
MLINSPECT_ROOT_DIR = get_project_root()

# DBMS related:
UMBRA_USER = "postgres"
UMBRA_PW = " "
UMBRA_DB = ""
UMBRA_PORT = 5433
UMBRA_HOST = "/tmp/"

POSTGRES_USER = "luca"
POSTGRES_PW = "password"
POSTGRES_DB = "healthcare_benchmark"
POSTGRES_PORT = 5432
POSTGRES_HOST = "localhost"

ROOT_DIR = pathlib.Path(__file__).resolve().parent
PLOT_DIR = ROOT_DIR / "plots"
LOG = ROOT_DIR / "plots" / "aRunLog.csv"

COLOR_SET = ["#B93313", "#188DD3", "#64B351", "#7F6AA7", "#D6BD62", "#BBDDAA"]
rc = {
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.labelcolor': 'dimgrey',
    'lines.solid_capstyle': 'round',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.right': False,
    'figure.figsize': (4., 4.)}

seaborn.set_theme(context="paper", font='Franklin Gothic Book', font_scale=1.2, style="whitegrid", palette=COLOR_SET,
                  rc=rc)
plt.legend(loc="upper left", prop={'size': 10}, title=None)


def plot_compare(title, x, all_y, all_y_names, colors=None, x_label="dataset size (rows)", y_label="runtime (ms)",
                 save=True):
    """
    Based on: mlinspect/experiments/performance/performance_benchmarks.ipynb
    Args:
        title (str): Title of the plot
        x (list): List of values for the x-axis
        all_y (list): List of lists containing the values for the y-axis
        all_y_names (list): List of strings for the names of the plots - same order as the list all_y
        colors (list or None): optional colors for the graphs passed in all_y
        x_label (str)
        y_label (str):
    """
    if colors is None:
        colors = COLOR_SET

    figure, axis = plt.subplots()

    axis.set_yscale('log')  # sets the scale to be logarithmic with powers of 10

    for j, (y, y_name) in enumerate(zip(all_y, all_y_names)):
        axis.plot(y, marker='8', color=colors[j], markersize=4, label=y_name, linewidth=2)
    # for marker type see: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    # plot function: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
    # first arguments set y_value range , x is set to 0..N-1 for now

    axis.set_xticks(range(0, len(x)))
    axis.set_xticklabels(x)
    axis.set_axisbelow(True)

    # get the current labels
    labels = [item.get_text() for item in axis.get_xticklabels()]

    # Beat them in form and add back
    # see: https://stackoverflow.com/questions/36480077/python-plot-how-to-denote-ticks-on-the-axes-as-powers
    def get_exp(label):
        return int(np.log10(int(label)))

    # axis.set_xticklabels(
    #     [str(f"${int(int(label) / 10 ** get_exp(label))} \\times 10^{get_exp(label)}$") for label in labels])
    axis.set_xticklabels(
        [str(f"$10^{get_exp(label)}$") for label in labels])

    axis.set_facecolor('white')
    axis.axis('equal')
    axis.set(xlabel=x_label, ylabel=y_label)
    axis.grid(True, color='lightgrey')

    if save:
        save_path = ROOT_DIR / f"plots/{title}.png"
        if save_path.exists():
            save_path = ROOT_DIR / f"plots/{title}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')}.png"
        figure.savefig(save_path, bbox_inches='tight', dpi=800)

    return plt


def bar_plot_compare(title, data, colors=None, save=True, shape=(4., 4.), y_axis_ticks=None):
    """
    Based on: mlinspect/experiments/performance/performance_benchmarks.ipynb
    Args:
        title (str): Title of the plot
        x (list): List of values for the x-axis
        all_y (list): List of lists containing the values for the y-axis
        all_y_names (list): List of strings for the names of the plots - same order as the list all_y
        colors (list or None): optional colors for the graphs passed in all_y
        x_label (str)
        y_label (str):
    """

    plt.rcParams["figure.figsize"] = shape

    if colors is None:
        colors = COLOR_SET

    figure, axis = plt.subplots()
    axis = seaborn.barplot(x="operation", y="runtime (ms)", hue="engine", data=data, linewidth=0)
    axis.set_axisbelow(True)

    if y_axis_ticks is not None:
        axis.set_yticks(y_axis_ticks)

    axis.set(xlabel=None)
    plt.legend(title=None)


    # axis.set_yscale('log')  # sets the scale to be logarithmic with powers of 10
    plt.xticks(rotation=30)

    axis.set_facecolor('white')
    # axis.axis('equal')
    # axis.set(xlabel=x_label, ylabel=y_label)
    axis.grid(True, color='lightgrey')

    if save:
        save_path = ROOT_DIR / f"plots/{title}.png"
        if save_path.exists():
            save_path = ROOT_DIR / f"plots/{title}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')}.png"
        figure.savefig(save_path, bbox_inches='tight', dpi=800)

    return plt


def write_to_log(pipeline_name, only_pandas, inspection, size, mode, materialize, engine, time, csv_file_paths,
                 with_train):
    if only_pandas:
        only_pandas = "ONLY_PANDAS"
    else:
        only_pandas = "FULL"

    if inspection:
        inspection = "INSPECTION"
        if with_train:
            inspection = "INSPECTION_TRAIN"
    else:
        inspection = "PURE_RUN"
    if materialize:
        materialize = "MATERIALIZED"
    else:
        materialize = "NON-MATERIALIZED"

    time_pure, std, var = time

    with LOG.open("a") as file:
        file.write(
           f"{pipeline_name}, {only_pandas}, {inspection}, {size}, {mode}, {materialize}, {engine}, {time_pure}, "
           f"{std}, {var}, {str(csv_file_paths).replace(',', ';')}\n")

def write_brake_to_log():
    with LOG.open("a") as file:
        file.write("\n###############################\n")
