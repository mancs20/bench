import ast
import re
import shutil
from collections import Counter

import pandas as pd
import seaborn as sns
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
import os
from scipy.stats import ttest_rel, wilcoxon
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class StrategySpec:
    label: str                 # name used in plots/tables
    source: str                # value in CSV column front_generator
    query: Optional[str] = None  # pandas query, e.g., "use_lex == True"

# noinspection PyTypeChecker
def find_closest_time_index_to_time_t(x_all_times, id_times, t):
    for i in range(id_times, len(x_all_times)):
        if x_all_times[i] > t:
            return i - 1
    return len(x_all_times) - 1


# noinspection PyTypeChecker
def is_dominated(point, other_point, maximize=True):
    for dimension in range(len(point)):
        if maximize and point[dimension] > other_point[dimension]:
            return False
        elif not maximize and point[dimension] < other_point[dimension]:
            return False
    return True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def format_number(x):
    if isinstance(x, (int, float)):  # Ensure it's a number
        return int(x) if x == int(x) else f"{x:.2f}"
    return x  # Keep non-numeric values unchanged


#
# stats = ["time(s)", "sum_solutions_resolution_time(s)", "sum_solutions_nodes",
#          "sum_solutions_backtracks", "sum_solutions_fails",
#          "sum_number_solutions"]
# stats_exhaustive_pretty_name = ["time(s)", "resolution_time(s)", "nodes",
#                      "backtracks", "fails",
#                      "number_solutions"]
#
# stats = ["time(s)", "sum_solutions_nodes",
#          "sum_solutions_backtracks"]
# stats_exhaustive_pretty_name = ["time(s)", "nodes",
#                      "backtracks"]
#
# stats = ["time(s)", "sum_solutions_nodes", "front_cardinality"]
# stats_exhaustive_pretty_name = ["time(s)", "nodes"]
#
# stats_non_exhaustive = ["hypervolume", "front_cardinality", "exhaustive"]
# stats_non_exhaustive_pretty_name = ["Hyp", "Points", "Compl"]


def get_info_similar_instances_ukp_moolibrary():
    objs_elements = {2: [50], 3: [30, 40, 50], 4: [20, 30, 40], 5: [10, 20]}
    pattern_template = "KP_p-{obj}_n-{elements}_ins-"
    return objs_elements, pattern_template


def get_info_similar_instances_bi_ukp_voptlib():
    objs_elements = {2: [50]}
    pattern_template = "K5050W"
    return objs_elements, pattern_template


def get_info_similar_instances_nqueens():
    objs_elements = {2: [8, 10, 12, 14], 3: [8, 10, 12, 14], 4: [8, 10, 12, 14], 5: [8, 10, 12, 14]}
    pattern_template = "n_queens_p-{obj}_q-{elements}_ins-"
    return objs_elements, pattern_template


def get_info_similar_instances_rcpsp():
    objs_elements = {2: [30, 60, 90]}
    pattern_template = "J{elements}"
    return objs_elements, pattern_template


def get_info_similar_instances_sims():
    objs_elements = {"cost_clouds": [30, 50, 100, 150, 200], "cost_clouds_angle": [30, 50, 100, 150, 200]}
    pattern_template = "_{elements}_{obj}$"
    return objs_elements, pattern_template


# noinspection PyTypeChecker
def filter_initial_k_points(front, k, maximize=True):
    """
    Removes initial k points if any of them are dominated or duplicated later.
    """
    front = np.array(front)
    to_keep = np.ones(len(front), dtype=bool)

    if len(front) <= k:
        return front

    for i in range(k):
        suspicious_pt = front[i]
        for j in range(k, len(front)):
            if is_dominated(suspicious_pt, front[j], maximize):
                to_keep[i] = False
                break

    return front[to_keep]


def compute_hv(front, reference_point, double_check_non_dominance=True):
    front = np.array(front)
    if front.size == 0:
        return 0.0

    reference_point = np.array(reference_point)
    front = np.array(front)

    if is_maximization_problem(front, reference_point):
        # convert maximization to minimization to calculate the Hypervolume using the pymoo library
        reference_point = -reference_point
        front = -front

    # Filter non-dominated only (safe redundancy)
    if double_check_non_dominance:
        idx = NonDominatedSorting().do(front, only_non_dominated_front=True)
        front = front[idx]

    return Hypervolume(ref_point=reference_point)(front)


def compute_igd(front, reference_point, pareto_front, maximize=None, plus=False):
    if len(front) == 0 or len(pareto_front) == 0:
        return float('inf')
    reference_point = np.array(reference_point)
    front = np.array(front)
    pareto_front = np.array(pareto_front)

    if maximize is True or (maximize is None and is_maximization_problem(front, reference_point)):
        # convert maximization to minimization to calculate the Hypervolume using the pymoo library
        front = -front
        pareto_front = -pareto_front
    if plus:
        ind = IGDPlus(pareto_front)
    else:
        ind = IGD(pareto_front)
    return ind(front)


def is_maximization_problem(front, reference_point):
    # check if the front is np array
    if not isinstance(front, np.ndarray):
        front_np = np.array(front)
    else:
        front_np = front

    if not isinstance(reference_point, np.ndarray):
        reference_point_np = np.array(reference_point)
    else:
        reference_point_np = reference_point

    if len(front) > 1:
        comparisson_array = reference_point_np < front_np[1]
    else:
        comparisson_array = reference_point_np < front_np[0]

    return comparisson_array[0]


def set_general_plot_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "lines.markersize": 8,
        # grid style + enable it globally
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
    })


class MoAnalysis:

    def __init__(self, strategies=None, calculate_std=True, benchmark='benchmark', problem='problem', instance='instance', solver_name='solver',
                 front_strategy='front_generator',
                 hypervolume='hypervolume', pareto_front='pareto_front',
                 hypervolume_evolution='hypervolume_evolution', number_of_solutions='front_cardinality',
                 exhaustive='exhaustive', time='time(s)', solutions_in_time='solutions_in_time',
                 time_solver_sec=None, pareto_solutions_time_list=None, all_solutions='all_solutions'):
        self.calculate_std = calculate_std
        self.joining_latex_string_avg_std = "\\pm "
        self.benchmark = benchmark
        self.problem = problem
        self.hypervolume = hypervolume
        self.instance = instance
        self.solver_name = solver_name
        self.front_strategy = front_strategy
        self.exhaustive = exhaustive
        self.number_of_solutions = number_of_solutions
        self.hypervolume_evolution = hypervolume_evolution
        self.solutions_in_time = solutions_in_time
        self.pareto_front = pareto_front
        self.time = time
        self.timeout = "timeout"
        # for table stats
        self.stats_exhaustive = ["time(s)", "sum_solutions_nodes", "front_cardinality"]
        self.stats_exhaustive_pretty_name = ["time(s)", "nodes"]

        self.stats_non_exhaustive = ["hypervolume", "front_cardinality", "exhaustive"]
        self.stats_non_exhaustive_pretty_name = ["Hyp", "Points", "Compl"]

        # to store the front for each solver
        self._joint_front_cache = {}

        if time_solver_sec is None:
            self.time_solver_sec = time
        else:
            self.time_solver_sec = time_solver_sec
        if pareto_solutions_time_list is None:
            self.pareto_solutions_time_list = solutions_in_time
        else:
            self.pareto_solutions_time_list = pareto_solutions_time_list
        if all_solutions is None:
            self.all_solutions = pareto_front
        else:
            self.all_solutions = all_solutions

        # strategies's colors
        if strategies is None:
            self.strategies = self.create_default_strategies()
        else:
            self.strategies = strategies
        self.fixed_strategies = self.strategies.strategies_better_name
        self.strategy_colors = self.strategies.colors
        self.strategy_markers = self.strategies.markers

    def set_stats_non_exhaustive_for_metric(self, metric: str):
        """
        Switch the non-exhaustive table metric away from hypervolume.

        Assumption: your analysis code expects the metric column to already exist in the df
        and to be numeric (NaN allowed).
        """
        # what you compare for non-exhaustive (replace "hypervolume" by metric)
        # keep the same second column ("front_cardinality") if you want it unchanged.
        if metric == "igd_plus":
            metric_pretty_name = "IGD$^+$"
        elif metric == "igd":
            metric_pretty_name = "IGD"
        else:
            raise ValueError(f"Unknown metric for non-exhaustive table: {metric}")

        # self.stats_non_exhaustive = ["hypervolume", "front_cardinality", "exhaustive"]
        self.stats_non_exhaustive[0] = metric
        # self.stats_non_exhaustive_pretty_name = ["Hyp", "Points", "Compl"]
        self.stats_non_exhaustive_pretty_name[0] = metric_pretty_name

    def create_default_strategies(self):
        default_strategies = {"GIA":"GIA", "GIA_boundedLazy":"GIAubL", "GIA_bounded":"GIAub",
                              "ParetoDisjunctiveProgramming":"DisjProg", "ParetoGavanelliGlobalConstraint":"Gavanelli",
                              "SaugmeconNoR":"Saugmecon"}
        return Strategies(default_strategies)

    @staticmethod
    def csv_to_df(file_path):
        df = pd.read_csv(file_path, delimiter=',')
        return df

    @staticmethod
    def count_common_digits(values):
        """Finds how many leading digits are the same in all values (excluding exponent)."""
        values = sorted(set(abs(v) for v in values if v >= 10 ** 4))  # Ignore small numbers
        if len(values) < 2:
            return 0  # No comparison possible with one or no values

        str_values = [f"{v:.15e}" for v in values]  # Convert to high-precision scientific notation
        min_len = min(len(s) for s in str_values)

        common_digits = 0
        # noinspection PyTypeChecker
        for i in range(min_len):
            chars_at_pos = set(s[i] for s in str_values)
            if len(chars_at_pos) < len(str_values):  # Some agreement
                common_digits += 1
            else:
                break

        return max(0, common_digits - 2)  # Subtract exponent and decimal point

    @staticmethod
    def format_number_dynamic(num, values_in_column, base_precision=5, min_precision=2):
        """Formats number in scientific notation, ensuring at least two extra digits beyond common digits."""
        if num is None or np.isnan(num):
            return ""

        if num < 10 ** 4:  # Keep normal formatting for small numbers
            return format_number(num)

        exponent = int(np.floor(np.log10(abs(num))))  # Get exponent
        coefficient = num / (10 ** exponent)  # Normalize to get coefficient

        # Compute required precision dynamically
        common_digits = MoAnalysis.count_common_digits(values_in_column)

        precision = max(min_precision, common_digits + 2)  # Ensure 2 extra digits
        precision = min(precision, base_precision)  # Don't exceed base precision

        return f"{coefficient:.{precision}f} \\times 10^{{{exponent}}}"

    @staticmethod
    def disjunctive_paper_style_format_row_latex(row, list_header_rows, minimize=True):
        """
        Format a LaTeX row, making the best MEAN per metric bold.
        If std columns exist, they must be in the 2nd level of the MultiIndex with suffix '_std'
        (e.g., ('SAUGMECON-RealS', 'time(s)_std')).
        Std cells are formatted but never bolded (bolding is decided by mean only).
        """
        formatted_row = []

        data_columns_start_pos = len(list_header_rows)
        numeric_values = row.iloc[data_columns_start_pos:].apply(pd.to_numeric, errors='coerce')

        # Get unique statistics (e.g., "time(s)", "nodes") from column headers
        stats_columns = {}
        for col in row.index[data_columns_start_pos:]:
            stat_name = col[1]  # Second level of MultiIndex (e.g., "time(s)")
            if isinstance(stat_name, str) and stat_name.endswith("_std"):
                continue  # do not include std columns in the "min/max per stat" logic
            if stat_name not in stats_columns:
                stats_columns[stat_name] = []
            stats_columns[stat_name].append(col)

        # Compute min per statistic group
        values_per_stat_to_highlight = {}
        for stat, columns in stats_columns.items():
            if minimize:
                values_per_stat_to_highlight[stat] = numeric_values[columns].min(skipna=True)
            else:
                values_per_stat_to_highlight[stat] = numeric_values[columns].max(skipna=True)

        # Joining string for mean+std (fallback if not defined)
        join_str = getattr(MoAnalysis, "joining_latex_string_avg_std", "\\pm ")
        # # close_str = "" if (")" in join_str or "\\pm" in join_str) else ")"

        # Format row values
        for col, value in zip(row.index, row):
            # Do not print std columns as separate columns (they will be merged into the mean cell)
            if col in row.index[data_columns_start_pos:]:
                stat_name = col[1]
                if isinstance(stat_name, str) and stat_name.endswith("_std"):
                    continue

            if is_number(value) and not pd.isna(value):  # skip np.nan
                num_value = float(value)
                # Get values for this stat if available, otherwise use just the current number
                column_values = numeric_values[stats_columns[col[1]]] if col[1] in stats_columns else [num_value]
                # formatted_value = (
                #     f"${MoAnalysis.format_number_dynamic(num_value, column_values, base_precision=8, min_precision=2)}$"
                # )
                mean_str = MoAnalysis.format_number_dynamic(num_value, column_values, base_precision=8, min_precision=2)

                # Attach std if available
                std_col = (col[0], f"{col[1]}_std")
                if std_col in row.index:
                    std_raw = row[std_col]
                    if is_number(std_raw) and not pd.isna(std_raw):
                        std_val = float(std_raw)
                        std_str = MoAnalysis.format_number_dynamic(
                            std_val, [std_val], base_precision=2, min_precision=2
                        )
                        cell_content = f"{mean_str}{join_str}{std_str}"
                    else:
                        cell_content = f"{mean_str}"
                else:
                    cell_content = f"{mean_str}"

                formatted_value = f"${cell_content}$"

                # Apply bold if this MEAN is the best for its stat
                stat_name = col[1]
                if stat_name not in ["NaN", "nan", None] and num_value == values_per_stat_to_highlight.get(stat_name,
                                                                                                           None):
                    formatted_value = f"$\\mathbf{{{cell_content}}}$" if col[0] != "" else formatted_value

                formatted_row.append(formatted_value)
            elif isinstance(value, str) and join_str in value:
                formatted_row.append(f"${value}$")
            else:
                formatted_row.append("-" if pd.isna(value) else str(value))

        return " & ".join(map(str, formatted_row)) + " \\\\"

    @staticmethod
    def disjunctive_paper_style_dataframe_to_latex(table, header_columns_list, minimize=True,
                                                   title="Comparison of Strategies"):
        """Convert a DataFrame to a formatted LaTeX table."""
        # Apply formatting row by row
        latex_rows = table.apply(
            lambda row: MoAnalysis.disjunctive_paper_style_format_row_latex(row, header_columns_list, minimize),
            axis=1
        ).tolist()

        # remove columns with _std
        for col in table.columns:
            if isinstance(col[1], str) and col[1].endswith("_std"):
                table = table.drop(columns=col)

        # Construct LaTeX table
        # noinspection PyTypeChecker
        latex_table = "\\begin{table}[h]\n\\centering\n\\caption{" + title + "}\n\\begin{tabular}{" + "r" * len(
            table.columns) + "}\n\\hline\n"

        # Add headers
        strategy_counts = Counter([col[0] for col in table.columns if col[0] != ""])  # Count occurrences of each
        # strategy

        header_first_row = []
        col_idx = 0
        while col_idx < len(table.columns):
            strategy = table.columns[col_idx][0]  # Get strategy name
            if isinstance(strategy, str) and strategy:  # Only merge strategy names, not empty headers
                colspan = strategy_counts[strategy]  # How many columns does this strategy span?
                header_first_row.append(
                    "\\multicolumn{{{}}}{{c}}{{{}}}".format(colspan, strategy.replace("_", "\\_")))
                col_idx += colspan  # Skip next columns since they are merged
            else:
                header_first_row.append("")  # Non-strategy columns remain the same
                col_idx += 1

        header_first_row = " & ".join(header_first_row) + " \\\\\n"
        header_second_row = " & ".join([
            "{}".format(col[1].replace("_", "\\_")) for col in table.columns
        ]) + " \\\\\n\\hline\n"

        # Add headers to LaTeX table
        latex_table += header_first_row
        latex_table += header_second_row

        latex_table += "\n".join(
            latex_rows) + "\n\\hline\n\\end{tabular}\n\\end{table}"

        return latex_table

    # Function to calculate score for each front_strategy
    def calculate_hypervolume_score(self, group):
        best_hypervolume = group.max()
        return group / best_hypervolume

    def plot_lexicographic_best(self, df):
        # Calculate hypervolume scores and reset index
        df_score_by_front_strategy = df.copy()

        # Calculate total fronts by solver
        df_total_front_by_solver = df.groupby([self.solver_name, self.front_strategy])[
            self.instance].count().reset_index()

        # Calculate the best fronts by solver
        df_best_front_by_solver1 = (df_score_by_front_strategy[df_score_by_front_strategy[Cols.LEX_SCORE] == 1.0].
                                    groupby([self.solver_name, self.front_strategy]).size().rename(Cols.LEX_BEST).
                                    to_frame().reset_index())

        df_best_front_by_solver = df_score_by_front_strategy.groupby(
            [self.solver_name, self.front_strategy]).size().rename(
            Cols.LEX_BEST).to_frame().reset_index()

        # Count the number of times self.exhaustive was true for each strategy
        df_best_front_by_solver2 = df_score_by_front_strategy.groupby(
            [self.solver_name, self.front_strategy])[self.exhaustive].sum().rename(
            f"total_{self.exhaustive}").to_frame().reset_index()

        # Merge to align indices properly
        df_best_front_by_solver = pd.merge(df_best_front_by_solver, df_best_front_by_solver1,
                                           on=[self.solver_name, self.front_strategy], how='left',
                                           suffixes=('', '_new'))

        df_best_front_by_solver = pd.merge(df_best_front_by_solver, df_best_front_by_solver2,
                                           on=[self.solver_name, self.front_strategy], how='left',
                                           suffixes=('', '_new'))

        df_best_front_by_solver[Cols.LEX_BEST] = df_best_front_by_solver[f"{Cols.LEX_BEST}_new"].fillna(0).astype(int)
        df_best_front_by_solver = df_best_front_by_solver.drop(columns=f"{Cols.LEX_BEST}_new")

        # Calculate average scores by front strategy
        df_avg_score_by_front_strategy = df_score_by_front_strategy.groupby([self.solver_name, self.front_strategy])[
            Cols.LEX_SCORE].mean().rename(Cols.LEX_AVG_SCORE).to_frame().reset_index()

        # Merge dataframes to get total best and average scores
        df_total_best = pd.merge(df_total_front_by_solver, df_best_front_by_solver,
                                 on=[self.solver_name, self.front_strategy])
        df_total_best_avg_score = pd.merge(df_total_best, df_avg_score_by_front_strategy,
                                           on=[self.solver_name, self.front_strategy])

        print(df_total_best_avg_score)

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=self.solver_name, y=Cols.LEX_BEST, hue=self.front_strategy, data=df_total_best_avg_score)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        plt.title('Times each strategy was the best')
        plt.xlabel('Solver name')
        plt.ylabel('Times best')
        plt.legend(title='Front strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return df_total_best_avg_score, fig

    def plot_lexicographic_score_best_average(self, df_total_best_avg_score):
        # Create the second graph
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=self.solver_name, y=Cols.LEX_AVG_SCORE,
                         hue=self.front_strategy, data=df_total_best_avg_score)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        # Adding labels and legend
        plt.title('Average score for each front strategy')
        plt.xlabel('Solver name')
        plt.ylabel('Average score')
        plt.legend(title='Front strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return fig

    def plot_hypervolume_best(self, df):
        # Calculate hypervolume scores and reset index
        df_score_by_front_strategy = df.copy()

        # Calculate total fronts by solver
        df_total_front_by_solver = df.groupby([self.solver_name, self.front_strategy])[
            self.instance].count().reset_index()

        # Calculate the best fronts by solver
        df_best_front_by_solver1 = df_score_by_front_strategy[df_score_by_front_strategy[Cols.HV_SCORE] == 1.0].groupby(
            [self.solver_name, self.front_strategy]).size().rename(Cols.HV_BEST).to_frame().reset_index()

        df_best_front_by_solver = df_score_by_front_strategy.groupby(
            [self.solver_name, self.front_strategy]).size().rename(
            Cols.HV_BEST).to_frame().reset_index()

        # Merge to align indices properly
        df_best_front_by_solver = pd.merge(df_best_front_by_solver, df_best_front_by_solver1,
                                           on=[self.solver_name, self.front_strategy], how='left',
                                           suffixes=('', '_new'))

        df_best_front_by_solver[Cols.HV_BEST] = df_best_front_by_solver[f"{Cols.HV_BEST}_new"].fillna(0).astype(int)
        df_best_front_by_solver = df_best_front_by_solver.drop(columns=f"{Cols.HV_BEST}_new")

        # Calculate average scores by front strategy
        df_avg_score_by_front_strategy = df_score_by_front_strategy.groupby([self.solver_name, self.front_strategy])[
            Cols.HV_SCORE].mean().rename(Cols.HV_AVG_SCORE).to_frame().reset_index()

        # Merge dataframes to get total best and average scores
        df_total_best = pd.merge(df_total_front_by_solver, df_best_front_by_solver,
                                 on=[self.solver_name, self.front_strategy])
        df_total_best_avg_score = pd.merge(df_total_best, df_avg_score_by_front_strategy,
                                           on=[self.solver_name, self.front_strategy])

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=self.solver_name, y=Cols.HV_BEST, hue=self.front_strategy, data=df_total_best_avg_score)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        plt.title('Times each strategy had the best hypervolume')
        plt.xlabel('Solver name')
        plt.ylabel('Times best')
        plt.legend(title='Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return df_total_best_avg_score, fig

    def plot_lexicographic_hv_time_score_per_instance(self, df):
        # Set up the plot
        fig = plt.figure(figsize=(10, 20))
        sns.set_theme(style="whitegrid")

        # Create a color palette for front_strategy
        palette = sns.color_palette("husl", len(df[self.front_strategy].unique()))

        # Calculate scores

        # Get average scores for plotting
        df_avg_score_by_front_strategy = (df.groupby([self.problem, self.instance, self.front_strategy])[
                                              Cols.LEX_SCORE].mean().reset_index())
        df_avg_score_by_front_strategy['problem_instance'] = (
                df_avg_score_by_front_strategy[self.problem] + ' - ' + df_avg_score_by_front_strategy[self.instance]
        )

        # Plot hypervolume scores
        ax = sns.barplot(x=Cols.LEX_SCORE, y='problem_instance', hue=self.front_strategy,
                         data=df_avg_score_by_front_strategy,
                         palette=palette, orient='h')

        plt.title('Lexicographic hypervolume time score by front strategy for each problem-instance')
        plt.xlabel('Score')
        plt.ylabel('Problem-instance')
        plt.legend(title='Front strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set the background to white and remove the grid lines
        sns.despine(left=True, bottom=True)
        ax.grid(False)
        plt.show()
        return fig

    def plot_hypervolume_score_per_instance(self, df):
        # Set up the plot
        fig = plt.figure(figsize=(10, 20))
        sns.set_theme(style="whitegrid")

        # Create a color palette for front_strategy
        palette = sns.color_palette("husl", len(df[self.front_strategy].unique()))

        # Calculate scores
        df_score_by_front_strategy = df.copy()

        # Get average scores for plotting
        df_avg_score_by_front_strategy = (df_score_by_front_strategy.groupby([self.problem, self.instance,
                                                                              self.front_strategy])
                                          [Cols.HV_SCORE].mean().reset_index())
        df_avg_score_by_front_strategy['problem_instance'] = (
                df_avg_score_by_front_strategy[self.problem] + ' - ' + df_avg_score_by_front_strategy[self.instance]
        )

        # Plot hypervolume scores
        ax = sns.barplot(x=Cols.HV_SCORE, y='problem_instance', hue=self.front_strategy,
                         data=df_avg_score_by_front_strategy,
                         palette=palette, orient='h')

        plt.title('Hypervolume score by front strategy for each problem-instance')
        plt.xlabel('Hypervolume score')
        plt.ylabel('Problem-instance')
        plt.legend(title='Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set the background to white and remove the grid lines
        sns.despine(left=True, bottom=True)
        ax.grid(False)
        plt.show()
        return fig

    def plot_hypervolume_best_average(self, df_total_best_avg_score):
        # Create the second graph
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=self.solver_name, y=Cols.HV_AVG_SCORE,
                         hue=self.front_strategy, data=df_total_best_avg_score)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        # Adding labels and legend
        plt.title('Average hypervolume score for each front strategy')
        plt.xlabel('Solver name')
        plt.ylabel('Average score')
        plt.legend(title='Front strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return fig

    # Plot the time and the number of solutions for each instance
    def get_time_number_solutions(self, df):
        # Apply the function to calculate scores
        df_time_number_solutions = df.groupby([self.problem, self.instance, self.solver_name]).apply(
            self.calculate_number_of_solutions_score).reset_index(drop=True)
        df_time = (df.groupby([self.problem, self.instance, self.solver_name]).apply(self.calculate_time_score).
                   reset_index(drop=True))
        df_solver_front = df.groupby([self.problem, self.instance, self.solver_name]).apply(
            self.merge_solver_front_strategy_names).reset_index(
            drop=True)

        # Merge the results
        df_time_number_solutions[Cols.TIME_SCORE] = df_time[Cols.TIME_SCORE]
        df_time_number_solutions['solver_front_strategy'] = df_solver_front['solver_front_strategy']
        df_time_number_solutions['problem_instance'] = (
                df_time_number_solutions[self.problem] + ' - ' + df_time_number_solutions[self.instance]
        )

        columns_to_select = [
            'problem_instance', 'solver_front_strategy', Cols.TIME_SCORE, self.time_solver_sec,
            'number_of_solutions_score',
            self.number_of_solutions, self.exhaustive]
        df_time_number_solutions = df_time_number_solutions[columns_to_select]

        return df_time_number_solutions

    def calculate_time_score(self, group):
        best_time = group.loc[group[Cols.TIME_FOR_TIME_SCORE].idxmin(), self.time_solver_sec]
        group[Cols.TIME_SCORE] = group[Cols.TIME_FOR_TIME_SCORE] / best_time
        return group

    def calculate_time_score_for_lex_score(self, group):
        best_time = group.min()
        return best_time / group

    def calculate_number_of_solutions_score(self, group):
        best_number_of_solutions = group.loc[group[self.number_of_solutions].idxmin(), self.number_of_solutions]
        group['number_of_solutions_score'] = group[self.number_of_solutions] / best_number_of_solutions
        return group

    def merge_solver_front_strategy_names(self, group):
        group['solver_front_strategy'] = group[self.solver_name] + ' ' + group[self.front_strategy]
        return group

    # Plot time score
    def plot_strategy_time_score_to_get_the_front(self, df_time_number_solutions):
        # Set up the plot
        fig = plt.figure(figsize=(10, 20))
        sns.set_theme(style="whitegrid")

        # Define a threshold for applying a logarithmic scale
        log_threshold = 10  # Adjust the threshold as needed

        # Create a color palette for solver_front_strategy
        palette = sns.color_palette("husl", len(df_time_number_solutions['solver_front_strategy'].unique()))

        # Plot time scores with a conditional logarithmic x-axis
        ax = sns.barplot(x=Cols.TIME_SCORE, y='problem_instance', hue='solver_front_strategy',
                         data=df_time_number_solutions,
                         palette=palette, orient='h')

        plt.title('Time score by solver front strategy')
        plt.xlabel('Time score')
        plt.ylabel('problem_instance')
        plt.legend(title='Solver - Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Apply a logarithmic scale only for values greater than the threshold
        if df_time_number_solutions[Cols.TIME_SCORE].max() > log_threshold:
            plt.xscale('log')

        # Set the background to white and remove the grid lines
        sns.despine(left=True, bottom=True)
        ax.grid(False)
        plt.show()
        return fig

    def plot_general_metric_times_best(self, df, metric_col, maximize=True, figs=None, csvs=None,
                                       group_by_solver=False):
        """
        Generates N+1 plots and tables: N plots for each unique problem and 1 general plot considering all instances.

        Parameters:
        df (pd.DataFrame): The dataframe containing results, including a 'problem' column.
        metric_col (str): The name of the column representing the metric.
        maximize (bool): If True, higher values are better. If False, lower values are better.
        group_by_solver (bool): If True, the x-axis will show the solver + front strategy combination.
                                If False, it will only show the front strategy.
        figs (dict): A dictionary to store the plot figures.
        csvs (dict): A dictionary to store the CSV file paths for later saving.
        """
        if figs is None:
            figs = {}
        if csvs is None:
            csvs = {}

        df = self.apply_timeout_limit(df, metric_col)

        # Step 1: Create a general plot considering all instances
        general_problem_name = 'All instances'
        general_df = self.apply_timeout_limit(df, metric_col)  # Use the entire DataFrame for the general case
        self.plot_metric_for_problem(general_df, metric_col, maximize, group_by_solver, general_problem_name, figs,
                                     csvs)

        # Step 2: Create individual plots for each problem
        problems = df['problem'].unique()  # Get unique problems
        for problem in problems:
            df_problem = df[df['problem'] == problem]  # Filter DataFrame for the current problem
            df_problem = self.apply_timeout_limit(df_problem, metric_col)
            self.plot_metric_for_problem(df_problem, metric_col, maximize, group_by_solver, problem, figs, csvs)

        return figs, csvs

    def get_strategy_times_best_for_similar_instances_ukp(self, df, non_stats_headers, metric, maximize=True):
        objs_list = []
        pattern_list = []
        objs_elements_vol, pattern_template_vol = get_info_similar_instances_bi_ukp_voptlib()
        objs_list.append(objs_elements_vol)
        pattern_list.append(pattern_template_vol)
        objs_elements_mol, pattern_template_mol = get_info_similar_instances_ukp_moolibrary()
        objs_list.append(objs_elements_mol)
        pattern_list.append(pattern_template_mol)
        table_list = []
        # noinspection PyTypeChecker
        for i in range(len(objs_list)):
            objs_elements = objs_list[i]
            pattern_template = pattern_list[i]
            table = self.get_strategy_times_best_for_similar_instances(objs_elements, pattern_template, df,
                                                                       metric, maximize, non_stats_headers=None)
            table_list.append(table)
        # Concatenate all results into a single DataFrame
        if len(table_list) == 0:
            exit("Something is wrong the results are empty")
        df_best_final = pd.concat(table_list, ignore_index=True)
        return df_best_final

    def get_strategy_times_best_for_similar_instances_nqueens(self, df, non_stats_headers, metric, maximize=True):
        objs_elements, pattern_template = get_info_similar_instances_rcpsp()
        return self.get_strategy_times_best_for_similar_instances(objs_elements, pattern_template, df,
                                                                  metric, maximize, non_stats_headers=None)

    def get_strategy_times_best_for_similar_instances_rcpsp(self, df, non_stats_headers, metric, maximize=True):
        objs_elements, pattern_template = get_info_similar_instances_rcpsp()
        return self.get_strategy_times_best_for_similar_instances(objs_elements, pattern_template, df,
                                                                  metric, maximize, non_stats_headers=None)

    def get_strategy_times_best_for_similar_instances(self, objs_elements, pattern_template, df,
                                                      metric, maximize=True, non_stats_headers=None):

        if non_stats_headers is None:
            non_stats_headers = ["K", "n", "Instances"]
        if "K" not in non_stats_headers or "n" not in non_stats_headers:
            raise Exception("The non_stats_headers must contain the following: 'K', 'n'. Which represent the number "
                            "of objectives and the number of items/queens.")

        stats = ["Times best", "Exhaustive"]
        best_table_rows = []
        for obj, list_elements in objs_elements.items():
            for elements in list_elements:
                # Create the instance name pattern dynamically
                pattern = pattern_template.format(obj=obj, elements=elements)

                # Filter instances based on the pattern
                df_similar_instances = df[df[self.instance].str.contains(pattern, regex=True)]
                if not df_similar_instances.empty:
                    best_count, absolute_best_count, shared_best_count = self.get_best_strategies_absolute_shared(
                        df_similar_instances,
                        metric,
                        maximize,
                        False)
                    # For each strategy, count the number of times exhaustive is true in the similar instances
                    # Count total instances per strategy
                    # total_similar_instances = df_similar_instances.groupby(self.front_strategy).size()
                    total_exhaustive = df_similar_instances.groupby(self.front_strategy)[self.exhaustive].sum()
                    # Merge both counts into a single DataFrame
                    # df_exhaustive = pd.DataFrame({
                    #     self.front_strategy: total_exhaustive.index,
                    #     'Exhaustive': total_exhaustive.astype(str) + "/" + total_similar_instances.astype(str)
                    # }).reset_index(drop=True)

                    df_best = pd.DataFrame({
                        self.front_strategy: list(best_count.keys()),
                        # Front strategy or solver + front based on grouping
                        'Times best': list(best_count.values()),
                    })
                    # Merge `df_best` with `total_exhaustive` on `self.front_strategy`
                    df_combined = pd.merge(df_best, total_exhaustive, on=self.front_strategy, how="outer").fillna("0/0")
                    # non_stats_headers = ["K", "n", "Instances"]
                    df_combined["K"] = obj
                    df_combined["n"] = elements
                    df_combined["Instances"] = len(df_similar_instances[self.instance].unique())
                    # rename exhaustive column
                    # noinspection PyTypeChecker
                    df_combined.rename(columns={self.exhaustive: stats[1]}, inplace=True)

                    best_table_rows.append(df_combined)

        # Concatenate all results into a single DataFrame
        if len(best_table_rows) == 0:
            exit("Something is wrong the results are empty")
        df_best_final = pd.concat(best_table_rows, ignore_index=True)

        return self.create_data_frame_pretty_table_like_disjunctive_paper(df_best_final, stats, non_stats_headers)

    def get_best_strategies_absolute_shared(self, df, metric_col, maximize=True, group_by_solver=False):
        # Group by instance to compare solvers and strategies for each instance
        grouped = df.groupby(self.instance)

        # Initialize dictionaries to store the number of best occurrences for each strategy
        best_count = {}
        absolute_best_count = {}
        shared_best_count = {}

        # Iterate over each instance group
        for instance, group in grouped:
            # Find the best value (max or min) depending on the maximize flag
            if maximize:
                best_value = group[metric_col].max()  # Highest value is better
            else:
                best_value = group[metric_col].min()  # Lowest value is better

            # Find strategies that have this best value
            best_strategies = group[group[metric_col] == best_value][[self.solver_name, self.front_strategy]]

            # Define the key for counting (solver + front or only front)
            for _, row in best_strategies.iterrows():
                if group_by_solver:
                    best_solver_front = (row[self.solver_name], row[self.front_strategy])
                else:
                    best_solver_front = row[self.front_strategy]

                # Increment the count (absolute if only one, shared if multiple)
                if len(best_strategies) == 1:
                    absolute_best_count[best_solver_front] = absolute_best_count.get(best_solver_front, 0) + 1
                else:
                    shared_best_count[best_solver_front] = shared_best_count.get(best_solver_front, 0) + 1

        # Combine absolute and shared best counts
        all_strategies = set(absolute_best_count.keys()).union(shared_best_count.keys())
        for strategy in all_strategies:
            best_count[strategy] = (
                    absolute_best_count.get(strategy, 0) + shared_best_count.get(strategy, 0)
            )
        return best_count, absolute_best_count, shared_best_count

    def plot_metric_for_problem(self, df, metric_col, maximize, group_by_solver, problem_name, figs, csvs):
        """
        Plots the metric for the given problem or for all instances if `problem_name` is "All instances".
        Saves the plot and the dataframe.
        """
        best_count, absolute_best_count, shared_best_count = self.get_best_strategies_absolute_shared(df, metric_col,
                                                                                                      maximize,
                                                                                                      group_by_solver)

        # Convert the best_count dictionaries to pandas DataFrame for saving
        best_df = pd.DataFrame({
            'Strategy': list(best_count.keys()),  # Front strategy or solver + front based on grouping
            'Total Best': list(best_count.values()),
            'Absolute Best': [absolute_best_count.get(strategy, 0) for strategy in best_count],
            'Shared Best': [shared_best_count.get(strategy, 0) for strategy in best_count],
        })

        # Save the DataFrame as a CSV file if required
        if csvs is not None:
            csvs[f"best_considering_{metric_col}_{problem_name}"] = best_df

        # Plot the results using a stacked bar chart
        fig, ax = plt.subplots()
        best_df.set_index('Strategy')[['Absolute Best', 'Shared Best']].plot(
            kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])

        # Add total counts on top of each bar
        for idx, row in best_df.iterrows():
            ax.text(idx, row['Total Best'] + 0.1, str(int(row['Total Best'])), ha='center')

        # Add labels and title
        ax.set_ylabel(f"Number of times best : {metric_col}")
        ax.set_xlabel('Algorithms')
        ax.grid(True, axis='y')
        ax.grid(False, axis='x')
        ax.set_title(f"{problem_name}. Total instances: {len(df['instance'].unique())}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if best_df['Shared Best'].sum() > 0:
            # Add legend for absolute and shared best
            ax.legend(['Absolute Best', 'Shared Best'], title='Best Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.get_legend().remove()

        # Save the figure in figs if provided
        if figs is not None:
            figs[f"best_considering_{metric_col}_{problem_name}"] = fig

    def plot_metric_by_instance_problem(self, df, metric_col, maximize, figs=None, csvs=None, group_by_solver=False):
        """
        Plot the raw metric values for each problem and instance.
        """
        df = self.apply_timeout_limit(df, metric_col)
        problems = df[self.problem].unique()

        # Loop through each problem and create a plot for it
        for problem in problems:
            df_problem = df[df[self.problem] == problem]  # Filter DataFrame for the current problem

            # Use the raw metric values for plotting
            metric_data = df_problem[metric_col]

            # Call the helper function to create the plot
            figs, csvs = self._plot_instance_problem_helper(df_problem, metric_col, metric_data, maximize, figs, csvs,
                                                            problem,
                                                            group_by_solver)

        return figs, csvs

    def plot_metric_by_instance_problem_normalized(self, df, metric_col, maximize, figs=None, csvs=None,
                                                   group_by_solver=False):
        """
        Plot the normalized metric values for each problem and instance.
        Each value is normalized relative to the best value in the instance.
        """
        df = self.apply_timeout_limit(df, metric_col)
        problems = df[self.problem].unique()

        # Loop through each problem and create a plot for it
        for problem in problems:
            df_problem = df[df[self.problem] == problem]  # Filter DataFrame for the current problem
            metric_data = df_problem.copy()  # Make a copy to avoid modifying the original df

            # Normalize the values per instance
            for instance in df_problem[self.instance].unique():
                df_instance = df_problem[df_problem[self.instance] == instance]  # Filter for each instance
                best_value = df_instance[metric_col].max() if maximize else df_instance[metric_col].min()

                # Apply normalization
                metric_data.loc[df_instance.index, 'normalized'] = df_instance[metric_col] / best_value

            # Use the normalized values for plotting
            metric_data_values = metric_data['normalized']

            # Call the helper function to create the plot
            figs, csvs = self._plot_instance_problem_helper(
                metric_data, 'normalized', metric_data_values, maximize, figs, csvs, problem, group_by_solver,
                is_normalized=True
            )

        return figs, csvs

    def apply_timeout_limit(self, df, metric_col):
        """
        Apply timeout limit to the specified metric column if it's one of the time-related columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the metrics.
        - metric_col (str): The name of the metric column to be checked.
        """
        if metric_col in [self.time, self.time_solver_sec, "sum_solutions_resolution_time(s)"]:
            # Convert the metric column and the timeout column to numeric, coerce invalid values to NaN
            df.loc[:, metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            df.loc[:, self.timeout] = pd.to_numeric(df[self.timeout], errors='coerce')
            # Apply the timeout limit: If the metric value exceeds the timeout, set it to the timeout
            df.loc[df[metric_col] > df[self.timeout], metric_col] = df[self.timeout]
        return df

    def _plot_instance_problem_helper(self, df, metric_col, metric_data, maximize, figs, csvs, problem_name,
                                      group_by_solver, is_normalized=False):
        """
        Helper function to plot metric values for each problem and instance using Seaborn.

        Parameters:
        - df: Filtered DataFrame for the current problem.
        - metric_col: Name of the column containing the metric.
        - metric_data: The values to be plotted (either raw or normalized).
        - figs: Dictionary to store figures.
        - problem_name: Name of the current problem (for the title).
        - group_by_solver: If True, use solver + strategy combination on the plot.
        - is_normalized: If True, indicates that the values are normalized and extra rows (Average, Std Dev) should be added.
        """

        if figs is None:
            figs = {}

        if csvs is None:
            csvs = {}

        # Copy the DataFrame to avoid the SettingWithCopyWarning
        df1 = df.copy()

        # Handle missing values (if any) in problem or instance columns using .loc
        df1.loc[:, self.problem] = df1[self.problem].fillna('Unknown Problem')
        df1.loc[:, self.instance] = df1[self.instance].fillna('Unknown Instance')

        # Ensure problem and instance are strings for concatenation
        df1.loc[:, 'problem_instance'] = df1[self.instance].astype(str)

        # If grouping by solver and strategy, create a combined label using .loc
        if group_by_solver:
            df1.loc[:, 'strategy_combination'] = df1[self.solver_name] + '-' + df1[self.front_strategy]
            strategy_col = 'strategy_combination'
        else:
            strategy_col = self.front_strategy

        # Create a DataFrame for metrics by instance and strategy
        metric_table = df1.pivot_table(index=self.instance, columns=strategy_col, values=metric_col)

        # Initialize a list to track how many times each strategy is the best
        best_counts = {strategy: 0 for strategy in metric_table.columns}

        # For each instance, find the strategy with the best value (max or min depending on context)
        for instance in metric_table.index:
            row = metric_table.loc[instance]

            # # Drop any NaN values before finding the best strategy
            # row_clean = row.dropna()
            # if row_clean.empty:
            #     # If the row is empty after dropping NaNs, skip this instance
            #     continue

            if maximize:  # Assuming maximization
                best_value = row.max()
                best_strategies = row[row == best_value].index  # All strategies with the best value
            else:  # Minimization
                best_value = row.min()
                best_strategies = row[row == best_value].index  # All strategies with the best value

            # Increment the count for each best strategy
            for strategy in best_strategies:
                best_counts[strategy] += 1

        # Add Average and Standard Deviation rows only if it's a normalized plot
        if is_normalized:
            metric_table.loc["Average"] = metric_table.mean()
            metric_table.loc["Std Dev"] = metric_table.std()

        # Append the row for best strategy counts
        metric_table.loc["X Best"] = pd.Series(best_counts)

        # Apply the table printing logic
        print(metric_table)  # Display the table for inspection
        # metric_table = self._print_metric_table(metric_table, maximize) // this print the formatted table only in the online Jupyter Notebook

        # Now include the plotting logic

        # Create a color palette for the strategies
        palette = sns.color_palette("husl", len(df1[strategy_col].unique()))

        # Dynamically adjust figure height based on the number of instances
        num_instances = df1['problem_instance'].nunique()
        base_height = 5  # Minimum height
        height_per_instance = 0.25  # Adjust this value to control bar spacing
        fig_height = max(base_height, height_per_instance * num_instances)  # Calculate the figure height

        # Set up the plot using Seaborn with dynamic height
        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.set_theme(style="whitegrid")

        # Plot using Seaborn barplot
        sns.barplot(x=metric_data, y='problem_instance', hue=strategy_col, data=df1, palette=palette, orient='h', ax=ax)

        # Set titles and labels
        ax.set_title(f'{metric_col} by front strategy for each problem-instance ({problem_name})')
        ax.set_xlabel(metric_col)
        ax.set_ylabel('Instances')

        # Add a legend
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set the background to white and remove the grid lines
        sns.despine(left=True, bottom=True)
        ax.grid(False)

        # Adjust layout
        plt.tight_layout()

        # Save the figure in the figs dictionary
        figs[f"{problem_name}_metric_{metric_col}"] = fig
        # Add to csvs for later export
        csvs[f"{problem_name}_metric_{metric_col}"] = metric_table

        return figs, csvs

    def _print_metric_table(self, metric_table, maximize=True):
        """
        Helper function to print the metric table with best values highlighted in bold (for Jupyter Notebook).

        Parameters:
        - metric_table: The DataFrame containing metric values for each instance and strategy.
        - maximize: If True, the best value is the maximum. If False, the best value is the minimum.
        - is_normalized: If True, adds extra rows (Average, Std Dev).
        """

        # Style the DataFrame to highlight the best value in each row
        def highlight_best(s):
            if s.name == "X Best":
                return ['font-weight: bold' if v == s.max() else '' for v in
                        s]  # Always highlight the max in X Best row
            else:
                is_best: bool = (s == s.max()) if maximize else (s == s.min())
                return ['font-weight: bold' if v else '' for v in is_best]

        # Apply the styling
        styled_table = metric_table.style.apply(highlight_best, axis=1)

        # Print the table to Jupyter Notebook
        display(styled_table)

        return metric_table  # Return the table for further use or saving

    # Plot certain instances to check the number of solutions in time and the hypervolume
    def plot_solutions_in_time(self, df, instances_list, figs):
        generic_key = "solutions_in_time"

        # Step 1: Combine solver_name and front_strategy into a new column
        # df.loc['solver_strategy'] = df.loc[self.solver_name] + ' ' + df.loc[self.front_strategy]
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        # Map solver_strategy combinations to y-values
        unique_combinations = df_copy['solver_strategy'].unique()
        combination_to_y = {comb: i for i, comb in enumerate(unique_combinations)}

        for instances in instances_list:
            filtered_df = df_copy[df_copy['problem_instance'] == instances]

            # Set up the plot
            fig = plt.figure(figsize=(10, 5))
            y_ticks_labels = []

            # Step 3: Plot data
            for combination in unique_combinations:
                # check if the combination is in the filtered_df
                if combination not in filtered_df['solver_strategy'].values:
                    y_ticks_labels.append(f"{combination} - 0")
                    continue
                # Directly access the row for the current combination
                row = filtered_df[filtered_df['solver_strategy'] == combination].iloc[0]

                # Process the solutions time list and pareto times
                x = [float(time) for time in row[self.solutions_in_time].replace('[', '').replace(']', '').split(',')]
                y = [combination_to_y[combination]] * len(x)  # Use the mapped y-value for this combination

                # Update y-tick labels to include the number of Pareto front points
                pareto_front_str = row[self.pareto_front]
                pareto_front_str = pareto_front_str.replace(' ', '')
                pareto_front_count = len(pareto_front_str.split('],['))
                y_ticks_labels.append(f"{combination} - {pareto_front_count}")
                plt.scatter(x, y, facecolors='none', edgecolors='b')  # Plot all points for this combination
                # Step 4: Highlight special points
                pareto_times = row[self.pareto_solutions_time_list]
                pareto_times = pareto_times.replace('[', '').replace(']', '').split(',')
                pareto_times = [float(time) for time in pareto_times]
                special_x = [time for time in x if time not in pareto_times]
                special_y = [combination_to_y[combination]] * len(special_x)
                plt.scatter(special_x, special_y, facecolors='none', edgecolors='r')  # Filled points for special data

            # Step 5: Customize y-axis labels
            plt.yticks(range(len(unique_combinations)), y_ticks_labels)

            plt.xlabel('Solution time')
            plt.ylabel('Solver and strategy combination')
            plt.title(f'Comparison of solution times by solver and strategy for problem-instance {instances}')

            plt.xticks([])

            # Show legend with bbox_to_anchor outside the plot area
            all_points = Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=10, label='All points', markerfacecolor='none')
            not_pareto_front = Line2D([], [], color='red', marker='o', linestyle='None',
                                      markersize=10, label='Not Pareto front points', markerfacecolor='none')

            plt.legend(handles=[all_points, not_pareto_front], title="Legend", bbox_to_anchor=(1.05, 1),
                       loc='upper left')

            plt.show()

            fig_key = f"{generic_key}_{instances}"
            figs[fig_key] = fig
        return figs

    def plot_best_strategy_count_vs_time(self, df, time_points, figs):
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        problem_list = df_copy[self.problem].unique()
        all_strategy_counts_over_time = {problem: {strategy: np.zeros(len(time_points)) for strategy in
                                                   df_copy['solver_strategy'].unique()} for problem in problem_list}
        # Dictionary to track how many instances each strategy is the best over time
        for problem_to_process in problem_list:
            instances_list = df_copy[df_copy[self.problem] == problem_to_process][self.instance].unique()
            for instance_to_process in instances_list:
                self.get_best_strategy_count_vs_time_for_instance(df_copy, instance_to_process, time_points,
                                                                  all_strategy_counts_over_time[problem_to_process])
        all_strategy_counts_over_time["all_problems"] = {strategy: np.zeros(len(time_points)) for strategy in
                                                         df_copy['solver_strategy'].unique()}
        for problem in problem_list:
            for strategy in df_copy['solver_strategy'].unique():
                all_strategy_counts_over_time["all_problems"][strategy] += all_strategy_counts_over_time[problem][
                    strategy]

        generic_key = "best_strategy_count_vs_time"
        for problem in all_strategy_counts_over_time.keys():
            strategy_counts_over_time = all_strategy_counts_over_time[problem]

            fig, ax = plt.subplots(figsize=(10, 5))
            for strategy, counts in strategy_counts_over_time.items():
                ax.plot(time_points, counts, label=strategy)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Number of instances where strategy has best Hypervolume')
            ax.set_title(f"Count best strategy Hypervolume vs Time - {problem}")
            ax.legend(title='Solver strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

            fig_key = f"{generic_key}-{problem}"
            figs[fig_key] = fig

        return figs

    def get_best_strategy_count_vs_time_for_instance(self, df_with_instance, instance, time_points,
                                                     strategy_counts_over_time):
        filtered_df = df_with_instance[df_with_instance[self.instance] == instance]

        last_time_index_per_solver_strategy = {}
        x_all_times_per_strategy = {}
        hypervolumes_evolution_per_strategy = {}
        for combination in filtered_df['solver_strategy'].unique():
            last_time_index_per_solver_strategy[combination] = 0
            row = filtered_df[filtered_df['solver_strategy'] == combination].iloc[0]
            x_all_times_per_strategy[combination], hypervolumes_evolution_per_strategy[combination] = (
                self.get_hypervolume_vs_time_values_from_table_row(row))

        best_hypervolume = 0
        for time_idx, t in enumerate(time_points):
            best_strategies = []
            for combination in filtered_df['solver_strategy'].unique():
                x_all_times = x_all_times_per_strategy[combination]
                hypervolumes = hypervolumes_evolution_per_strategy[combination]

                # If no valid data, skip this strategy
                if len(x_all_times) == 0 or len(hypervolumes) == 0:
                    continue

                if ((last_time_index_per_solver_strategy[combination] == len(x_all_times) - 1) and len(x_all_times)
                        > 1):
                    closest_time_idx = len(x_all_times) - 1
                else:
                    closest_time_idx = find_closest_time_index_to_time_t(x_all_times,
                                                                         last_time_index_per_solver_strategy[
                                                                             combination], t)
                if closest_time_idx > 0:
                    hv_at_time = hypervolumes[closest_time_idx]
                    last_time_index_per_solver_strategy[combination] = closest_time_idx
                else:
                    hv_at_time = 0

                if hv_at_time > best_hypervolume:
                    best_hypervolume = hv_at_time
                    best_strategies = [combination]
                elif hv_at_time == best_hypervolume:
                    best_strategies.append(combination)

            # Increment count for each best strategy at this time point
            for strategy in best_strategies:
                strategy_counts_over_time[strategy][time_idx] += 1

    def get_hypervolume_vs_time_values_from_table_row(self, row):
        # Extract hypervolumes and times, handling "Not available." or invalid entries
        x_all_times = []
        hypervolumes = []

        # Convert time and hypervolumes, skipping invalid entries
        for time_str, hv_str in zip(
                row[self.solutions_in_time].replace('[', '').replace(']', '').split(','),
                row[self.hypervolume_evolution].replace('[', '').replace(']', '').split(',')):

            time_str = time_str.strip()
            hv_str = hv_str.strip()

            # Skip if time or hypervolume is "Not available." or other invalid strings
            if time_str.lower() == "not available." or hv_str.lower() == "not available.":
                continue

            try:
                x_all_times.append(float(time_str))
                hypervolumes.append(float(hv_str))
            except ValueError:
                # Skip any invalid values that can't be converted to float
                continue

        x_all_times = np.array(x_all_times)
        hypervolumes = np.array(hypervolumes)
        return x_all_times, hypervolumes

    def plot_hypervolume_vs_time(self, df, instances_list, figs, zoom_in_y=False):
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        generic_key = "hypervolume_vs_time"
        for instance_to_process in instances_list:
            filtered_df = df_copy[df_copy[self.instance] == instance_to_process]

            # get the problem name
            problem_for_instance_list = filtered_df[self.problem].unique()
            if len(problem_for_instance_list) != 1:
                raise ValueError(f"More than one problem for instance {instance_to_process} or no problem at all")
            problem_for_instance = problem_for_instance_list[0]

            name_plot = f"{problem_for_instance}-{instance_to_process}"
            # Plot 1: Regular Scale
            fig, ax = plt.subplots(figsize=(10, 5))
            self.plot_hypervolume_evolution(filtered_df, ax, name_plot)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Hypervolume')
            ax.legend(title='Solver Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            fig_key = f"{generic_key}_{name_plot}"
            figs[fig_key] = fig

            if zoom_in_y:
                # Plot 2: Logarithmic Scale
                fig, ax = plt.subplots(figsize=(10, 5))
                self.plot_hypervolume_evolution(filtered_df, ax, name_plot, zoom_in_y=True)
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Log of hypervolume')
                ax.legend(title='Solver strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.show()
                fig_key_zoom = f"{generic_key}_log_{name_plot}"
                figs[fig_key_zoom] = fig

        return figs

    def plot_fronts(self, df, instances_list, figs,
                    plot_reference_front=True):
        """
        For each instance, plot all strategies' fronts in the same figure.
        Assumes df is already in the final strategy naming you want to show.

        - Uses Strategies order/markers/colors if self.strategies exists and provides them.
        - Legend: '<strategy> - <n> points' plus '*' if exhaustive.
        - plot_reference_front:
            if an exhaustive front exists -> use it as reference,
            else -> build joint non-dominated union and plot as hollow black circles.
        """

        import ast
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        def _parse_front(v):
            if pd.isna(v):
                return None
            if isinstance(v, (list, tuple)):
                return list(v)
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") and s.endswith("}"):
                    s = s.replace("{", "[").replace("}", "]")
                try:
                    return ast.literal_eval(s)
                except (SyntaxError, ValueError):
                    return None
            return None

        # order + style from Strategies (labels)
        if hasattr(self, "strategies") and self.strategies is not None:
            strategies_order = [s for s in self.strategies.strategies_better_name
                                if s in set(df[self.front_strategy].unique())]
            colors = self.strategies.colors
            markers = self.strategies.markers
        else:
            strategies_order = list(df[self.front_strategy].unique())
            colors = getattr(self, "strategy_colors", {})
            markers = getattr(self, "strategy_markers", {})

        for instance_to_process in instances_list:
            df_inst = df[df[self.instance] == instance_to_process]
            if df_inst.empty:
                continue

            # problem name (for key/title)
            problem_vals = df_inst[self.problem].dropna().unique()
            if len(problem_vals) != 1:
                raise ValueError(f"More than one problem for instance {instance_to_process} or no problem at all")
            problem_name = problem_vals[0]
            name_plot = f"{problem_name}-{instance_to_process}"

            # determine dimension from first parsable front
            dim = None
            for _, r in df_inst.iterrows():
                pts = _parse_front(r.get(self.pareto_front, None))
                if pts:
                    dim = len(pts[0])
                    break
            if dim is None or dim > 3 or dim < 2:
                continue

            if dim == 3:
                # fig = plt.figure(figsize=(12, 7))
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))

            # determine maximize/minimize (only needed for joint front)
            maximize = False
            if plot_reference_front:
                try:
                    row0 = df_inst.iloc[0]
                    ref_point = row0.get("reference_point", None)
                    front0 = _parse_front(row0.get(self.pareto_front, None))
                    if isinstance(ref_point, str):
                        ref_point = ast.literal_eval(ref_point)
                    if front0:
                        maximize = is_maximization_problem(front0, ref_point)
                except Exception:
                    maximize = False

            # plot each strategy
            at_least_one_strategy_exhaustive = False
            for strategy in strategies_order:
                df_s = df_inst[df_inst[self.front_strategy] == strategy]
                if df_s.empty:
                    continue

                pts_all = []
                exhaustive_flag = False

                for _, r in df_s.iterrows():
                    pts = _parse_front(r.get(self.pareto_front, None))
                    if pts:
                        pts_all.extend(pts)
                    if bool(r.get(self.exhaustive, False)):
                        exhaustive_flag = True
                        at_least_one_strategy_exhaustive = True

                if not pts_all:
                    continue

                # unique points
                pts_all = list({tuple(p) for p in pts_all})
                pts_arr = np.array(pts_all)

                star = "*" if exhaustive_flag else ""
                label = f"{strategy} - {len(pts_all)} points{star}"

                c = colors.get(strategy, None)
                m = markers.get(strategy, "o")

                if dim == 3:
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                               label=label,
                               marker=m,
                               s=28,
                               facecolors="none",
                               edgecolors=c if c is not None else None,
                               linewidths=1.0)
                else:
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                               label=label,
                               marker=m,
                               s=28,
                               facecolors="none",
                               edgecolors=c if c is not None else None,
                               linewidths=1.0)

            # reference / joint front
            # joint front (computed from all points), plotted last so it ends in the legend
            if plot_reference_front and not at_least_one_strategy_exhaustive:
                all_points = set()
                for _, r in df_inst.iterrows():
                    pts = _parse_front(r.get(self.pareto_front, None))
                    if not pts:
                        continue
                    for p in pts:
                        all_points.add(tuple(p))

                all_points = list(all_points)
                ref_pts = None
                if all_points:
                    ref_pts = MoAnalysis.remove_dominated_points(all_points, maximize)

                if ref_pts:
                    ref_pts = list({tuple(p) for p in ref_pts})
                    ref_arr = np.array(ref_pts)
                    ref_label = f"Joint front - {len(ref_pts)} points"

                    if dim == 3:
                        ax.scatter(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2],
                                   label=ref_label,
                                   marker="o",
                                   s=90,  # bigger than others
                                   facecolors="none",  # unfilled
                                   edgecolors="black",  # black contour
                                   linewidths=1.4)
                    else:
                        ax.scatter(ref_arr[:, 0], ref_arr[:, 1],
                                   label=ref_label,
                                   marker="o",
                                   s=90,
                                   facecolors="none",
                                   edgecolors="black",
                                   linewidths=1.4)

            # labels / legend / grid
            if dim == 3:
                ax.set_xlabel("Objective 1")
                ax.set_ylabel("Objective 2")
                ax.set_zlabel("Objective 3")
            else:
                ax.set_xlabel("Objective 1")
                ax.set_ylabel("Objective 2")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

            # ax.set_title(f"Pareto fronts for instance {name_plot}", fontsize=16)
            ax.set_title(f"{name_plot}", fontsize=16)

            ax.legend(bbox_to_anchor=(1.05, 1),
                      loc="upper left",
                      frameon=True,
                      fontsize=9,  # smaller, since you add #points/*
                      markerscale=0.9,
                      labelspacing=0.25,
                      handletextpad=0.5,
                      borderaxespad=0.3)

            if dim == 3:
                fig.subplots_adjust(left=0, right=0.9, top=0.90, bottom=0.05)
            else:
                fig.subplots_adjust(right=0.75)
            plt.show()
            figs[f"{name_plot}"] = fig

        return figs

    def check_fronts_are_equal(self, data):
        """
        Check if all strategies have the same Pareto front for each instance.
        Return a boolean and, if false, the strategies and instance with discrepancies.
        """
        inconsistent_instances = []
        # duplicate_issues = []

        for instance, group in data.groupby(self.instance):
            # fronts = group[self.pareto_front].apply(eval).apply(lambda x: sorted(map(tuple, x)))
            fronts = group[self.pareto_front].apply(eval).apply(lambda x: tuple(sorted(map(tuple, x))))

            # Check for duplicate elements in each front

            has_duplicates = group[self.pareto_front].apply(
                lambda x: len(x) != len(set(map(tuple, eval(x))))
            )

            # if has_duplicates.any():  # At least one strategy has duplicates
            #     if not has_duplicates.all():  # Not all strategies have duplicates
            #         duplicate_issues.append(instance)
            #         continue  # Skip further checks for this instance
            #     else:
            #         # All strategies have duplicates; mark as a potential issue
            #         duplicate_issues.append(instance)

            # Check if all fronts are identical (including duplicates)
            if len(fronts.unique()) > 1:  # More than one unique front
                if group[self.exhaustive].all():  # Only consider if all strategies are exhaustive
                    inconsistent_instances.append(instance)

        if inconsistent_instances:
            print(f"Instances with differing Pareto fronts despite being exhaustive: {inconsistent_instances}")
            return False
        # if duplicate_issues:
        #     print(f"Instances with inconsistent duplicates across strategies: {duplicate_issues}")
        #     return False
        return True

    @staticmethod
    def check_points_in_front_are_not_dominated(front, maximize=True, verbose=True, return_indices=False):
        dominated_points = []
        id_dominated_points = []

        for i, point in enumerate(front):
            for j, other_point_in_front in enumerate(front):
                if i != j and is_dominated(point, other_point_in_front, maximize):
                    if verbose:
                        print(f"Point {point} is dominated by {other_point_in_front}")
                    dominated_points.append(point)
                    id_dominated_points.append(i)
                    break
        if return_indices:
            return dominated_points, id_dominated_points
        else:
            return dominated_points

    @staticmethod
    def remove_dominated_points(front, maximize=None):
        # remove the dominated points from the front using the id_dominated_points
        _, id_dominated_points = MoAnalysis.check_points_in_front_are_not_dominated(
            front, maximize=maximize, verbose=False, return_indices=True
        )
        # return [p for i, p in enumerate(front) if p not in dominated_points]
        return [pt for i, pt in enumerate(front) if i not in id_dominated_points]

    def plot_specific_front(self, df, instance_to_process, figs, margin=0.05):
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        generic_key = "pareto_front"
        filtered_df = df_copy[df_copy[self.instance] == instance_to_process]

        # get the problem name
        problem_for_instance_list = filtered_df[self.problem].unique()
        if len(problem_for_instance_list) != 1:
            raise ValueError(f"More than one problem for instance {instance_to_process} or no problem at all")
        problem_for_instance = problem_for_instance_list[0]

        # Initialize lists to find global min and max for x and y
        all_x = []
        all_y = []

        for _, row in filtered_df.iterrows():
            pareto_front_str = row[self.pareto_front]
            if pareto_front_str.startswith('{') and pareto_front_str.endswith('}'):
                pareto_front_str = pareto_front_str.replace('{', '[').replace('}', ']')

            try:
                pareto_front = ast.literal_eval(pareto_front_str)
                pareto_front = np.array(pareto_front)
                all_x.extend(pareto_front[:, 0])
                all_y.extend(pareto_front[:, 1])
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing pareto_front for {row['solver_strategy']}: {e}")

        # Determine global min and max for x and y
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        for _, row in filtered_df.iterrows():
            solver_strategy = row['solver_strategy']
            pareto_front_str = row[self.pareto_front]

            if pareto_front_str.startswith('{') and pareto_front_str.endswith('}'):
                pareto_front_str = pareto_front_str.replace('{', '[').replace('}', ']')

            try:
                pareto_front = ast.literal_eval(pareto_front_str)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing pareto_front for {solver_strategy}: {e}")
                continue

            exhaustive_star = ''
            if row[self.exhaustive]:
                exhaustive_star = '*'
            label = f"{solver_strategy} - {len(pareto_front)} points{exhaustive_star}"

            fig, ax = plt.subplots(figsize=(10, 5))
            pareto_front = np.array(pareto_front)
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], label=label)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_title(f'Pareto front for {solver_strategy} in instance {problem_for_instance}-{instance_to_process}')
            ax.legend(title='Solver strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

            fig_key = f"{generic_key}_{problem_for_instance}_{instance_to_process}_{solver_strategy.replace(' ', '_')}"
            figs[fig_key] = fig

        return figs

    def plot_hypervolume_evolution(self, filtered_df, ax, instance_to_process, consider_only_pareto=False,
                                   zoom_in_y=False):
        unique_combinations = filtered_df['solver_strategy'].unique()
        ax.set_title(f'Instance: {instance_to_process}')
        all_y_values = []
        for idx, combination in enumerate(unique_combinations):
            if combination in filtered_df['solver_strategy'].values:
                # Get the row corresponding to the combination
                row = filtered_df[filtered_df['solver_strategy'] == combination].iloc[0]

                # Process the solutions time list and pareto times
                x_all_times = [float(time) for time in
                               row[self.solutions_in_time].replace('[', '').replace(']', '').split(',')]
                all_solutions_string = row[self.all_solutions]
                # check if it has the string "Unfeasible"
                if "Unfeasible" in all_solutions_string:
                    # remove the "Unfeasible" string from the all_solutions_string
                    all_solutions_string = all_solutions_string.replace(',Unfeasible', '')
                    all_solutions_string = all_solutions_string.replace('Unfeasible,', '')
                all_solutions = all_solutions_string.replace(' ', '').split('],[')
                x_all_times = [x_all_times[i] for i in range(len(all_solutions)) if
                               "Unfeasible" not in all_solutions[i]]
                hypervolumes = [float(hv) for hv in
                                row[self.hypervolume_evolution].replace('[', '').replace(']', '').split(',')]

                y = hypervolumes
                x = x_all_times
                if consider_only_pareto:
                    x_pareto = [float(time) for time in
                                row[self.pareto_solutions_time_list].replace('[', '').replace(']', '').split(',')]
                    x_pareto_id = [x_all_times.index(time) for time in x_pareto]
                    hypervolumes_pareto = [hypervolumes[i] for i in x_pareto_id]
                    y = hypervolumes_pareto
                    x = x_pareto

                # for each y value, add it to the all_y_values list
                all_y_values.extend(y)
                # Plot the data
                exhaustive_star = ''
                if row[self.exhaustive]:
                    exhaustive_star = '*'
                label = f"{combination} - {len(x_all_times)} points{exhaustive_star}"
                ax.plot(x_all_times, y, marker='o', linestyle='-', label=label)

        if zoom_in_y:
            median_y = np.median(all_y_values)
            max_y = max(all_y_values)
            # reference value can be the max or the median
            reference_value = max_y
            # based on the reference value, set the y-axis limits
            ax.set_ylim(reference_value * 0.99, reference_value * 1.01)
            plt.draw()

    # save the data as pdf and csv
    @staticmethod
    def save_pictures_and_tables(figs, folder_name, csvs):
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # save df_total_best_avg_score to a csv file
        for name, csv in csvs:
            csv.to_csv(f'{folder_name}/{name}.csv', sep=';')

        for key, fig in figs.items():
            image_path = os.path.join(folder_name, f"{key}.pdf")
            fig.tight_layout()
            fig.savefig(image_path)
            plt.close(fig)

        print(f"All images have been saved in the '{folder_name}' folder.")

    def add_extra_fields_for_best_analysis(self, df):
        # add necessary columns to the df, that evaluate the performance of the front strategy for each combination of
        # problem, instance and solver

        # 1. TimeForScore: If this.exaustive is True, then the time is the time is self.time_solver_sec, otherwise it is
        # equal to the value of the colum self.timeout
        df_extrafields = df.copy()
        df_extrafields[Cols.TIME_FOR_TIME_SCORE] = df_extrafields.apply(
            lambda x: x[self.time_solver_sec] if x[self.exhaustive] else x[self.timeout], axis=1)

        # 2. HypervolumeScore: The hypervolume score is calculated as the ratio between the hypervolume of the front
        # generated by the front strategy and the best hypervolume for the same problem, instance and solver
        df_extrafields[Cols.HV_SCORE] = df_extrafields.groupby([self.problem, self.instance, self.solver_name])[
            self.hypervolume].transform(self.calculate_hypervolume_score)

        # 3. TimeScore: The time score is calculated as the ratio between the value of TimeForScore and the best
        # TimeForScore for the same problem, instance and solver
        df_extrafields[Cols.TIME_SCORE_FOR_LEX_SCORE] = df_extrafields.groupby(
            [self.problem, self.instance, self.solver_name]
        )[Cols.TIME_FOR_TIME_SCORE].transform(self.calculate_time_score_for_lex_score)

        # 4. LexScore: There are 4 cases:
        # 1. All have exhaustive = False. In this case, the LexScore is equal to the HV_SCORE
        # 2. All have exhaustive = True. In this case, the LexScore is equal to the TIME_SCORE_FOR_LEX_SCORE
        # 3. Only one has exhaustive = True. In this case, the LexScore is equal to the HV_SCORE if the second highest
        # HV_SCORE is lower than the TIME_SCORE_FOR_LEX_SCORE, otherwise it is equal to the TIME_SCORE_FOR_LEX_SCORE
        # 4. More than one has exhaustive = True. In this case, the LexScore is equal to the TIME_SCORE_FOR_LEX_SCORE

        # df_extrafields[Cols.LEX_SCORE] = df_extrafields.apply(
        #     lambda x: x[Cols.TIME_SCORE_FOR_LEX_SCORE] if x[self.exhaustive] else x[Cols.HV_SCORE], axis=1)
        df_extrafields[Cols.LEX_SCORE] = 0.0  # Initialize with default value
        grouped = df_extrafields.groupby([self.problem, self.instance, self.solver_name])

        for name, group in grouped:
            exhaustive_count = group[self.exhaustive].sum()
            if exhaustive_count == 0:  # All have exhaustive = False
                df_extrafields.loc[group.index, Cols.LEX_SCORE] = group[Cols.HV_SCORE].astype(float)
            elif exhaustive_count == len(group):  # All have exhaustive = True
                df_extrafields.loc[group.index, Cols.LEX_SCORE] = group[Cols.TIME_SCORE_FOR_LEX_SCORE].astype(float)
            elif exhaustive_count == 1:  # Only one has exhaustive = True
                # Find the row with the second highest HV_SCORE
                second_highest_hv_score = group.nlargest(2, Cols.HV_SCORE).iloc[-1]
                # Determine which score to use for the whole group
                if second_highest_hv_score[Cols.HV_SCORE] < second_highest_hv_score[Cols.TIME_SCORE_FOR_LEX_SCORE]:
                    df_extrafields.loc[group.index, Cols.LEX_SCORE] = group[Cols.HV_SCORE].astype(float)
                else:
                    df_extrafields.loc[group.index, Cols.LEX_SCORE] = group[Cols.TIME_SCORE_FOR_LEX_SCORE].astype(float)
            else:  # More than one has exhaustive = True
                df_extrafields.loc[group.index, Cols.LEX_SCORE] = group[Cols.TIME_SCORE_FOR_LEX_SCORE].astype(float)

        return df_extrafields

    def print_general_figs_and_tables(self, df_original, figs):
        df = self.add_extra_fields_for_best_analysis(df_original)

        strategy_str = self.get_unique_values_chained(df, self.front_strategy)
        problems_front_strategy_str = self.get_problems_front_strategy_str_for_fig_name(df)

        df_total_best_avg_score, fig = self.plot_lexicographic_best(df)
        figs[f"0-Lex_HV_time_score_best_{strategy_str}"] = fig

        fig = self.plot_lexicographic_score_best_average(df_total_best_avg_score)
        figs[f"01-Lex_HV_time_score_avg_{strategy_str}"] = fig

        fig = self.plot_lexicographic_hv_time_score_per_instance(df)
        figs[f"02-Lex_HV_time_score_{problems_front_strategy_str}"] = fig

        fig = self.plot_hypervolume_score_per_instance(df)
        figs[f"1-HV_score_{problems_front_strategy_str}"] = fig

        df_total_best_hv_avg_score, fig = self.plot_hypervolume_best(df)
        figs[f"2-HV_score_best_{strategy_str}"] = fig

        fig = self.plot_hypervolume_best_average(df_total_best_hv_avg_score)
        figs[f"3-HV_score_avg_{strategy_str}"] = fig

        df_time_number_solutions = self.get_time_number_solutions(df)
        fig = self.plot_strategy_time_score_to_get_the_front(df_time_number_solutions)
        figs[f"4-Time_score_{problems_front_strategy_str}"] = fig

        return figs, df_total_best_avg_score

    def plot_solutions_points_in_time(self, df, figs):
        # define, the instances to plot, by default all instances are plotted
        df_copy = df.copy()
        df_copy['problem_instance'] = df_copy[self.problem] + ' - ' + df_copy[self.instance]
        instances_list = df_copy['problem_instance'].unique()
        # instances_list = ['paris_30']
        figs = self.plot_solutions_in_time(df_copy, instances_list, figs)
        return figs

    def get_problems_front_strategy_str_for_fig_name(self, df):
        # get all the unique values of all self.problem and self.front_strategy from the df
        problems_str = self.get_unique_values_chained(df, self.problem)
        front_strategies_str = self.get_unique_values_chained(df, self.front_strategy)
        return f"{problems_str}_{front_strategies_str}"

    def get_unique_values_chained(self, df, df_col):
        unique_values = df[df_col].unique()
        unique_values_str = ''
        for value in unique_values:
            unique_values_str += value + '--'
        unique_values_str = unique_values_str[:-2]
        return unique_values_str

    def normalize_statistics_for_metrics(self, data, metrics):
        """
        Normalize the statistics for the given metrics and calculate the mean and std for each strategy.
        Return a normalized dataframe.
        """
        results = []
        normalized_metric_names = []
        for metric in metrics:
            normalized_metric_names.append(metric.name)

        for instance, group in data.groupby(self.instance):
            normalized_group = group.copy()
            for metric in metrics:
                if metric.minimization:
                    best_value = group[metric.name].min()
                else:
                    best_value = group[metric.name].max()
                normalized_group[metric.name] = (group[metric.name] / best_value).fillna(1)

            normalized_group = normalized_group[
                [self.instance, self.problem, self.front_strategy] + normalized_metric_names]
            results.append(normalized_group)
        # Combine all normalized groups
        normalized_data = pd.concat(results)
        summary = normalized_data.groupby(self.front_strategy).agg(
            {col: ['mean', 'std'] for col in normalized_metric_names}
        )

        return normalized_data, summary

    def calculate_percentage_best_for_metrics(self, data, metrics):
        """
        Calculate the percentage of instances where each strategy was the best for each metric.
        Return a summary dataframe.
        """
        best_counts = {}

        for metric in metrics:
            if metric.minimization:
                best_strategy = data.groupby(self.instance).apply(
                    lambda x: x.loc[x[metric.name].idxmin(), self.front_strategy]
                )
            else:
                best_strategy = data.groupby(self.instance).apply(
                    lambda x: x.loc[x[metric.name].idxmax(), self.front_strategy]
                )
            best_counts[metric.name] = best_strategy.value_counts(normalize=True) * 100

        best_percentage = pd.DataFrame(best_counts).fillna(0)
        return best_percentage

    def compare_strategies_ttest(self, df, metric_column, strategies):
        strategy_1 = df[df[self.front_strategy] == strategies[0]][metric_column].values
        strategy_2 = df[df[self.front_strategy] == strategies[1]][metric_column].values

        ttest_result = ttest_rel(strategy_1, strategy_2)
        results = pd.DataFrame({
            "Metric": [metric_column],
            "Strategy 1": [strategies[0]],
            "Strategy 2": [strategies[1]],
            "T-Statistic": [ttest_result.statistic],
            "P-Value": [ttest_result.pvalue]
        })
        return results

    def compare_strategies_wilcoxon(self, df, metric_column, strategies):
        strategy_1 = df[df[self.front_strategy] == strategies[0]][metric_column].values
        strategy_2 = df[df[self.front_strategy] == strategies[1]][metric_column].values

        wilcoxon_result = wilcoxon(strategy_1, strategy_2)
        results = pd.DataFrame({
            "Metric": [metric_column],
            "Strategy 1": [strategies[0]],
            "Strategy 2": [strategies[1]],
            "W-Statistic": [wilcoxon_result.statistic],
            "P-Value": [wilcoxon_result.pvalue]
        })
        return results

    def pairwise_comparison_ttest_wilcoxon(self, df, metrics, strategies, alpha=0.05):
        """
        Perform T-Test and Wilcoxon Test for all metrics between two strategies.
        Combine results into a single DataFrame.

        Parameters:
        - df: DataFrame containing the data.
        - metrics: List of metrics to analyze.
        - strategies: List of two strategies to compare.
        - alpha: Significance level for statistical tests (default is 0.05).

        Returns:
        - A DataFrame with test results for all metrics.
        """
        results = []  # Store results for each metric

        for metric in metrics:
            # Extract metric values for the two strategies
            strategy_1 = df[df[self.front_strategy] == strategies[0]][metric.name].values
            strategy_2 = df[df[self.front_strategy] == strategies[1]][metric.name].values

            # Perform T-Test
            ttest_result = ttest_rel(strategy_1, strategy_2)
            ttest_significant = "Yes" if ttest_result.pvalue < alpha else "No"

            # Perform Wilcoxon Test
            try:
                wilcoxon_result = wilcoxon(strategy_1, strategy_2)
                wilcoxon_statistic = wilcoxon_result.statistic
                wilcoxon_pvalue = wilcoxon_result.pvalue
                wilcoxon_significant = "Yes" if wilcoxon_result.pvalue < alpha else "No"
            except ValueError as e:
                wilcoxon_statistic = e
                wilcoxon_pvalue = e
                wilcoxon_significant = "N/A"
                continue

            # Append results for the current metric
            results.append({
                "Metric": metric.name,
                "Strategy 1": strategies[0],
                "Strategy 2": strategies[1],
                "T-Test Statistic": ttest_result.statistic,
                "T-Test P-Value": ttest_result.pvalue,
                "T-Test Significant": ttest_significant,
                "Wilcoxon Statistic": wilcoxon_statistic,
                "Wilcoxon P-Value": wilcoxon_result,
                "Wilcoxon Significant": wilcoxon_significant
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def get_tables_normalized_analysis(self, data, metrics=False):
        """
        Perform the complete analysis, including checking Pareto fronts, normalizing statistics,
        and calculating the percentage of best strategies.
        """
        # Ensure Pareto fronts are consistent
        self.check_fronts_are_equal(data)

        if not metrics:
            metrics = self.build_metrics()

        # Normalize the statistics
        normalized_data, summary = self.normalize_statistics_for_metrics(data, metrics)

        # Calculate the percentage of being the best
        best_percentage = self.calculate_percentage_best_for_metrics(normalized_data, metrics)

        # stats metrics
        pairwise_comparison_ttest_wilcoxon = self.pairwise_comparison_ttest_wilcoxon(normalized_data, metrics,
                                                                                     data[self.front_strategy].unique())

        return normalized_data, summary, best_percentage, pairwise_comparison_ttest_wilcoxon

    def gets_table_comparing_average_metric_moolibrary(self, data, strategies, metrics=None):
        if metrics is None:
            metrics = ["time(s)",
                       "sum_solutions_resolution_time(s)",
                       "sum_solutions_nodes",
                       "sum_solutions_backtracks"]
        # group data by instance and strategy
        grouped = data.groupby(self.instance)

        # for instance, group in grouped:
        #     for metric in metrics:

    @staticmethod
    def build_metrics():
        metrics_names = [
            "time(s)", "sum_solutions_resolution_time(s)", "sum_solutions_nodes",
            "average_node_per_second", "sum_solutions_building_time(s)", "sum_solutions_fails",
            "sum_solutions_backtracks", "sum_number_solutions", "sum_solutions_restarts",
            "sum_solutions_backjumps"
        ]
        metrics = [Metrics(name, minimization=True) for name in metrics_names]
        for metric in metrics:
            if metric.name == "average_node_per_second":
                metric.minimization = False
        return metrics

    def process_similar_instances_for_average(self, df, obj, elements, pattern, non_stats_headers,
                                              is_exhaustive=True):
        df_filtered = df[df[self.instance].str.contains(pattern, regex=True)]
        if df_filtered.empty:
            return None

        if is_exhaustive:
            stats_depending_exhaustive = self.stats_exhaustive.copy()
            stats_pretty_name_depending_exhaustive = self.stats_exhaustive_pretty_name
        else:
            stats_depending_exhaustive = self.stats_non_exhaustive
            stats_pretty_name_depending_exhaustive = self.stats_non_exhaustive_pretty_name

        agg_dict = {col: "mean" for col in stats_depending_exhaustive}
        if not is_exhaustive:
            agg_dict["exhaustive"] = "sum"
        grouped_df = df_filtered.groupby(self.front_strategy, as_index=False).agg(agg_dict)

        if self.calculate_std:
            std_cols = [c for c in stats_depending_exhaustive if not (not is_exhaustive and c == "exhaustive")]
            agg_std = {col: "std" for col in std_cols}
            grouped_std = df_filtered.groupby(self.front_strategy, as_index=False).agg(agg_std)
            for col in std_cols:
                grouped_df[f"{col}_std"] = grouped_std[col]

        # Handle front_cardinality only for exhaustive cases
        if is_exhaustive:
            # average_number_pareto_optimal = grouped_df["front_cardinality"].iloc[0]
            average_number_pareto_optimal = round(float(grouped_df["front_cardinality"].iloc[0]), 2)
            del grouped_df["front_cardinality"]

            if self.calculate_std:
                # std_number_pareto_optimal = grouped_df["front_cardinality_std"].iloc[0]
                std_number_pareto_optimal = round(float(grouped_df["front_cardinality_std"].iloc[0]), 2)
                del grouped_df["front_cardinality_std"]
            # Rename columns
            if "p" in non_stats_headers and is_exhaustive:
                if self.calculate_std:
                    average_number_pareto_optimal = (f"{average_number_pareto_optimal}{self.joining_latex_string_avg_std}"
                                                     f"{std_number_pareto_optimal}")
                grouped_df["p"] = average_number_pareto_optimal
            stats_depending_exhaustive.remove("front_cardinality")

        for i, stat in enumerate(stats_depending_exhaustive):
            grouped_df.rename(columns={stat: stats_pretty_name_depending_exhaustive[i]}, inplace=True)
            if self.calculate_std:
                grouped_df.rename(columns={f"{stat}_std": f"{stats_pretty_name_depending_exhaustive[i]}_std"}, inplace=True)

        # Add metadata columns
        grouped_df["K"] = obj
        grouped_df["n"] = elements
        if "instances" in non_stats_headers:
            total_instances = len(df[self.instance][df[self.instance].str.contains(pattern, regex=True)].unique())
            averaged_instances = len(df_filtered[self.instance].unique())
            grouped_df["instances"] = f"{averaged_instances}/{total_instances}"

        return grouped_df

    def average_similar_instances(self, df, objs_elements, pattern_template, non_stats_headers,
                                  is_exhaustive=True):
        avg_table_rows = []
        for obj, list_elements in objs_elements.items():
            for elements in list_elements:
                pattern = pattern_template.format(obj=obj, elements=elements)
                df_result = self.process_similar_instances_for_average(df, obj, elements, pattern, non_stats_headers,
                                                                       is_exhaustive)
                if df_result is not None:
                    avg_table_rows.append(df_result)

        if not avg_table_rows:
            return None

        return pd.concat(avg_table_rows, ignore_index=True)

    def average_similar_instances_exhaustive_nonexhaustive(self, df, objs_elements, pattern_template,
                                                           non_stats_headers=None):
        """
        Computes the average statistics for both exhaustive and non-exhaustive instances.

        :param df: DataFrame containing data.
        :param objs_elements: Dictionary mapping objectives to decision variables in the problem (n).
        :param pattern_template: Template for instance name patterns.
        :param non_stats_headers: Headers for the problem date, objectives, n, etc.
        :return: List of DataFrames [exhaustive_results, non_exhaustive_results].
        """
        if non_stats_headers is None:
            non_stats_headers = ["K", "n", "instances", "p"]
        if "K" not in non_stats_headers or "n" not in non_stats_headers:
            raise Exception("The non_stats_headers must contain 'K' and 'n'.")

        # Get exhaustive and non-exhaustive data
        exhaustive_df, non_exhaustive_df = self.get_all_exhaustive_and_all_non_exhaustive_instances_df_only_stats(df)

        data_to_return = [None, None]
        # Process exhaustive instances
        non_stats_headers = ["K", "n", "instances", "p"]
        df_exhaustive = self.average_similar_instances(exhaustive_df, objs_elements, pattern_template,
                                                       non_stats_headers,
                                                       is_exhaustive=True)
        if df_exhaustive is not None:
            data_exhaustive = self.create_data_frame_pretty_table_like_disjunctive_paper(df_exhaustive,
                                                                                         self.stats_exhaustive_pretty_name,
                                                                                         non_stats_headers)
            data_to_return[0] = data_exhaustive
        else:
            print("No exhaustive instances found.")

        # Process non-exhaustive instances
        non_exhaustive_headers = ["K", "n", "instances"]  # Modified headers
        df_non_exhaustive = self.average_similar_instances(non_exhaustive_df, objs_elements, pattern_template,
                                                           non_exhaustive_headers, is_exhaustive=False)

        if df_non_exhaustive is not None:
            data_non_exhaustive = self.create_data_frame_pretty_table_like_disjunctive_paper(df_non_exhaustive,
                                                                                             self.stats_non_exhaustive_pretty_name,
                                                                                             non_exhaustive_headers)
            data_to_return[1] = data_non_exhaustive
        else:
            print("No non-exhaustive instances found.")

        return data_to_return

    def average_similar_ukp_moolibrary_instances(self, df, non_stats_headers=None):
        objs_elements, pattern_template = get_info_similar_instances_ukp_moolibrary()
        return self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                       non_stats_headers)

    def average_similar_ukp_moolibrary_voptlib_instances(self, df, non_stats_headers=None):
        objs_list = []
        pattern_list = []
        objs_elements_vol, pattern_template_vol = get_info_similar_instances_bi_ukp_voptlib()
        objs_list.append(objs_elements_vol)
        pattern_list.append(pattern_template_vol)
        objs_elements_mol, pattern_template_mol = get_info_similar_instances_ukp_moolibrary()
        objs_list.append(objs_elements_mol)
        pattern_list.append(pattern_template_mol)
        table_list_exhaustive = []
        table_list_non_exhaustive = []
        for i in range(len(objs_list)):
            objs_elements = objs_list[i]
            pattern_template = pattern_list[i]
            table = self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                            non_stats_headers)
            # table is a list with 2 dataframes, one for exhaustive and one for non-exhaustive
            if table[0] is not None:
                table_list_exhaustive.append(table[0])
            if table[1] is not None:
                table_list_non_exhaustive.append(table[1])
        # Concatenate all results into a single DataFrame
        if len(table_list_exhaustive) > 1:
            table_list_exhaustive = pd.concat(table_list_exhaustive, ignore_index=True)
        elif len(table_list_exhaustive) == 0:
            table_list_exhaustive = None
        else:
            table_list_exhaustive = table_list_exhaustive[0]
        if len(table_list_non_exhaustive) > 1:
            table_list_non_exhaustive = pd.concat(table_list_non_exhaustive, ignore_index=True)
        elif len(table_list_non_exhaustive) == 0:
            table_list_non_exhaustive = None
        else:
            table_list_non_exhaustive = table_list_non_exhaustive[0]
        return [table_list_exhaustive, table_list_non_exhaustive]

    def average_similar_bi_ukp_voptlib_instances(self, df, non_stats_headers=None):
        objs_elements, pattern_template = get_info_similar_instances_bi_ukp_voptlib()
        return self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                       non_stats_headers)

    def average_similar_nqueens_instances(self, df, non_stats_headers=None):
        objs_elements, pattern_template = get_info_similar_instances_nqueens()
        return self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                       non_stats_headers)

    def average_similar_rcpsp_instances(self, df, non_stats_headers=None):
        objs_elements, pattern_template = get_info_similar_instances_rcpsp()
        return self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                       non_stats_headers)

    def average_similar_sims_instances(self, df, non_stats_headers=None):
        objs_elements, pattern_template = get_info_similar_instances_sims()
        return self.average_similar_instances_exhaustive_nonexhaustive(df, objs_elements, pattern_template,
                                                                       non_stats_headers)

    def get_all_exhaustive_and_all_non_exhaustive_instances_df_only_stats(self, df):
        # gruop by instance
        grouped = df.groupby(self.instance)
        # create a two new df, one where are the instances where all the rows in the group have the value
        # column self.exhaustive True and the other where is False or NaN
        exhaustive_groups = []
        non_exhaustive_groups = []
        for instance, group in grouped:
            exhaustive = group[self.exhaustive].unique()
            # if len(exhaustive) == 1 and bool(exhaustive[0]) is True:
            if len(exhaustive) == 1 and not (pd.isna(exhaustive[0])) and bool(exhaustive[0]) is True:
                # add the group to the exhaustive df
                exhaustive_groups.append(group)
            else:
                # add the group to the non exhaustive df
                non_exhaustive_groups.append(group)
        keep_cols = [self.instance, self.front_strategy]
        keep_cols.extend(self.stats_exhaustive)

        exhaustive_df = pd.concat(exhaustive_groups, ignore_index=True)[keep_cols] if exhaustive_groups \
            else pd.DataFrame(columns=df.columns)
        # todo how to show the non exhaustive instances
        keep_cols = [self.instance, self.front_strategy]
        keep_cols.extend(self.stats_non_exhaustive)
        non_exhaustive_df = pd.concat(non_exhaustive_groups, ignore_index=True)[keep_cols] if non_exhaustive_groups \
            else pd.DataFrame(columns=df.columns)
        return exhaustive_df, non_exhaustive_df

    def create_data_frame_pretty_table_like_disjunctive_paper(self, exhaustive_df, stats_columns, non_stats_headers):
        std_columns = [f"{col}_std" for col in stats_columns if f"{col}_std" in exhaustive_df.columns]
        stats_columns = stats_columns + std_columns

        strategies = exhaustive_df[self.front_strategy].unique()
        column_tuples = []
        for header in non_stats_headers:
            column_tuples.append((np.nan, header))

        for strategy in strategies:
            for stat in stats_columns:
                column_tuples.append((strategy, stat))
        columns = pd.MultiIndex.from_tuples(column_tuples)

        table = pd.DataFrame(columns=columns)
        for group_keys, group in exhaustive_df.groupby(non_stats_headers):
            row = list(group_keys)
            for strategy in strategies:
                filtered_group = group[group[self.front_strategy] == strategy]
                if not filtered_group.empty:
                    row.extend(filtered_group.iloc[0][stats_columns].values)
                else:
                    row.extend([np.nan] * len(stats_columns))
            table.loc[len(table)] = row

        table = table.map(lambda x: "--" if pd.isna(x) else format_number(x))
        return table

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Time vs Instances solvede------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_time_vs_completed_instances_for_problem(self, df, problem_name, objs_elements, pattern_template, figs):
        """
        For a given problem, plots time vs number of completed instances (exhaustive = True)
        only considering instances that appear across all strategies.
        One plot per number of objectives and a global one for all objectives.
        """

        df_problem = df
        strategies = self.fixed_strategies  # use consistent strategy list

        overall_times = {strategy: [] for strategy in strategies}
        overall_counts = {strategy: [] for strategy in strategies}

        for obj, elements_list in objs_elements.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            title = f"{problem_name} - Instances with {obj} objectives"

            local_times = {strategy: [] for strategy in strategies}
            local_counts = {strategy: [] for strategy in strategies}

            obj_strategies = None
            for elements in elements_list:
                pattern = pattern_template.format(obj=obj, elements=elements)
                matched_df = df_problem[df_problem[self.instance].str.contains(pattern, regex=True)]
                if obj_strategies is None:
                    obj_strategies = matched_df[self.front_strategy].unique()

                # Find instance names common to all strategies
                instance_counts = matched_df.groupby(self.instance)[self.front_strategy].nunique()
                common_instances = instance_counts[instance_counts == len(obj_strategies)].index

                for inst in common_instances:
                    inst_df = matched_df[matched_df[self.instance] == inst]
                    for strategy in obj_strategies:
                        strat_row = inst_df[inst_df[self.front_strategy] == strategy]
                        if strat_row.empty:
                            continue
                        row = strat_row.iloc[0]
                        if not row[self.exhaustive]:
                            continue
                        resolution_time = min(row[self.time], row[self.timeout])
                        local_times[strategy].append(resolution_time)
                        overall_times[strategy].append(resolution_time)

            # Plot per-objective
            for strategy in strategies:
                if not local_times[strategy]:
                    continue  # ✅ skip strategies that didn't appear in this objective group
                times_sorted = sorted(local_times[strategy])
                completed = list(range(1, len(times_sorted) + 1))
                local_counts[strategy] = completed
                ax.plot(times_sorted, completed,
                        color=self.strategy_colors[strategy],
                        linestyle='-', linewidth=1.5)
                ax.scatter(times_sorted, completed,
                           label=strategy,
                           edgecolor=self.strategy_colors[strategy],
                           facecolor='none',
                           marker=self.strategy_markers[strategy],
                           s=80)

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time (s)", fontsize=14)
            ax.set_ylabel("Completed Instances", fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
            figs[f"time_vs_completed_{problem_name}_{obj}obj"] = fig
            plt.tight_layout()
            plt.show()

        # Global plot for all objectives
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            if not overall_times[strategy]:
                continue  # ✅ skip strategies that didn't appear globally
            times_sorted = sorted(overall_times[strategy])
            completed = list(range(1, len(times_sorted) + 1))
            overall_counts[strategy] = completed
            ax.plot(times_sorted, completed,
                    color=self.strategy_colors[strategy],
                    linestyle='-', linewidth=1.5)
            ax.scatter(times_sorted, completed,
                       label=strategy,
                       edgecolor=self.strategy_colors[strategy],
                       facecolor='none',
                       marker=self.strategy_markers[strategy],
                       s=80)

        ax.set_title(f"{problem_name} - All instances", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Completed Instances", fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
        figs[f"time_vs_completed_{problem_name}_all"] = fig
        plt.tight_layout()
        plt.show()

        return figs

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Hypervolume vs Time------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_normalized_hypervolume_evolution(self, df, problem_name, objs_elements, pattern_template, figs,
                                              plot_variance=False):
        """
        Plots normalized hypervolume evolution over time using precomputed data from get_normalized_hv_evolution_data.
        """
        data = self.get_normalized_hv_evolution_data(df, problem_name, objs_elements, pattern_template, plot_variance)

        normalized_time = data["normalized_time"]
        strategies = list(data["global"].keys())
        strategy_colors = dict(zip(strategies, sns.color_palette("colorblind", len(strategies))))
        strategy_markers = dict(zip(strategies, ['o', 's', '^', 'D', 'X', '*']))

        plot_styles = ["lines", "markers"]

        for style in plot_styles:
            for obj, obj_data in data["objectives"].items():
                fig, ax = plt.subplots(figsize=(10, 6))
                title = f"{problem_name} - Instances with {obj} objectives"

                for strategy in strategies:
                    means = obj_data[strategy]["mean"]
                    stds = obj_data[strategy]["std"]

                    if style == "lines":
                        ax.plot(
                            normalized_time, means,
                            label=strategy,
                            color=strategy_colors[strategy],
                            linestyle='-', linewidth=3,
                            markerfacecolor='none'
                        )

                        if plot_variance and stds is not None:
                            ax.fill_between(
                                normalized_time,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2,
                                color=strategy_colors[strategy]
                            )

                    elif style == "markers":
                        ax.plot(
                            normalized_time, means,
                            label=strategy,
                            color=self.strategy_colors[strategy],
                            marker=self.strategy_markers[strategy],
                            linestyle='None',
                            markersize=4,
                            markerfacecolor='none'
                        )

                        if plot_variance and stds is not None:
                            ax.errorbar(
                                normalized_time,
                                means,
                                yerr=stds,
                                fmt=self.strategy_markers[strategy],
                                color=self.strategy_colors[strategy],
                                markersize=3,
                                capsize=2,
                                linestyle='None'
                            )

                ax.set_title(title, fontsize=16)
                ax.set_xlabel("Normalized Time", fontsize=14)
                ax.set_ylabel("Normalized Hypervolume", fontsize=14)
                ax.tick_params(axis='both', labelsize=13)
                ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                figs[f"normalized_hv_vs_time_{problem_name}_{obj}obj_{style}"] = fig
                plt.show()

            # Global Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            title = f"{problem_name} - All instances"

            for strategy in strategies:
                means = data["global"][strategy]["mean"]
                stds = data["global"][strategy]["std"]

                if style == "lines":
                    ax.plot(
                        normalized_time, means,
                        label=strategy,
                        color=strategy_colors[strategy],
                        linestyle='-', linewidth=3,
                        markerfacecolor='none'
                    )
                    if plot_variance and stds is not None:
                        ax.fill_between(
                            normalized_time,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2,
                            color=strategy_colors[strategy]
                        )
                elif style == "markers":
                    ax.plot(
                        normalized_time, means,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='None',
                        markersize=4,
                        markerfacecolor='none'
                    )
                    if plot_variance and stds is not None:
                        ax.errorbar(
                            normalized_time,
                            means,
                            yerr=stds,
                            fmt=self.strategy_markers[strategy],
                            color=self.strategy_colors[strategy],
                            markersize=3,
                            capsize=2,
                            linestyle='None'
                        )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Normalized Time", fontsize=14)
            ax.set_ylabel("Normalized Hypervolume", fontsize=14)
            ax.tick_params(axis='both', labelsize=13)
            ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            figs[f"normalized_hv_vs_time_{problem_name}_all_{style}"] = fig
            plt.show()
        return figs

    def get_normalized_hv_evolution_data(self, df, problem_name, objs_elements, pattern_template, plot_variance=True):
        from scipy.interpolate import interp1d

        df_problem = df
        strategies = df_problem[self.front_strategy].unique()
        normalized_time = np.linspace(0, 1, 100)

        all_data = {"normalized_time": normalized_time, "objectives": {},
                    "global": {strategy: [[] for _ in range(100)] for strategy in strategies}}

        for obj, elements_list in objs_elements.items():
            objective_data = {strategy: [[] for _ in range(100)] for strategy in strategies}

            for elements in elements_list:
                pattern = pattern_template.format(obj=obj, elements=elements)
                matched_df = df_problem[df_problem[self.instance].str.contains(pattern, regex=True)]
                instance_counts = matched_df.groupby(self.instance)[self.front_strategy].nunique()
                common_instances = instance_counts[instance_counts == len(strategies)].index

                for inst in common_instances:
                    inst_df = matched_df[matched_df[self.instance] == inst]
                    max_time = inst_df[self.time].max()
                    max_hv = 0
                    temp_store = []
                    for strategy in strategies:
                        row = inst_df[inst_df[self.front_strategy] == strategy]
                        if row.empty:
                            continue
                        row = row.iloc[0]
                        if "hv_computed_evolution" in df_problem.columns:
                            hv_data = ast.literal_eval(row["hv_computed_evolution"])
                            times = [t for t, _ in hv_data]
                            hypervolumes = [hv for _, hv in hv_data]
                        else:
                            times, hypervolumes = self.compute_hypervolume_vs_time_from_fronts(row)
                        if len(times) == 0 or len(hypervolumes) == 0:
                            continue
                        max_hv = max(max_hv, hypervolumes[-1])
                        temp_store.append((strategy, times, hypervolumes))

                    for strategy, times, hypervolumes in temp_store:
                        times_norm = np.array(times) / max_time
                        hypervolumes_norm = np.array(hypervolumes) / max_hv
                        f_interp = interp1d(times_norm, hypervolumes_norm, kind='previous', bounds_error=False,
                                            fill_value=(0.0, hypervolumes_norm[-1]))
                        for i, t_norm in enumerate(normalized_time):
                            hv = f_interp(t_norm)
                            objective_data[strategy][i].append(hv)
                            all_data["global"][strategy][i].append(hv)

            # Store data per objective
            all_data["objectives"][obj] = {
                strategy: {
                    "mean": [np.mean(vals) if vals else 0 for vals in objective_data[strategy]],
                    "std": [np.std(vals) if vals else 0 for vals in objective_data[strategy]] if plot_variance else None
                }
                for strategy in strategies
            }

        # Global stats
        all_data["global"] = {
            strategy: {
                "mean": [np.mean(vals) if vals else 0 for vals in all_data["global"][strategy]],
                "std": [np.std(vals) if vals else 0 for vals in all_data["global"][strategy]] if plot_variance else None
            }
            for strategy in strategies
        }

        return all_data

    def compute_hypervolume_vs_time_from_fronts(self, row):
        try:
            data = ast.literal_eval(row[self.hypervolume_evolution])
        except Exception:
            return [], []

        if not isinstance(data, list) or len(data) == 0:
            return [], []

        ref_point = row["reference_point"]
        if isinstance(ref_point, str):
            ref_point = ast.literal_eval(ref_point)
        num_possible_points_to_filter = len(ref_point)
        times = []
        hypervolumes = []
        # convert from string to np array
        pareto_front = row[self.pareto_front]  # is a string
        if isinstance(pareto_front, str):
            pareto_front = ast.literal_eval(pareto_front)

        maximize = is_maximization_problem(pareto_front, ref_point)

        removed_indexes = set()

        for t, front in data:
            if not front:
                continue
            if front[0][0] < 0:
                # front elements cannot be negative, convert to positive
                front = np.abs(front)
            front = np.array(front)
            # reduce the number of points to filter if there have been determined some points to filter in previous
            # iterations
            if (("saugmecon" in row[self.front_strategy].lower() or "disjunctive" in row[self.front_strategy].lower())
                    and num_possible_points_to_filter > 0):

                # Check new ones only if we haven't removed k yet
                if len(removed_indexes) < num_possible_points_to_filter:
                    for i in range(num_possible_points_to_filter):
                        if i in removed_indexes:
                            continue
                        for j in range(num_possible_points_to_filter, len(front)):
                            if is_dominated(front[i], front[j], maximize):
                                removed_indexes.add(i)
                                break

                    # After checking, remove again
                    if removed_indexes:
                        front = np.delete(front, list(removed_indexes), axis=0)

            hv = compute_hv(front, reference_point=ref_point, double_check_non_dominance=False)
            times.append(t)
            hypervolumes.append(hv)

        return times, hypervolumes

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Cumulative time vs cumulative normalized HV-----------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_cumulative_hv_vs_time(self, data, problem_name, figs):
        # strategies = data["global"].keys()
        strategies = self.fixed_strategies
        # colors = dict(zip(strategies, sns.color_palette("colorblind", len(strategies))))
        # markers = dict(zip(strategies, ['o', 's', '^', 'D', 'X', '*']))

        for obj, obj_data in data["objectives"].items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - {obj} objectives", fontsize=16)
            for strategy in strategies:
                hv = obj_data[strategy]["hv"]
                time = obj_data[strategy]["time"]
                ax.plot(time, hv,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        linestyle='-', linewidth=3)
            ax.set_xlabel("Cumulative time (s)", fontsize=14)
            ax.set_ylabel("Cumulative normalized hypervolume", fontsize=14)
            ax.tick_params(axis='both', labelsize=13)
            ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            figs[f"cumulative_hv_vs_time_{problem_name}_{obj}obj"] = fig
            plt.show()

        # Global plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - All instances", fontsize=16)
        for strategy in strategies:
            hv = data["global"][strategy]["hv"]
            time = data["global"][strategy]["time"]
            ax.plot(time, hv,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    linestyle='-', linewidth=3)
        ax.set_xlabel("Cumulative time (s)", fontsize=14)
        ax.set_ylabel("Cumulative normalized hypervolume", fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        figs[f"cumulative_hv_vs_time_{problem_name}_all"] = fig
        plt.show()
        return figs

    def get_cumulative_hv_vs_time_data(self, df, problem_name, objs_elements, pattern_template):
        df_problem = df
        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"hv": [], "time": []} for strategy in strategies}
        }

        for strategy in strategies:
            # Get all instances solved by the strategy
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            # Normalize globally by per-instance best HV
            instance_max_hv = df_problem.groupby(self.instance)[self.hypervolume].max()
            strat_df["normalized_hv"] = strat_df[self.hypervolume] / strat_df[self.instance].map(instance_max_hv)
            strat_df = strat_df.dropna(subset=["normalized_hv", self.time])

            # ---------- GLOBAL ----------
            strat_df_sorted = strat_df.sort_values(by=["normalized_hv", self.time], ascending=[False, True])
            cum_hv = strat_df_sorted["normalized_hv"].cumsum()
            cum_time = strat_df_sorted[self.time].cumsum()
            data["global"][strategy]["hv"] = cum_hv.tolist()
            data["global"][strategy]["time"] = cum_time.tolist()

            # ---------- PER OBJECTIVE ----------
            for obj, elements_list in objs_elements.items():
                pattern_mask = strat_df[self.instance].apply(
                    lambda x: any(re.search(pattern_template.format(obj=obj, elements=el), x) for el in elements_list)
                )
                strat_df_obj = strat_df[pattern_mask].copy()

                strat_df_obj_sorted = strat_df_obj.sort_values(by=["normalized_hv", self.time], ascending=[False, True])
                cum_hv_obj = strat_df_obj_sorted["normalized_hv"].cumsum()
                cum_time_obj = strat_df_obj_sorted[self.time].cumsum()

                if obj not in data["objectives"]:
                    data["objectives"][obj] = {}
                data["objectives"][obj][strategy] = {
                    "hv": cum_hv_obj.tolist(),
                    "time": cum_time_obj.tolist()
                }

        return data

    # --------------------------------------------------------------------------------------------------------------------
    # ----------------- Histograms Best Hypervolume ---------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    def plot_best_hypervolume_histogram(self, df, problem_name, objs_elements, pattern_template, figs):
        """
        Plots histogram showing how often each strategy is exclusive best, shared best, or second best.
        One plot per objective + a global one.
        Returns: figs dictionary
        """

        global_df, per_objective_data = self.categorize_strategy_performance_grouped_by_objective(
            df, objs_elements, pattern_template
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Exclusive, Shared, Second

        def _plot(df_counts, title):
            df_counts = df_counts.set_index("Strategy")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_counts[["Exclusive best", "Shared best", "Second best"]].plot(
                kind='bar', stacked=True, color=colors, ax=ax
            )
            ax.set_title(title, fontsize=16)
            ax.set_ylabel("Number of Instances", fontsize=14)
            ax.set_xlabel("Strategy", fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

        # Global plot
        fig_global = _plot(global_df, f"{problem_name} – All objectives – Best hypervolume per strategy")
        figs[f"best_hv_global_{problem_name}"] = fig_global

        # Per-objective plots
        for obj, df_counts in per_objective_data.items():
            fig = _plot(df_counts, f"{problem_name} – {obj} objectives – Best Hypervolume per Strategy")
            figs[f"best_hv_obj_{obj}_{problem_name}"] = fig

        return figs

    def categorize_strategy_performance_grouped_by_objective(self, df, objs_elements, pattern_template):
        """
        Categorize each strategy as Exclusive Best, Shared Best, or Second Best per objective.
        Also return the global aggregation over all objectives.
        """
        results_per_objective = {}
        global_exclusive = Counter()
        global_shared = Counter()
        global_second = Counter()

        for obj, elements_list in objs_elements.items():
            exclusive = Counter()
            shared = Counter()
            second = Counter()

            for elements in elements_list:
                pattern = pattern_template.format(obj=obj, elements=elements)
                matched_df = df[df[self.instance].str.contains(pattern, regex=True)]

                for inst, inst_group in matched_df.groupby(self.instance):
                    sorted_group = inst_group.sort_values(by=self.hypervolume, ascending=False)
                    top_hv = sorted_group[self.hypervolume].iloc[0]
                    top_strats = sorted_group[sorted_group[self.hypervolume] == top_hv][self.front_strategy].tolist()

                    if len(top_strats) == 1:
                        exclusive[top_strats[0]] += 1
                    else:
                        for s in top_strats:
                            shared[s] += 1

                    # Second best
                    unique_hv = sorted_group[self.hypervolume].unique()
                    if len(unique_hv) > 1:
                        second_best_hv = unique_hv[1]
                        second_best_strats = sorted_group[sorted_group[self.hypervolume] == second_best_hv][
                            self.front_strategy].tolist()
                        for s in second_best_strats:
                            second[s] += 1

            # Save results per objective
            all_strats = set(df[self.front_strategy])
            results_per_objective[obj] = pd.DataFrame({
                "Strategy": list(all_strats),
                "Exclusive best": [exclusive[s] for s in all_strats],
                "Shared best": [shared[s] for s in all_strats],
                "Second best": [second[s] for s in all_strats],
            })

            # Update global
            for k in exclusive: global_exclusive[k] += exclusive[k]
            for k in shared: global_shared[k] += shared[k]
            for k in second: global_second[k] += second[k]

        all_strats = set(df[self.front_strategy])
        global_df = pd.DataFrame({
            "Strategy": list(all_strats),
            "Exclusive best": [global_exclusive[s] for s in all_strats],
            "Shared best": [global_shared[s] for s in all_strats],
            "Second best": [global_second[s] for s in all_strats],
        })

        return global_df, results_per_objective

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Normalized HV per strategy ---------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_normalized_hypervolume_per_strategy(self, df, problem_name, objs_elements, pattern_template, figs,
                                                 data=None):
        strategies = self.fixed_strategies
        if data is None:
            data = self.get_normalized_hv_per_strategy_data(df, problem_name, objs_elements, pattern_template)

        for_plotting = data["objectives"]

        # --------- PER OBJECTIVE ----------
        for obj, obj_data in for_plotting.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - Normalized HV per instance - {obj} objectives", fontsize=16)

            for i, strategy in enumerate(strategies):
                if strategy not in obj_data:
                    continue
                hv_sorted = obj_data[strategy]["hv"]
                hv_values = [val for (_, val) in hv_sorted]
                x = list(range(1, len(hv_values) + 1))

                ax.plot(x, hv_values,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='-',
                        markersize=6,
                        markerfacecolor='none',  # Unfilled marker
                        linewidth=1.5)

            self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by normalized HV)",
                                          ylabel="HV / best HV")
            figs[f"normalized_hv_sorted_per_instance_{problem_name}_{obj}obj"] = fig
            plt.show()

        # --------- GLOBAL ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - Normalized HV per instance - All instances", fontsize=16)

        for i, strategy in enumerate(strategies):
            if strategy not in data["global"]:
                continue
            hv_sorted = data["global"][strategy]["hv"]
            hv_values = [val for (_, val) in hv_sorted]
            x = list(range(1, len(hv_values) + 1))

            ax.plot(x, hv_values,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle='-',
                    markersize=6,
                    markerfacecolor='none',
                    linewidth=1.5)

        self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by normalized HV)",
                                      ylabel="HV / best HV")
        figs[f"normalized_hv_sorted_per_instance_{problem_name}_all"] = fig
        plt.show()

        return figs

    def get_normalized_hv_per_strategy_data(self, df, problem_name, objs_elements, pattern_template,
                                            remove_completed_instances=False):
        df_problem = df.copy()

        # Optional remove instances completed by all the strategies
        if remove_completed_instances:
            df_problem = self.remove_instances_completed_by_all_strategies(df_problem)

        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"hv": []} for strategy in strategies}
        }

        # Precompute best HV per instance
        instance_max_hv = df_problem.groupby(self.instance)[self.hypervolume].max()

        for strategy in strategies:
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            # todo delete this line later is just to get HV values
            # print(f"HV values for strategy {strategy}: {strat_df[self.hypervolume]}")

            # Normalize hypervolume
            strat_df["normalized_hv"] = strat_df[self.hypervolume] / strat_df[self.instance].map(instance_max_hv)
            strat_df = strat_df.dropna(subset=["normalized_hv", self.time])

            # ---------- GLOBAL ----------
            strat_df_sorted = strat_df.sort_values(by=["normalized_hv", self.time], ascending=[False, True])
            data["global"][strategy]["hv"] = list(zip(strat_df_sorted[self.instance], strat_df_sorted["normalized_hv"]))

            # ---------- PER OBJECTIVE ----------
            for obj, elements_list in objs_elements.items():
                pattern_mask = strat_df[self.instance].apply(
                    lambda x: any(re.search(pattern_template.format(obj=obj, elements=el), x) for el in elements_list)
                )
                strat_df_obj = strat_df[pattern_mask].copy()
                if strat_df_obj.empty:
                    continue

                strat_df_obj_sorted = strat_df_obj.sort_values(by=["normalized_hv", self.time], ascending=[False, True])
                normalized_hvs = list(zip(strat_df_obj_sorted[self.instance], strat_df_obj_sorted["normalized_hv"]))

                if obj not in data["objectives"]:
                    data["objectives"][obj] = {}
                data["objectives"][obj][strategy] = {
                    "hv": normalized_hvs
                }

        return data

    def remove_instances_completed_by_all_strategies(self, df):
        strategies = df[self.front_strategy].unique()
        num_strategies = len(strategies)

        instances_to_remove = []

        for instance, group in df.groupby(self.instance):
            all_strategies_present = group[self.front_strategy].nunique() == num_strategies
            all_exhaustive = group[self.exhaustive].all()

            if all_strategies_present and all_exhaustive:
                instances_to_remove.append(instance)

        return df[~df[self.instance].isin(instances_to_remove)]

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Normalized contribution per strategy -----------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_normalized_contribution_per_strategy(self, df, problem_name, objs_elements, pattern_template, figs,
                                                  data=None):
        strategies = self.fixed_strategies
        if data is None:
            data = self.get_contribution_to_joint_front_data(df, problem_name, objs_elements, pattern_template)

        for_plotting = data["objectives"]

        # --------- PER OBJECTIVE ----------
        for obj, obj_data in for_plotting.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - Contribution to joint front - {obj} objectives", fontsize=16)

            for i, strategy in enumerate(strategies):
                if strategy not in obj_data:
                    continue
                contrib_list = obj_data[strategy]["contribution"]
                contrib_values = [val for _, val in contrib_list]  # ignore instance names
                x = list(range(1, len(contrib_values) + 1))

                ax.plot(x, contrib_values,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='-',
                        markersize=6,
                        markerfacecolor='none',  # Unfilled marker
                        linewidth=1.5)

            self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by contribution)",
                                          ylabel="Contribution to joint front", ylim=[0, 1.05])
            figs[f"normalized_contribution_sorted_{problem_name}_{obj}obj"] = fig
            plt.show()

        # --------- GLOBAL ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - Contribution to joint front - All instances", fontsize=16)

        for i, strategy in enumerate(strategies):
            if strategy not in data["global"]:
                continue
            contrib_list = data["global"][strategy]["contribution"]
            contrib_values = [val for _, val in contrib_list]
            x = list(range(1, len(contrib_values) + 1))

            ax.plot(x, contrib_values,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle='-',
                    markersize=6,
                    markerfacecolor='none',
                    linewidth=1.5)

        self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by contribution)",
                                      ylabel="Contribution to joint front", ylim=[0, 1.05])
        figs[f"normalized_contribution_sorted_{problem_name}_all"] = fig
        plt.show()

        return figs

    def get_contribution_to_joint_front_data(self, df, problem_name, objs_elements, pattern_template, remove_completed_instances=False):
        df_problem = df

        # Optional remove instances completed by all the strategies
        if remove_completed_instances:
            df_problem = self.remove_instances_completed_by_all_strategies(df_problem)

        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"contribution": []} for strategy in strategies}
        }

        instances = df_problem[self.instance].unique()

        # Check whether the problem is maximization or minimization
        row = df_problem.iloc[0]
        ref_point = row["reference_point"]
        if isinstance(ref_point, str):
            ref_point = ast.literal_eval(ref_point)
        # convert from string to np array
        pareto_front = row[self.pareto_front]  # is a string
        if isinstance(pareto_front, str):
            pareto_front = ast.literal_eval(pareto_front)
        maximize = is_maximization_problem(pareto_front, ref_point)

        for strategy in strategies:
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            for instance in instances:
                df_instance = df_problem[df_problem[self.instance] == instance]
                fronts_by_strategy = {}
                true_front = None

                for _, row in df_instance.iterrows():
                    strat = row[self.front_strategy]
                    if pd.isna(row[self.pareto_front]):
                        continue

                    front_points = ast.literal_eval(row[self.pareto_front])
                    fronts_by_strategy.setdefault(strat, []).extend(front_points)
                    if row.get(self.exhaustive, False):
                        true_front = front_points

                if true_front:
                    joint_front = true_front
                else:
                    cache_key = (problem_name, instance)
                    if cache_key not in self._joint_front_cache:
                        all_points = set()
                        for points in fronts_by_strategy.values():
                            for pt in points:
                                all_points.add(tuple(pt))  # convert lists to tuples to make them hashable
                        joint_front = MoAnalysis.remove_dominated_points(list(all_points), maximize)
                        # cache the joint front
                        self._joint_front_cache[cache_key] = joint_front
                    else:
                        joint_front = self._joint_front_cache[cache_key]

                joint_front_set = {tuple(pt) for pt in joint_front}
                strat_points = [tuple(p) for p in fronts_by_strategy.get(strategy, [])]
                contribution_count = sum(1 for pt in strat_points if pt in joint_front_set)
                total_joint = len(joint_front_set)

                contribution_ratio = (
                    contribution_count / total_joint if total_joint > 0 else 0
                )
                data["global"][strategy]["contribution"].append((instance, contribution_ratio))

                # --------- Per Objective (pattern match) ---------
                for obj, elements_list in objs_elements.items():
                    if not any(
                            re.search(pattern_template.format(obj=obj, elements=el), instance) for el in elements_list):
                        continue

                    if obj not in data["objectives"]:
                        data["objectives"][obj] = {s: {"contribution": []} for s in strategies}

                    data["objectives"][obj][strategy]["contribution"].append((instance, contribution_ratio))

        # Sort contributions by value (descending)
        for strategy in data["global"]:
            data["global"][strategy]["contribution"].sort(key=lambda x: x[1], reverse=True)

        for obj in data["objectives"]:
            for strategy in data["objectives"][obj]:
                data["objectives"][obj][strategy]["contribution"].sort(key=lambda x: x[1], reverse=True)

        return data

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- IGD data----------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    def plot_igd_per_strategy(self, df, problem_name, objs_elements, pattern_template, figs, data=None):
        strategies = self.fixed_strategies
        if data is None:
            data = self.compute_igd_per_strategy(df, problem_name, objs_elements, pattern_template)

        for_plotting = data["objectives"]

        # --------- PER OBJECTIVE ----------
        for obj, obj_data in for_plotting.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - IGD per instance - {obj} objectives", fontsize=16)

            for i, strategy in enumerate(strategies):
                if strategy not in obj_data:
                    continue
                igd_sorted = obj_data[strategy]["igd"]
                igd_values = [val for (_, val) in igd_sorted]
                x = list(range(1, len(igd_values) + 1))

                ax.plot(x, igd_values,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='-',
                        markersize=6,
                        markerfacecolor='none',
                        linewidth=1.5)

            self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by IGD)",
                                          ylabel="IGD")
            figs[f"igd_sorted_per_instance_{problem_name}_{obj}obj"] = fig
            plt.show()

        # --------- GLOBAL ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - IGD per instance - All instances", fontsize=16)

        for i, strategy in enumerate(strategies):
            if strategy not in data["global"]:
                continue
            igd_sorted = data["global"][strategy]["igd"]
            igd_values = [val for (_, val) in igd_sorted]
            x = list(range(1, len(igd_values) + 1))

            ax.plot(x, igd_values,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle='-',
                    markersize=6,
                    markerfacecolor='none',
                    linewidth=1.5)

        self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by IGD)",
                                      ylabel="IGD")
        figs[f"igd_sorted_per_instance_{problem_name}_all"] = fig
        plt.show()

        return figs

    def compute_igd_per_strategy(self, df, problem_name, objs_elements, pattern_template):
        df_problem = df
        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"igd": []} for strategy in strategies}
        }

        instances = df_problem[self.instance].unique()

        # Check whether the problem is maximization or minimization
        row = df_problem.iloc[0]
        ref_point = row["reference_point"]
        if isinstance(ref_point, str):
            ref_point = ast.literal_eval(ref_point)
        # convert from string to np array
        pareto_front = row[self.pareto_front]  # is a string
        if isinstance(pareto_front, str):
            pareto_front = ast.literal_eval(pareto_front)
        maximize = is_maximization_problem(pareto_front, ref_point)

        for strategy in strategies:
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            for instance in instances:
                df_instance = df_problem[df_problem[self.instance] == instance]
                fronts_by_strategy = {}
                true_front = None

                for _, row in df_instance.iterrows():
                    strat = row[self.front_strategy]
                    if pd.isna(row[self.pareto_front]):
                        continue

                    front_points = ast.literal_eval(row[self.pareto_front])
                    fronts_by_strategy.setdefault(strat, []).extend(front_points)
                    if row.get(self.exhaustive, False):
                        true_front = front_points

                if true_front:
                    joint_front = true_front
                else:
                    cache_key = (problem_name, instance)
                    if cache_key not in self._joint_front_cache:
                        all_points = set()
                        for points in fronts_by_strategy.values():
                            for pt in points:
                                all_points.add(tuple(pt))  # convert lists to tuples to make them hashable
                        joint_front = MoAnalysis.remove_dominated_points(list(all_points), maximize)
                        # cache the joint front
                        self._joint_front_cache[cache_key] = joint_front
                    else:
                        joint_front = self._joint_front_cache[cache_key]

                # get the igd
                strat_instance_igd = compute_igd(fronts_by_strategy.get(strategy, []), ref_point,
                                                                joint_front, maximize)

                data["global"][strategy]["igd"].append((instance, strat_instance_igd))

                # --------- Per Objective (pattern match) ---------
                for obj, elements_list in objs_elements.items():
                    if not any(
                            re.search(pattern_template.format(obj=obj, elements=el), instance) for el in elements_list):
                        continue

                    if obj not in data["objectives"]:
                        data["objectives"][obj] = {s: {"igd": []} for s in strategies}

                    data["objectives"][obj][strategy]["igd"].append((instance, strat_instance_igd))

        # Sort contributions by value (descending)
        for strategy in data["global"]:
            data["global"][strategy]["igd"].sort(key=lambda x: x[1], reverse=False)

        for obj in data["objectives"]:
            for strategy in data["objectives"][obj]:
                data["objectives"][obj][strategy]["igd"].sort(key=lambda x: x[1], reverse=False)

        return data

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- IGD+ data----------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    def plot_igd_plus_per_strategy(self, df, problem_name, objs_elements, pattern_template, figs, data=None):
        strategies = self.fixed_strategies
        if data is None:
            data = self.compute_igd_plus_per_strategy(df, problem_name, objs_elements, pattern_template)

        for_plotting = data["objectives"]

        # --------- PER OBJECTIVE ----------
        for obj, obj_data in for_plotting.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - IGD+ per instance - {obj} objectives", fontsize=16)

            for i, strategy in enumerate(strategies):
                if strategy not in obj_data:
                    continue
                igd_sorted = obj_data[strategy]["igd_plus"]
                igd_values = [val for (_, val) in igd_sorted]
                x = list(range(1, len(igd_values) + 1))

                ax.plot(x, igd_values,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='-',
                        markersize=6,
                        markerfacecolor='none',
                        linewidth=1.5)

            self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by IGD+)",
                                          ylabel="IGD+")
            figs[f"igd_plus_sorted_per_instance_{problem_name}_{obj}obj"] = fig
            plt.show()

        # --------- GLOBAL ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - IGD+ per instance - All instances", fontsize=16)

        for i, strategy in enumerate(strategies):
            if strategy not in data["global"]:
                continue
            igd_sorted = data["global"][strategy]["igd_plus"]
            igd_values = [val for (_, val) in igd_sorted]
            x = list(range(1, len(igd_values) + 1))

            ax.plot(x, igd_values,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle='-',
                    markersize=6,
                    markerfacecolor='none',
                    linewidth=1.5)

        self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by IGD+)",
                                      ylabel="IGD+")
        figs[f"igd_plus_sorted_per_instance_{problem_name}_all"] = fig
        plt.show()

        return figs

    def compute_igd_plus_per_strategy(self, df, problem_name, objs_elements, pattern_template, remove_completed_instances=False):
        df_problem = df

        # Optional remove instances completed by all the strategies
        if remove_completed_instances:
            df_problem = self.remove_instances_completed_by_all_strategies(df_problem)

        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"igd_plus": []} for strategy in strategies}
        }

        instances = df_problem[self.instance].unique()

        # Check whether the problem is maximization or minimization
        row = df_problem.iloc[0]
        ref_point = row["reference_point"]
        if isinstance(ref_point, str):
            ref_point = ast.literal_eval(ref_point)
        # convert from string to np array
        pareto_front = row[self.pareto_front]  # is a string
        if isinstance(pareto_front, str):
            pareto_front = ast.literal_eval(pareto_front)
        maximize = is_maximization_problem(pareto_front, ref_point)

        for strategy in strategies:
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            for instance in instances:
                df_instance = df_problem[df_problem[self.instance] == instance]
                fronts_by_strategy = {}
                true_front = None

                for _, row in df_instance.iterrows():
                    strat = row[self.front_strategy]
                    if pd.isna(row[self.pareto_front]):
                        continue

                    front_points = ast.literal_eval(row[self.pareto_front])
                    fronts_by_strategy.setdefault(strat, []).extend(front_points)
                    if row.get(self.exhaustive, False):
                        true_front = front_points

                if true_front:
                    joint_front = true_front
                else:
                    cache_key = (problem_name, instance)
                    if cache_key not in self._joint_front_cache:
                        all_points = set()
                        for points in fronts_by_strategy.values():
                            for pt in points:
                                all_points.add(tuple(pt))  # convert lists to tuples to make them hashable
                        joint_front = MoAnalysis.remove_dominated_points(list(all_points), maximize)
                        # cache the joint front
                        self._joint_front_cache[cache_key] = joint_front
                    else:
                        joint_front = self._joint_front_cache[cache_key]

                # get the igd
                strat_instance_igd = compute_igd(fronts_by_strategy.get(strategy, []), ref_point,
                                                 joint_front, maximize, plus=True)

                # delete this line later is just to get IGD+ values
                # print(f"IGD+ values for strategy {strategy} on instance {instance}: {strat_instance_igd}")

                data["global"][strategy]["igd_plus"].append((instance, strat_instance_igd))

                # --------- Per Objective (pattern match) ---------
                for obj, elements_list in objs_elements.items():
                    if not any(
                            re.search(pattern_template.format(obj=obj, elements=el), instance) for el in elements_list):
                        continue

                    if obj not in data["objectives"]:
                        data["objectives"][obj] = {s: {"igd_plus": []} for s in strategies}

                    data["objectives"][obj][strategy]["igd_plus"].append((instance, strat_instance_igd))

        # Sort contributions by value (descending)
        for strategy in data["global"]:
            data["global"][strategy]["igd_plus"].sort(key=lambda x: x[1], reverse=False)

        for obj in data["objectives"]:
            for strategy in data["objectives"][obj]:
                data["objectives"][obj][strategy]["igd_plus"].sort(key=lambda x: x[1], reverse=False)

        return data

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------- Normalized Time per strategy -------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    def get_normalized_time_per_strategy_data(self, df, objs_elements, pattern_template, remove_all_timeout_instances=False):
        df_problem = df.copy()
        strategies = df_problem[self.front_strategy].unique()

        data = {
            "objectives": {},
            "global": {strategy: {"time": []} for strategy in strategies}
        }

        capped_time = "capped_time"
        normalized_time = "normalized_time"

        # --- 1) capped_time = min(time, timeout) ---
        # Make sure time/timeout are numeric (in case they come as strings)
        df_problem[self.time] = pd.to_numeric(df_problem[self.time], errors="coerce")
        df_problem[self.timeout] = pd.to_numeric(df_problem[self.timeout], errors="coerce")

        df_problem[capped_time] = np.minimum(df_problem[self.time], df_problem[self.timeout])
        # If time is NaN -> capped_time should be NaN (np.minimum already does that)
        # --- OPTIONAL: drop instances where ALL strategies timed out ---
        if remove_all_timeout_instances:
            # timed_out row = time >= timeout (ignore NaNs)
            df_problem["_timed_out"] = (
                    df_problem[self.time].notna()
                    & df_problem[self.timeout].notna()
                    & (df_problem[self.time] >= df_problem[self.timeout])
            )

            # per instance: True iff every row (strategy) is timed out
            all_timeout_by_instance = df_problem.groupby(self.instance)["_timed_out"].all()

            # keep only instances that are NOT all-timeout
            keep_instances = all_timeout_by_instance[~all_timeout_by_instance].index
            df_problem = df_problem[df_problem[self.instance].isin(keep_instances)].copy()

            df_problem = df_problem.drop(columns=["_timed_out"])

        # --- 2) best (min) capped time per instance across ALL strategies ---
        instance_min_time = df_problem.groupby(self.instance)[capped_time].min()

        # Guard: if an instance has best time 0, normalization would blow up.
        # Replace 0 by NaN so we drop those rows (or you can decide another policy).
        instance_min_time = instance_min_time.replace(0, np.nan)

        for strategy in strategies:
            strat_df = df_problem[df_problem[self.front_strategy] == strategy].copy()
            if strat_df.empty:
                continue

            # Normalize time (lower is better, so >= 1.0, where 1.0 is best)
            strat_df[normalized_time] = strat_df[capped_time] / strat_df[self.instance].map(instance_min_time)
            strat_df = strat_df.dropna(subset=[normalized_time, capped_time])

            # ---------- GLOBAL ----------
            # best on the left: smaller normalized_time first; tie-break by smaller capped_time
            strat_df_sorted = strat_df.sort_values(by=[normalized_time, capped_time], ascending=[True, True])
            data["global"][strategy]["time"] = strat_df_sorted[normalized_time].tolist()

            # ---------- PER OBJECTIVE ----------
            for obj, elements_list in objs_elements.items():
                pattern_mask = strat_df[self.instance].apply(
                    lambda x: any(re.search(pattern_template.format(obj=obj, elements=el), x) for el in elements_list)
                )
                strat_df_obj = strat_df[pattern_mask].copy()
                if strat_df_obj.empty:
                    continue

                strat_df_obj_sorted = strat_df_obj.sort_values(by=[normalized_time, capped_time],
                                                               ascending=[True, True])
                normalized_times = strat_df_obj_sorted[normalized_time].tolist()

                if obj not in data["objectives"]:
                    data["objectives"][obj] = {}
                data["objectives"][obj][strategy] = {
                    "time": normalized_times
                }

        return data

    def plot_normalized_time_per_strategy(self, df, problem_name, objs_elements, pattern_template,
                                          figs, data=None, log_threshold=50):
        strategies = self.fixed_strategies
        if data is None:
            data = self.get_normalized_time_per_strategy_data(df, objs_elements, pattern_template)

        for_plotting = data["objectives"]

        # --------- PER OBJECTIVE ----------
        for obj, obj_data in for_plotting.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{problem_name} - Normalized Time per instance - {obj} objectives", fontsize=16)

            # Collect exactly the y-values that will be plotted in this figure
            all_vals = []
            for i, strategy in enumerate(strategies):
                if strategy not in obj_data:
                    continue
                time_sorted = obj_data[strategy]["time"]
                all_vals.extend(time_sorted)
                x = list(range(1, len(time_sorted) + 1))

                ax.plot(x, time_sorted,
                        label=strategy,
                        color=self.strategy_colors[strategy],
                        marker=self.strategy_markers[strategy],
                        linestyle='-',
                        markersize=6,
                        markerfacecolor='none',
                        linewidth=1.5)

            # log scale only if max/min >= log_threshold, and set readable ticks
            self.apply_y_scale_and_ticks(ax, all_vals, log_threshold=log_threshold)
            self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by normalized time)",
                                          ylabel="Runtime / best runtime")
            figs[f"normalized_time_sorted_per_instance_{problem_name}_{obj}obj"] = fig
            plt.show()

        # --------- GLOBAL ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{problem_name} - Normalized Time per instance - All instances", fontsize=16)

        # collect exactly the y-values that will be plotted in this figure
        all_vals = []
        for i, strategy in enumerate(strategies):
            if strategy not in data["global"]:
                continue
            time_sorted = data["global"][strategy]["time"]
            all_vals.extend(time_sorted)
            x = list(range(1, len(time_sorted) + 1))

            ax.plot(x, time_sorted,
                    label=strategy,
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle='-',
                    markersize=6,
                    markerfacecolor='none',
                    linewidth=1.5)

        # log scale only if max/min >= log_threshold, and set readable ticks
        self.apply_y_scale_and_ticks(ax, all_vals, log_threshold=log_threshold)
        self.finalize_plot_for_screen(fig, ax, xlabel="Instance rank (per strategy) (sorted by normalized time)",
                                      ylabel="Runtime / best runtime")
        figs[f"normalized_time_sorted_per_instance_{problem_name}_all"] = fig
        plt.show()

        return figs

    def apply_y_scale_and_ticks(self, ax, all_vals, log_threshold=50, tick_multipliers=(1, 2, 5)):
        use_log = False
        vals = np.array(all_vals, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 0:
            use_log = (vals.max() / vals.min()) >= log_threshold

        if use_log:
            ax.set_yscale("log")  # base 10 default
            maxv = vals.max()
            ticks = []
            for k in range(0, int(np.ceil(np.log10(maxv))) + 1):
                ticks.extend([m * (10 ** k) for m in tick_multipliers])
            ticks = [t for t in ticks if 1 <= t <= maxv * 1.0001]
            if len(ticks) >= 2:
                ax.set_yticks(ticks)

    def finalize_plot_for_screen(self, fig, ax, xlabel, ylabel,
                                 ylim=None,
                                 legend_title="Strategy",
                                 legend_loc="upper left",
                                 legend_bbox_to_anchor=(1.05, 1),
                                 add_grid=False):
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        ax.tick_params(axis='both', labelsize=14)

        if add_grid:
            ax.grid(True, which='both')

        ax.legend(title=legend_title, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)

        fig.tight_layout()

    # ------------------------------ End of paper plotting functions ---------------------------------------------------

    def set_stats_exhaustive(self, stats_exhaustive, pretty_name):
        self.stats_exhaustive = stats_exhaustive
        self.stats_exhaustive_pretty_name = pretty_name

    def set_stats_non_exhaustive(self, stats_non_exhaustive, pretty_name):
        self.stats_non_exhaustive = stats_non_exhaustive
        self.stats_non_exhaustive_pretty_name = pretty_name

    def set_fixed_strategies(self, fixed_strategies=None, strategy_colors=None, strategy_markers=None):
        if fixed_strategies is not None:
            self.fixed_strategies = fixed_strategies

        if strategy_colors is not None:
            self.strategy_colors = strategy_colors

        if strategy_markers is not None:
            self.strategy_markers = strategy_markers


class SaveAllResultsCP2025:

    def __init__(self, csv_paths_problem, config=None, strategies=None):
        self.csv_paths_problem = csv_paths_problem  # expecting list of (csv_path, problem_name) tuples
        # Ex: csvs = [("rcpsp.csv", "rcpsp"), ("ukp.csv", "ukp")]
        # runner = SaveAllResultsCP2025(csvs)
        if config is None:
            raise ValueError("Config is required")
        self.config = config
        if strategies is None:
            raise ValueError("Strategies is required")
        self.analysis = MoAnalysis(strategies)
        self.strategies = strategies
        self.normalized_fronts = {}
        self.normalized_fronts_folder = "normalized_fronts"

    def save_all_results(self):
        print("Starting to save all results. Here we go!")
        print(f"With evolution: {self.config.print_hv_evolution}")
        figs, data = {}, {}
        for csv_path, problem in self.csv_paths_problem:
            self.normalized_fronts[problem] = False
            figs[problem], data[problem] = self.save_results(csv_path, problem)
        return figs, data

    def save_results(self, csv_path, problem):
        df = self.analysis.csv_to_df(csv_path)
        print(f"Processing {csv_path} for problem {problem}")

        current_sources = set(self.analysis.strategies.strategies_original_name_list)
        # todo delete below after testing
        print(f"Current sources to consider: {current_sources}")
        df = df.loc[df[self.analysis.front_strategy].isin(current_sources)]

        # 2) expand/filter/rename into the *actual* compared strategies
        df = self.strategies.apply_specs(df, strategy_col=self.analysis.front_strategy)

        # 3) keep only instances where all compared strategies exist
        if not self.analysis.strategies.all_instances_per_strategy:
            df = self.keep_only_common_instances(
                df,
                instance_col=self.analysis.instance,
                strategy_col=self.analysis.front_strategy,
                required_strategies=self.analysis.strategies.strategies_better_name
            )

        figs = {}
        latex_text = ""
        if self.config.print_hv_evolution:
            df = self.add_hv_computed_evolution_if_not_present(df, csv_path)

        print("Safety checks")
        if "sims" in problem.lower():
            df = self.rename_sims_strategies_to_indicate_objectives(df)

        if self.config.do_safety_checks:
            if self.analysis.strategies.all_instances_per_strategy:
                if not self.safety_check_all_instances_same_number_strategies(df):
                    print(f"Safety checks for all instances having the same number of strategies has failed for {csv_path}")
                    return None, None
            passed_exhaustive_check, sorted_front = self.safety_check_exhaustive(df)
            if not passed_exhaustive_check:
                print(f"Safety checks for exhaustive conditions has failed for {csv_path}")
                return None, None
            if not self.safety_check_fronts(df, sorted_front):
                print(f"Safety checks for fronts has failed for {csv_path}")
                return None, None
            if not self.safety_check_no_duplicates(df, sorted_front):
                print(f"Safety checks for duplicates has failed for {csv_path}")
                return None, None
            print("Safety checks passed")
        else:
            print(f"⚠️ Unless you are sure, it is recommended to enable safety checks")

        figs, data = self.save_all_figs(df, problem)

        metric = self.analysis.hypervolume
        if not (self.config.print_cumulative_hv or self.config.print_hv_histogram or
                self.config.print_sorted_normalized_hv_per_strategy):
            # not hv computed with normalized fronts
            df, metric = self.add_metric_from_plot_data(df, data)

        print("Creating pretty tables")
        if self.config.print_latex_tables:
            self.save_latex_table_text(df, problem, metric)
            print("Creating pretty tables done")

        print("Finished processing")
        return figs, data

    def add_metric_from_plot_data(self, df, data, prefer=("igd_plus", "igd"), include_solver=False):
        """
        Adds one metric column to df from your already-computed plot 'data' dict.

        Returns:
          (df_out, chosen_metric_name or None)

        prefer: ordered list of metric names to try.
        """

        # 1) choose metric that exists in data
        chosen_metric = None
        for m in prefer:
            if m in data:
                chosen_metric = m
                break
        if chosen_metric is None:
            return df, None

        metric_block = data.get(chosen_metric)
        global_block = metric_block.get("global")

        rows = []
        for strat, payload in global_block.items():
            # payload is expected to contain {chosen: [(inst,val), ...]}
            pairs = payload.get(chosen_metric)
            for inst, val in pairs:
                row = {
                    self.analysis.instance: inst,
                    self.analysis.front_strategy: strat,
                    chosen_metric: float(val) if val is not None else np.nan,
                }
                if include_solver:
                    # only if your dict is solver-specific (usually it isn't)
                    row[self.analysis.solver_name] = payload.get("solver", None)
                rows.append(row)

        if not rows:
            return df, None

        df_metric = pd.DataFrame(rows)

        keys = [self.analysis.instance, self.analysis.front_strategy]
        if include_solver:
            keys = [self.analysis.instance, self.analysis.solver_name, self.analysis.front_strategy]

        df_out = df.merge(df_metric, on=keys, how="left")
        return df_out, chosen_metric

    def keep_only_common_instances(self, df, instance_col, strategy_col, required_strategies):
        required = set(required_strategies)
        present = (df.groupby(instance_col)[strategy_col]
                   .apply(lambda s: required.issubset(set(s))))
        keep_instances = present[present].index
        return df[df[instance_col].isin(keep_instances)].copy()

    def add_hv_computed_evolution_if_not_present(self, df, csv_file_path):
        if "hv_computed_evolution" not in df.columns:
            backup_path = csv_file_path.replace(".csv", "_backup.csv")
            shutil.copyfile(csv_file_path, backup_path)
            print(f"Backup of original CSV saved to {backup_path}")

            df["hv_computed_evolution"] = df.apply(
                lambda row: [[t, hv] for t, hv in zip(*self.analysis.compute_hypervolume_vs_time_from_fronts(row))],
                axis=1
            )
            # Save updated DataFrame to CSV
            df.to_csv(csv_file_path, index=False)
            print(f"Modified CSV file with hv_computed_evolution saved to {csv_file_path}")
        return df

    def beautify_strategies_names(self, df):
        df = df.copy()
        df["front_generator"] = df["front_generator"].replace(
            self.strategies.mapping_strategy_names, regex=False
        )
        return df

    def rename_sims_strategies_to_indicate_objectives(self, df):
        # In lagos nigereia there were 145 images, I want to rename them to 150 because it's similar to the other
        # instances with 150 images
        df = df.copy()
        df.instance = df.instance.str.replace("145", "150")
        return df

    def safety_check_all_instances_same_number_strategies(self, df):
        # Count the number of unique strategies
        number_of_strategies = df[self.analysis.front_strategy].nunique()

        # Verify all the groups have the same number of strategies
        group_sizes = df.groupby(self.analysis.instance).size()
        groups_with_missing_strategies = group_sizes[group_sizes != number_of_strategies]
        if groups_with_missing_strategies.size > 0:
            print("There are groups with different numbers of strategies")
            print(groups_with_missing_strategies)
            # save a dataframe with the details using save_temp_csv_for_debugging
            problem_instances = groups_with_missing_strategies.index
            cols_to_show = [
                self.analysis.problem,
                self.analysis.instance,
                self.analysis.front_strategy,
                'datetime'
            ]
            df_to_save = df[df[self.analysis.instance].isin(problem_instances)]
            self.save_temp_csv_for_debugging(df_to_save, problem_instances, cols_to_show,
                                             "different number of strategies for different instances")
            return False
        else:
            print("All the groups have the same number of strategies")
        return True

    # def safety_check_exhaustive_to_delete(self, df):
    #     # Filter only rows where exhaustive == True
    #     df_exhaustive = df[df[self.analysis.exhaustive] == True]
    #
    #     # Group exhaustive=True rows by instance and count unique hypervolumes
    #     hypervolume_diff = df_exhaustive.groupby(self.analysis.instance)[self.analysis.hypervolume].nunique()
    #     hypervolume_diff = hypervolume_diff[hypervolume_diff > 1]
    #
    #     if hypervolume_diff.size > 0:
    #         print("There are groups where exhaustive=True rows have different hypervolumes:")
    #         print(hypervolume_diff)
    #
    #         df_exhaustive = df_exhaustive.copy()
    #         df_exhaustive['pareto_front_size'] = df_exhaustive[self.analysis.pareto_front].apply(
    #             lambda v: 0 if pd.isna(v)
    #             else len(v) if isinstance(v, (list, tuple))
    #             else (len(ast.literal_eval(v)) if isinstance(v, str) else 0)
    #         )
    #
    #         cols_to_show = [
    #             self.analysis.problem,
    #             self.analysis.instance,
    #             self.analysis.front_strategy,
    #             'pareto_front_size',
    #             self.analysis.hypervolume,
    #             self.analysis.exhaustive,
    #             self.analysis.time,
    #             'sum_solutions_resolution_time(s)',
    #             'datetime',
    #             self.analysis.pareto_front,
    #         ]
    #         problem_instances = hypervolume_diff.index
    #         self.save_temp_csv_for_debugging(df_exhaustive, problem_instances, cols_to_show, "different hypervolumes when exhaustive=true")
    #         return False
    #     else:
    #         print("The strategies that are exhaustive produce the same hypervolume")
    #
    #     # 🔹 Count groups where NOT all rows have exhaustive = True
    #     count_not_all_exhaustive = df.groupby(self.analysis.instance)[self.analysis.exhaustive].apply(
    #         lambda x: not x.all()).sum()
    #
    #     total_groups = df[self.analysis.instance].nunique()
    #
    #     print(
    #         f"Number of groups where not all strategies are exhaustive = True: {count_not_all_exhaustive} / {total_groups}")
    #
    #     return True

    def safety_check_exhaustive(self, df):
        df_exhaustive = df[df[self.analysis.exhaustive] == True].copy()

        size_diff = df_exhaustive.groupby(self.analysis.instance)["front_cardinality"].nunique()
        size_diff = size_diff[size_diff > 1]

        if size_diff.size > 0:
            print("There are groups where exhaustive=True rows have different front_cardinality:")
            print(df_exhaustive[df_exhaustive[self.analysis.instance].isin(size_diff.index)]
                  .groupby(self.analysis.instance)["front_cardinality"]
                  .apply(lambda x: sorted(x.unique())))
            return False, None

        def _sorted_front(v):
            if pd.isna(v):
                return tuple()
            if isinstance(v, str):
                front = eval(v)
            else:
                front = v
            if front is None:
                return tuple()
            return tuple(sorted(map(tuple, front)))

        df_exhaustive["sorted_front"] = df_exhaustive[self.analysis.pareto_front].apply(_sorted_front)

        front_diff = df_exhaustive.groupby(self.analysis.instance)["sorted_front"].nunique()
        front_diff = front_diff[front_diff > 1]

        if front_diff.size > 0:
            print("There are groups where exhaustive=True rows have different Pareto fronts (after sorting):")
            print(front_diff)

            df_exhaustive["pareto_front_size"] = df_exhaustive[self.analysis.pareto_front].apply(
                lambda v: 0 if pd.isna(v)
                else len(v) if isinstance(v, (list, tuple))
                else (len(ast.literal_eval(v)) if isinstance(v, str) else 0)
            )

            cols_to_show = [
                self.analysis.problem,
                self.analysis.instance,
                self.analysis.front_strategy,
                "pareto_front_size",
                "front_cardinality",
                "sorted_front",
                self.analysis.exhaustive,
                self.analysis.time,
                "sum_solutions_resolution_time(s)",
                "datetime",
                self.analysis.pareto_front,
            ]

            problem_instances = front_diff.index
            self.save_temp_csv_for_debugging(
                df_exhaustive, problem_instances, cols_to_show,
                "different fronts when exhaustive=true"
            )
            return False, df_exhaustive[["sorted_front"]]
        else:
            print("The strategies that are exhaustive produce the same Pareto front (after sorting)")

        # 🔹 Count groups where NOT all rows have exhaustive = True
        count_not_all_exhaustive = df.groupby(self.analysis.instance)[self.analysis.exhaustive].apply(
            lambda x: not x.all()).sum()

        total_groups = df[self.analysis.instance].nunique()

        print(
            f"Number of groups where not all strategies are exhaustive = True: {count_not_all_exhaustive} / {total_groups}")

        return True, df_exhaustive[["sorted_front"]]

    def save_temp_csv_for_debugging(self, df_to_save, problem_instances, cols_to_show, issue_type):
        details = (
            df_to_save[df_to_save[self.analysis.instance].isin(problem_instances)][cols_to_show]
            .sort_values([self.analysis.instance, self.analysis.front_strategy, 'datetime'])
        )
        print(f"\nDetails for issue: {issue_type}")
        print(details.to_string(index=False))

        # Save details to a CSV file in a temp directory
        # unique problem names (sorted for stable filenames)
        problems = sorted(map(str, df_to_save[self.analysis.problem].dropna().unique()))

        # join and sanitize for filesystem safety
        problems_for_file_name = "-".join(problems)
        problems_for_file_name = re.sub(r"[^A-Za-z0-9._-]+", "_", problems_for_file_name)

        try:
            base_dir = Path(__file__).resolve().parent  # folder of the .py file
        except NameError:
            base_dir = Path.cwd()  # Jupyter / REPL fallback

        out_dir = base_dir / "temp"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{issue_type}_issues_{problems_for_file_name}.csv"
        details.to_csv(out_path, index=False)
        print(f"Saved exhaustive issues details to: {out_path}")

    def safety_check_fronts(self, df, sorted_front=None):
        if df.empty:
            print("DataFrame is empty, skipping problem.")
            return False
        row = df.iloc[0]
        ref_point = row["reference_point"]
        if isinstance(ref_point, str):
            ref_point = ast.literal_eval(ref_point)
        # convert from string to np array
        pareto_front = row[self.analysis.pareto_front]  # is a string
        if isinstance(pareto_front, str):
            pareto_front = ast.literal_eval(pareto_front)
        maximization_problems = is_maximization_problem(pareto_front, ref_point)

        df_filtered = df.dropna(subset=["pareto_front"])
        front_analysis_results = ["instance", "front_generator", "new_value"]
        results_df = pd.DataFrame(columns=front_analysis_results)
        for _, row in df_filtered.iterrows():
            pareto_front_points = row["pareto_front"]
            if isinstance(pareto_front_points, str):
                pareto_front_points = ast.literal_eval(pareto_front_points)

            dominated_points = MoAnalysis.check_points_in_front_are_not_dominated(pareto_front_points,
                                                                                  maximization_problems)
            if len(dominated_points) > 0:
                # Create a new row
                new_row = pd.DataFrame([{
                    "instance": row["instance"],
                    "front_generator": row["front_generator"],
                    "dominated_points": dominated_points
                }])
                # Append to results DataFrame
                results_df = pd.concat([results_df, new_row], ignore_index=True)
        if results_df.empty:
            print("All the points in the pareto front are not dominated")
        else:
            print("There are points in the pareto front that are dominated")
            print(results_df)
            return False
        return True

    def safety_check_no_duplicates(self, df, sorted_front=None):
        key_cols = [self.analysis.instance, self.analysis.front_strategy, self.analysis.timeout]
        dupes = df[df.duplicated(subset=key_cols, keep=False)].sort_values(
            [self.analysis.instance, self.analysis.front_strategy, self.analysis.timeout, 'datetime'])

        if dupes.empty:
            print("There are no duplicates")
            return True
        else:
            print("There are duplicates. Below the details:")
            print(dupes[[self.analysis.instance, self.analysis.front_strategy, self.analysis.timeout, 'datetime']].to_string(index=False))
            cols_to_show = [
                self.analysis.problem,
                self.analysis.instance,
                self.analysis.front_strategy,
                self.analysis.timeout,
                self.analysis.exhaustive,
                self.analysis.time,
                'sum_solutions_resolution_time(s)',
                'datetime',
                self.analysis.pareto_front,
            ]
            problem_instances = dupes.instance.unique()
            self.save_temp_csv_for_debugging(dupes, problem_instances, cols_to_show, "duplicate rows")
            return False

    def remove_duplicates_and_save(self, df, strategies, dates, csv_path):
        key_cols = [self.analysis.instance, self.analysis.front_strategy, self.analysis.timeout]
        for strategy, cutoff_str in zip(strategies, dates):
            df_strategy = df[df[self.analysis.front_strategy] == strategy]
            dupes = df_strategy[df_strategy.duplicated(subset=key_cols, keep=False)]
            if not dupes.empty:
                # cutoff_date = pd.to_datetime(cutoff_str)

                # groups (by key_cols) that have at least one row >= cutoff
                has_newer = (
                    df_strategy.groupby(key_cols)['datetime']
                    .transform(lambda s: (s >= cutoff_str).any())
                )

                # drop only the older rows in groups that have a newer (>= cutoff)
                drop_mask = (df_strategy['datetime'] < cutoff_str) & has_newer

                df_strategy = df_strategy[~drop_mask]

                df = df[df[self.analysis.front_strategy] != strategy]
                df = pd.concat([df, df_strategy], ignore_index=True)

        df.to_csv(csv_path, index=False)


    def save_latex_table_text(self, df, problem, metric=None):
        # Get evaluation of the experiments like in the Disjunctive Programming paper for exhasutive and non exhaustive instances
        # todo select the part of the df in which you're interested
        df_for_tables = df

        # stats to display
        stats_exhaustive = [["time(s)", "front_cardinality"], ["sum_solutions_nodes", "front_cardinality"]]
        stats_exhaustive_pretty_name = [["time(s)"], ["nodes"]]

        text_to_save = ""
        non_stats_headers = None  # they are inside the code
        # title_exhaustive = "Comparison strategies when all exhaustive"
        # title_non_exhaustive = "Comparison strategies when not all exhaustive"

        # If metric is provided, switch non-exhaustive tables to use it (instead of hypervolume)
        if metric is not None and metric is not self.analysis.hypervolume:
            self.analysis.set_stats_non_exhaustive_for_metric(metric)

        for stats, stats_pretty_name in zip(stats_exhaustive, stats_exhaustive_pretty_name):
            self.analysis.set_stats_exhaustive(stats, stats_pretty_name)
            if problem == "MUKP":
                table_results = self.analysis.average_similar_ukp_moolibrary_instances(df, non_stats_headers)
            elif problem == "MN-Queens":
                table_results = self.analysis.average_similar_nqueens_instances(df, non_stats_headers)
            elif problem == "MORCPSP":
                table_results = self.analysis.average_similar_rcpsp_instances(df, non_stats_headers)
            elif problem == "SIMS":
                # todo deal with sims correctly
                table_results = self.analysis.average_similar_sims_instances(df, non_stats_headers)
            else:
                print("The problem is not recognized")
                table_results = [None, None]

            problem_capitalized = problem.upper()
            title_exhaustive = f"{problem_capitalized}. Comparison strategies when all exhaustive"
            title_non_exhaustive = f"{problem_capitalized}. Comparison strategies when not all exhaustive"

            if table_results[0] is not None:
                non_stats_headers = ["K", "n", "instances", "p"]
                table_exhaustive_latex = self.analysis.disjunctive_paper_style_dataframe_to_latex(table_results[0],
                                                                                                  non_stats_headers,
                                                                                                  True,
                                                                                                  title_exhaustive)
                print(table_exhaustive_latex)
                text_to_save += "\n\n"
                text_to_save += table_exhaustive_latex
        if table_results[1] is not None:
            if self.config.perform_front_normalization and not self.normalized_fronts[problem]:
                self.normalize_fronts_in_df(df, problem)
            non_stats_headers = ["K", "n", "instances"]
            table_non_exhaustive_latex = self.analysis.disjunctive_paper_style_dataframe_to_latex(table_results[1],
                                                                                                  non_stats_headers,
                                                                                                  False,
                                                                                                  title_non_exhaustive)
            print("---------------Comparison strategies when not all exhaustive----------------------")
            print(table_non_exhaustive_latex)
            text_to_save += "\n\n"
            text_to_save += table_non_exhaustive_latex
        # Save the text to a file
        output_dir = os.path.join(self.config.folder_path, problem)
        os.makedirs(output_dir, exist_ok=True)
        if self.normalized_fronts[problem]:
            normalized_fronts_dir = os.path.join(output_dir, self.normalized_fronts_folder)
            os.makedirs(normalized_fronts_dir, exist_ok=True)
            text_file_path = os.path.join(normalized_fronts_dir, f"table_results_{problem}.tex")
        else:
            text_file_path = os.path.join(output_dir, f"table_results_{problem}.tex")
        with open(text_file_path, "w") as text_file:
            text_file.write(text_to_save)
        print(f"Saved table results to '{text_file_path}'")

    def save_all_figs(self, df, problem):
        print("Creating plots")
        set_general_plot_style()
        if problem == "MUKP":
            objs_elements, pattern_template = get_info_similar_instances_ukp_moolibrary()
        elif problem == "MN-Queens":
            objs_elements, pattern_template = get_info_similar_instances_nqueens()
        elif problem == "MORCPSP":
            objs_elements, pattern_template = get_info_similar_instances_rcpsp()
        elif problem == "SIMS":
            # todo deal with sims correctly
            objs_elements, pattern_template = get_info_similar_instances_sims()
        else:
            raise ValueError("The problem is not recognized")

        figs = {}
        data = {}
        if self.config.perform_front_normalization and (not self.normalized_fronts[problem] and (self.config.print_hv_evolution or self.config.print_cumulative_hv or self.config.print_hv_histogram or
                                                                                                 self.config.print_sorted_normalized_hv_per_strategy or
                                                                                                 self.config.print_sorted_igd_per_strategy or self.config.print_sorted_igd_plus_per_strategy)):
            # normalize fronts and compute new hypervolumes and reference point
            self.normalize_fronts_in_df(df, problem)

        if self.config.print_time_vs_instances:
            figs = self.analysis.plot_time_vs_completed_instances_for_problem(df, problem, objs_elements,
                                                                              pattern_template, figs)
        if self.config.print_hv_evolution:
            figs = self.analysis.plot_normalized_hypervolume_evolution(df, problem, objs_elements, pattern_template,
                                                                       figs, plot_variance=False)
        if self.config.print_cumulative_hv:
            data['cumulative_hv_vs_time'] = self.analysis.get_cumulative_hv_vs_time_data(df, problem, objs_elements, pattern_template)
            figs = self.analysis.plot_cumulative_hv_vs_time(data['cumulative_hv_vs_time'], problem, figs)
        if self.config.print_hv_histogram:
            figs = self.analysis.plot_best_hypervolume_histogram(df, problem, objs_elements, pattern_template, figs)
        if self.config.print_sorted_normalized_hv_per_strategy:
            data['hv_per_strategy'] = self.analysis.get_normalized_hv_per_strategy_data(df, problem, objs_elements, pattern_template, True)
            figs = self.analysis.plot_normalized_hypervolume_per_strategy(df, problem, objs_elements, pattern_template,
                                                                        figs, data['hv_per_strategy'])
        if self.config.print_sorted_contribution_per_strategy:
            data['contribution_joint_front'] = self.analysis.get_contribution_to_joint_front_data(df, problem, objs_elements, pattern_template, True)
            figs = self.analysis.plot_normalized_contribution_per_strategy(df, problem, objs_elements, pattern_template,
                                                                       figs, data['contribution_joint_front'])
        if self.config.print_sorted_igd_per_strategy:
            data['igd'] = self.analysis.compute_igd_per_strategy(df, problem, objs_elements, pattern_template)
            figs = self.analysis.plot_igd_per_strategy(df, problem, objs_elements, pattern_template, figs, data['igd'])
        if self.config.print_sorted_igd_plus_per_strategy:
            data['igd_plus'] = self.analysis.compute_igd_plus_per_strategy(df, problem, objs_elements, pattern_template, True)
            figs = self.analysis.plot_igd_plus_per_strategy(df, problem, objs_elements, pattern_template, figs, data['igd_plus'])
        if self.config.print_sorted_normalized_time_per_strategy:
            data['time_per_strategy'] = self.analysis.get_normalized_time_per_strategy_data(df, objs_elements, pattern_template, True)
            figs = self.analysis.plot_normalized_time_per_strategy(df, problem, objs_elements, pattern_template,
                                                                   figs, data['time_per_strategy'], self.config.log_threshold)

        if figs:
            import os
            import matplotlib.pyplot as plt

            output_dir = os.path.join(self.config.folder_path, problem)
            os.makedirs(output_dir, exist_ok=True)

            for name, fig in figs.items():
                # Infer plot type from name
                if "igd_plus_sorted_per_instance" in name:
                    plot_type = "igd_plus"
                elif "igd_sorted_per_instance" in name:
                    plot_type = "igd"
                elif "normalized_hv_sorted_per_instance" in name:
                    plot_type = "hv"
                elif "normalized_contribution_sorted" in name:
                    plot_type = "hv_contribution"
                elif "time_sorted_per_instance" in name:
                    plot_type = "time"
                else:
                    plot_type = None
                fig = self.clean_figure_for_paper(fig, plot_type)

                if self.normalized_fronts[problem]:
                    normalized_fronts_dir = os.path.join(output_dir, self.normalized_fronts_folder)
                    os.makedirs(normalized_fronts_dir, exist_ok=True)
                    fig_path = os.path.join(normalized_fronts_dir, f"{name}.pdf")
                else:
                    fig_path = os.path.join(output_dir, f"{name}.pdf")
                fig.savefig(fig_path, format='pdf', bbox_inches='tight')
                plt.close(fig)  # optional: frees memory if you're done
            print(f"Saved {len(figs)} figures to '{output_dir}'")
        print("Creating plots done")
        return figs, data

    def clean_figure_for_paper(self, fig, plot_type=None):
        for ax in fig.axes:
            # Remove title for paper version
            ax.set_title("")

            # ax.set_xlabel("Instance rank (per strategy)", fontsize=18)
            if plot_type is not None:
                ax.set_xlabel("Instance rank (per strategy)")

            # Enable grid with rcParams styling
            ax.grid(True, which='both')

            # Adjust legend
            legend = ax.get_legend()
            if legend:
                # Legend position by plot type + compact sizing
                if plot_type in ["igd", "igd_plus", "time"]:
                    ax.legend(loc='upper left', frameon=True,
                              handlelength=1.6, markerscale=0.8,
                              handletextpad=0.6, labelspacing=0.3, borderaxespad=0.3)
                elif plot_type in ["hv", "hv_contribution"]:
                    ax.legend(loc='lower left', frameon=True,
                              handlelength=1.6, markerscale=0.8,
                              handletextpad=0.6, labelspacing=0.3, borderaxespad=0.3)
                else:
                    ax.legend(loc='best', frameon=True,
                              handlelength=1.6, markerscale=0.8,
                              handletextpad=0.6, labelspacing=0.3, borderaxespad=0.3)

                # Re-fetch legend AFTER ax.legend(...) so we style the final one
                legend = ax.get_legend()
                if legend:
                    legend.set_title("")
                    for text in legend.get_texts():
                        text.set_fontfamily('serif')
                        text.set_fontsize(18)

        return fig

    def normalize_fronts_in_df(self, df, problem=None):
        # df already filtered to compared strategies; modify in place
        print("Normalizing fronts in DataFrame")

        reference_point = "reference_point"
        ideal_col = "ideal_point"
        nadir_col = "nadir_point"

        # ---- detect maximize/minimize (need real arrays, not strings)
        row0 = df.iloc[0]
        ref_point0 = row0[reference_point]
        front0 = row0[self.analysis.pareto_front]

        if isinstance(ref_point0, str):
            ref_point0 = ast.literal_eval(ref_point0)
        if isinstance(front0, str):
            front0 = ast.literal_eval(front0)

        maximize = is_maximization_problem(np.array(front0), np.array(ref_point0))

        reference_point_ratio = 1.1  # leave some space beyond worst point so edge points contribute to HV

        # duplicate absolute values before normalization
        df[self.analysis.pareto_front + "_absolute_obj"] = df[self.analysis.pareto_front]
        df[reference_point + "_absolute_obj"] = df[reference_point]
        # df[self.analysis.hypervolume + "_absolute_obj"] = df[self.analysis.hypervolume]

        # ---- per instance: combined ideal / combined nadir for normalization
        grouped = df.groupby(self.analysis.instance)

        ideal_points = {}
        nadir_points = {}
        instance_all_exhaustive = {}
        for instance, group in grouped:
            # if exhaustive exists, take its ideal (first one)
            exhaustive_strategies = group[group[self.analysis.exhaustive] == True]

            if not exhaustive_strategies.empty:
                hv_vals = group[self.analysis.hypervolume]
                if len(exhaustive_strategies) == len(group) and hv_vals.nunique(dropna=False) == 1:
                    if hv_vals.iloc[0] == 0:
                        df.loc[group.index, self.analysis.hypervolume] = 1.0
                    instance_all_exhaustive[instance] = True
                    continue
                ip = exhaustive_strategies.iloc[0][ideal_col]
                ideal_point = ast.literal_eval(ip) if isinstance(ip, str) else ip
            else:
                # best ideal among strategies
                ideal_point = None
                for _, r in group.iterrows():
                    ic = r[ideal_col]
                    ideal_candidate = ast.literal_eval(ic) if isinstance(ic, str) else ic
                    if ideal_candidate is None or (isinstance(ideal_candidate, float) and np.isnan(ideal_candidate)):
                        continue
                    if ideal_point is None:
                        ideal_point = list(ideal_candidate)
                    else:
                        if maximize:
                            ideal_point = [max(i, j) for i, j in zip(ideal_point, ideal_candidate)]
                        else:
                            ideal_point = [min(i, j) for i, j in zip(ideal_point, ideal_candidate)]

            # worst nadir among strategies
            nadir_point = None
            for _, r in group.iterrows():
                nc = r[nadir_col]
                nadir_candidate = ast.literal_eval(nc) if isinstance(nc, str) else nc
                if nadir_candidate is None or (isinstance(nadir_candidate, float) and np.isnan(nadir_candidate)):
                    continue
                if nadir_point is None:
                    nadir_point = list(nadir_candidate)
                else:
                    if maximize:
                        nadir_point = [min(i, j) for i, j in zip(nadir_point, nadir_candidate)]
                    else:
                        nadir_point = [max(i, j) for i, j in zip(nadir_point, nadir_candidate)]

            ideal_points[instance] = ideal_point
            nadir_points[instance] = nadir_point

        # ---- normalize fronts + set normalized reference point + recompute HV
        # normalized reference: [1.1, 1.1, ...] (since normalized space is [0,1], worst=1)
        # compute_hv expects minimization (good): our normalization produces 0=best, 1=worst for BOTH problem types
        for instance, group in grouped:
            if instance in instance_all_exhaustive:
                # all strategies exhaustive with same HV -> skip normalization
                continue
            ideal_point = ideal_points.get(instance)
            nadir_point = nadir_points.get(instance)
            if ideal_point is None or nadir_point is None:
                raise ValueError(f"There should be an ideal and nadir point in instance '{instance}'.")

            ideal_arr = np.array(ideal_point, dtype=float)
            nadir_arr = np.array(nadir_point, dtype=float)

            denom = abs(nadir_arr - ideal_arr)
            # avoid division by zero per objective (degenerate dimension)
            denom_safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)

            norm_ref = np.ones_like(ideal_arr, dtype=float) * reference_point_ratio

            # iterate rows of this instance and update df in place
            for idx, r in group.iterrows():
                front = r[self.analysis.pareto_front + "_absolute_obj"]
                if pd.isna(front):
                    raise ValueError(f"Front cannot be NaN in instance '{instance}' and strategy '{r[self.analysis.front_strategy]}'.")
                if isinstance(front, str):
                    try:
                        front = ast.literal_eval(front)
                    except Exception:
                        raise ValueError(f"Front cannot be parsed in instance '{instance}' and strategy '{r[self.analysis.front_strategy]}': {front}")

                front_arr = np.array(front, dtype=float)
                if front_arr.size == 0:
                    raise ValueError(f"Front cannot be empty in instance '{instance}' and strategy '{r[self.analysis.front_strategy]}'.")

                # ---- min-max normalization to 0(best) .. 1(worst)
                # normalized = (value - ideal) / (nadir - ideal)
                norm_front = abs(front_arr - ideal_arr) / denom_safe

                # clip numerical noise slightly outside [0,1]
                norm_front = np.clip(norm_front, 0.0, 1.0)

                # write normalized front + normalized reference point
                # df.at[idx, self.analysis.pareto_front] = norm_front.tolist()
                # df.at[idx, reference_point] = norm_ref.tolist()
                df.at[idx, self.analysis.pareto_front] = str(norm_front.tolist())
                df.at[idx, reference_point] = str(norm_ref.tolist())

                # recompute HV on normalized front if it is required for later analysis
                if self.config.print_cumulative_hv or self.config.print_hv_histogram or self.config.print_sorted_normalized_hv_per_strategy:
                    try:
                        hv = compute_hv(norm_front, norm_ref, double_check_non_dominance=False)
                        df.at[idx, self.analysis.hypervolume] = float(hv)
                    except Exception:
                        # keep old value if something goes wrong
                        raise ValueError(f"Could not compute hypervolume for normalized front in instance '{instance}' and strategy '{r[self.analysis.front_strategy]}'.")
            print(f"Computed normalization for instance '{instance}'.")

        if problem is not None:
            self.normalized_fronts[problem] = True

    @staticmethod
    def check_if_two_fronts_are_equal(f1, f2):
        """
        Returns True iff both fronts contain the same points (as a *set*, i.e., ignoring order).
        Also prints:
          - original lengths
          - unique-set lengths
          - points in f1 not in f2
          - points in f2 not in f1

        Notes:
          - This treats points as identical if their coordinate triples are identical.
          - If you want "same multiset" (i.e., duplicates must match too), you can add a Counter check,
            but you asked to stop at the set difference lists.
        """
        # Basic length check (list lengths; can differ even if sets are equal because of duplicates)
        print("len(f1) =", len(f1), "len(f2) =", len(f2))

        # Convert to sets of tuples (hashable)
        s1 = set(tuple(p) for p in f1)
        s2 = set(tuple(p) for p in f2)

        print("len(set(f1)) =", len(s1), "len(set(f2)) =", len(s2))

        if s1 == s2:
            print("same set", s1 == s2)
            return True
        print("different sets")

        # Collect explicit differences by membership checks (as you requested)
        front1_not_in_f2 = []
        for p in s1:
            if p not in s2:
                front1_not_in_f2.append(p)

        front2_not_in_f1 = []
        for p in s2:
            if p not in s1:
                front2_not_in_f1.append(p)

        print("points in f1 (unique) not in f2:", len(front1_not_in_f2))
        print("points in f2 (unique) not in f1:", len(front2_not_in_f1))

        # Optional: print a few examples (helps debugging huge fronts)
        if front1_not_in_f2:
            print("example f1_not_in_f2:", front1_not_in_f2[0])
        if front2_not_in_f1:
            print("example f2_not_in_f1:", front2_not_in_f1[0])

        # If you want the full lists accessible to the caller, return them too.
        # (Returning them is often more useful than printing only.)
        return (s1 == s2), front1_not_in_f2, front2_not_in_f1

    def plot_pareto_fronts(self, problem_name, plot_reference_front=True, instances_list=None, normalize=True):
        for csv_path, problem in self.csv_paths_problem:
            if problem == problem_name:
                problem_path = csv_path
                break
        else:
            raise ValueError(f"Problem name '{problem_name}' not found in csv paths")

        df_front = self.analysis.csv_to_df(problem_path)

        # --- SAME PIPELINE AS save_results() ---

        # 1) keep only sources that MoAnalysis knows about
        current_sources = set(self.analysis.strategies.strategies_original_name_list)
        df_front = df_front.loc[df_front[self.analysis.front_strategy].isin(current_sources)]

        # 2) expand/filter/rename into the compared strategies (better names)
        df_front = self.strategies.apply_specs(df_front, strategy_col=self.analysis.front_strategy)

        # 3) keep only instances where all compared strategies exist
        if not self.analysis.strategies.all_instances_per_strategy:
            df_front = self.keep_only_common_instances(
                df_front,
                instance_col=self.analysis.instance,
                strategy_col=self.analysis.front_strategy,
                required_strategies=self.analysis.strategies.strategies_better_name
            )

        # default: all instances
        if instances_list is None:
            instances_list = df_front[self.analysis.instance].unique()

        if normalize:
            self.normalize_fronts_in_df(df_front)

        figs = {}
        return self.analysis.plot_fronts(
            df_front,
            instances_list,
            figs,
            plot_reference_front=plot_reference_front
        )
        # for saving:
        # fig.savefig(fig_path, format='pdf', bbox_inches='tight')


class FiguresTablesToPrint:
    def __init__(
            self,
            do_safety_checks=True,
            log_threshold=50,
            print_time_vs_instances=True,
            print_cumulative_hv=True,
            print_hv_histogram=True,
            print_latex_tables=True,
            print_sorted_normalized_hv_per_strategy=True,
            print_sorted_contribution_per_strategy=True,
            print_sorted_igd_per_strategy=True,
            print_sorted_igd_plus_per_strategy=True,
            print_sorted_normalized_time_per_strategy=True,
            perform_front_normalization=True,
            folder_path="cp2025",
            print_hv_evolution=False
    ):
        self.do_safety_checks = do_safety_checks
        self.log_threshold = log_threshold
        self.print_time_vs_instances = print_time_vs_instances
        self.print_cumulative_hv = print_cumulative_hv
        self.print_hv_histogram = print_hv_histogram
        self.print_latex_tables = print_latex_tables
        self.print_sorted_normalized_hv_per_strategy = print_sorted_normalized_hv_per_strategy
        self.print_sorted_contribution_per_strategy = print_sorted_contribution_per_strategy
        self.print_sorted_igd_per_strategy = print_sorted_igd_per_strategy
        self.print_sorted_igd_plus_per_strategy = print_sorted_igd_plus_per_strategy
        self.print_sorted_normalized_time_per_strategy = print_sorted_normalized_time_per_strategy
        self.perform_front_normalization = perform_front_normalization
        self.folder_path = folder_path
        self.print_hv_evolution = print_hv_evolution
        self.print_config_lines()

    def __repr__(self):
        return (f"FiguresTablesToPrint("
                f"do_safety_checks={self.do_safety_checks}, "
                f"log_threshold={self.log_threshold}, "
                f"time_vs_instances={self.print_time_vs_instances}, "
                f"hv_evolution={self.print_hv_evolution}, "
                f"cumulative_hv={self.print_cumulative_hv}, "
                f"hv_histogram={self.print_hv_histogram}, "
                f"latex_tables={self.print_latex_tables}),"
                f"sorted_normalized_hv_per_strategy={self.print_sorted_normalized_hv_per_strategy},"
                f"sorted_contribution_hv_per_strategy={self.print_sorted_contribution_per_strategy},"
                f"print_sorted_igd_per_strategy={self.print_sorted_igd_per_strategy},"
                f"print_sorted_igd_plus_per_strategy={self.print_sorted_igd_plus_per_strategy},"
                f"print_sorted_normalized_time_per_strategy={self.print_sorted_normalized_time_per_strategy},"
                f"perform_front_normalization={self.perform_front_normalization}, ")

    def print_config_lines(self):
        def symbol(value, is_safety=False):
            if value:
                return "✅"
            else:
                return "⚠️" if is_safety else "❌"

        print("\n🛠️ Configuration for printing figures and tables:")
        print(f"{symbol(self.do_safety_checks, is_safety=True)} Do safety checks: {self.do_safety_checks}")
        print(f"{symbol(self.log_threshold)} Threshold to use log scale: {self.log_threshold}")
        print(f"{symbol(self.print_time_vs_instances)} Print time vs instances: {self.print_time_vs_instances}")
        print(f"{symbol(self.print_hv_evolution)} Print HV evolution: {self.print_hv_evolution}")
        print(f"{symbol(self.print_cumulative_hv)} Print cumulative HV: {self.print_cumulative_hv}")
        print(f"{symbol(self.print_hv_histogram)} Print HV histogram: {self.print_hv_histogram}")
        print(f"{symbol(self.print_latex_tables)} Print LaTeX tables: {self.print_latex_tables}")
        print(
            f"{symbol(self.print_sorted_normalized_hv_per_strategy)} Print sorted normalized HV per strategy: {self.print_sorted_normalized_hv_per_strategy}")
        print(
            f"{symbol(self.print_sorted_contribution_per_strategy)} Print sorted contribution HV per strategy: {self.print_sorted_contribution_per_strategy}")
        print(
            f"{symbol(self.print_sorted_igd_per_strategy)} Print sorted IGD per strategy: {self.print_sorted_igd_per_strategy}")
        print(
            f"{symbol(self.print_sorted_igd_plus_per_strategy)} Print sorted IGD+ per strategy: {self.print_sorted_igd_plus_per_strategy}")
        print(
            f"{symbol(self.print_sorted_normalized_time_per_strategy)} Print sorted normalized time per strategy: {self.print_sorted_normalized_time_per_strategy}")
        print(
            f"{symbol(self.perform_front_normalization)} Perform front normalization: {self.perform_front_normalization}")
        if not self.perform_front_normalization:
                print("⚠️ Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: Front normalization is disabled. This may lead to misleading comparisons in HV and IGD+ if the objectives have different scales!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Folder path: {self.folder_path}")
        print("")  # Just a clean newline


class Strategies:
    def __init__(self, mapping_or_specs):
        if isinstance(mapping_or_specs, dict):
            self.specs = [StrategySpec(label=v, source=k, query=None)
                          for k, v in mapping_or_specs.items()]
        else:
            self.specs = list(mapping_or_specs)

        self.all_instances_per_strategy = all(getattr(s, "query", None) is None for s in self.specs)
        self.strategies_original_name_list = list(dict.fromkeys(s.source for s in self.specs))
        self.strategies_better_name = [s.label for s in self.specs]

        # keep your existing colors/markers logic (just keyed by label now)
        self.colors = self._build_colors(self.strategies_better_name)
        self.markers = self._build_markers(self.strategies_better_name)

    def _build_colors(self, labels):
        return dict(zip(labels, sns.color_palette("colorblind", len(labels))))

    def _build_markers(self, labels):
        base = ['o', 's', '^', 'D', 'X', '*', 'P', 'v', '>', '<', 'h', 'H', 'd', '8', 'p']
        return {lab: base[i % len(base)] for i, lab in enumerate(labels)}

    def apply_specs(self, df, strategy_col="front_generator"):
        parts = []
        for s in self.specs:
            d = df[df[strategy_col] == s.source]
            if s.query:
                d = d.query(s.query)
            d = d.copy()
            d[strategy_col] = s.label
            parts.append(d)
        if not parts:
            return df.iloc[0:0].copy()
        return pd.concat(parts, ignore_index=True)

class Cols:
    TIME_FOR_TIME_SCORE = 'time_for_time_score'
    TIME_SCORE_FOR_LEX_SCORE = 'time_score_for_lex_score'  # here the best score is the maximum: 1, representing the
    # minimum time
    TIME_SCORE = 'time_score'  # here the best score is the minimum: 1, representing the minimum time
    HV_SCORE = 'hypervolume_score'
    HV_AVG_SCORE = 'hypervolume_average_score'
    HV_BEST = 'hypervolume_best'
    LEX_SCORE = 'score'
    LEX_BEST = 'lex_best'
    LEX_AVG_SCORE = 'lex_average_score'


class Metrics:

    def __init__(self, name, minimization):
        self.name = name
        if not minimization:
            self.minimization = True
        else:
            self.minimization = minimization


def for_test():
    csv_file_path_problem = [
        # (
        #     "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/ukp/saugmecon_gava_gias/mo_saugmecon_gava_gias_solutions_and_stats.csv",
        #     "MUKP"
        # )  # ,
        # (
        #     "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/nqueens/saug_gava_gias_disj/mo_saug_gava_gias_disj_solutions_and_stats.csv",
        #     "MN-Queens"
        # ),
        # (
        #     "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/rcpsp/saug_gava_disjunctive_gias_10800/mo_saug_gava_disjunctive_gias_18000_solutions_and_stats.csv",
        #     "MORCPSP"
        # ),
        (
            "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/sims/fix_saug_10800_timeout/mo_fix_saug_10800_timeout_solutions_and_stats.csv",
            "SIMS"
        )
    ]
    print_config = FiguresTablesToPrint(
        do_safety_checks=False,
        log_threshold=50,
        print_time_vs_instances=False,
        print_cumulative_hv=False,
        print_hv_histogram=False,
        print_latex_tables=True,
        print_sorted_normalized_hv_per_strategy=True,
        print_sorted_contribution_per_strategy=False,
        print_sorted_igd_per_strategy=False,
        print_sorted_igd_plus_per_strategy=True,
        print_sorted_normalized_time_per_strategy=False,
        perform_front_normalization=True,
        folder_path="debug-and-testing",
        print_hv_evolution=False
    )

    # mapping_strategy_names = {
    #     "GIA_boundedLazy": "GIAubL",
    #     "GIA_bounded": "GIAub",
    #     "ParetoGavanelliGlobalConstraint": "Gavanelli",
    #     "ParetoDisjunctiveProgramming": "DisjProg",
    #     "SaugmeconNoR": "SaugmeconNoR",
    #     "GIA": "GIA",
    # }
    strategy_specs = [
        StrategySpec(label="SAUGMECON", source="Saugmecon",
                     query="solver_version == 'v5.0.0' and search_strategy == 'domOverWDegSearch'"),
        StrategySpec(label="MOBAB-CP-wd", source="Gavanelli",
                     query="solver_version == 'v5.0.0' and search_strategy == 'domOverWDegSearch'"),
    ]

    mapping_strategy_names = {
        "ParetoGavanelliGlobalConstraint": "MOBAB-CP",
        "GIA": "GIA",
    }

    strategies_styled = Strategies(strategy_specs)
    # strategies_styled = Strategies(mapping_strategy_names)

    # runner = SaveAllResultsCP2025(csv_file_path_problem, print_config, strategies_styled)
    runner = SaveAllResultsCP2025(csv_file_path_problem, print_config, strategies_styled)
    figs_paper, data_paper = runner.save_all_results()
    figs_fronts = runner.plot_pareto_fronts("SIMS")



# add main function to run the analysis
if __name__ == '__main__':
    just_test = True
    if just_test:
        for_test()
    else:
        analysis = MoAnalysis()
        # prepare folder to save data
        figs_general_stats = {}
        csvs = {}
        fig_hv_time = {}
        figs_fronts = {}

        # todo for test copy code here for quick test
        csv_file_path = "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/ukp/saugmecon_gava_gias/mo_saugmecon_gava_gias_solutions_and_stats.csv"

        problem = "ukp"  # ukp, nqueens, rcpsp, sims_cost_clouds, automotive, flowshop_permutation

        df = analysis.csv_to_df(csv_file_path)

        df.front_generator = df.front_generator.str.replace("GIA_boundedLazy", "GIAubL")
        df.front_generator = df.front_generator.str.replace("GIA_bounded", "GIAub")
        df.front_generator = df.front_generator.str.replace("ParetoGavanelliGlobalConstraint", "Gavanelli")
        df.front_generator = df.front_generator.str.replace("ParetoDisjunctiveProgramming", "DisjProg")

        # if problem == "MUKP":
        #     objs_elements, pattern_template = get_info_similar_instances_ukp_moolibrary()
        # elif problem == "MN-Queens":
        #     objs_elements, pattern_template = get_info_similar_instances_nqueens()
        # elif problem == "MORCPSP":
        #     objs_elements, pattern_template = get_info_similar_instances_rcpsp()
        # elif problem == "sims":
        #     # todo deal with sims correctly
        #     objs_elements, pattern_template = get_info_similar_instances_sims()
        #     to_rename = SaveAllResultsCP2025([(csv_file_path, problem)], FiguresTablesToPrint())
        #     df = to_rename.rename_sims_strategies_to_indicate_objectives(df)
        #
        # data = analysis.compute_igd_per_strategy(df, problem, objs_elements, pattern_template)
        # figs = {}
        # figs = analysis.plot_normalized_contribution_per_strategy(df, problem, objs_elements, pattern_template, figs, data)

        # figsHVTime = {}
        # figsHVTime = analysis.plot_normalized_hypervolume_evolution(df, problem, objs_elements, pattern_template, figsHVTime,
        #                                                             plot_variance=True)
        # Get evaluation of the experiments like in the Disjunctive Programming paper for exhasutive and non exhaustive instances
        # todo select the part of the df in which you're interested
        df_for_tables = df
        comparison_strategies = ["GIA", "GIAubL", "GIAub"]
        df_reduced = df[df[analysis.front_strategy].isin(comparison_strategies)]
        df_for_tables = df_reduced

        # stats to display
        stats_exhaustive = [["time(s)", "front_cardinality"], ["sum_solutions_nodes", "front_cardinality"],
                            ["sum_solutions_propagations", "front_cardinality"]]
        stats_exhaustive_pretty_name = [["time(s)"], ["nodes"], ["propagations"]]

        non_stats_headers = None  # they are inside the code
        title_exhaustive = "Comparison strategies when all exhaustive"
        title_non_exhaustive = "Comparison strategies when not all exhaustive"
        for stats, stats_pretty_name in zip(stats_exhaustive, stats_exhaustive_pretty_name):
            analysis.set_stats_exhaustive(stats, stats_pretty_name)
            if problem == "MUKP":
                table_results = analysis.average_similar_ukp_moolibrary_instances(df_for_tables, non_stats_headers)
            elif problem == "MN-Queens":
                table_results = analysis.average_similar_nqueens_instances(df_for_tables, non_stats_headers)
            elif problem == "MORCPSP":
                table_results = analysis.average_similar_rcpsp_instances(df_for_tables, non_stats_headers)
            elif problem == "sims_cost_clouds":
                # todo deal with sims correctly
                table_results = analysis.average_similar_sims_instances(df_for_tables, non_stats_headers)
            else:
                print("The problem is not recognized")
                table_results = [None, None]

            if table_results[0] is not None:
                non_stats_headers = ["K", "n", "instances", "p"]
                table_exhaustive_latex = analysis.disjunctive_paper_style_dataframe_to_latex(table_results[0],
                                                                                             non_stats_headers, True,
                                                                                             title_exhaustive)
                print(table_exhaustive_latex)
        if table_results[1] is not None:
            non_stats_headers = ["K", "n", "instances"]
            table_non_exhaustive_latex = analysis.disjunctive_paper_style_dataframe_to_latex(table_results[1],
                                                                                             non_stats_headers, False,
                                                                                             title_non_exhaustive)
            print("---------------Comparison strategies when not all exhaustive----------------------")
            print(table_non_exhaustive_latex)
