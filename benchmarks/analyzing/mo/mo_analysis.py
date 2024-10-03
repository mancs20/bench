import ast

import pandas as pd
import seaborn as sns
import numpy as np  # Needed for log transformation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
import os


class MoAnalysis:

    def __init__(self, benchmark='benchmark', problem='problem', instance='instance', solver_name='solver',
                 front_strategy='front_generator',
                 hypervolume='hypervolume', pareto_front='pareto_front',
                 hypervolume_evolution='hypervolume_evolution', number_of_solutions='front_cardinality',
                 exhaustive='exhaustive', time='time(s)', solutions_in_time='solutions_in_time',
                 time_solver_sec=None, pareto_solutions_time_list=None, all_solutions='all_solutions'):
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

    @staticmethod
    def csv_to_df(file_path):
        df = pd.read_csv(file_path, delimiter=',')
        return df

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
        general_df = df  # Use the entire DataFrame for the general case
        general_problem_name = 'All instances'
        self.plot_metric_for_problem(general_df, metric_col, maximize, group_by_solver, general_problem_name, figs,
                                     csvs)

        # Step 2: Create individual plots for each problem
        problems = df['problem'].unique()  # Get unique problems
        for problem in problems:
            df_problem = df[df['problem'] == problem]  # Filter DataFrame for the current problem
            self.plot_metric_for_problem(df_problem, metric_col, maximize, group_by_solver, problem, figs, csvs)

        return figs, csvs

    def plot_metric_for_problem(self, df, metric_col, maximize, group_by_solver, problem_name, figs, csvs):
        """
        Plots the metric for the given problem or for all instances if `problem_name` is "All instances".
        Saves the plot and the dataframe.
        """
        df = self.apply_timeout_limit(df, metric_col)

        # Group by instance to compare solvers and strategies for each instance
        grouped = df.groupby('instance')

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
        ax.grid(False)
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
            figs, csvs = self._plot_instance_problem_helper(df_problem, metric_col, metric_data, maximize, figs, csvs, problem,
                                                            group_by_solver)

        return figs, csvs

    def plot_metric_by_instance_problem_normalized(self, df, metric_col, maximize, figs=None, csvs=None, group_by_solver=False):
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
                is_best = (s == s.max()) if maximize else (s == s.min())
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

    # Plot hypervolume vs time
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

    def plot_fronts(self, df, instances_list, figs):
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        generic_key = "pareto_front"
        for instance_to_process in instances_list:
            filtered_df = df_copy[df_copy[self.instance] == instance_to_process]

            # get the problem name
            problem_for_instance_list = filtered_df[self.problem].unique()
            if len(problem_for_instance_list) != 1:
                raise ValueError(f"More than one problem for instance {instance_to_process} or no problem at all")
            problem_for_instance = problem_for_instance_list[0]
            name_plot = f"{problem_for_instance}-{instance_to_process}"

            fig, ax = plt.subplots(figsize=(10, 5))

            for _, row in filtered_df.iterrows():
                solver_strategy = row['solver_strategy']
                pareto_front_str = row[self.pareto_front]

                # Parse the pareto_front field
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
                pareto_front = np.array(pareto_front)
                ax.scatter(pareto_front[:, 0], pareto_front[:, 1], label=label)

            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_title(f'Pareto fronts for instance {name_plot}')
            ax.legend(title='Solver strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            fig_key = f"{generic_key}_{name_plot}"
            figs[fig_key] = fig

        return figs

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

    def plot_hypervolume_evolution(self, filtered_df, ax, instance_to_process, consider_only_pareto=False, zoom_in_y=False):
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
