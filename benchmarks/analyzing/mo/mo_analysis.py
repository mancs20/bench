import pandas as pd
import seaborn as sns
import numpy as np  # Needed for log transformation
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import os


class MoAnalysis:

    def __init__(self, instance='instance', solver_name='solver', front_strategy='front_generator',
                 hypervolume='hypervolume', pareto_front='pareto_front',
                 hypervolume_evolution='hypervolume_evolution', number_of_solutions='front_cardinality',
                 exhaustive='exhaustive', time='time(s)', solutions_in_time='solutions_in_time',
                 time_solver_sec=None, pareto_solutions_time_list=None):
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

        if time_solver_sec is None:
            self.time_solver_sec = time
        else:
            self.time_solver_sec = time_solver_sec
        if pareto_solutions_time_list is None:
            self.pareto_solutions_time_list = solutions_in_time
        else:
            self.pareto_solutions_time_list = pareto_solutions_time_list


    @staticmethod
    def csv_to_df(file_path):
        df = pd.read_csv(file_path, delimiter=',')
        return df

    # Function to calculate score for each front_strategy
    def calculate_score(self, group):
        # best_hypervolume = group.loc[group[self.hypervolume].idxmax(), self.hypervolume]
        # group['score'] = group[self.hypervolume] / best_hypervolume
        # return group
        best_hypervolume = group[self.hypervolume].max()
        group['score'] = group[self.hypervolume] / best_hypervolume
        return group

    def plot_hypervolume_best(self, df):
        # Apply the function to calculate scores
        df_score_by_front_strategy = df.groupby([self.instance, self.solver_name]).apply(self.calculate_score)
        # df_score_by_front_strategy = df.groupby([self.instance, self.solver_name]).apply(calculate_score)
        # df_score_by_front_strategy =  df_score_by_front_strategy.drop(columns=self.solver_name)

        df_total_front_by_solver = df.groupby([self.solver_name, self.front_strategy])[self.instance].count()
        df_best_front_by_solver1 = df_score_by_front_strategy[df_score_by_front_strategy['score'] == 1.0].groupby(
            [self.solver_name, self.front_strategy]).size().rename('best').to_frame()
        df_best_front_by_solver = df_score_by_front_strategy.groupby(
            [self.solver_name, self.front_strategy]).size().rename(
            'best').to_frame()
        df_best_front_by_solver['best'] = df_best_front_by_solver1['best'].astype(int)
        # replace NaN values with 0 and convert to int
        df_best_front_by_solver['best'] = df_best_front_by_solver['best'].fillna(0).astype(int)
        df_avg_score_by_front_strategy = df_score_by_front_strategy.groupby([self.solver_name, self.front_strategy])[
            'score'].mean().rename('average_score').to_frame()

        df_total_best = pd.merge(df_total_front_by_solver, df_best_front_by_solver, left_index=True, right_index=True)
        # add a column with the average score for each front strategy
        df_total_best_avg_score = pd.merge(df_total_best, df_avg_score_by_front_strategy, left_index=True,
                                           right_index=True)
        print(df_total_best_avg_score)

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        # Adding text labels on top of bars
        ax = sns.barplot(x=df_total_best_avg_score.index.get_level_values(self.solver_name), y='best',
                         hue=df_total_best_avg_score.index.get_level_values(self.front_strategy),
                         data=df_total_best_avg_score)
        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        # Adding labels and legend
        plt.title('Times each strategy was the best')
        plt.xlabel('Solver Name')
        plt.ylabel('Times best')
        plt.legend(title='Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return df_total_best_avg_score, fig

    def plot_hypervolume_best_average(self, df_total_best_avg_score):
        # Create the second graph
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=df_total_best_avg_score.index.get_level_values(self.solver_name), y='average_score',
                         hue=df_total_best_avg_score.index.get_level_values(self.front_strategy),
                         data=df_total_best_avg_score)
        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        # Adding labels and legend
        plt.title('Average Score for each front strategy')
        plt.xlabel('Solver Name')
        plt.ylabel('Average score')
        plt.legend(title='Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        return fig

    # Plot the time and the number of solutions for each instance
    # todo delete after test
    def get_time_number_solutions1(self, df):
        # Apply the function to calculate scores
        # df_time_number_solutions = df.groupby([self.instance]).apply(self.calculate_number_of_solutions_score,
        #                                                              include_groups=False)
        df_time_number_solutions = df.groupby([self.instance]).apply(self.calculate_number_of_solutions_score,)

        # df_time = df.groupby([self.instance]).apply(self.calculate_time_score, include_groups=False)
        df_time = df.groupby([self.instance]).apply(self.calculate_time_score)
        df_time_number_solutions['time_score'] = df_time['time_score']

        # df_solver_front = df.groupby([self.instance]).apply(self.merge_solver_front_strategy_names,
        #                                                     include_groups=False)
        df_solver_front = df.groupby([self.instance]).apply(self.merge_solver_front_strategy_names)
        df_time_number_solutions['solver_front_strategy'] = df_solver_front['solver_front_strategy']

        df_time_number_solutions = df_time_number_solutions[['solver_front_strategy', 'time_score', self.time_solver_sec,
                                                             'number_of_solutions_score', self.number_of_solutions,
                                                             self.exhaustive]]
        return df_time_number_solutions

    def get_time_number_solutions(self, df):
        # Apply the function to calculate scores
        df_time_number_solutions = df.groupby([self.instance]).apply(
            self.calculate_number_of_solutions_score).reset_index(drop=True)
        df_time = df.groupby([self.instance]).apply(self.calculate_time_score).reset_index(drop=True)
        df_solver_front = df.groupby([self.instance]).apply(self.merge_solver_front_strategy_names).reset_index(
            drop=True)

        # Merge the results
        df_time_number_solutions['time_score'] = df_time['time_score']
        df_time_number_solutions['solver_front_strategy'] = df_solver_front['solver_front_strategy']

        columns_to_select = [
            self.instance, 'solver_front_strategy', 'time_score', self.time_solver_sec, 'number_of_solutions_score',
            self.number_of_solutions, self.exhaustive]
        df_time_number_solutions = df_time_number_solutions[columns_to_select]

        return df_time_number_solutions

    def calculate_time_score(self, group):
        best_time = group.loc[group[self.time_solver_sec].idxmin(), self.time_solver_sec]
        group['time_score'] = group[self.time_solver_sec] / best_time
        return group

    def calculate_number_of_solutions_score(self, group):
        best_number_of_solutions = group.loc[group[self.number_of_solutions].idxmin(), self.number_of_solutions]
        group['number_of_solutions_score'] = group[self.number_of_solutions] / best_number_of_solutions
        return group

    def merge_solver_front_strategy_names(self, group):
        group['solver_front_strategy'] = group[self.solver_name] + ' ' + group[self.front_strategy]
        return group

    # Plot number of solutions in time
    def plot_strategy_time_score_to_get_the_front(self, df_time_number_solutions):
        # Set up the plot
        fig = plt.figure(figsize=(10, 20))
        sns.set_theme(style="whitegrid")

        # Define a threshold for applying a logarithmic scale
        log_threshold = 10  # Adjust the threshold as needed

        # Create a color palette for solver_front_strategy
        palette = sns.color_palette("husl", len(df_time_number_solutions['solver_front_strategy'].unique()))

        # Plot time scores with a conditional logarithmic x-axis
        ax = sns.barplot(x='time_score', y=self.instance, hue='solver_front_strategy', data=df_time_number_solutions,
                         palette=palette, orient='h')

        plt.title('Time Score by Solver Front Strategy')
        plt.xlabel('Time Score')
        plt.ylabel(self.instance)
        plt.legend(title='Solver Front Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Apply a logarithmic scale only for values greater than the threshold
        if df_time_number_solutions['time_score'].max() > log_threshold:
            plt.xscale('log')

        # Set the background to white and remove the grid lines
        sns.despine(left=True, bottom=True)
        ax.grid(False)
        plt.show()
        return fig

    # Plot certain instances to check the number of solutions in time and the hypervolume
    def plot_solutions_in_time(self, df, instances_list, figs):
        # Step 1: Combine solver_name and front_strategy into a new column
        # df.loc['solver_strategy'] = df.loc[self.solver_name] + ' ' + df.loc[self.front_strategy]
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        # Map solver_strategy combinations to y-values
        unique_combinations = df_copy['solver_strategy'].unique()
        combination_to_y = {comb: i for i, comb in enumerate(unique_combinations)}

        for instances in instances_list:
            filtered_df = df_copy[df_copy[self.instance] == instances]

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
                pareto_front_count = len(row['pareto_front'].split('],['))
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

            plt.xlabel('Solution Time')
            plt.ylabel('Solver and Strategy Combination')
            plt.title(f'Comparison of Solution Times by Solver and Strategy for instance {instances}')

            plt.xticks([])

            # Show legend with bbox_to_anchor outside the plot area
            all_points = Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=10, label='All points', markerfacecolor='none')
            not_pareto_front = Line2D([], [], color='red', marker='o', linestyle='None',
                                      markersize=10, label='Not Pareto front points', markerfacecolor='none')

            plt.legend(handles=[all_points, not_pareto_front], title="Legend", bbox_to_anchor=(1.05, 1),
                       loc='upper left')

            plt.show()
            figs.append(fig)
        return figs

    # Plot hypervolume vs time
    def plot_hypervolume_vs_time(self, df, instances_list, figs):
        df_copy = df.copy()
        df_copy['solver_strategy'] = df_copy[[self.solver_name, self.front_strategy]].agg(' '.join, axis=1)

        for instance_to_process in instances_list:
            filtered_df = df_copy[df_copy[self.instance] == instance_to_process]

            # Plot 1: Regular Scale
            fig, ax = plt.subplots(figsize=(10, 5))
            self.plot_data(filtered_df, ax, instance_to_process)
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Hypervolume')
            ax.legend(title='Solver Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            figs.append(fig)

            # Plot 2: Logarithmic Scale
            fig, ax = plt.subplots(figsize=(10, 5))
            self.plot_data(filtered_df, ax, instance_to_process, zoom_in_y=True)
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Log of Hypervolume')
            ax.legend(title='Solver Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            figs.append(fig)

        return figs

    def plot_data(self, filtered_df, ax, instance_to_process, consider_only_pareto=False, zoom_in_y=False):
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
                all_solutions_string = row['all_solutions']
                # check if it has the string "Unfeasible"
                if "Unfeasible" in all_solutions_string:
                    # remove the "Unfeasible" string from the all_solutions_string
                    all_solutions_string = all_solutions_string.replace(',Unfeasible', '')
                    all_solutions_string = all_solutions_string.replace('Unfeasible,', '')
                all_solutions = all_solutions_string.replace('{', '').replace('}', '').split('],[')
                # print(f'all_solutions for {combination}: {all_solutions} has a length of {len(all_solutions)}')
                # remove from x_all_times the times where all_solutions is equal to "Unfeasible"
                x_all_times = [x_all_times[i] for i in range(len(all_solutions)) if "Unfeasible" != all_solutions[i]]
                # print(f'x_all_times for {combination}: {x_all_times} has a length of {len(x_all_times)}')
                hypervolumes = [float(hv) for hv in
                                row[self.hypervolume_evolution].replace('[', '').replace(']', '').split(',')]
                # print(f'hypervolumes for {combination}: {hypervolumes} has a length of {len(hypervolumes)}')

                y = hypervolumes
                x = x_all_times
                if consider_only_pareto:
                    x_pareto = [float(time) for time in
                                row[self.pareto_solutions_time_list].replace('[', '').replace(']', '').split(',')]
                    x_pareto_id = [x_all_times.index(time) for time in x_pareto]
                    hypervolumes_pareto = [hypervolumes[i] for i in x_pareto_id]
                    y = hypervolumes_pareto
                    x = x_pareto

                # if not consider_only_pareto:# Find indices of x_all_times that correspond to x_pareto
                #     x_pareto_id = [x_all_times.index(time) for time in x_pareto]
                #
                #     # Prepare the y values for plotting, mapping each time in x_all_times to its corresponding hypervolume
                #     y = []
                #     last_index = 0  # Track the last index of x_pareto that was used
                #     for i in range(len(x_all_times)):
                #         if last_index < len(x_pareto_id) - 1 and i > x_pareto_id[last_index]:
                #             last_index += 1
                #         y.append(hypervolumes[last_index])
                # else:
                #     y = hypervolumes
                #     x_all_times = x_pareto

                # for each y value, add it to the all_y_values list
                all_y_values.extend(y)
                # Plot the data
                ax.plot(x_all_times, y, marker='o', linestyle='-', label=combination)

        if zoom_in_y:
            # get the median of all y values
            median_y = np.median(all_y_values)
            # get the maximum y value
            max_y = max(all_y_values)
            # based on this median value, set the y-axis limits
            ax.set_ylim(max_y * 0.99, max_y * 1.01)
            plt.draw()

        # %%

    # save the data as pdf and csv
    @staticmethod
    def save_pictures_and_tables(figs, folder_name, df_total_best_avg_score):
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # save df_total_best_avg_score to a csv file
        df_total_best_avg_score.to_csv(f'{folder_name}/df_total_best_avg_score.csv', sep=';')

        # Save each figure
        for i, fig in enumerate(figs):
            image_path = os.path.join(folder_name, f"image_{i + 1}.pdf")
            # make sure the fig is displayed correctly and that it is not cut off
            fig.tight_layout()
            fig.savefig(image_path)
            plt.close(fig)

        print(f"All images have been saved in the '{folder_name}' folder.")

    def print_all_figs_and_tables(self, df, figs):
        df_total_best_avg_score, fig = self.plot_hypervolume_best(df)
        figs.append(fig)
        figs.append(self.plot_hypervolume_best_average(df_total_best_avg_score))
        df_time_number_solutions = self.get_time_number_solutions(df)
        figs.append(self.plot_strategy_time_score_to_get_the_front(df_time_number_solutions))
        # define, the instances to plot, by default all instances are plotted
        instances_list = df[self.instance].unique()
        # instances_list = ['paris_30']
        figs = self.plot_solutions_in_time(df, instances_list, figs)
        return figs, df_total_best_avg_score
