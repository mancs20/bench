import json
import csv
import os.path
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import Hypervolume
import re


def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False


def calculate_hypervolume(front, reference_point):
    # check if it is a maximization problem
    comparisson_array = reference_point < front[0]
    if comparisson_array[0]:
        # convert maximization to minimization to calculate the Hypervolume using the pymoo library
        reference_point = -reference_point
        front = -front
    return Hypervolume(ref_point=reference_point)(front)


def get_solutions_in_time_for_choco(solver_messages):
    sum_times = []
    previous_resolution_time = 0

    for index, message in enumerate(solver_messages):
        building_time, resolution_time = extract_times_choco(message)
        if index == 0:
            previous_resolution_time = building_time + resolution_time
            sum_times.append(previous_resolution_time)
        else:
            # check if there were any solutions found
            current_sum_time = resolution_time + previous_resolution_time
            previous_resolution_time += resolution_time
            if "No solution" in message:
                continue
            sum_times.append(current_sum_time)
    return sum_times


def extract_times_choco(message):
    building_time = float(re.search(r'Building time\s*:\s*([\d.]+)s', message).group(1))
    resolution_time = float(re.search(r'Resolution time\s*:\s*([\d.]+)s', message).group(1))
    return building_time, resolution_time


if __name__ == "__main__":
    output_dir = sys.argv[1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    input_file_path = Path(sys.argv[2])
    output_file_path = Path(output_dir)
    uid = os.path.basename(os.path.normpath(output_file_path))
    # uid = input_file_path.stem
    sol_stats_filename = Path(output_dir + "/mo_" + uid + "_solutions_and_stats.csv")
    # stats_filename = Path(output_dir + "/" + uid + "_stats.yml")

    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    json_objects = []
    current_json_mo_solution_details = {}
    unknowns = []
    errors = []
    statistics = {}

    # We keep successive experiments in the JSON file, even when they fail.
    # For the statistics, we are only interested by the latest experiment,
    # starting with {"type": "lattice-land", "lattice-land": "start"}.
    for line in lines:
        if is_valid_json(line):
            output = json.loads(line)
            if output["type"] == "lattice-land":
                if output["lattice-land"] == "start":
                    statistics = {}
                    unknowns = []
                    errors = []
                    current_json_mo_solution_details = {}
            elif output["type"] == "statistics":
                statistics.update(output["statistics"])
            elif output["type"] == "solutions-details":
                current_json_mo_solution_details.update(output["solutions-details"])
            elif output["type"] == "error":
                errors += line
        else:
            unknowns.append(line)

    statistics_fields = list(statistics.keys())
    solution_details_fields = list(current_json_mo_solution_details.keys())
    if "solver_messages" in solution_details_fields:
        solution_details_fields.remove("solver_messages")

    filtered_data = {}
    solutions_in_time = []
    if statistics:
        filtered_data = OrderedDict({field: statistics.get(field, None) for field in statistics_fields})
        # filtered_data = {field: statistics.get(field, None) for field in statistics_fields}
    if current_json_mo_solution_details:
        if "solver_messages" in current_json_mo_solution_details:
            if "choco" in statistics["solver"]:
                solutions_in_time = get_solutions_in_time_for_choco(current_json_mo_solution_details["solver_messages"])
            current_json_mo_solution_details.pop("solver_messages")
        if "reference_point" in current_json_mo_solution_details:
            reference_point = current_json_mo_solution_details.get("reference_point", None)
            current_json_mo_solution_details.pop("reference_point")
        else:
            reference_point = None
            print("No reference_point found in the solutions_details. Impossible to calculate hypervolume.")
        filtered_data.update({field: current_json_mo_solution_details.get(field, None) for field
                              in solution_details_fields})
    all_fields = list(filtered_data.keys())

    # extra multi-objective metrics
    new_fields = ['hypervolume', 'front_cardinality', 'hypervolume_evolution', 'solutions_in_time']
    all_fields.extend(new_fields)
    hypervolume = calculate_hypervolume(np.array(filtered_data['pareto_front']), np.array(reference_point))
    front_cardinality = len(filtered_data['pareto_front'])

    # calculate hypervolume evolution
    front = filtered_data['pareto_front']
    hypervolume_evolution = [0] * len(front)
    temp_front = []
    for index, point in enumerate(front):
        temp_front.append(point)
        hypervolume_evolution[index] = calculate_hypervolume(np.array(temp_front), np.array(reference_point))
    filtered_data.update({'hypervolume': hypervolume, 'front_cardinality': front_cardinality,
                          'hypervolume_evolution': hypervolume_evolution, 'solutions_in_time': solutions_in_time})

    output_file_exists = os.path.isfile(sol_stats_filename)
    existing_data = []
    if output_file_exists:
        with open(sol_stats_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_data = [row for row in reader]
            existing_headers = reader.fieldnames
            headers = existing_headers + [field for field in all_fields if field not in existing_headers]
    else:
        headers = all_fields

    with open(sol_stats_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        if existing_data:
            for row in existing_data:
                writer.writerow(row)
        writer.writerow(filtered_data)
