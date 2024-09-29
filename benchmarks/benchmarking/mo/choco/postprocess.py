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
    if len(front) > 1:
        comparisson_array = reference_point < front[1]
    else:
        comparisson_array = reference_point < front[0]
    if comparisson_array[0]:
        # convert maximization to minimization to calculate the Hypervolume using the pymoo library
        reference_point = -reference_point
        front = -front
    return Hypervolume(ref_point=reference_point)(front)


def get_solutions_in_time_for_choco(solver_messages, all_solutions, gavanelli_front_strategy):
    sum_times = []
    previous_resolution_time = 0

    for index, message in enumerate(solver_messages):
        building_time, resolution_time = extract_times_choco(message)
        if index == 0:
            previous_resolution_time = building_time
        if not gavanelli_front_strategy:
            previous_resolution_time += resolution_time
        else:
            previous_resolution_time = resolution_time
        if all_solutions or ("no solution" not in message.lower()):
            sum_times.append(previous_resolution_time)

    return sum_times


def extract_times_choco(message):
    building_time = float(re.search(r'Building time\s*:\s*([\d,.]+)s', message).group(1).replace(',', ''))
    resolution_time = float(re.search(r'Resolution time\s*:\s*([\d,.]+)s', message).group(1).replace(',', ''))
    return building_time, resolution_time


if __name__ == "__main__":
    calculate_evolution_for_gavanelli = False # if the evolution of the hypervolume is not needed, set this to False, as it could take a lot of time
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
    exceptions = []
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
                    exceptions = []
                    errors = []
                    current_json_mo_solution_details = {}
            elif output["type"] == "statistics":
                statistics.update(output["statistics"])
            elif output["type"] == "solutions-details":
                current_json_mo_solution_details.update(output["solutions-details"])
            elif output["type"] == "error":
                errors += line
        elif "exception" in line.lower():
            exceptions.append(line)

    if errors != [] or exceptions != []:
        print(f"Error in {sys.argv[2]}", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        for exception in exceptions:
            print(exception, file=sys.stderr)

    statistics_fields = list(statistics.keys())
    solution_details_fields = list(current_json_mo_solution_details.keys())
    if "solver_messages" in solution_details_fields:
        solution_details_fields.remove("solver_messages")

    filtered_data = {}
    solutions_in_time = []
    front_metrics = {}
    if statistics:
        filtered_data = OrderedDict({field: statistics.get(field, None) for field in statistics_fields})
    if current_json_mo_solution_details:
        if 'pareto_front' in current_json_mo_solution_details:
            if "reference_point" in current_json_mo_solution_details:
                reference_point = current_json_mo_solution_details.get("reference_point", None)
                current_json_mo_solution_details.pop("reference_point")
                hypervolume = calculate_hypervolume(np.array(current_json_mo_solution_details.get('pareto_front')), np.array(reference_point))
                front_metrics.update({"hypervolume": hypervolume})
                if 'all_solutions' in current_json_mo_solution_details:
                    solutions = current_json_mo_solution_details.get('all_solutions')
                else:
                    solutions = current_json_mo_solution_details.get('pareto_front')
                front_metrics.update({"front_cardinality": len(current_json_mo_solution_details.get('pareto_front'))})
                # calculate hypervolume evolution
                if calculate_evolution_for_gavanelli or filtered_data['front_generator'] != 'ParetoGavanelliGlobalConstraint':
                    hypervolume_evolution = [0] * len(solutions)
                    temp_front = []
                    for index, point in enumerate(solutions):
                        temp_front.append(point)
                        hypervolume_evolution[index] = float(calculate_hypervolume(np.array(temp_front),
                                                                             np.array(reference_point)))
                    front_metrics.update({"hypervolume_evolution": hypervolume_evolution})
                else:
                    front_metrics.update({"hypervolume_evolution": "Not available."})
                front_metrics["all_solutions"] = solutions
        front_metrics["solutions_in_time"] = "Not available."
        if "solver_messages" in current_json_mo_solution_details:
            if "choco" in statistics["solver"] and (calculate_evolution_for_gavanelli or filtered_data['front_generator'] != 'ParetoGavanelliGlobalConstraint'):
                solutions_in_time = get_solutions_in_time_for_choco(current_json_mo_solution_details["solver_messages"],
                                                                    'all_solutions' in current_json_mo_solution_details,
                                                                    filtered_data['front_generator'] == 'ParetoGavanelliGlobalConstraint')
                front_metrics["solutions_in_time"] = solutions_in_time
            current_json_mo_solution_details.pop("solver_messages")
        filtered_data.update({field: current_json_mo_solution_details.get(field, None) for field
                              in solution_details_fields})
        filtered_data.update(front_metrics)
        all_fields = list(filtered_data.keys())

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
