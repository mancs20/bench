import json
import csv
import os.path
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import Hypervolume
import re

fields_to_check = {
    'sum_solutions_building_time(s)': 'Building time',
    'sum_solutions_fails': 'Fails',
    'sum_solutions_backtracks': 'Backtracks',
    'sum_number_solutions': 'Solutions',
    'sum_solutions_resolution_time(s)': 'Resolution time',
    'sum_solutions_nodes': 'Nodes',
    'sum_solutions_restarts': 'Restarts',
    'sum_solutions_backjumps': 'Backjumps',
}


def process_json_file(input_json_file_path, output_stats_filename):
    with open(input_json_file_path, 'r') as file:
        lines = file.readlines()

    current_json_mo_solution_details = {}
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
    front_metrics = {}
    if statistics:
        filtered_data = OrderedDict({field: statistics.get(field, None) for field in statistics_fields})

    if not current_json_mo_solution_details:
        instance_path = Path(sys.argv[2])
        instance_file = instance_path.name
        print(f"No experiment data for: {instance_file}. File: {sys.argv[2]}", file=sys.stderr)

    reference_point = False
    # if current_json_mo_solution_details:
    if 'pareto_front' in current_json_mo_solution_details:
        if "reference_point" in current_json_mo_solution_details:
            reference_point = current_json_mo_solution_details.get("reference_point", None)
            current_json_mo_solution_details.pop("reference_point", None)
            hypervolume = calculate_hypervolume(np.array(current_json_mo_solution_details.get('pareto_front')),
                                                np.array(reference_point))
            front_metrics.update({"hypervolume": hypervolume})
            if 'all_solutions' in current_json_mo_solution_details:
                solutions = current_json_mo_solution_details.get('all_solutions')
            else:
                solutions = current_json_mo_solution_details.get('pareto_front')
            front_metrics.update({"front_cardinality": len(current_json_mo_solution_details.get('pareto_front'))})
            # calculate hypervolume evolution
            if calculate_evolution and (calculate_evolution_for_gavanelli or filtered_data['front_generator'] !=
                                        'ParetoGavanelliGlobalConstraint'):
                hypervolume_evolution = [0.0] * len(solutions)
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
        if "choco" in statistics["solver"] and calculate_evolution and (
                calculate_evolution_for_gavanelli or filtered_data['front_generator'] !=
                'ParetoGavanelliGlobalConstraint'):
            solutions_in_time = get_solutions_in_time_for_choco(current_json_mo_solution_details["solver_messages"],
                                                                'all_solutions' in current_json_mo_solution_details,
                                                                filtered_data['front_generator'] ==
                                                                'ParetoGavanelliGlobalConstraint')
            front_metrics["solutions_in_time"] = solutions_in_time
        current_json_mo_solution_details.pop("solver_messages")
    filtered_data.update({field: current_json_mo_solution_details.get(field, None) for field
                          in solution_details_fields})
    if reference_point:
        filtered_data["reference_point"] = reference_point

    filtered_data.update(front_metrics)
    all_fields = list(filtered_data.keys())
    # Move 'reference_point' to the last position
    if "reference_point" in all_fields:
        all_fields.remove("reference_point")
    all_fields.append("reference_point")

    if 'choco' in statistics['solver'] and current_json_mo_solution_details:
        double_check_front_strategy_stats_are_correct_for_choco(filtered_data,
                                                                output["solutions-details"]["solver_messages"])

    output_file_exists = os.path.isfile(output_stats_filename)
    existing_data = []
    if output_file_exists:
        with open(output_stats_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_data = [row for row in reader]
            existing_headers = reader.fieldnames
            headers = existing_headers + [field for field in all_fields if field not in existing_headers]
            # Ensure 'reference_point' is last
            if "reference_point" in headers:
                headers.remove("reference_point")
            headers.append("reference_point")
    else:
        headers = all_fields

    with open(output_stats_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        if existing_data:
            for row in existing_data:
                writer.writerow(row)
        writer.writerow(filtered_data)
    print(f"Processed {sys.argv[2]}")


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


def double_check_front_strategy_stats_are_correct_for_choco(processed_data, original_solution_details_data):
    if 'gavanelli' in processed_data['front_generator'].lower():
        double_check_gavanelli_front_strategy_stats_are_correct_for_choco(
            processed_data, original_solution_details_data)
    else:
        double_check_non_gavanelli_front_strategy_stats_are_correct_for_choco(processed_data,
                                                                              original_solution_details_data)


def double_check_non_gavanelli_front_strategy_stats_are_correct_for_choco(processed_data,
                                                                          original_solution_details_data):
    check_processed_data = dict.fromkeys(fields_to_check, 0.0)
    check_processed_data['average_node_per_second'] = 0.0
    temp_processed_data = processed_data.copy()
    for solution_details_message in original_solution_details_data:
        get_stats_from_solution_message(solution_details_message, temp_processed_data)
        for key, value in check_processed_data.items():
            check_processed_data[key] = check_processed_data[key] + temp_processed_data[key]
    # todo fix the average node per second and sum_solutions_building_time(s) fields
    check_processed_data['sum_solutions_building_time(s)'] = temp_processed_data['sum_solutions_building_time(s)']
    check_processed_data['average_node_per_second'] = (check_processed_data['average_node_per_second'] /
                                                       len(original_solution_details_data))
    for key, value in fields_to_check.items():
        if processed_data[key] != check_processed_data[key]:
            if key == 'average_node_per_second' and int(processed_data[key]) == int(check_processed_data[key]):
                continue
            instance_msg = processed_data.get('instance', 'Unknown instance')
            front_generator = processed_data.get('front_generator', 'Unknown front generator')
            raise ValueError(
                f"Error: Field '{value}' in the solution details message for instance '{instance_msg}' for "
                f"front generator {front_generator} does not match the value in the processed data."
            )


def double_check_gavanelli_front_strategy_stats_are_correct_for_choco(processed_data, original_solution_details_data):
    last_solution_details_messages = original_solution_details_data[-1]
    get_stats_from_solution_message(last_solution_details_messages, processed_data)


def get_stats_from_solution_message(solution_message, processed_data):
    instance_msg = processed_data.get('instance', 'Unknown instance')
    front_generator = processed_data.get('front_generator', 'Unknown front generator')
    msg_number_match = re.search(r'Model solved # (\d+)', solution_message)
    if msg_number_match:
        msg_number = int(msg_number_match.group(1))
    else:
        raise (ValueError(f"Error: Number of model solved not found in the solution details message for instance "
                          f"'{instance_msg}' for front strategy {front_generator}."))
    error_msg = (f"Error: Field '{{}}' not found in the solution details message for model solved {msg_number} for "
                 f"instance '{instance_msg}' for front strategy {front_generator}.")
    for key, value in fields_to_check.items():
        try:
            if key == 'sum_solutions_nodes':  # Special case for 'Nodes'
                # Extract both the total nodes and the average nodes per second.
                match = re.search(r'Nodes:\s*([\d,]+)\s*\(([\d,.]+)\s*n/s\)', solution_message)
                if match:
                    total_nodes = int(match.group(1).replace(',', ''))
                    average_nodes = float(match.group(2).replace(',', ''))
                    processed_data['sum_solutions_nodes'] = total_nodes
                    processed_data['average_node_per_second'] = average_nodes
                else:
                    raise ValueError(
                        error_msg.format(value)
                    )
            else:
                match = re.search(rf'{re.escape(value)}\s*:\s*([\d,]+(?:\.\d+)?)', solution_message)
                if match:
                    value_in_message = float(match.group(1).replace(',', ''))
                    if processed_data[key] != value_in_message:
                        processed_data[key] = value_in_message
                else:
                    raise ValueError(
                        error_msg.format(value)
                    )
        except ValueError as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    calculate_evolution_for_gavanelli = False  # if the evolution of the hypervolume is not needed, set this to False,
    # as it could take a lot of time
    calculate_evolution = False  # turn off the computation of the hypervolume evolution, it could be very expensive
    output_dir = sys.argv[1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    input_file_path = Path(sys.argv[2])
    output_file_path = Path(output_dir)
    uid = os.path.basename(os.path.normpath(output_file_path))
    sol_stats_filename = Path(output_dir + "/mo_" + uid + "_solutions_and_stats.csv")
    process_json_file(input_file_path, sol_stats_filename)
