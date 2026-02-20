import json
import csv
import os.path
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import Hypervolume
import re
import ast

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

LEXICOGRAPHIC_LINE = "Lexicographic optimization is used. Saugmecon objective is bigger than Integer.MAX_VALUE"


def extract_hypervolume_lines(lines):
    """Extracts all HYPERVOLUME lines and parses them into [second, points]."""
    hv_data = []
    found_first_hypervolume_line = False
    for line in lines:
        if line.startswith("HYPERVOLUME:"):
            found_first_hypervolume_line = True
            match = re.match(r'HYPERVOLUME: \[(\d+), (.+)\]', line.strip())
            if match:
                second = int(match.group(1))
                points = json.loads(match.group(2).replace("'", '"'))
                hv_data.append([second, points])
        elif found_first_hypervolume_line:
            break

    return hv_data


def calculate_hypervolume_evolution(lines, reference_point):
    hv_data = extract_hypervolume_lines(lines)
    # hypervolume_evolution = []
    # temp_front = []
    # for index, point in enumerate(solutions):
    #     temp_front.append(point)
    #     hypervolume_evolution[index] = float(calculate_hypervolume(np.array(temp_front),
    #                                                                np.array(reference_point)))
    return hv_data


def process_json_file(input_json_file_path, output_stats_filename):
    with open(input_json_file_path, 'r') as file:
        lines = file.readlines()

    current_json_mo_solution_details = {}
    exceptions = []
    errors = []
    statistics = {}
    lexicographic_opt = None
    search_strategy = None

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
                    lexicographic_opt = None
            elif output["type"] == "statistics":
                statistics.update(output["statistics"])
            elif output["type"] == "solutions-details":
                current_json_mo_solution_details.update(output["solutions-details"])
            elif output["type"] == "error":
                errors += line
        elif LEXICOGRAPHIC_LINE in line:
            lexicographic_opt = True
        elif "solverSearchStrategy" in line:
            # I want to get what is after solverSearchStrategy, it is like this solverSearchStrategy: domOverWDegSearch
            search_strategy = line.split("solverSearchStrategy:")[-1].strip().split(",")[0]
        elif "exception" in line.lower():
            exceptions.append(line)
        elif "error" in line.lower():
            errors.append(line)


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
        if "front_generator" in statistics and "saugmecon" in str(statistics["front_generator"]).lower():
            # If we never saw the line, it means "checked and not present"
            filtered_data["lexicographic_opt"] = (lexicographic_opt is True)
        if search_strategy is not None:
            filtered_data["search_strategy"] = search_strategy

    if not current_json_mo_solution_details:
        instance_path = Path(sys.argv[2])
        instance_file = instance_path.name
        print(f"No experiment data for: {instance_file}. File: {sys.argv[2]}", file=sys.stderr)

    reference_point = False
    if 'pareto_front' in current_json_mo_solution_details:
        # simple verification if the search was exhaustive. If the last search was incomplete exhaustive has to be False
        if "exhaustive" in current_json_mo_solution_details:
            exhaustive = current_json_mo_solution_details.get("exhaustive")
            if exhaustive:
                # if current_json_mo_solution_details['solver_messages'][-1].find("Incomplete search") != -1:
                if "Incomplete search" in current_json_mo_solution_details['solver_messages'][-1]:
                    current_json_mo_solution_details['exhaustive'] = False

        if "reference_point" in current_json_mo_solution_details:
            reference_point = current_json_mo_solution_details.get("reference_point", None)
            current_json_mo_solution_details.pop("reference_point", None)
            # verify if the last point in the pareto front is non-dominated by the front. If the timeout is reached the
            # algorithms will return the approximate solution as the last point in the front, which is not necessarily
            # non-dominated.
            hypervolume_evolution = None
            if len(np.array(current_json_mo_solution_details.get('pareto_front'))) != 0:
                pareto_front = get_true_non_dominated_front(np.array(current_json_mo_solution_details.get('pareto_front')),
                                                        current_json_mo_solution_details.get('exhaustive'),
                                                        reference_point)
            else:
                # check if there is a pareto front, if not get the last entrance in hypervolume evolution
                hypervolume_evolution = calculate_hypervolume_evolution(lines, reference_point) # We're
                # interested in the front not in the hypervolume evolution, so with
                # hv_data = extract_hypervolume_lines(lines) and read the last element should be enough
                pareto_front = np.array(hypervolume_evolution[-1][1])
                if len(pareto_front) > 1:
                    # check if the first number objectives points are not dominated by the rest of the front
                    maximization = is_maximization_problem(pareto_front, reference_point)
                    number_objectives = len(reference_point)
                    dominated_initial_points = [False] * number_objectives
                    for index, point in enumerate(pareto_front):
                        if index >= number_objectives:
                            for i in range(number_objectives):
                                test_point = pareto_front[i]
                                if not dominated_initial_points[i] and dominates(point, test_point, maximization):
                                    dominated_initial_points[i] = True
                                    break
                    # remove the dominated points from the front
                    filtered_front = []
                    for i in range(number_objectives):
                        if not dominated_initial_points[i]:
                            filtered_front.append(pareto_front[i])
                    # Add the remaining points after the initial ones
                    filtered_front.extend(pareto_front[number_objectives:])
                    pareto_front = np.array(filtered_front)

            if len(pareto_front) != len(current_json_mo_solution_details.get('pareto_front')):
                # pareto_front to string to avoid numpy array serialization issues
                current_json_mo_solution_details['pareto_front'] = pareto_front.tolist()
            # hypervolume = calculate_hypervolume(pareto_front, np.array(reference_point))
            front_metrics.update({"hypervolume": 1.0})
            if 'all_solutions' in current_json_mo_solution_details:
                solutions = current_json_mo_solution_details.get('all_solutions')
            else:
                solutions = pareto_front
            front_metrics.update({"front_cardinality": len(pareto_front)})

            try:
                ideal_pt, nadir_pt, _ = compute_ideal_nadir_from_front(
                    np.array(pareto_front), reference_point
                )
                front_metrics.update({"ideal_point": ideal_pt})
                front_metrics.update({"nadir_point": nadir_pt})
            except Exception:
                # this cannot happen as there is at least one point in the front exit with error
                print(f"Error computing ideal and nadir points for "
                      f"instance "f"{filtered_data.get('instance', 'unknown')}", file=sys.stderr)
                exit(1)

            if calculate_evolution and (calculate_evolution_for_gavanelli or filtered_data['front_generator'] !=
                                        'ParetoGavanelliGlobalConstraint'):
                # calculate hypervolume evolution
                if hypervolume_evolution is None:
                    hypervolume_evolution = calculate_hypervolume_evolution(lines, reference_point)
                # hypervolume_evolution = [0.0] * len(solutions)
                # temp_front = []
                # for index, point in enumerate(solutions):
                #     temp_front.append(point)
                #     hypervolume_evolution[index] = float(calculate_hypervolume(np.array(temp_front),
                #                                                                np.array(reference_point)))
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


def is_maximization_problem(front, reference_point):
    if len(front) > 1:
        comparisson_array = reference_point < front[1]
    else:
        comparisson_array = reference_point < front[0]
    return comparisson_array[0]


def dominates(point, test_point, maximization):
    if maximization:
        return np.all(point >= test_point) and np.any(point > test_point)
    else:
        return np.all(point <= test_point) and np.any(point < test_point)


def get_true_non_dominated_front(pareto_front, exhaustive, reference_point):
    if not exhaustive:
        # remove the last point if it is dominated by the rest of the front
        maximization = is_maximization_problem(pareto_front, reference_point)
        for index, point in enumerate(pareto_front):
            if index == len(pareto_front) - 2:
                break
            if dominates(point, pareto_front[-1], maximization):
                pareto_front = pareto_front[:-1]
                break
    return pareto_front


def calculate_hypervolume(front, reference_point):
    # check if it is a maximization problem
    if is_maximization_problem(front, reference_point):
        # convert maximization to minimization to calculate the Hypervolume using the pymoo library
        reference_point = -reference_point
        front = -front
    return Hypervolume(ref_point=reference_point)(front)


def _safe_literal_list(v):
    """Parse list-like values stored as python-literal or json-ish strings."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # sometimes stored as "{...}" in csv
        if s.startswith("{") and s.endswith("}"):
            s = s.replace("{", "[").replace("}", "]")
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def compute_ideal_nadir_from_front(front, reference_point):
    """
    front: np.array shape (m, k)
    Returns (ideal_list, nadir_list, maximize_bool) in ORIGINAL objective space.
    """
    maximize = is_maximization_problem(front, np.array(reference_point))
    if maximize:
        ideal = np.max(front, axis=0)
        nadir = np.min(front, axis=0)
    else:
        ideal = np.min(front, axis=0)
        nadir = np.max(front, axis=0)
    return ideal.tolist(), nadir.tolist(), maximize



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
            if value != 'Restarts':
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


def proceed_to_write_experiment_in_csv(csv_path, key_data, allow_replace=False, force=False):
    """
    Returns a tuple where the first element indicates if we should proceed to write the csv or not, the second one indicates
    if we are replacing an existing entry, if it is true we have to delete.
    """
    if not os.path.isfile(csv_path):
        return False, False

    key_fields = ["problem", "instance", "solver_version", "front_generator", "timeout", "search_strategy"]
    target_key = tuple(key_data.get(k) for k in key_fields)
    target_datetime = key_data.get("datetime")

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_key = tuple(row.get(k) for k in key_fields)
            if row_key == target_key:
                if force or (allow_replace and ("datetime" in row and row["datetime"] < target_datetime)):
                    return True, True
                else:
                    return False, False
    return True, False

def remove_row_from_csv(csv_path, key_data):
    key_fields = ["problem", "instance", "solver_version", "front_generator", "timeout"]
    target_key = tuple(key_data.get(k) for k in key_fields)

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader if tuple(row.get(k) for k in key_fields) != target_key]
        headers = reader.fieldnames

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def backfill_ideal_nadir_in_csv(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []

    # ensure columns exist
    for col in ["ideal_point", "nadir_point"]:
        if col not in headers:
            headers.append(col)

    for row in rows:
        if row.get("ideal_point") and row.get("nadir_point"):
            continue  # already filled

        front = _safe_literal_list(row.get("pareto_front"))
        refp = _safe_literal_list(row.get("reference_point"))

        if not front or not refp:
            row["ideal_point"] = row.get("ideal_point", "")
            row["nadir_point"] = row.get("nadir_point", "")
            continue

        front = np.array(front)
        try:
            ideal_pt, nadir_pt, _ = compute_ideal_nadir_from_front(front, refp)
            # store as JSON-ish strings (robust in csv)
            row["ideal_point"] = json.dumps(ideal_pt)
            row["nadir_point"] = json.dumps(nadir_pt)
        except Exception:
            row["ideal_point"] = ""
            row["nadir_point"] = ""

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)



if __name__ == "__main__":
    # set to True to force replace existing entries in the CSV, regardless of datetime. Use with caution.
    # It should be set back to False after use.
    force_replace = False

    csv.field_size_limit(sys.maxsize)
    calculate_evolution_for_gavanelli = False  # if the evolution of the hypervolume is not needed, set this to False,
    # as it could take a lot of time
    calculate_evolution = False  # turn off the computation of the hypervolume evolution, it could be very expensive
    output_dir_or_existing_csv = sys.argv[1]
    input_file_path = Path(sys.argv[2])
    allow_replace = len(sys.argv) > 3 and sys.argv[3].lower() == "true"
    # check if the output_dir_or_existing_csv is a directory or a file
    if os.path.isfile(output_dir_or_existing_csv):
        sol_stats_filename = Path(output_dir_or_existing_csv)
    elif os.path.isdir(output_dir_or_existing_csv):
        if output_dir_or_existing_csv[-1] == "/":
            output_dir_or_existing_csv = output_dir_or_existing_csv[:-1]
        output_file_path = Path(output_dir_or_existing_csv)
        uid = os.path.basename(os.path.normpath(output_file_path))
        sol_stats_filename = Path(output_dir_or_existing_csv + "/mo_" + uid + "_solutions_and_stats.csv")
    else:
        # If the path is neither a file nor a directory, exit with an error.
        print(f"❌ The path {output_dir_or_existing_csv} is neither a valid file nor a directory.")
        sys.exit(1)

    # Extract statistics line to identify the experiment
    metadata = {}
    read_one_line = False
    search_strategy = None
    with open(input_file_path, 'r') as file:
        for line in file:
            if '"type": "statistics"' in line:
                try:
                    json_line = json.loads(line)
                    metadata = json_line["statistics"]
                    if read_one_line:
                        break
                    else:
                        read_one_line = True
                except json.JSONDecodeError:
                    continue
            elif "solverSearchStrategy" in line:
                # I want to get what is after solverSearchStrategy, it is like this solverSearchStrategy: domOverWDegSearch
                tmp_search_strategy = line.split("solverSearchStrategy:")[-1].strip().split(",")[0]
                if tmp_search_strategy != "default":
                    search_strategy = tmp_search_strategy
                if read_one_line:
                    break
                else:
                    read_one_line = True

    if not metadata:
        print(f"❌ Could not extract statistics from {input_file_path}")
        sys.exit(1)
    metadata["search_strategy"] = search_strategy

    # todo delete this is temporary to write the ideal and nadir points per row in existing csv
    # backfill_ideal_nadir_in_csv(sol_stats_filename)
    # exit(0)

    proceed_to_write, overwrite = proceed_to_write_experiment_in_csv(sol_stats_filename, metadata, allow_replace, force=force_replace)
    if not proceed_to_write:
        print(f"⚠️ Skipping {metadata['problem']}/{metadata['instance']}/{metadata['front_generator']}/{metadata['timeout']} — already exists in CSV.")
    else:
        if overwrite:
            print(f"♻️ Overwriting existing entry for {metadata['problem']}/{metadata['instance']}/{metadata['front_generator']}/{metadata['timeout']}.")
            remove_row_from_csv(sol_stats_filename, metadata)
        process_json_file(input_file_path, sol_stats_filename)
