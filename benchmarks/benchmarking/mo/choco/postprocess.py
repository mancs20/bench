import json
import csv
import os.path
import sys
from collections import OrderedDict
from pathlib import Path


def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False


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
    if "solver-messages" in solution_details_fields:
        solution_details_fields.remove("solver-messages")

    filtered_data = {}
    if statistics:
        filtered_data = OrderedDict({field: statistics.get(field, None) for field in statistics_fields})
        # filtered_data = {field: statistics.get(field, None) for field in statistics_fields}
    if current_json_mo_solution_details:
        if "solver-messages" in current_json_mo_solution_details:
            current_json_mo_solution_details.pop("solver-messages")
        filtered_data.update({field: current_json_mo_solution_details.get(field, None) for field
                              in solution_details_fields})
    all_fields = list(filtered_data.keys())

    # extra multi-objective metrics
    new_fields = ['Hypervolume', 'Front cardinality', 'Hypervolume evolution']
    all_fields.extend(new_fields)

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
