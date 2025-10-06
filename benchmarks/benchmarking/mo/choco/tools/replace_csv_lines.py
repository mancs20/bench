def replace_csv_lines(original_csv_path, csv_withh_lines_to_replace_path):
    with open(original_csv_path, "r") as original_csv:
        original_lines = original_csv.readlines()
    with open(csv_withh_lines_to_replace_path, "r") as csv_with_lines_to_replace:
        lines_to_replace = csv_with_lines_to_replace.readlines()
    lines_to_replace = lines_to_replace[1:]  # Remove the header
    for line in lines_to_replace:
        id_line_origianl_csv = get_id_line_of_original_csv_to_replace(original_lines, line)
        original_lines[id_line_origianl_csv] = line
    # Write the updated csv
    with open(original_csv_path, "w") as original_csv:
        original_csv.writelines(original_lines)


def get_id_line_of_original_csv_to_replace(lines_original_csv, line_to_replace):
    line_data_list = line_to_replace.split(",")
    instance = line_data_list[2]
    strategy = line_data_list[6]
    for id_line, line in enumerate(lines_original_csv):
        if instance in line and strategy in line:
            return id_line
    raise ValueError(f"Line to replace not found in the original csv file: {line_to_replace}")


if __name__ == "__main__":
    original_csv_path = "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/nqueens/saug_gava_gias_disj/mo_saug_gava_gias_disj_solutions_and_stats.csv"
    csv_with_lines_to_replace_path = "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/nqueens/saug_gava_gias_disj/to_replace_saug/mo_to_replace_saug_solutions_and_stats.csv"
    replace_csv_lines(original_csv_path, csv_with_lines_to_replace_path)