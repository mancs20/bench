import os
import re
import random
import networkx as nx


def add_dummy_initial_end_tasks(file_path):
    """ Parses a .dzn file and extracts n_tasks, d, and suc. """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract number of tasks and write the new value increased by 2
    match = re.search(r"n_tasks\s*=\s*(\d+);", content)
    if match:
        old_n_tasks = int(match.group(1))
        n_tasks = old_n_tasks + 2  # Increase tasks by 2
        content = re.sub(r"n_tasks\s*=\s*\d+;", f"n_tasks = {n_tasks};", content)

    # Modify `d` (durations) → Add `0` at the start and end
    content = re.sub(
        r"d\s*=\s*\[(.*?)\];",
        lambda m: f"d = [0, {m.group(1).strip()}, 0];",
        content,
        flags=re.DOTALL
    )
    d_match = re.search(r"d\s*=\s*\[(.*?)\];", content, re.DOTALL)
    durations = list(map(int, d_match.group(1).split(','))) if d_match else []

    # Modify `rr` (resource requirements) → Add `0s` at start and end of each row
    def modify_rr(match, as_array2d=True):
        """ Modify `rr` to add 0s at the start and end of each row.
            - If `as_array2d=True`: Uses `array2d(...)` format.
            - Else: Uses `|` row format.
        """
        rows = match.group(1).strip().split("\n")
        modified_matrix = []

        for row in rows:
            numbers = re.findall(r"-?\d+", row)
            if numbers:
                modified_numbers = [0] + list(map(int, numbers)) + [0]  # Add 0s at start and end
                modified_matrix.append(modified_numbers)

        n_rows = len(modified_matrix)  # Get number of rows
        n_cols = len(modified_matrix[0]) if modified_matrix else 0  # Get number of columns (updated size)

        if as_array2d:
            # Convert to MiniZinc `array2d(...)` format
            flat_values = [num for row in modified_matrix for num in row]  # Flatten the list
            formatted_rr = f"rr = array2d(1..{n_rows}, 1..{n_cols}, [{', '.join(map(str, flat_values))}]);\n"
        else:
            # Convert to MiniZinc `|` format
            formatted_rr = "rr = [\n"
            for row in modified_matrix:
                formatted_rr += "   | " + ", ".join(map(str, row)) + "\n"
            formatted_rr += "   |];\n"

        return formatted_rr

    content = re.sub(r"rr\s*=\s*\[\|(.*?)\|\];", modify_rr, content, flags=re.DOTALL)

    # Extract successors
    suc_match = re.search(r"suc\s*=\s*\[(.*?)\];", content, re.DOTALL)
    suc_str = suc_match.group(1).split("\n") if suc_match else []

    successors = []
    start_activities = set(range(2, n_tasks))

    for line in suc_str:
        nums = {int(x) + 1 for x in re.findall(r"\d+", line)}  # Convert to int and increment by 1
        start_activities.difference_update(nums)  # Remove incremented numbers from start_set
        if nums:
            successors.append(nums)
        else:
            successors.append({n_tasks})

    successors = [start_activities] + successors

    # Modify `suc` (successors)
    modified_suc = ["suc = ["]
    for nums in successors:
        modified_suc.append(f"   {nums},")
    modified_suc.append("   {  } ];")

    content = re.sub(r"suc\s*=\s*\[\s*(.*?)\s*\];", "\n".join(modified_suc), content, flags=re.DOTALL)

    successors = successors + [set()]

    return n_tasks, durations, successors, content


def compute_forward_pass(n_tasks, durations, successors):
    """ Computes the earliest start time for each task using a forward pass. """
    earliest_start = [0] * n_tasks  # Initialize start times

    final_task = len(successors) - 1  # Get the final task
    for task in range(n_tasks):
        for succ in successors[task]:
            # safety check
            if task + 1 >= succ:
                raise ValueError(f"Task {task + 1} has a successor {succ} that is not allowed.")
                exit(1)
            earliest_start[succ - 1] = max(earliest_start[succ - 1], earliest_start[task] + durations[task])

    project_duration = earliest_start[final_task]  # Project makespan
    return earliest_start, project_duration


def topological_sort(n_tasks, successors):
    """ Performs topological sorting to determine task execution order. """
    G = nx.DiGraph()

    # Add nodes (tasks)
    G.add_nodes_from(range(n_tasks))

    # Add edges based on the successors
    for task, succ in enumerate(successors):
        for s in succ:
            G.add_edge(task, s - 1)  # Convert to 0-based indexing

    # Get topological sorting order
    return list(nx.topological_sort(G))


def generate_deadline(n_tasks, sorted_tasks, project_duration, due_date_factor):
    """ Generates the deadline array with topologically sorted start times and random earliness/tardiness costs. """
    max_due_date = int(due_date_factor * project_duration)

    # Generate n_tasks random numbers between 1 and max_due_date, then sort them
    due_dates = sorted(random.randint(1, max_due_date) for _ in range(n_tasks))

    # Assign sorted due dates to tasks in topological order
    deadline = [0] * (n_tasks * 3)
    for idx, task in enumerate(sorted_tasks):
        deadline[task * 3] = due_dates[idx]  # Desired start time
        deadline[task * 3 + 1] = random.randint(0, 5)  # Earliness cost
        deadline[task * 3 + 2] = random.randint(0, 5)  # Tardiness cost

    return deadline


def modify_dzn(file_path, due_date_factor):
    """ Modifies the .dzn file to add the deadline array. """
    n_tasks, durations, successors, content = add_dummy_initial_end_tasks(file_path)
    earliest_start, project_duration = compute_forward_pass(n_tasks, durations, successors)
    # sorted_tasks = topological_sort(n_tasks, successors)
    sorted_tasks = list(range(n_tasks))
    deadline = generate_deadline(n_tasks, sorted_tasks, project_duration, due_date_factor)

    # Convert deadline array to MiniZinc format
    deadline_str = "deadline = array2d(1..{}, 1..3, [{}]);\n".format(n_tasks, ', '.join(map(str, deadline)))

    # Check if 'deadline' already exists, replace it; otherwise, append it
    if "deadline = array2d(" in content:
        content = re.sub(r"deadline\s*=\s*array2d\(.*?\);", deadline_str, content, flags=re.DOTALL)
    else:
        content += "\n" + deadline_str

    # Write back the modified file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Modified: {file_path}")


def process_folder(folder_path, due_date_factor):
    """ Iterates over all .dzn files in a folder and modifies them. """
    for filename in os.listdir(folder_path):
        if filename.endswith(".dzn"):
            modify_dzn(os.path.join(folder_path, filename), due_date_factor)


if __name__ == "__main__":
    # Example usage
    # get current directory
    due_date_factor = 2.5
    current_directory = os.path.dirname(os.path.abspath(__file__))
    directory_psplib = "minizinc_psplib_due_dates"
    directory_psplib = os.path.join(current_directory, directory_psplib)
    folder_name = ["j30", "j60", "j90", "j120"]
    # folder_name = ["test"]
    for folder in folder_name:
        folder_path = os.path.join(directory_psplib, folder)
        process_folder(folder_path, due_date_factor)
        print("Done! in folder: ", folder)

