import os
import re
import random
import networkx as nx


def parse_dzn(file_path):
    """ Parses a .dzn file and extracts n_tasks, d, and suc. """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract number of tasks
    n_tasks = int(re.search(r"n_tasks\s*=\s*(\d+);", content).group(1))

    # Extract durations
    d_match = re.search(r"d\s*=\s*\[(.*?)\];", content, re.DOTALL)
    durations = list(map(int, d_match.group(1).split(','))) if d_match else []

    # Extract successors
    suc_match = re.search(r"suc\s*=\s*\[(.*?)\];", content, re.DOTALL)
    suc_str = suc_match.group(1).split("\n") if suc_match else []

    successors = []
    for line in suc_str:
        nums = re.findall(r"\d+", line)
        successors.append(set(map(int, nums))) if nums else successors.append(set())

    return n_tasks, durations, successors, content


def compute_forward_pass(n_tasks, durations, successors):
    """ Computes the earliest start time for each task using a forward pass. """
    earliest_start = [0] * n_tasks  # Initialize start times

    final_tasks = [task for task in range(n_tasks) if not successors[task]]  # Tasks without successors
    for task in range(n_tasks):
        for succ in successors[task]:
            # safety check
            if task + 1 >= succ:
                raise ValueError(f"Task {task + 1} has a successor {succ} that is not allowed.")
                exit(1)
            earliest_start[succ - 1] = max(earliest_start[succ - 1], earliest_start[task] + durations[task])

    project_duration = max(earliest_start[i] + durations[i] for i in final_tasks)  # Project makespan
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
        deadline[task * 3 + 1] = random.randint(1, 10)  # Earliness cost
        deadline[task * 3 + 2] = random.randint(1, 10)  # Tardiness cost

    return deadline


def modify_dzn(file_path, due_date_factor):
    """ Modifies the .dzn file to add the deadline array. """
    n_tasks, durations, successors, content = parse_dzn(file_path)
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
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = "test"
    folder_path = os.path.join(current_directory, folder_name)
    due_date_factor = 2.5
    process_folder(folder_path, due_date_factor)

    print("Done!")
