import os
import shutil


def copy_file_dzn(src_path, dest_dir, append_to_name):
    os.makedirs(dest_dir, exist_ok=True)
    base_name = os.path.basename(src_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{append_to_name}{ext}"
    dest_path = os.path.join(dest_dir, new_name)
    shutil.copy2(src_path, dest_path)
    print(f"File copied and renamed to: {dest_path}")


def remove_lines_from_dzn(file_path, lines_starting_with):
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        for line in lines:
            if not any(line.startswith(prefix) for prefix in lines_starting_with):
                file.write(line)


def add_experiment_name_path_to_csv(benchmark_name, problem, instances, input_dir, output_file):
    # the line is in this way "benchmark_name","problem","instance","instance_relative_path"
    # check if data folder is in the input_dir path
    if "/data/mo/" not in input_dir:
        raise ValueError("The input_dir path must contain the data folder, /data/mo/")
    instances_path = [0]*len(instances)

    for id_instance, instance in enumerate(instances):
        # check if there is a file dzn in the input_dir that contains the instance name
        print(f"Searching for instance {instance} in the input_dir with id {id_instance}...")
        found_instance = False
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if instance in file:
                    instances_path[id_instance] = os.path.join(root, file)
                    found_instance = True
                    break
            if found_instance:
                break
        if not found_instance:
            print(f"Instance {instance} not found in the input_dir")
            raise ValueError(f"Instance {instance} not found in the input_dir")

    # append to the output_file the instances found
    for instance, instance_path in zip(instances, instances_path):
        if instance_path != 0:
            # modify the instance path to be relative to the data folder
            instance_path = instance_path.split("/data/mo/")[1]
            instance_path = "../data/mo/" + instance_path
            line = f'"{benchmark_name}","{problem}","{instance}","{instance_path}"\n'
            with open(output_file, "a") as file:
                file.write(line)
            print(f"Line added to the csv file: {line}")





