import math
import os
import random


def generate_instance(k, elements):
    obj_coef = [[random.randint(1, 1000) for _ in range(elements)] for _ in range(k)]
    constraint_weights = [random.randint(1, 1000) for _ in range(elements)]
    capacity = math.ceil(sum(constraint_weights) / 2)

    # Format the instance as a string
    lines = []
    lines.append(str(k))
    lines.append(str(elements))
    lines.append(str(capacity))

    # Manually format the 2D objective list
    lines.append("[" + str(obj_coef[0]) + ",")  # open outer list
    for i in range(1, k):
        line = str(obj_coef[i])
        if i < k - 1:
            line += ","
        else:
            line += "]"
        lines.append(line)

    lines.append(str(constraint_weights))

    return "\n".join(lines)


def create_instances_like_moolibrary(k, elements, number_instances, folder_path):
    os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
    for i in range(number_instances):
        instance = generate_instance(k, elements)
        instance_name = f"KP_p-{k}_n-{elements}_ins-{i+1}.dat"
        instance_path = os.path.join(folder_path, instance_name)
        with open(instance_path, "w") as file:
            file.write(instance)


if __name__ == "__main__":
    k = 2
    elements = 50
    number_instances = 30
    output_folder_path = "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/data/mo/MOOLibrary/ukp"
    create_instances_like_moolibrary(k, elements, number_instances, output_folder_path)
    print("Done! in folder: ", output_folder_path)
