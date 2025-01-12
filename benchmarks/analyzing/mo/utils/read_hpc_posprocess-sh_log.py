import pandas as pd


front_generator_strategies = ["GIA", "GIA_bounded", "Saugmecon", "ParetoGavanelliGlobalConstraint"]

def get_error_instances_and_cause(log_hpc_terminal):
    error_experiment_instances = {}
    error_causes = []
    with open(log_hpc_terminal, "r") as file:
        lines = file.readlines()

    for i in range(len(lines)):
        if lines[i].startswith("Error in"):
            error_instance_cause = lines[i].split("choco-solver-orgv4-10-14")[-1].split("cores")
            error_instance = error_instance_cause[0].strip()
            error_line_cause = []
            if error_instance not in error_experiment_instances:
                if i + 1 < len(lines) and lines[i + 1].startswith("Exception in"):
                    error_line_cause.append(lines[i + 1])
                else:
                    error_line_cause.append(error_instance_cause[1].split(".json")[-1].strip())
                error_experiment_instances[error_instance] = error_line_cause
            elif error_instance_cause[1].strip().endswith("no processing"):
                continue
            else:
                error_experiment_instances[error_instance].append(error_instance_cause[1].split(".json")[-1].strip())
        elif lines[i].startswith("Exception in") and i-1 > 0 and not lines[i-1].startswith("Error in"):
            raise ValueError(f"Exception found in line {i} without previous error instance")

    return error_experiment_instances


def separate_instance_front_generator(instances_front_error_dict):
    instance = []
    front_generator = []
    error_cause = []
    for key, value in error_instances_dict.items():
        found_strategy = False
        for strategy in front_generator_strategies:
            strategy_id = key.find(strategy)
            if strategy_id > 0:
                found_strategy = True
                instance.append(key[0:strategy_id-1])
                front_generator.append(key[strategy_id:])
                error_cause.append(value)
                break
        if not found_strategy:
            raise ValueError(f"Front generator strategy not found in the instance name: {key}")
    return instance, front_generator, error_cause


if __name__ == "__main__":
    log_file = "/Users/manuel.combarrosimon/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Thesis ideas/code/bench/benchmarks/campaign/aion/mo/choco-solver.org-v4.10.14/solutions_and_stats_bi-ukp_100plus_sims_cost_clouds_hard/postprocess_error_log"
    error_instances_dict = get_error_instances_and_cause(log_file)
    instance, front_generator, error_cause = separate_instance_front_generator(error_instances_dict)
    # create pandas dataframe with the instances and the error causes
    report = pd.DataFrame({"instance": instance, "front_generator": front_generator, "Error cause": error_cause})
    print(report)

