import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from packaging import version

# A tentative to have unique experiment names.
def make_uid(config, mzn_solver, version, machine, timeout_ms, eps_num_subproblems, or_nodes, and_nodes):
  uid = mzn_solver + '_' + str(version) + '_' + machine
  if (int(timeout_ms) % 1000) == 0:
    uid += "_" + str(int(int(timeout_ms)/1000)) + "s"
  else:
    uid += "_" + str(int(timeout_ms)) + "ms"
  if 'java11' in config:
    uid += '_java11'
  if mzn_solver == 'turbo.gpu.release':
    uid += '_' + str(int(eps_num_subproblems)) + '_' + str(int(or_nodes)) + '_' + str(int(and_nodes))
    if 'noatomics' in config:
      uid += '_noatomics'
    if 'globalmem' in config:
      uid += '_globalmem'
    return uid
  else:
    if(or_nodes > 1):
      uid += '_' + str(int(or_nodes)) + "threads"
    return uid

def make_short_uid(uid):
  components = uid.split('_')
  mzn_solver = components[0]
  if mzn_solver == 'turbo.gpu.release':
    mzn_solver = 'turbo.gpu'
  elif mzn_solver == 'turbo.cpu.release':
    mzn_solver = 'turbo.cpu'
  elif mzn_solver == 'com.google.ortools.sat':
    mzn_solver = 'ortools'
  elif mzn_solver == 'com.google.ortools.sat.noglobal':
    mzn_solver = 'ortools.noglobal'
  elif mzn_solver == 'org.choco.choco':
    mzn_solver = 'choco'
  elif mzn_solver == 'org.choco.choco.noglobal':
    mzn_solver = 'choco.noglobal'
  extra = '_'.join(components[4:]) # We remove the machine name and timeout info.
  if extra != '':
    extra = "_" + extra
  return mzn_solver + '_' + components[1] + extra

def read_experiments(experiments):
  all_xp = pd.DataFrame()
  for e in experiments:
    df = pd.read_csv(e)
    df.rename(columns={"timeout_sec": "timeout_ms"}) # due to a mistake in the naming of that column.
    if "java11" in e:
      df['configuration'] = df['configuration'].apply(lambda x: x + "_java11")
    # Find how many subproblems are running in parallel.
    if 'or_nodes' not in df:
      if 'threads' in df:
        df['or_nodes'] = df['threads']
      else:
        df['or_nodes'] = df['configuration'].apply(lambda x: 1) # We suppose it is only 1 thread if the column is missing.
    if 'version' not in df:
      df['version'] = df['configuration'].apply(determine_version)
    else:
      df['version'] = df.apply(lambda row: determine_version(row["configuration"]) if not isinstance(row['version'], str) else row['version'], axis=1)
    if 'machine' not in df:
      df['machine'] = os.path.basename(os.path.dirname(e))
    if 'timeout_ms' not in df:
      df['timeout_ms'] = "300000"
    else:
      df['timeout_ms'] = df.apply(lambda row: "300000" if not isinstance(row['version'], int) else row['version'], axis=1)
     # estimating the number of nodes (lower bound).
    if 'nodes' not in df:
      df['nodes'] = (df['failures'] + df['nSolutions']) * 2 - 1
    if df[df['status'] == 'ERROR'].shape[0] > 0:
      print(e, ': Number of erroneous rows: ', df[df['status'] == 'ERROR'].shape[0])
      print(e, df[df['status'] == 'ERROR']['data_file'])
      df = df[df['status'] != 'ERROR']
    failed_xps = df[(df['mzn_solver'] == "turbo.gpu.release") & df['or_nodes'].isna()]
    if len(failed_xps) > 0:
      failed_xps_path = f"failed_{Path(e).name}"
      failed_xps[['problem', 'model', 'data_file']].to_csv(failed_xps_path, index=False)
      df = df[(df['mzn_solver'] != "turbo.gpu.release") | (~df['or_nodes'].isna())]
      print(f"{e}: {len(failed_xps)} failed experiments using turbo.gpu.release have been removed (the faulty experiments have been stored in {failed_xps_path}).")
    # print(df[(df['mzn_solver'] == "turbo.gpu.release") & df['and_nodes'].isna()])
    # df = df[(df['mzn_solver'] != "turbo.gpu.release") | (~df['and_nodes'].isna())]
    all_xp = pd.concat([df, all_xp], ignore_index=True)
  all_xp['version'] = all_xp['version'].apply(version.parse)
  all_xp['nodes'] = all_xp['nodes'].fillna(0).astype(int)
  if 'memory_configuration' not in all_xp:
    all_xp['memory_configuration'] = 'RAM'
  all_xp['and_nodes'] = all_xp['and_nodes'].fillna(1).astype(int) if 'and_nodes' in all_xp else 1
  all_xp['propagator_mem'] = all_xp['propagator_mem'].fillna(0).astype(int) if 'propagator_mem' in all_xp else 0
  all_xp['store_mem'] = all_xp['store_mem'].fillna(0).astype(int) if 'store_mem' in all_xp else 0
  all_xp['fixpoint_iterations'] = pd.to_numeric(all_xp['fixpoint_iterations'], errors='coerce').fillna(0).astype(int) if 'fixpoint_iterations' in all_xp else 0
  all_xp['eps_num_subproblems'] = pd.to_numeric(all_xp['eps_num_subproblems'], errors='coerce').fillna(1).astype(int) if 'eps_num_subproblems' in all_xp else 1
  all_xp['num_blocks_done'] = pd.to_numeric(all_xp['num_blocks_done'], errors='coerce').fillna(0).astype(int) if 'num_blocks_done' in all_xp else 0
  all_xp['hardware'] = all_xp['machine'].apply(determine_hardware)
  all_xp['uid'] = all_xp.apply(lambda row: make_uid(row['configuration'], row['mzn_solver'], row['version'], row['machine'], row['timeout_ms'],
                                                    row['eps_num_subproblems'], row['or_nodes'], row['and_nodes']), axis=1)
  all_xp['short_uid'] = all_xp['uid'].apply(make_short_uid)
  all_xp['nodes_per_second'] = all_xp['nodes'] / all_xp['solveTime']
  all_xp['fp_iterations_per_node'] = all_xp['fixpoint_iterations'] / all_xp['nodes']
  all_xp['fp_iterations_per_second'] = all_xp['fixpoint_iterations'] / all_xp['solveTime']
  all_xp['normalized_nodes_per_second'] = 0
  all_xp['normalized_fp_iterations_per_second'] = 0
  all_xp['normalized_fp_iterations_per_node'] = 0
  all_xp['normalized_propagator_mem'] = 0
  all_xp['normalized_store_mem'] = 0
  return all_xp

def intersect(df):
  # Group by 'mzn_solver' and convert the 'data_file' column to a set
  solver_instance = df.groupby('uid')['data_file'].apply(set)

  # Intersection of the solvers' instances.
  target_set = solver_instance.iloc[0]
  for i in range(0, len(solver_instance)):
    # print(f"{i} has {len(solver_instance.iloc[i])} instances.")
    target_set = target_set.intersection(solver_instance.iloc[i])

  return df[df['data_file'].isin(target_set)]

def determine_hardware(machine_name):
  if machine_name == 'precision5820':
    return 'Intel Core i9-10900X 10-Core@3.7GHz;24GO DDR4;NVIDIA RTX A5000'
  elif machine_name == 'meluxina':
    return 'AMD EPYC 7452 32-Core@2.35GHz;RAM 512GO;NVIDIA A100 40GB HBM'
  elif machine_name == 'aion':
    return 'AMD EPYC ROME 7H12 64-Core@2.6GHz;RAM 256GO'
  else:
    return 'unknown'

def plot_overall_result(df):
  grouped = df.groupby(['short_uid', 'status']).size().unstack(fill_value=0)
  grouped['OPTIMAL/UNSAT'] = grouped.get('OPTIMAL_SOLUTION', 0) + grouped.get('UNSATISFIABLE', 0)

  # Ensure 'SATISFIED' and 'UNKNOWN' columns exist, even if they are all zeros
  if 'SATISFIED' not in grouped.columns:
      grouped['SATISFIED'] = 0
  if 'UNKNOWN' not in grouped.columns:
      grouped['UNKNOWN'] = 0

  grouped = grouped[['OPTIMAL/UNSAT', 'SATISFIED', 'UNKNOWN']]

  # Sort the DataFrame by 'OPTIMAL/UNSAT' and then 'SATISFIED'
  grouped.sort_values(by=['OPTIMAL/UNSAT', 'SATISFIED'], ascending=[True, True], inplace=True)

  # Plot
  colors = {'OPTIMAL/UNSAT': 'green', 'SATISFIED': 'lightblue', 'UNKNOWN': 'orange'}
  ax = grouped.plot(kind='barh', stacked=True, color=[colors[col] for col in grouped.columns])
  plt.title('Problem Status by Configuration')
  plt.ylabel('Configuration')
  plt.xlabel('Number of Problems')
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
  plt.tight_layout()
  plt.show()

def determine_version(solver_info):
  if 'choco' in solver_info:
    return "4.10.15"
  elif 'ortools' in solver_info:
    return "9.9"
  else:
    return "unknown"

def percent(new, old):
  if old == 0:
    return "inf"
  else:
    v = round(((new - old) / old) * 100)
    if v > 0:
      return f"+{v}%"
    elif v < 0:
      return f"{v}%"
    else:
      return "0%"

def print_table_line(m1, m2, header, key, unit):
  print(f"| {header} | {m2['avg_normalized_'+key]:.2f} | {percent(m2['avg_normalized_'+key], m1['avg_normalized_'+key])} | {m2['best_'+key]} | {m2['avg_'+key]:.2f}{unit} | {percent(m2['avg_'+key], m1['avg_'+key])} | {m2['median_'+key]:.2f}{unit} | {percent(m2['median_'+key], m1['median_'+key])} |")

def compute_normalized_value(df, row, uid1, uid2, key):
  # uid1 is the solver we are created the normalized value for, and uid2 is the solver we compare to.
  uid1_ = uid1 if row['uid'] == uid1 else uid2
  uid2_ = uid2 if row['uid'] == uid1 else uid1
  uid1 = uid1_
  uid2 = uid2_

  # Retrieve the value for each solver for the particular data_file of the row examined.
  nps1 = row[key]
  nps_vals = df[(df['uid'] == uid2) & (df['data_file'] == row['data_file'])][key].values
  assert(len(nps_vals) == 1)
  nps2 = nps_vals[0]

  if nps1 < nps2:
    return nps1 / nps2 * 100
  else:
    return 100

def normalize(df, uid1, uid2, key):
  df2 = df[df['uid'].isin([uid1, uid2])]
  df2['normalized_'+key] = df2.apply(lambda row: compute_normalized_value(df2, row, uid1, uid2, key), axis=1)
  # print(df2[['uid', 'data_file', 'normalized_'+key, key]])
  return df2

def comparison_table_md(df, uid1, uid2):
  df2 = normalize(df, uid1, uid2, 'nodes_per_second')
  df2 = normalize(df2, uid1, uid2, 'fp_iterations_per_second')
  df2 = normalize(df2, uid1, uid2, 'fp_iterations_per_node')
  df2 = normalize(df2, uid1, uid2, 'propagator_mem')
  df2 = normalize(df2, uid1, uid2, 'store_mem')
  m1 = metrics_table(df2[df2['uid'] == uid1]).squeeze()
  m2 = metrics_table(df2[df2['uid'] == uid2]).squeeze()
  total_pb = df2[df2['uid'] == uid1].shape[0]
  print(f"| Metrics | Normalized average [0,100] | Δ v{m1['version']} | #best (_/{total_pb}) | Average | Δ v{m1['version']} | Median | Δ v{m1['version']} |")
  print("|---------|----------------------------|----------|--------------|---------|----------|--------|----------|")
  print_table_line(m1, m2, "Nodes per second", "nodes_per_second", "")
  print_table_line(m1, m2, "Fixpoint iterations per second", "fp_iterations_per_second", "")
  print_table_line(m1, m2, "Fixpoint iterations per node", "fp_iterations", "")
  print_table_line(m1, m2, "Propagators memory", "propagator_mem_mb", "MB")
  print_table_line(m1, m2, "Variables store memory", "store_mem_kb", "KB")
  print("")
  print(f"| Metrics | Count | Δ v{m1['version']} |")
  print("|---------|-------|----------|")
  print(f"| #Problems at optimality | {m2['problem_optimal']} | {m1['problem_optimal']} | ")
  print(f"| #Problems satisfiable | {m2['problem_sat']} | {m1['problem_sat']}  |")
  print(f"| #Problems unknown | {m2['problem_unknown']} | {m1['problem_unknown']}  |")
  print(f"| #Problem with store in shared memory | {m2['problem_with_store_shared']} | {m1['problem_with_store_shared']}  |")
  print(f"| #Problem with prop in shared memory | {m2['problem_with_props_shared']} | {m1['problem_with_props_shared']}  |")
  print(f"| #Problems with IDLE SMs at timeout | {m2['idle_eps_workers']} | {m1['idle_eps_workers']} |")

def metrics_table(df):
  grouped = df.groupby(['uid'])

  # Calculate metrics
  metrics = grouped.agg(
    version=('version', 'first'),
    machine=('machine', 'first'),
    short_uid=('short_uid', 'first'),
    avg_nodes_per_second=('nodes_per_second', lambda x: x[x != 0].mean()),
    median_nodes_per_second=('nodes_per_second', lambda x: x[x != 0].median()),
    avg_normalized_nodes_per_second=('normalized_nodes_per_second', lambda x: x[x != 0].mean()),
    best_nodes_per_second=('normalized_nodes_per_second', lambda x: x[x >= 100.0].count()),
    avg_fp_iterations_per_second=('fp_iterations_per_second', 'mean'),
    median_fp_iterations_per_second=('fp_iterations_per_second', 'median'),
    avg_normalized_fp_iterations_per_second=('normalized_fp_iterations_per_second', 'mean'),
    best_fp_iterations_per_second=('normalized_fp_iterations_per_second', lambda x: x[x >= 100.0].count()),
    avg_fp_iterations=('fp_iterations_per_node', 'mean'),
    median_fp_iterations=('fp_iterations_per_node', 'median'),
    avg_normalized_fp_iterations=('normalized_fp_iterations_per_node', 'mean'),
    best_fp_iterations=('normalized_fp_iterations_per_node', lambda x: x[x < 100.0].count()),
    avg_propagator_mem_mb=('propagator_mem', lambda x: x.mean() / 1000000),
    median_propagator_mem_mb=('propagator_mem', lambda x: x.median() / 1000000),
    avg_normalized_propagator_mem_mb=('normalized_propagator_mem', 'mean'),
    best_propagator_mem_mb=('normalized_propagator_mem', lambda x: x[x < 100.0].count()),
    avg_store_mem_kb=('store_mem', lambda x: x.mean() / 1000),
    avg_normalized_store_mem_kb=('normalized_store_mem', 'mean'),
    best_store_mem_kb=('normalized_store_mem', lambda x: x[x < 100.0].count()),
    median_store_mem_kb=('store_mem', lambda x: x.median() / 1000),
    problem_optimal=('status', lambda x: (x == 'OPTIMAL_SOLUTION').sum() + (x == 'UNSATISFIABLE').sum()),
    problem_sat=('status', lambda x: (x == 'SATISFIED').sum()),
    problem_unknown=('status', lambda x: (x == 'UNKNOWN').sum()),
    problem_with_store_shared=('memory_configuration', lambda x: (x == "store_shared").sum()),
    problem_with_props_shared=('memory_configuration', lambda x: (x == "store_pc_shared").sum())
  )

  # Count problems with non-zero num_blocks_done and not solved to optimality or proven unsatisfiable
  condition = (df['num_blocks_done'] != 0) & (~df['status'].isin(['OPTIMAL_SOLUTION', 'UNSATISFIABLE']))
  idle_eps_workers = df[condition].groupby(['uid']).size().reset_index(name='idle_eps_workers')

  # Merge metrics with idle_eps_workers
  overall_metrics = metrics.merge(idle_eps_workers, on=['uid'], how='left').fillna(0)
  overall_metrics = overall_metrics.sort_values(by=['avg_nodes_per_second', 'version', 'machine'], ascending=[False, False, True])

  return overall_metrics

def compare_solvers_pie_chart(df, uid1, uid2):
    """
    Compares the performance of two solvers based on objective value and optimality.

    Parameters:
    - df: DataFrame containing the data.
    - uid1: Name of the first solver (str).
    - uid2: Name of the second solver (str).

    Returns:
    - Displays a pie chart comparing the performance of the two solvers.
    """

    solvers_df = df[(df['uid'] == uid1) | (df['uid'] == uid2)]

    # Pivoting for 'objective', 'method', and 'status' columns
    pivot_df = solvers_df.pivot_table(index='data_file', columns='uid', values=['objective', 'method', 'status'], aggfunc='first')

    # Compare objective values based on method and optimality status
    conditions = [
        # Error
        (pivot_df['method', uid1] != pivot_df['method', uid2]),

        # Solver 1 better
        ((pivot_df['status', uid1] != "UNKNOWN") & (pivot_df['status', uid2] == "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] == "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] != "OPTIMAL_SOLUTION")),

        # Solver 2 better
        ((pivot_df['status', uid1] == "UNKNOWN") & (pivot_df['status', uid2] != "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] != "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] == "OPTIMAL_SOLUTION")),

        # Equal
        (pivot_df['status', uid1] == pivot_df['status', uid2])
    ]

    choices = ['Error', f'{uid1} better', f'{uid2} better', 'Equal']

    pivot_df['Comparison'] = np.select(conditions, choices, default='Unknown')

    # Get problems with "Unknown" comparison (should not happen, this is for debugging).
    # unknown_problems = pivot_df[pivot_df['Comparison'] == 'Unknown'].index.tolist()
    # if unknown_problems:
    #     print(f"The comparison is 'Unknown' for the following problems: {', '.join(unknown_problems)}")
    # else:
    #     print("There are no problems with 'Unknown' comparison.")

    # Get counts for each category
    category_counts = pivot_df['Comparison'].value_counts()

    color_mapping = {
        f'{uid1} better': 'green' if category_counts.get(f'{uid1} better', 0) >= category_counts.get(f'{uid2} better', 0) else 'orange',
        f'{uid2} better': 'green' if category_counts.get(f'{uid2} better', 0) > category_counts.get(f'{uid1} better', 0) else 'orange',
        'Equal': (0.678, 0.847, 0.902), # light blue
        'Unknown': 'red',
        'Error': 'red'
    }
    colors = [color_mapping[cat] for cat in category_counts.index]

    # Plot pie chart
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f'Objective Value and Optimality Comparison between {uid1} and {uid2}')
    plt.ylabel('')
    plt.show()

    return pivot_df

# List the problems on which the key is greater on the second solver
def list_problem_where_leq(df, key, uid1, uid2):
  df1 = df[df['uid'] == uid1]
  df2 = df[df['uid'] == uid2]
  # Merge the two dataframes on the 'data_file' to compare them
  comparison_df = pd.merge(df1[['data_file', key]], df2[['data_file', key]], on='data_file', suffixes=('_1', '_2'))
  # Find where key is greater on df2
  return comparison_df[comparison_df[key+"_1"] > comparison_df[key+"_2"]]
