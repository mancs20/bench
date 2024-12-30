include "alldifferent.mzn";
%-----------------------------------------------------------------------------%
% flowshop.mzn based on the jobshop model in Minizinc benchmark problems
% https://github.com/MiniZinc/minizinc-benchmarks/blob/master/jobshop/jobshop.mzn 
% Originally by: Ralph Becket
% Flowshop permutation adaptation: Manuel COMBARRO SIMÓN
%
% The permutation flowshop scheduling problem.
% --------------------------------
%
% Flow shop is similar to job shop but all the jobs have to be executed in the
% order, task 1 goes to machine 1 and task m goes to machine m. In job shop the
% order of the task depends on the job, in one job it could be that task 1 goes to
% machine 3 and in another one to machine 5. In the permutation version the jobs
% have to be scheduled in the same order, that means that each machine will execute
% the tasks in the same order, for example if in machine 1 they are executed like this:
% task1 job1, task1 job3, task1 job2; then in the other machines have to be the same, e.g
% task2 job1, task2 job3, task2 job2;
% task3 job1, task3 job3, task3 job2;
% Ther order of the jobs is job1,job2 and job3
%
% A flow shop has some machines, each performing a different operation.
% There are some jobs to be performed.
% A job is a sequence of tasks.
% A task involves processing by a single machine for some duration.
% A machine can operate on at most one task at a time.
% Tasks cannot be interrupted.
%
% The goal is to schedule each job to minimise the finishing time.
%
%-----------------------------------------------------------------------------%

%-----------------------------------------------------------------------------%
% Model parameters.
%

int: n_machines;                        % The number of machines.
int: n_jobs;                            % The number of jobs.
array[jobs] of int: due_date_jobs;              % Each job has a due date.
set of int: jobs = 1..n_jobs;
set of int: tasks = 1..n_machines;

    % flow shop job order
array[1..n_jobs] of var 1..n_jobs: jobs_order;

    % job_task_duration[k, k] is the duration of task k of job j.
    %
array [jobs, tasks] of int: job_task_duration;

    % minimal/maximal duration : bounds on end time
    %
int: min_duration = 
    max([sum([job_task_duration[i, j] | j in tasks]) | i in jobs]);

int: max_duration = sum([job_task_duration[i, j] | i in jobs, j in tasks]);

array [jobs] of int: lb_job_tardiness = [sum([job_task_duration[i, j] | j in tasks]) - due_date_jobs[i] | i in jobs];
array [jobs] of int: ub_job_tardiness = [max_duration - due_date_jobs[i] | i in jobs];

%-----------------------------------------------------------------------------%
% Model variables.
%

    % The start time of each job task.
    %
array [jobs, tasks] of var 0.. max_duration: job_task_start;

    % The finishing time is the time of the last task to complete.
    %
var min_duration..max_duration: t_end;

var sum(lb_job_tardiness)..sum(ub_job_tardiness): sum_tardiness;

% var max(lb_job_tardiness)..max(ub_job_tardiness): max_tardiness;
%-----------------------------------------------------------------------------%
% Constraints.
%

    % Sanity check: tasks cannot take a negative amount of time.
    %
constraint
    forall ( j in jobs, k in tasks ) (
        job_task_duration[j, k]  >=  0
    );

    % Each job task must complete before the next.
    %
constraint
    forall ( j in jobs, k in 1..(n_machines - 1) ) (
        job_task_start[j, k] + job_task_duration[j, k]  <=
            job_task_start[j, k + 1]
    );

    % The first job task can start no earlier than time step 0.
    %
constraint
    forall ( j in jobs ) (
        0  <=  job_task_start[j, 1]
    );

        
    % Flow shop. Stablish job order
constraint alldifferent(jobs_order);

    % flow shop constraint, where for each job the task are executed on the same machines    
constraint
    forall (
        j in 1..(n_jobs - 1),
        k in tasks
    ) (
        job_task_start[jobs_order[j], k] + job_task_duration[jobs_order[j], k] <=
        job_task_start[jobs_order[j+1], k]
    );
    
    % Flow shop. Task 1 of the first job has to start at 0.
constraint job_task_start[jobs_order[1],1] = 0;

    % Flow shop. The tasks of the first job have to start one inmediatelly after the other.
constraint
    forall(
        k in 2..n_machines
    ) (
        job_task_start[jobs_order[1],k] = job_task_start[jobs_order[1],k-1] + job_task_duration[jobs_order[1],k-1]
    );
    
    % Flow shop. The first task of each job have to start inmediatelly after the previous first job task has ended.
constraint
    forall(
        j in 2..n_jobs
    ) (
        job_task_start[jobs_order[j],1] = job_task_start[jobs_order[j-1],1] + job_task_duration[jobs_order[j-1],1]
    );
    
    % flow shop constraint. The job_task_start[j][t+1] has to be equal to 
    %max(task_end[j][t], task_end[previous_job][t+1])
constraint
    forall (
        j in 1..(n_jobs - 1),
        k in 1..(n_machines - 1)
    ) (
        job_task_start[jobs_order[j+1], k+1] = max([job_task_start[jobs_order[j+1], k] + job_task_duration[jobs_order[j+1], k], job_task_start[jobs_order[j], k+1] + job_task_duration[jobs_order[j], k+1]])
    );

    % The finishing time must be no earlier than the finishing time
    % of any task.
    %
% constraint t_end = max([job_task_start[j, n_machines] +
%             job_task_duration[j, n_machines] | j in jobs]);

    % Permuting flowshop. The finishing time is equal to the last job's task finishing time     
constraint t_end = job_task_start[jobs_order[n_jobs], n_machines] + job_task_duration[jobs_order[n_jobs], n_machines];

%-----------------------------------------------------------------------------%
% Multi-objective optimization
%
array[1..2] of var int: objs;
constraint objs[1] = t_end;

constraint sum_tardiness = sum([job_task_start[j, n_machines] +
            job_task_duration[j, n_machines] - due_date_jobs[j] | j in jobs]);
constraint objs[2] = sum_tardiness;

% constraint max_tardiness = max([job_task_start[j, n_machines] + 
%             job_task_duration[j, n_machines] - due_date_jobs[j] | j in jobs]);
% constraint objs[3] = max_tardiness;


% minimization
var bool: minimization;

% decision variables
array [jobs, tasks] of var 0.. max_duration: decision_vars;
constraint decision_vars = job_task_start;

%-----------------------------------------------------------------------------%
% Objective.
%

% solve minimize t_end;
solve satisfy;

% output [
%     "job_task_start = ", show(job_task_start), "\n",
%     "job_order = ", show(jobs_order), "\n",
%     "decision_vars = ", show(decision_vars), "\n",
%     "objs = ", show(objs)
% ];
%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%