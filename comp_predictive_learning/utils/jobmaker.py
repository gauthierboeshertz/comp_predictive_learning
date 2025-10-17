import itertools
import os 

def make_combination_string(combination,style="hydra"):
    """
    Generate a string representation of a combination of parameters.
    
    Args:
        combination (dict): A dictionary representing a combination of parameters.
    
    Returns:
        str: A string representation of the combination.
    """
    args = ""
    if style == "hydra":
        for key, value in combination.items():
            args = args + f" {key}={value}"
    elif style == "argparse":
        for key, value in combination.items():
            args = args + f" --{key} {value}"    
    else:
        raise ValueError("style should be either 'hydra' or 'argparse (for absl too)'")
    args = args.replace(", ",",")
    return args

def make_combinations(param_dict):
    """
    Generate all combinations of parameters from a dictionary of lists.
    
    Args:
        param_dict (dict): A dictionary where keys are parameter names and values are lists of parameter values.
    
    Returns:
        list: A list of dictionaries, each representing a unique combination of parameters.
    """
    keys = param_dict.keys()
    values = param_dict.values()
    
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    print(f"Generated {len(combinations)} combinations.")
    return combinations


def make_whole_command(command, param_dict, style="hydra"):
    args = make_combination_string(param_dict,style)
    whole_command = f" {command}  {args}  "
    return whole_command



def write_all_commands_to_file(file_path,
                               param_dicts,
                               command,
                               combination_style="hydra",
                               gpus=[0],
                               parallel=1):
    """
    Generate all commands from a list of parameter dictionaries.
    
    Args:
        file_path (str): The file path where the commands will be written.
        param_dicts (list): A list of dictionaries where keys are parameter names and values are their respective values.
        command (str): The base command to be used for each combination.
    
    Returns:
        list: A list of commands generated from the parameter dictionaries.
    """
    
    combinations = []
    if isinstance(param_dicts, dict):
        combinations.extend(make_combinations(param_dicts))
    else:
        assert isinstance(param_dicts, (list, tuple)), "param_dicts should be a list or tuple of dictionaries if not a single dictionary"
        for param_dict in param_dicts:
            combinations.extend(make_combinations(param_dict))

    if os.path.exists(file_path):
        print(f"Warning: {file_path} already exists. Deleting it.")
        os.remove(file_path)

    assert parallel >= 1, "`parallel` must be at least 1"
    num_gpus = len(gpus)

    with open(file_path, "w") as f:
        if parallel <= 1:
            for idx, param_dict in enumerate(combinations):
                gpu_id = gpus[idx % num_gpus]
                full_cmd = make_whole_command(command, param_dict, combination_style)
                line = f"CUDA_VISIBLE_DEVICES={gpu_id} {full_cmd}"
                f.write(line + "\n")
            return combinations

        total = len(combinations)
        for batch_start in range(0, total, parallel):
            batch = combinations[batch_start : batch_start + parallel]
            for j, param_dict in enumerate(batch):
                gpu_id = gpus[j % num_gpus]
                full_cmd = make_whole_command(command, param_dict, combination_style)
                line = f"CUDA_VISIBLE_DEVICES={gpu_id} {full_cmd} &"
                f.write(line + "\n")
            f.write("wait\n")
    return combinations

def write_combinations_for_slurm(file_path,
                               param_dicts,
                               command,
                               parallel=1,
                               combination_style="hydra",
                               time="00:30:00",
                               mem_per_cpu="8GB",
                               cpus_per_task=4,
                               time_between=240):
    """
    Generate all commands from a list of parameter dictionaries.
    
    Args:
        file_path (str): The file path where the commands will be written.
        param_dicts (list): A list of dictionaries where keys are parameter names and values are their respective values.
        command (str): The base command to be used for each combination.
    
    Returns:
        list: A list of commands generated from the parameter dictionaries.
    """
    sbatch_options = f"-n 1 --cpus-per-task={cpus_per_task} --gres=gpu:rtx8000:1 --output=./slurms/%j_%x.out --error=./slurms/%j_%x.err --time={time} --mem-per-cpu={mem_per_cpu}"

    wrap_prefix = f"""--wrap=\" singularity exec  --nv --overlay /scratch/ghb2756/prl/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c \'
    source /ext3/env.sh
    conda activate \n """
    
    combinations = []
    if isinstance(param_dicts, dict):
        combinations.extend(make_combinations(param_dicts))
    else:
        assert isinstance(param_dicts, (list, tuple)), "param_dicts should be a list or tuple of dictionaries if not a single dictionary"
        for param_dict in param_dicts:
            combinations.extend(make_combinations(param_dict))

    if os.path.exists(file_path):
        print(f"Warning: {file_path} already exists. Deleting it.")
        os.remove(file_path)

    total = len(combinations)
    with open(file_path, "w") as f:
        sbatch_call_index = 0
        for start in range(0, total, parallel):
            batch = combinations[start:start + parallel]
            if parallel <= 1:
                payload = make_whole_command(command, batch[0], combination_style)
            else:
                parts = [make_whole_command(command, pd, combination_style) + " &" for pd in batch]
                parts.append("wait")
                payload = " ".join(parts)

            current_delay_seconds = sbatch_call_index * time_between
            begin_option = f"--begin=now+{current_delay_seconds}"
            sbatch_call_index += 1
            line = f"sbatch  {sbatch_options}  {begin_option}  {wrap_prefix}{payload} \' \" \n"
            f.write(line)

    return combinations

def write_combinations_for_qsub(qsub_file_path,
                                qsub_headers,
                                param_dicts,
                                commands):
    """
    Generate all commands from a list of parameter dictionaries for qsub.
    
    Args:
        qsub_file_path (str): The file path where the qsub commands will be written.
        qsub_headers (list): A list of qsub headers to be included in the command. This function add #PBS -J 1-N with N = number of combinations.
        param_dicts (list): A list of dictionaries where keys are parameter names and values are their respective values.
        command (str): The base command to be used for each combination.
    
    Returns:
        list: A list of commands generated from the parameter dictionaries.
    """
    
    combinations = []
    for param_dict in param_dicts:
        combinations.extend(make_combinations(param_dict))
    
    combinations_txt = os.path.basename(qsub_file_path).split(".")[0]
    combinations_txt = os.path.join(os.path.dirname(qsub_file_path), combinations_txt + "_combinations.txt")
    with open(combinations_txt, "w") as f:
        for param_dict in combinations:
            combination_string = make_combination_string(param_dict)
            f.write(combination_string + "\n")
    
    qsub_headers = qsub_headers + [f"#PBS -J 1-{len(combinations)}"]
    with open(qsub_file_path, "w") as f:
        for header in qsub_headers:
            f.write(header + "\n")
        
        if isinstance(commands, str):
            f.write(f" {commands} ${combinations_txt}")
        else:
            for command in commands[:-1]:
                f.write(f"{command} \n")
            f.write(f"{commands[-1]}" +' $(sed -n "${PBS_ARRAY_INDEX}p" combinations_txt)')

    return combinations

def make_qsub_headers(walltime="24:00:00",
                    ncpu=1,
                    ngpu=1,
                    mem="4gb",
                    output_file="my_job_output.txt",
                    error_file="my_job_error.txt"):
    
    """
    Generate a list of typical qsub headers.
    Remember that for a job array, these are for each jobs.
    Returns:
        list: A list of qsub headers.
    """
    qsub_headers = [
        "#!/bin/bash",
        f"#PBS -l walltime={walltime}",
        f"#PBS -l select=1:ncpus={ncpu}:mem={mem}gb:ngpus={ngpu}:gpu_type=RTX6000",
        f"#PBS -o {output_file}",
        f"#PBS -e {error_file}"]
    
    return qsub_headers

def make_conda_commands(conda_env_name):
    """
    Generate a list of commands to activate a conda environment.
    
    Args:
        conda_env_name (str): The name of the conda environment to be activated.
    
    Returns:
        list: A list of commands to activate the conda environment.
    """
    return [
        "export PATH=$HOME/miniconda3/bin:$PATH",
        "source ~/miniconda3/bin/activate",
        f"conda activate {conda_env_name}"]