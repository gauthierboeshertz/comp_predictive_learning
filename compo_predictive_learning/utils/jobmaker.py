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
                               parallel=1,
                               env_variables=""):
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
                line = f"CUDA_VISIBLE_DEVICES={gpu_id} {env_variables} {full_cmd}"
                f.write(line + "\n")
            return combinations
        else:
            total = len(combinations)
            for batch_start in range(0, total, parallel):
                batch = combinations[batch_start : batch_start + parallel]
                line = ""
                for j, param_dict in enumerate(batch):
                    gpu_id = gpus[j % num_gpus]
                    full_cmd = make_whole_command(command, param_dict, combination_style)
                    # line = f"CUDA_VISIBLE_DEVICES={gpu_id} {env_variables} {full_cmd} &"
                    line = line + f" CUDA_VISIBLE_DEVICES={gpu_id} {env_variables} {full_cmd} & "
                line = line + " wait \n" 
                f.write(line)
        return combinations



def write_combinations_for_slurm_torch(file_path,
                               param_dicts,
                               command,
                               parallel=1,
                               combination_style="hydra",
                               time="00:40:00",
                               mem_per_cpu="8GB",
                               cpus_per_task=4,
                               time_between=240,
                               use_gpu=True,
                               other_env_variables=""):
    """
    Generate all commands from a list of parameter dictionaries.
    
    Args:
        file_path (str): The file path where the commands will be written.
        param_dicts (list): A list of dictionaries where keys are parameter names and values are their respective values.
        command (str): The base command to be used for each combination.
    
    Returns:
        list: A list of commands generated from the parameter dictionaries.
    """
    sbatch_options = f"-n 1 --comment=\"preemption=yes;requeue=true\" --account=ghb2756 --cpus-per-task={cpus_per_task} {'--gres=gpu:rtx8000:1' if use_gpu else ''} --output=./slurms/%j_%x.out --error=./slurms/%j_%x.err --time={time} --mem-per-cpu={mem_per_cpu}"

    wrap_prefix = f"""--wrap=\" singularity exec  {'--nv' if use_gpu else ''} --overlay /scratch/ghb2756/prl/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c \'
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
                payload = other_env_variables + make_whole_command(command, batch[0], combination_style)
            else:
                parts = [other_env_variables + make_whole_command(command, pd, combination_style) + " &" for pd in batch]
                parts.append("wait")
                payload = " ".join(parts)

            current_delay_seconds = sbatch_call_index * time_between
            begin_option = f"--begin=now+{current_delay_seconds}"
            sbatch_call_index += 1
            line = f"sbatch  {sbatch_options}  {begin_option}  {wrap_prefix}{payload} \' \" \n"
            f.write(line)

    return combinations

def write_combinations_for_slurm_greene(file_path,
                               param_dicts,
                               command,
                               parallel=1,
                               combination_style="hydra",
                               time="00:40:00",
                               mem_per_cpu="8GB",
                               cpus_per_task=4,
                               time_between=240,
                               use_gpu=True,
                               other_env_variables="",
                               one_hour_sleep_every_n_jobs=None):
    """
    Generate all commands from a list of parameter dictionaries.
    
    Args:
        file_path (str): The file path where the commands will be written.
        param_dicts (list): A list of dictionaries where keys are parameter names and values are their respective values.
        command (str): The base command to be used for each combination.
    
    Returns:
        list: A list of commands generated from the parameter dictionaries.
    """
    sbatch_options = f"-n 1 --cpus-per-task={cpus_per_task} {'--gres=gpu:rtx8000:1' if use_gpu else ''} --output=./slurms/%j_%x.out --error=./slurms/%j_%x.err --time={time} --mem-per-cpu={mem_per_cpu}"

    wrap_prefix = f"""--wrap=\" singularity exec  {'--nv' if use_gpu else ''} --overlay /scratch/ghb2756/prl/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c \'
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
                payload = other_env_variables + make_whole_command(command, batch[0], combination_style)
            else:
                parts = [other_env_variables + make_whole_command(command, pd, combination_style) + " &" for pd in batch]
                parts.append("wait")
                payload = " ".join(parts)

            current_delay_seconds = sbatch_call_index * time_between
            begin_option = f"--begin=now+{current_delay_seconds}"
            sbatch_call_index += 1
            line = f"sbatch  {sbatch_options}  {begin_option}  {wrap_prefix}{payload} \' \" \n"
            if one_hour_sleep_every_n_jobs is not None:
                if (sbatch_call_index % one_hour_sleep_every_n_jobs) == 0 and sbatch_call_index > 0:
                    line = line + " sleep 3600 \n"
            f.write(line)

    return combinations

import os

def write_combinations_for_qsub_stdin(qsub_submit_file_path,
                                      qsub_headers,
                                      param_dicts,
                                      command,
                                      combination_style="hydra",
                                      extra_body_lines=None,
                                      pass_env=True):
    """
    Writes a *submission script* containing `qsub <<'PBS' ... PBS` blocks
    (i.e. qsub reads the job script from stdin; no .pbs file is created).

    Outputs:
      1) <name>_combinations.txt: one line per combination (Hydra/argparse args)
      2) qsub_submit_file_path: a shell script you run (bash ...) to submit the array job

    Args:
        qsub_submit_file_path (str): where to write the submission shell script (e.g. submit_jobs.sh)
        qsub_headers (list[str]): PBS headers like "#PBS -l walltime=..." (do NOT include -J; added here)
        param_dicts (list[dict] or list of dict-of-lists): parameter grids
        command (str): base command, e.g. "python train.py"
        combination_style (str): "hydra" or "argparse"
        extra_body_lines (list[str] | None): optional extra lines in the PBS body (module load, cd, etc.)
        pass_env (bool): if True, add qsub -V (export current env into job)
    """
    # ---- build combinations ----
    combinations = []
    for pd in param_dicts:
        combinations.extend(make_combinations(pd))

    # ---- write combinations file ----
    base = os.path.splitext(os.path.basename(qsub_submit_file_path))[0]
    combos_path = os.path.join(os.path.dirname(qsub_submit_file_path), f"{base}_combinations.txt")

    with open(combos_path, "w") as f:
        for param_dict in combinations:
            f.write(make_combination_string(param_dict, style=combination_style) + "\n")

    # ---- build PBS headers (add array) ----
    headers = list(qsub_headers)
    headers.append(f"#PBS -J 1-{len(combinations)}")

    # ---- body: pick args for this array index + run ----
    body_lines = []
    # (optional but common) run from submission dir if PBS_O_WORKDIR exists
    body_lines.append('cd "${PBS_O_WORKDIR:-$PWD}"')

    if extra_body_lines:
        body_lines.extend(extra_body_lines)

    # fetch the correct line for this array index
    body_lines.append(f'ARGS="$(sed -n \\"${{PBS_ARRAY_INDEX}}p\\" {combos_path})"')
    body_lines.append(f'{command} $ARGS')

    # ---- write a submission shell script that submits via stdin here-doc ----
    # You run: bash submit_jobs.sh
    qsub_cmd = "qsub"
    if pass_env:
        qsub_cmd += " -V"

    with open(qsub_submit_file_path, "w") as f:
        f.write("#!/bin/bash\nset -euo pipefail\n\n")
        f.write(f"{qsub_cmd} <<'PBS'\n")
        for h in headers:
            f.write(h + "\n")
        f.write("\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write("PBS\n")

    # make it executable (nice-to-have)
    try:
        os.chmod(qsub_submit_file_path, 0o755)
    except Exception:
        pass

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


def write_combinations_for_qsub_stdin_one_per_combo(
    qsub_submit_file_path,
    qsub_headers,
    param_dicts,
    command,
    combination_style="hydra",
    extra_body_lines=None,
    pass_env=True,
    job_name_prefix=None,
    throttle_sleep_seconds=0,
):
    """
    Writes a *submission script* with one `qsub <<'PBS' ... PBS` block per combo.

    Output:
      - qsub_submit_file_path: a shell script you run (bash ...) to submit all jobs

    Args:
        qsub_submit_file_path (str): e.g. "submit_jobs.sh"
        qsub_headers (list[str]): PBS headers (#!/bin/bash, #PBS -l ..., #PBS -o..., #PBS -e...)
                                  If you want distinct names, do NOT include #PBS -N here; this function adds it.
        param_dicts: dict-of-lists OR list of dict-of-lists (your existing grids)
        command (str): base command, e.g. "python train.py"
        combination_style (str): "hydra" or "argparse"
        extra_body_lines (list[str] | None): module loads, conda activate, cd, etc.
        pass_env (bool): if True, submit with `qsub -V`
        job_name_prefix (str | None): if provided, sets #PBS -N <prefix>_<i>
        throttle_sleep_seconds (int): sleep between submissions to avoid hammering scheduler
    """
    # ---- build combinations ----
    combinations = []
    if isinstance(param_dicts, dict):
        combinations.extend(make_combinations(param_dicts))
    else:
        assert isinstance(param_dicts, (list, tuple)), "param_dicts should be a dict or list/tuple of dicts"
        for pd in param_dicts:
            combinations.extend(make_combinations(pd))

    qsub_cmd = "qsub" + (" -V" if pass_env else "")

    with open(qsub_submit_file_path, "w") as f:
        f.write("#!/bin/bash\nset -euo pipefail\n\n")

        for i, combo in enumerate(combinations, start=1):
            args = make_combination_string(combo, style=combination_style)

            f.write(f"{qsub_cmd} <<'PBS'\n")

            # Headers (strip array + name if present; we add name optionally)
            for h in qsub_headers:
                hs = h.strip()
                if hs.startswith("#PBS -J"):
                    continue
                if hs.startswith("#PBS -N"):
                    continue
                f.write(h + "\n")

            # Optional per-job name
            if job_name_prefix is not None:
                f.write(f"#PBS -N {job_name_prefix}_{i}\n")

            f.write("\n")

            # Body
            f.write('cd "${PBS_O_WORKDIR:-$PWD}"\n')
            if extra_body_lines:
                for line in extra_body_lines:
                    f.write(line + "\n")

            # Run command for this combo
            f.write(f"{command} {args}\n")
            f.write("PBS\n\n")

            if throttle_sleep_seconds and throttle_sleep_seconds > 0:
                f.write(f"sleep {int(throttle_sleep_seconds)}\n\n")

    # # make executable (nice-to-have)
    # try:
    #     os.chmod(qsub_submit_file_path, 0o755)
    # except Exception:
    #     pass

    return combinations

def make_qsub_headers(walltime="24:00:00",
                    ncpu=1,
                    ngpu=1,
                    mem="4gb",
                    output_file=None,
                    error_file=None):
    
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
        f"#PBS -o {output_file}" if output_file else "",
        f"#PBS -e {error_file}" if error_file else "",]
    
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