#!/bin/bash
set -euo pipefail

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=pred rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=auto rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.0001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.001 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.01 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=0.1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=1 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.0001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.001 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.01 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=0.1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=1 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=0
PBS

sleep 600

qsub -V <<'PBS'
#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000



cd "${PBS_O_WORKDIR:-$PWD}"
module load miniforge/3
module load CUDA/12.1.1
eval "$(~/miniforge3/bin/conda shell.bash hook)" 
conda activate pre
cd comp_predictive_learning
python3 -u scripts/train_rnn.py  compute_metrics=True compute_clustering=True train_loop.num_steps=20000 train_loop.eval_every=1000 train_loop.compute_metrics_every=1000 dataset.next_shape_offset=[0,1,2] dataset.next_wall_offset=[0,1,2] dataset.next_object_offset=[0,1,2] model=rnnae model.type=mem rnn.noise=0.05 encoder.hidden_dims=[16,16,16] decoder.hidden_dims=[16,16,16] train_loop.pretrain_decay=10 train_loop.pretrain_act_decay=10 train_loop.pretrain_weight_l1=0 train_loop.pretrain_act_l1=0 encoder.output_dim=512 rnn.hidden_dim=512 train_loop.batch_size=128 seed=1
PBS

sleep 600

