/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
Output shape from conv layers 2048
[2026-02-26 17:56:27,411][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 1
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.0001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 17:56:27,416][__main__][INFO] - -----------------
[2026-02-26 17:56:27,416][__main__][INFO] - Pretraining
[2026-02-26 17:56:27,416][__main__][INFO] - -----------------
[2026-02-26 17:56:28,474][__main__][INFO] - Step 1/15000, Training loss: 0.6951,  Regularization loss: 0.6951, Validation loss: 0.6898
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:56:31,117][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:56:31,117][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:56:31,117][__main__][INFO] - wall_hue_val_classifier_gener : 0.3333333333333333
[2026-02-26 17:56:31,117][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:56:31,117][__main__][INFO] - object_hue_val_classifier_gener: 0.16666666666666666
[2026-02-26 17:56:31,117][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:56:31,117][__main__][INFO] - shape_val_classifier_gener    : 0.2222222222222222
[2026-02-26 17:56:31,117][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:56:31,117][__main__][INFO] - optimal_n_clusters            : 2
Output shape from conv layers 2048
[2026-02-26 17:56:31,577][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 0
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.0001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 17:56:31,583][__main__][INFO] - -----------------
[2026-02-26 17:56:31,583][__main__][INFO] - Pretraining
[2026-02-26 17:56:31,583][__main__][INFO] - -----------------
[2026-02-26 17:56:32,753][__main__][INFO] - Step 1/15000, Training loss: 0.6323,  Regularization loss: 0.6323, Validation loss: 0.6242
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:56:36,072][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:56:36,072][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:56:36,072][__main__][INFO] - wall_hue_val_classifier_gener : 0.5
[2026-02-26 17:56:36,072][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:56:36,072][__main__][INFO] - object_hue_val_classifier_gener: 0.2
[2026-02-26 17:56:36,072][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:56:36,072][__main__][INFO] - shape_val_classifier_gener    : 0.2222222222222222
[2026-02-26 17:56:36,072][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:56:36,072][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 17:57:52,820][__main__][INFO] - Step 1001/15000, Training loss: 0.0644,  Regularization loss: 0.0645, Validation loss: 0.0159
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:57:55,533][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:57:55,534][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:57:55,534][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 17:57:55,534][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:57:55,534][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 17:57:55,534][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:57:55,534][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 17:57:55,534][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:57:55,534][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 17:58:00,373][__main__][INFO] - Step 1001/15000, Training loss: 0.0588,  Regularization loss: 0.0590, Validation loss: 0.0162
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:58:03,118][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:58:03,118][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:58:03,118][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 17:58:03,118][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:58:03,118][__main__][INFO] - object_hue_val_classifier_gener: 0.9
[2026-02-26 17:58:03,118][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:58:03,118][__main__][INFO] - shape_val_classifier_gener    : 0.2777777777777778
[2026-02-26 17:58:03,118][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:58:03,118][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 17:59:14,408][__main__][INFO] - Step 2001/15000, Training loss: 0.0118,  Regularization loss: 0.0119, Validation loss: 0.0215
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:59:16,968][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:59:16,968][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:59:16,968][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 17:59:16,968][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:59:16,969][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 17:59:16,969][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:59:16,969][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 17:59:16,969][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:59:16,969][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 17:59:33,928][__main__][INFO] - Step 2001/15000, Training loss: 0.0126,  Regularization loss: 0.0127, Validation loss: 0.0119
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 17:59:36,370][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 17:59:36,371][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 17:59:36,371][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 17:59:36,371][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 17:59:36,371][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 17:59:36,371][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 17:59:36,371][__main__][INFO] - shape_val_classifier_gener    : 0.2222222222222222
[2026-02-26 17:59:36,371][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 17:59:36,371][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:00:32,922][__main__][INFO] - Step 3001/15000, Training loss: 0.0062,  Regularization loss: 0.0062, Validation loss: 0.0042
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:00:35,497][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:00:35,497][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:00:35,497][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:00:35,497][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:00:35,497][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:00:35,497][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:00:35,497][__main__][INFO] - shape_val_classifier_gener    : 0.8888888888888888
[2026-02-26 18:00:35,497][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:00:35,497][__main__][INFO] - optimal_n_clusters            : 3
[2026-02-26 18:01:05,641][__main__][INFO] - Step 3001/15000, Training loss: 0.0085,  Regularization loss: 0.0085, Validation loss: 0.0075
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:01:08,554][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:01:08,554][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:01:08,554][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:01:08,554][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:01:08,554][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:01:08,554][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:01:08,554][__main__][INFO] - shape_val_classifier_gener    : 0.6666666666666666
[2026-02-26 18:01:08,554][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:01:08,554][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 18:01:52,067][__main__][INFO] - Step 4001/15000, Training loss: 0.0029,  Regularization loss: 0.0030, Validation loss: 0.0017
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:01:55,027][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:01:55,027][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:01:55,028][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:01:55,028][__main__][INFO] - optimal_n_clusters            : 6
[2026-02-26 18:02:40,407][__main__][INFO] - Step 4001/15000, Training loss: 0.0044,  Regularization loss: 0.0045, Validation loss: 0.0043
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:02:42,924][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:02:42,924][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:02:42,924][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:02:42,924][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:02:42,924][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:02:42,924][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:02:42,924][__main__][INFO] - shape_val_classifier_gener    : 0.9444444444444444
[2026-02-26 18:02:42,924][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:02:42,924][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:03:32,155][__main__][INFO] - Step 5001/15000, Training loss: 0.0022,  Regularization loss: 0.0023, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:03:35,268][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:03:35,268][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:03:35,268][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:03:35,268][__main__][INFO] - optimal_n_clusters            : 3
[2026-02-26 18:04:14,664][__main__][INFO] - Step 5001/15000, Training loss: 0.0027,  Regularization loss: 0.0027, Validation loss: 0.0072
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:04:17,074][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:04:17,074][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:04:17,074][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:04:17,074][__main__][INFO] - optimal_n_clusters            : 6
[2026-02-26 18:04:53,165][__main__][INFO] - Step 6001/15000, Training loss: 0.0018,  Regularization loss: 0.0019, Validation loss: 0.0010
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:04:55,793][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:04:55,793][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:04:55,793][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:04:55,793][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:04:55,793][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:04:55,793][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:04:55,794][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:04:55,794][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:04:55,794][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 18:05:49,451][__main__][INFO] - Step 6001/15000, Training loss: 0.0016,  Regularization loss: 0.0016, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:05:51,984][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:05:51,985][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:05:51,985][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:05:51,985][__main__][INFO] - optimal_n_clusters            : 3
[2026-02-26 18:06:17,167][__main__][INFO] - Step 7001/15000, Training loss: 0.0010,  Regularization loss: 0.0011, Validation loss: 0.0009
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:06:20,382][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:06:20,382][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:06:20,382][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:06:20,383][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:07:25,811][__main__][INFO] - Step 7001/15000, Training loss: 0.0015,  Regularization loss: 0.0016, Validation loss: 0.0009
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:07:28,178][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:07:28,178][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:07:28,178][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:07:28,178][__main__][INFO] - optimal_n_clusters            : 12
[2026-02-26 18:07:37,816][__main__][INFO] - Step 8001/15000, Training loss: 0.0013,  Regularization loss: 0.0013, Validation loss: 0.0008
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:07:40,561][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:07:40,561][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:07:40,561][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:07:40,561][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:08:55,770][__main__][INFO] - Step 9001/15000, Training loss: 0.0009,  Regularization loss: 0.0009, Validation loss: 0.0008
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:08:58,133][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:08:58,134][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:08:58,134][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:08:58,134][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:09:01,249][__main__][INFO] - Step 8001/15000, Training loss: 0.0009,  Regularization loss: 0.0009, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:09:03,273][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:09:03,273][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:09:03,273][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:09:03,273][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:10:00,124][__main__][INFO] - Step 10001/15000, Training loss: 0.0009,  Regularization loss: 0.0010, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:10:02,312][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:10:02,312][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:10:02,312][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:10:02,312][__main__][INFO] - optimal_n_clusters            : 12
[2026-02-26 18:10:24,773][__main__][INFO] - Step 9001/15000, Training loss: 0.0012,  Regularization loss: 0.0013, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:10:26,848][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:10:26,848][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:10:26,848][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:10:26,848][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:11:05,003][__main__][INFO] - Step 11001/15000, Training loss: 0.0007,  Regularization loss: 0.0007, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:11:07,156][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:11:07,156][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:11:07,156][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:11:07,157][__main__][INFO] - optimal_n_clusters            : 12
[2026-02-26 18:11:48,538][__main__][INFO] - Step 10001/15000, Training loss: 0.0010,  Regularization loss: 0.0011, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:11:50,520][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:11:50,521][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:11:50,521][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:11:50,521][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:12:11,631][__main__][INFO] - Step 12001/15000, Training loss: 0.0005,  Regularization loss: 0.0005, Validation loss: 0.0004
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:12:13,735][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:12:13,735][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:12:13,735][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:12:13,735][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:13:12,062][__main__][INFO] - Step 11001/15000, Training loss: 0.0007,  Regularization loss: 0.0007, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:13:14,054][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:13:14,055][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:13:14,055][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:13:14,055][__main__][INFO] - optimal_n_clusters            : 14
[2026-02-26 18:13:15,897][__main__][INFO] - Step 13001/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:13:18,074][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:13:18,074][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:13:18,074][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:13:18,074][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:13:18,074][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:13:18,074][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:13:18,074][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:13:18,075][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:13:18,075][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:14:21,063][__main__][INFO] - Step 14001/15000, Training loss: 0.0005,  Regularization loss: 0.0005, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:14:23,191][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:14:23,191][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:14:23,191][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:14:23,191][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:14:35,809][__main__][INFO] - Step 12001/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:14:37,832][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:14:37,832][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:14:37,833][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:14:37,833][__main__][INFO] - optimal_n_clusters            : 12
[2026-02-26 18:15:25,711][__main__][INFO] - Step 15000/15000, Training loss: 0.0004,  Regularization loss: 0.0004, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:15:27,942][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:15:27,942][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:15:27,942][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:15:27,942][__main__][INFO] - optimal_n_clusters            : 8
Output shape from conv layers 2048
Optimal clusters: 8, silhouette scores: [0.19609 0.19248 0.19321 0.2314  0.2702  0.28458 0.30242 0.26525 0.29869 0.29842 0.28241 0.27618 0.27276 0.28034 0.27658 0.28559 0.28411 0.26838 0.24785 0.27416 0.25393 0.25049 0.25261]
Output shape from conv layers 2048
Output shape from conv layers 2048
Cluster 0: orig=0.0005  lesioned=0.0015
Output shape from conv layers 2048
Cluster 1: orig=0.0005  lesioned=0.0128
Output shape from conv layers 2048
Cluster 2: orig=0.0005  lesioned=0.0118
Output shape from conv layers 2048
Cluster 3: orig=0.0005  lesioned=0.0113
Output shape from conv layers 2048
Cluster 4: orig=0.0005  lesioned=0.0129
Output shape from conv layers 2048
Cluster 5: orig=0.0005  lesioned=0.0131
Output shape from conv layers 2048
Cluster 6: orig=0.0005  lesioned=0.0199
Output shape from conv layers 2048
Cluster 7: orig=0.0005  lesioned=0.0016
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/neuron_clusters/26-02-26-18-15-48.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/neuron_clusters_lesion/26-02-26-18-15-51.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/neuron_clusters_lesion_0_[0, 1, 0]/26-02-26-18-15-55.png
Output shape from conv layers 2048
Group 0 has [ 25  31  91 122 166 170 240 269 275 277 278 305 340 436 478] neurons
Group 1 has [  1   5   8  22  24  27  40  42  47  49  60  62  64  66  67  68  72  74  79  81  99 101 103 104 108 110 113 117 120 129 131 139 144 149 155 160 161 163 167 175 182 197 203 205 207 209 217 228 229
 232 241 243 246 247 252 263 273 285 292 297 299 317 328 329 341 343 344 345 352 353 365 367 368 375 389 390 394 409 410 413 418 423 427 434 438 446 453 464 470 488 495 497 498 503 510] neurons
Group 2 has [ 16  23  48 107 109 130 148 158 159 176 190 202 206 208 213 219 220 230 248 257 290 294 301 306 313 319 327 334 350 356 357 366 380 381 403 426 431 456 466 469 481 500 507] neurons
Group 3 has [ 50  65  73  76  80 115 138 143 168 178 199 204 214 225 227 268 303 304 355 398 400 450 467 468 472 476 482 485] neurons
Group 4 has [  3   6  14  36  77  83 105 153 162 177 183 186 239 256 316 322 376 396 416] neurons
Group 5 has [ 37  41  85  89  93  94 132 236 272 354 383 395 397 407 417 443 458 463 465 480 484 490 499 506] neurons
Group 6 has [ 46  56 123 124 152 154 185 189 194 216 224 251 267 270 342 348 361 377 384 386 405 432 451 457 479] neurons
Group 7 has [ 11  12  39 315 333 351 363 448 449] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/context_weights_heatmaps/26-02-26-18-15-58.png
[2026-02-26 18:15:59,797][__main__][INFO] - Step 13001/15000, Training loss: 0.0006,  Regularization loss: 0.0007, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
Output shape from conv layers 2048
[2026-02-26 18:16:01,851][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:16:01,851][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:16:01,851][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:16:01,851][__main__][INFO] - optimal_n_clusters            : 16
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/mean_cluster_activity_[[0, 1, 0], [0, 1, 2]]/26-02-26-18-16-05.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed0
figures_14999/global_cluster_correlation/26-02-26-18-16-05.png
[2026-02-26 18:16:13,060][__main__][INFO] - Saved model at step 14999.
[2026-02-26 18:16:13,111][__main__][INFO] - Training completed after 15000 steps.
[2026-02-26 18:17:17,452][__main__][INFO] - Step 14001/15000, Training loss: 0.0005,  Regularization loss: 0.0005, Validation loss: 0.0004
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:17:19,374][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:17:19,374][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:17:19,374][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:17:19,374][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:18:35,481][__main__][INFO] - Step 15000/15000, Training loss: 0.0005,  Regularization loss: 0.0005, Validation loss: 0.0004
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:18:37,304][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:18:37,304][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:18:37,304][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:18:37,305][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:18:37,305][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:18:37,305][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:18:37,305][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:18:37,305][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:18:37,305][__main__][INFO] - optimal_n_clusters            : 12
Output shape from conv layers 2048
Optimal clusters: 12, silhouette scores: [0.22597 0.19912 0.16296 0.16523 0.1583  0.17288 0.18472 0.22512 0.21562 0.23107 0.24072 0.23678 0.2077  0.21781 0.18943 0.20238 0.20438 0.1857  0.18829 0.20681 0.20703 0.20985 0.21113]
Output shape from conv layers 2048
Output shape from conv layers 2048
Cluster 0: orig=0.0004  lesioned=0.0017
Output shape from conv layers 2048
Cluster 1: orig=0.0004  lesioned=0.0014
Output shape from conv layers 2048
Cluster 2: orig=0.0004  lesioned=0.0009
Output shape from conv layers 2048
Cluster 3: orig=0.0004  lesioned=0.0025
Output shape from conv layers 2048
Cluster 4: orig=0.0004  lesioned=0.0206
Output shape from conv layers 2048
Cluster 5: orig=0.0004  lesioned=0.0147
Output shape from conv layers 2048
Cluster 6: orig=0.0004  lesioned=0.0020
Output shape from conv layers 2048
Cluster 7: orig=0.0004  lesioned=0.0134
Output shape from conv layers 2048
Cluster 8: orig=0.0004  lesioned=0.0106
Output shape from conv layers 2048
Cluster 9: orig=0.0004  lesioned=0.0060
Output shape from conv layers 2048
Cluster 10: orig=0.0004  lesioned=0.0020
Output shape from conv layers 2048
Cluster 11: orig=0.0004  lesioned=0.0173
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/neuron_clusters/26-02-26-18-19-01.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/neuron_clusters_lesion/26-02-26-18-19-04.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/neuron_clusters_lesion_0_[0, 1, 2]/26-02-26-18-19-05.png
Output shape from conv layers 2048
Group 0 has [  9  51  96 191 204 231 258 292 460 486 510 511] neurons
Group 1 has [ 95 110 167 184 189 308 346 376 454 505] neurons
Group 2 has [  4  17  21  39  62  78  84  87 115 126 159 229 288 303 328 349 363 374 379 398 401 459 479 494] neurons
Group 3 has [ 90 120 142 169 274 322 483] neurons
Group 4 has [ 11  15  19  26  52  59  64  89 141 150 156 186 230 266 271 275 276 317 369 371 414 415 433 434 464 508] neurons
Group 5 has [ 50  71  79 160 168 181 227 253 272 297 422 436 488 490 502] neurons
Group 6 has [ 24  28  32  40  57  70  86 112 133 134 147 148 165 182 200 201 213 224 243 252 261 273 281 282 287 298 300 307 309 312 319 330 339 341 351 361 362 365 373 378 385 397 399 402 407 420 430 438 440
 445 450 455 463 477 482 489 492 495] neurons
Group 7 has [ 12  23  49  68  83 121 153 262 263 269 286 311 446] neurons
Group 8 has [  1  27  97 122 193 194 289 386 411 432 480] neurons
Group 9 has [  2 130 145 183 207 235 290 344 375 461 496] neurons
Group 10 has [ 43 143 152 196 217 299 302 336 384 429 442 478] neurons
Group 11 has [  0  30  36  63  67 101 104 116 127 129 131 132 164 177 179 187 197 203 206 215 216 219 220 226 242 246 249 250 254 284 294 306 310 313 316 323 326 332 338 342 347 348 350 353 358 370 412 417 423
 426 437 443 444 447 452 462 485 497 504] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/context_weights_heatmaps/26-02-26-18-19-08.png
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/mean_cluster_activity_[[0, 1, 2], [0, 2, 1]]/26-02-26-18-19-14.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.0001_0_0_128/seed1
figures_14999/global_cluster_correlation/26-02-26-18-19-15.png
[2026-02-26 18:19:21,617][__main__][INFO] - Saved model at step 14999.
[2026-02-26 18:19:21,658][__main__][INFO] - Training completed after 15000 steps.
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
Output shape from conv layers 2048
[2026-02-26 18:21:00,841][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 1
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 18:21:00,847][__main__][INFO] - -----------------
[2026-02-26 18:21:00,847][__main__][INFO] - Pretraining
[2026-02-26 18:21:00,847][__main__][INFO] - -----------------
Output shape from conv layers 2048
[2026-02-26 18:21:01,874][__main__][INFO] - Step 1/15000, Training loss: 0.6951,  Regularization loss: 0.6951, Validation loss: 0.6900
[2026-02-26 18:21:02,025][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 0
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 18:21:02,030][__main__][INFO] - -----------------
[2026-02-26 18:21:02,031][__main__][INFO] - Pretraining
[2026-02-26 18:21:02,031][__main__][INFO] - -----------------
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:21:03,147][__main__][INFO] - Step 1/15000, Training loss: 0.6323,  Regularization loss: 0.6323, Validation loss: 0.6244
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:21:04,508][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:21:04,508][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:21:04,508][__main__][INFO] - wall_hue_val_classifier_gener : 0.3333333333333333
[2026-02-26 18:21:04,508][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:21:04,508][__main__][INFO] - object_hue_val_classifier_gener: 0.13333333333333333
[2026-02-26 18:21:04,508][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:21:04,508][__main__][INFO] - shape_val_classifier_gener    : 0.2777777777777778
[2026-02-26 18:21:04,508][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:21:04,508][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 18:21:05,850][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:21:05,850][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:21:05,850][__main__][INFO] - wall_hue_val_classifier_gener : 0.5333333333333333
[2026-02-26 18:21:05,850][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:21:05,850][__main__][INFO] - object_hue_val_classifier_gener: 0.13333333333333333
[2026-02-26 18:21:05,850][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:21:05,850][__main__][INFO] - shape_val_classifier_gener    : 0.2777777777777778
[2026-02-26 18:21:05,850][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:21:05,850][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 18:22:22,944][__main__][INFO] - Step 1001/15000, Training loss: 0.0697,  Regularization loss: 0.0703, Validation loss: 0.0174
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:22:24,374][__main__][INFO] - Step 1001/15000, Training loss: 0.0673,  Regularization loss: 0.0678, Validation loss: 0.0164
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:22:25,172][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:22:25,173][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:22:25,173][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:22:25,173][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:22:25,173][__main__][INFO] - object_hue_val_classifier_gener: 0.7333333333333333
[2026-02-26 18:22:25,173][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:22:25,173][__main__][INFO] - shape_val_classifier_gener    : 0.2777777777777778
[2026-02-26 18:22:25,173][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:22:25,173][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:22:26,564][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:22:26,564][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:22:26,564][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:22:26,564][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:22:26,564][__main__][INFO] - object_hue_val_classifier_gener: 0.9333333333333333
[2026-02-26 18:22:26,564][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:22:26,564][__main__][INFO] - shape_val_classifier_gener    : 0.3055555555555556
[2026-02-26 18:22:26,564][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:22:26,564][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:23:44,216][__main__][INFO] - Step 2001/15000, Training loss: 0.0123,  Regularization loss: 0.0125, Validation loss: 0.0142
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:23:45,581][__main__][INFO] - Step 2001/15000, Training loss: 0.0123,  Regularization loss: 0.0125, Validation loss: 0.0210
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:23:46,554][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:23:46,554][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:23:46,554][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:23:46,554][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:23:46,554][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:23:46,555][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:23:46,555][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 18:23:46,555][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:23:46,555][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:23:47,958][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:23:47,958][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:23:47,958][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:23:47,958][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:23:47,958][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:23:47,958][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:23:47,958][__main__][INFO] - shape_val_classifier_gener    : 0.3055555555555556
[2026-02-26 18:23:47,958][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:23:47,958][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:25:03,714][__main__][INFO] - Step 3001/15000, Training loss: 0.0090,  Regularization loss: 0.0092, Validation loss: 0.0063
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:25:05,053][__main__][INFO] - Step 3001/15000, Training loss: 0.0060,  Regularization loss: 0.0062, Validation loss: 0.0036
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:25:06,001][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:25:06,002][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:25:06,002][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:25:06,002][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:25:06,002][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:25:06,002][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:25:06,002][__main__][INFO] - shape_val_classifier_gener    : 0.6944444444444444
[2026-02-26 18:25:06,002][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:25:06,002][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:25:07,431][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:25:07,431][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:25:07,432][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:25:07,432][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:25:07,432][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:25:07,432][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:25:07,432][__main__][INFO] - shape_val_classifier_gener    : 0.7777777777777778
[2026-02-26 18:25:07,432][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:25:07,432][__main__][INFO] - optimal_n_clusters            : 6
[2026-02-26 18:26:23,009][__main__][INFO] - Step 4001/15000, Training loss: 0.0036,  Regularization loss: 0.0037, Validation loss: 0.0028
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:26:24,384][__main__][INFO] - Step 4001/15000, Training loss: 0.0026,  Regularization loss: 0.0027, Validation loss: 0.0039
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:26:25,407][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:26:25,407][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:26:25,407][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:26:25,407][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:26:25,407][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:26:25,407][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:26:25,407][__main__][INFO] - shape_val_classifier_gener    : 0.9166666666666666
[2026-02-26 18:26:25,407][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:26:25,407][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:26:26,747][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:26:26,747][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:26:26,747][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:26:26,748][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:26:26,748][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:26:26,748][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:26:26,748][__main__][INFO] - shape_val_classifier_gener    : 0.75
[2026-02-26 18:26:26,748][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:26:26,748][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:27:42,442][__main__][INFO] - Step 5001/15000, Training loss: 0.0028,  Regularization loss: 0.0029, Validation loss: 0.0015
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:27:43,830][__main__][INFO] - Step 5001/15000, Training loss: 0.0023,  Regularization loss: 0.0024, Validation loss: 0.0014
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:27:44,826][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:27:44,826][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:27:44,826][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:27:44,826][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:27:46,166][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:27:46,166][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:27:46,166][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:27:46,166][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:29:03,687][__main__][INFO] - Step 6001/15000, Training loss: 0.0017,  Regularization loss: 0.0018, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:29:04,962][__main__][INFO] - Step 6001/15000, Training loss: 0.0019,  Regularization loss: 0.0019, Validation loss: 0.0011
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:29:06,048][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:29:06,049][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:29:06,049][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:29:06,049][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:29:06,049][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:29:06,049][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:29:06,049][__main__][INFO] - shape_val_classifier_gener    : 0.9722222222222222
[2026-02-26 18:29:06,049][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:29:06,049][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:29:07,314][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:29:07,314][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:29:07,314][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:29:07,314][__main__][INFO] - optimal_n_clusters            : 17
[2026-02-26 18:30:25,851][__main__][INFO] - Step 7001/15000, Training loss: 0.0015,  Regularization loss: 0.0016, Validation loss: 0.0010
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:30:27,124][__main__][INFO] - Step 7001/15000, Training loss: 0.0014,  Regularization loss: 0.0014, Validation loss: 0.0014
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:30:28,271][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:30:28,271][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:30:28,271][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:30:28,271][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:30:28,271][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:30:28,271][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:30:28,272][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:30:28,272][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:30:28,272][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:30:29,430][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:30:29,430][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:30:29,430][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:30:29,430][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:30:29,430][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:30:29,430][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:30:29,431][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:30:29,431][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:30:29,431][__main__][INFO] - optimal_n_clusters            : 17
[2026-02-26 18:31:46,753][__main__][INFO] - Step 8001/15000, Training loss: 0.0011,  Regularization loss: 0.0012, Validation loss: 0.0009
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:31:48,049][__main__][INFO] - Step 8001/15000, Training loss: 0.0008,  Regularization loss: 0.0009, Validation loss: 0.0016
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:31:49,133][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:31:49,134][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:31:49,134][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:31:49,134][__main__][INFO] - optimal_n_clusters            : 14
[2026-02-26 18:31:50,311][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:31:50,312][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:31:50,312][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:31:50,312][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 18:33:07,066][__main__][INFO] - Step 9001/15000, Training loss: 0.0011,  Regularization loss: 0.0011, Validation loss: 0.0007
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:33:08,361][__main__][INFO] - Step 9001/15000, Training loss: 0.0013,  Regularization loss: 0.0013, Validation loss: 0.0010
Output shape from conv layers 2048
[2026-02-26 18:33:09,377][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:33:09,377][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:33:09,377][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:33:09,377][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:33:09,378][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:33:09,378][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:33:09,378][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:33:09,378][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:33:09,378][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:33:10,647][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:33:10,647][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:33:10,647][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:33:10,647][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:34:27,798][__main__][INFO] - Step 10001/15000, Training loss: 0.0009,  Regularization loss: 0.0009, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:34:29,065][__main__][INFO] - Step 10001/15000, Training loss: 0.0008,  Regularization loss: 0.0008, Validation loss: 0.0008
Output shape from conv layers 2048
[2026-02-26 18:34:30,096][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:34:30,096][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:34:30,096][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:34:30,096][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:34:31,382][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:34:31,382][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:34:31,382][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:34:31,382][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:35:48,213][__main__][INFO] - Step 11001/15000, Training loss: 0.0007,  Regularization loss: 0.0007, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:35:49,556][__main__][INFO] - Step 11001/15000, Training loss: 0.0008,  Regularization loss: 0.0009, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:35:50,610][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:35:50,610][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:35:50,610][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:35:50,610][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 18:35:51,979][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:35:51,979][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:35:51,979][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:35:51,979][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:37:08,527][__main__][INFO] - Step 12001/15000, Training loss: 0.0008,  Regularization loss: 0.0008, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 18:37:09,822][__main__][INFO] - Step 12001/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 18:37:10,768][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:37:10,768][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:37:10,769][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:37:10,769][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:37:12,008][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:37:12,008][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:37:12,008][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:37:12,008][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:37:12,008][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:37:12,009][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:37:12,009][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:37:12,009][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:37:12,009][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:38:29,445][__main__][INFO] - Step 13001/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0010
Output shape from conv layers 2048
[2026-02-26 18:38:30,801][__main__][INFO] - Step 13001/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0007
Output shape from conv layers 2048
[2026-02-26 18:38:31,720][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:38:31,721][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:38:31,721][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:38:31,721][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:38:32,994][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:38:32,994][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:38:32,994][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:38:32,994][__main__][INFO] - optimal_n_clusters            : 21
[2026-02-26 18:39:51,498][__main__][INFO] - Step 14001/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0007
Output shape from conv layers 2048
[2026-02-26 18:39:52,889][__main__][INFO] - Step 14001/15000, Training loss: 0.0005,  Regularization loss: 0.0005, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 18:39:53,846][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:39:53,846][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:39:53,846][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:39:53,847][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:39:55,183][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:39:55,183][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:39:55,183][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:39:55,183][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:39:55,183][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:39:55,183][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:39:55,184][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:39:55,184][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:39:55,184][__main__][INFO] - optimal_n_clusters            : 13
[2026-02-26 18:41:12,623][__main__][INFO] - Step 15000/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0004
Output shape from conv layers 2048
[2026-02-26 18:41:13,898][__main__][INFO] - Step 15000/15000, Training loss: 0.0004,  Regularization loss: 0.0005, Validation loss: 0.0004
Output shape from conv layers 2048
[2026-02-26 18:41:14,840][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:41:14,840][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:41:14,840][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:41:14,841][__main__][INFO] - optimal_n_clusters            : 11
Output shape from conv layers 2048
[2026-02-26 18:41:16,331][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:41:16,331][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:41:16,332][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:41:16,332][__main__][INFO] - optimal_n_clusters            : 14
Output shape from conv layers 2048
Optimal clusters: 11, silhouette scores: [0.21816 0.19578 0.19781 0.21421 0.23219 0.26549 0.30731 0.32969 0.3291  0.33299 0.30949 0.32441 0.30185 0.31666 0.2827  0.27017 0.26474 0.25554 0.25117 0.27532 0.27354 0.27682 0.26809]
Output shape from conv layers 2048
Optimal clusters: 14, silhouette scores: [0.20393 0.21588 0.20925 0.22153 0.26309 0.24133 0.25714 0.25676 0.28082 0.26323 0.27523 0.26808 0.28723 0.24966 0.26754 0.2604  0.25265 0.27526 0.26954 0.25202 0.26278 0.25569 0.24559]
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Cluster 0: orig=0.0004  lesioned=0.0066
Output shape from conv layers 2048
Cluster 0: orig=0.0004  lesioned=0.0020
Output shape from conv layers 2048
Cluster 1: orig=0.0004  lesioned=0.0156
Output shape from conv layers 2048
Cluster 1: orig=0.0004  lesioned=0.0008
Output shape from conv layers 2048
Cluster 2: orig=0.0004  lesioned=0.0011
Output shape from conv layers 2048
Cluster 2: orig=0.0004  lesioned=0.0004
Output shape from conv layers 2048
Cluster 3: orig=0.0004  lesioned=0.0022
Output shape from conv layers 2048
Cluster 3: orig=0.0004  lesioned=0.0176
Output shape from conv layers 2048
Cluster 4: orig=0.0004  lesioned=0.0072
Output shape from conv layers 2048
Cluster 4: orig=0.0004  lesioned=0.0063
Output shape from conv layers 2048
Cluster 5: orig=0.0004  lesioned=0.0129
Output shape from conv layers 2048
Cluster 5: orig=0.0004  lesioned=0.0095
Output shape from conv layers 2048
Cluster 6: orig=0.0004  lesioned=0.0005
Output shape from conv layers 2048
Cluster 6: orig=0.0004  lesioned=0.0004
Output shape from conv layers 2048
Cluster 7: orig=0.0004  lesioned=0.0056
Output shape from conv layers 2048
Cluster 7: orig=0.0004  lesioned=0.0083
Output shape from conv layers 2048
Cluster 8: orig=0.0004  lesioned=0.0016
Output shape from conv layers 2048
Cluster 8: orig=0.0004  lesioned=0.0010
Output shape from conv layers 2048
Cluster 9: orig=0.0004  lesioned=0.0072
Output shape from conv layers 2048
Cluster 9: orig=0.0004  lesioned=0.0101
Output shape from conv layers 2048
Cluster 10: orig=0.0004  lesioned=0.0016
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/neuron_clusters/26-02-26-18-41-41.png
Cluster 10: orig=0.0004  lesioned=0.0023
Output shape from conv layers 2048
Cluster 11: orig=0.0004  lesioned=0.0028
Output shape from conv layers 2048
Cluster 12: orig=0.0004  lesioned=0.0016
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/neuron_clusters_lesion/26-02-26-18-41-45.png
Cluster 13: orig=0.0004  lesioned=0.0057
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/neuron_clusters/26-02-26-18-41-47.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/neuron_clusters_lesion_0_[0, 1, 2]/26-02-26-18-41-50.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/neuron_clusters_lesion/26-02-26-18-41-51.png
Output shape from conv layers 2048
Group 0 has [ 37  53  71  76 110 129 166 187 220 258 263 286 289 294 332 344 361 363 365 377 398 426 446 465 477 505] neurons
Group 1 has [ 15  16  17  31  45  51  52  73  78  80  83  90 101 105 109 114 118 145 146 148 150 155 158 160 165 171 176 183 186 189 192 207 208 209 212 217 224 233 234 235 241 243 248 249 251 253 262 265 276
 278 280 287 297 307 310 311 312 314 322 324 350 352 357 367 369 370 371 372 386 388 392 402 415 421 435 437 445 451 454 463 473 478 480 484 488 489 494 495 496 511] neurons
Group 2 has [ 27  87 117 120 122 236 239 272 303 331 336 401 410 420 440 442 452 457 486 507] neurons
Group 3 has [ 66  89 115 116 130 139 141 188 191 218 229 259 279 281 315 317 328 330 348 400 403 418 434 467 508] neurons
Group 4 has [  3   7  12  35  40  42  97 132 135 143 164 174 179 231 232 237 250 252 264 277 298 299 319 329 333 345 384 443 444 483 490 492] neurons
Group 5 has [  4  21  28  62 142 181 283 304 313 327 358 385 425 453 459 461] neurons
Group 6 has [ 23  70  85 126 177 205 223 247 275 291 292 378 379 429 501] neurons
Group 7 has [ 46  69  86 125 169 194 246 300 436 491 509] neurons
Group 8 has [ 36  60  79  95 137 152 162 175 206 219 261 267 290 316 356 368 375 376 432 433] neurons
Group 9 has [ 11  19  39  64 108 112 178 184 228 238 266 288 339 449 482] neurons
Group 10 has [ 43  84 373 417 448 460 471 479] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/context_weights_heatmaps/26-02-26-18-41-53.png
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/neuron_clusters_lesion_0_[0, 1, 0]/26-02-26-18-41-55.png
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/mean_cluster_activity_[[0, 1, 2], [0, 2, 1]]/26-02-26-18-41-59.png
Group 0 has [ 65  94 123 156 160 168 322 374 397 400 406 455 458 460 507] neurons
Group 1 has [ 31  68 130 154 230 306 329 363 402 408 435 472 478] neurons
Group 2 has [ 76 108 153 191 307 317 379 449 450] neurons
Group 3 has [ 52  74 157 197 256 278 296 335 352 354 404 410 456 482] neurons
Group 4 has [ 13  22  38  49 107 144 215 292 336 393 425 468 496 505] neurons
Group 5 has [  2   6  10  11  14  29  33  40  41  44  53  75  79  82  85  97 103 104 111 114 116 133 134 138 149 155 166 169 179 180 190 195 203 220 232 243 250 252 257 260 275 283 287 290 293 302 304 305 308
 328 332 333 341 351 353 356 368 384 386 394 420 432 436 437 444 453 462 465 475 483 485 488 503] neurons
Group 6 has [ 32  47  50  64 167 175 205 291 312 340 383 389] neurons
Group 7 has [  1  19  23  25  27  34  37 112 142 145 187 208 219 227 241 319 350 370 413 417 434 448 466 470] neurons
Group 8 has [ 46  67  78 127 184 189 206 236 343] neurons
Group 9 has [158 176 223 274 279 282 295 349 388 480] neurons
Group 10 has [ 15  35  48  60  63  87  89 100 105 115 121 131 148 152 161 164 165 171 201 207 210 213 214 217 224 233 253 255 263 265 266 267 276 299 301 303 310 311 316 337 355 398 405 411 441 467 495 498] neurons
Group 11 has [ 12  20  30  73  93 199 228 242 246 346 358 359 371 409 442 492 499] neurons
Group 12 has [ 17  39  69  80  81 124 129 139 150 162 254 277 325 378 493] neurons
Group 13 has [ 42  90  99 106 146 170 186 209 222 225 331 360 367 381 395 407 422 426 487 490 509] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/context_weights_heatmaps/26-02-26-18-41-59.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed1
figures_14999/global_cluster_correlation/26-02-26-18-42-00.png
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/mean_cluster_activity_[[0, 1, 0], [0, 1, 2]]/26-02-26-18-42-05.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.001_0_0_128/seed0
figures_14999/global_cluster_correlation/26-02-26-18-42-06.png
[2026-02-26 18:42:06,254][__main__][INFO] - Saved model at step 14999.
[2026-02-26 18:42:06,301][__main__][INFO] - Training completed after 15000 steps.
[2026-02-26 18:42:07,732][__main__][INFO] - Saved model at step 14999.
[2026-02-26 18:42:07,775][__main__][INFO] - Training completed after 15000 steps.
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
Output shape from conv layers 2048
[2026-02-26 18:43:44,917][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 1
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.01
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 18:43:44,923][__main__][INFO] - -----------------
[2026-02-26 18:43:44,923][__main__][INFO] - Pretraining
[2026-02-26 18:43:44,923][__main__][INFO] - -----------------
[2026-02-26 18:43:45,902][__main__][INFO] - Step 1/15000, Training loss: 0.6951,  Regularization loss: 0.6952, Validation loss: 0.6916
Output shape from conv layers 2048
Output shape from conv layers 2048
[2026-02-26 18:43:46,678][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 0
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.0001
  pretrain_act_decay: 0.01
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 18:43:46,683][__main__][INFO] - -----------------
[2026-02-26 18:43:46,683][__main__][INFO] - Pretraining
[2026-02-26 18:43:46,683][__main__][INFO] - -----------------
[2026-02-26 18:43:47,706][__main__][INFO] - Step 1/15000, Training loss: 0.6323,  Regularization loss: 0.6324, Validation loss: 0.6273
[2026-02-26 18:43:48,079][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:43:48,079][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:43:48,079][__main__][INFO] - wall_hue_val_classifier_gener : 0.26666666666666666
[2026-02-26 18:43:48,079][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:43:48,079][__main__][INFO] - object_hue_val_classifier_gener: 0.03333333333333333
[2026-02-26 18:43:48,079][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:43:48,079][__main__][INFO] - shape_val_classifier_gener    : 0.3611111111111111
[2026-02-26 18:43:48,079][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:43:48,079][__main__][INFO] - optimal_n_clusters            : 2
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:43:50,221][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:43:50,221][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:43:50,221][__main__][INFO] - wall_hue_val_classifier_gener : 0.4
[2026-02-26 18:43:50,221][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:43:50,221][__main__][INFO] - object_hue_val_classifier_gener: 0.2
[2026-02-26 18:43:50,221][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:43:50,222][__main__][INFO] - shape_val_classifier_gener    : 0.16666666666666666
[2026-02-26 18:43:50,222][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:43:50,222][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 18:45:06,029][__main__][INFO] - Step 1001/15000, Training loss: 0.0654,  Regularization loss: 0.0664, Validation loss: 0.0146
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:45:08,357][__main__][INFO] - Step 1001/15000, Training loss: 0.0648,  Regularization loss: 0.0658, Validation loss: 0.0166
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
[2026-02-26 18:45:08,401][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:45:08,401][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:45:08,401][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:45:08,401][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:45:08,401][__main__][INFO] - object_hue_val_classifier_gener: 0.8333333333333334
[2026-02-26 18:45:08,401][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:45:08,401][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 18:45:08,401][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:45:08,401][__main__][INFO] - optimal_n_clusters            : 5
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:45:10,718][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:45:10,719][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:45:10,719][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:45:10,719][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:45:10,719][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:45:10,719][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:45:10,719][__main__][INFO] - shape_val_classifier_gener    : 0.3055555555555556
[2026-02-26 18:45:10,719][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:45:10,719][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 18:46:25,755][__main__][INFO] - Step 2001/15000, Training loss: 0.0109,  Regularization loss: 0.0113, Validation loss: 0.0092
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:46:28,027][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:46:28,027][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:46:28,027][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:46:28,027][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:46:28,027][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:46:28,027][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:46:28,027][__main__][INFO] - shape_val_classifier_gener    : 0.6944444444444444
[2026-02-26 18:46:28,027][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:46:28,027][__main__][INFO] - optimal_n_clusters            : 13
[2026-02-26 18:46:28,061][__main__][INFO] - Step 2001/15000, Training loss: 0.0105,  Regularization loss: 0.0110, Validation loss: 0.0076
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 18:46:30,347][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:46:30,347][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:46:30,347][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:46:30,347][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:46:30,347][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:46:30,347][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:46:30,347][__main__][INFO] - shape_val_classifier_gener    : 0.7777777777777778
[2026-02-26 18:46:30,347][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:46:30,347][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:47:45,896][__main__][INFO] - Step 3001/15000, Training loss: 0.0052,  Regularization loss: 0.0055, Validation loss: 0.0032
Output shape from conv layers 2048
[2026-02-26 18:47:48,101][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:47:48,101][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:47:48,101][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:47:48,101][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:47:48,101][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:47:48,101][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:47:48,102][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:47:48,102][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:47:48,102][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:47:48,350][__main__][INFO] - Step 3001/15000, Training loss: 0.0047,  Regularization loss: 0.0051, Validation loss: 0.0047
Output shape from conv layers 2048
[2026-02-26 18:47:50,630][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:47:50,631][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:47:50,631][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:47:50,631][__main__][INFO] - optimal_n_clusters            : 14
[2026-02-26 18:49:07,498][__main__][INFO] - Step 4001/15000, Training loss: 0.0026,  Regularization loss: 0.0029, Validation loss: 0.0020
Output shape from conv layers 2048
[2026-02-26 18:49:09,681][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:49:09,681][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:49:09,681][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:49:09,681][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:49:10,013][__main__][INFO] - Step 4001/15000, Training loss: 0.0027,  Regularization loss: 0.0030, Validation loss: 0.0021
Output shape from conv layers 2048
[2026-02-26 18:49:12,200][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:49:12,201][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:49:12,201][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:49:12,201][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:50:28,729][__main__][INFO] - Step 5001/15000, Training loss: 0.0022,  Regularization loss: 0.0024, Validation loss: 0.0020
Output shape from conv layers 2048
[2026-02-26 18:50:30,892][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:50:30,893][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:50:30,893][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:50:30,893][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:50:31,413][__main__][INFO] - Step 5001/15000, Training loss: 0.0021,  Regularization loss: 0.0023, Validation loss: 0.0017
Output shape from conv layers 2048
[2026-02-26 18:50:33,629][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:50:33,629][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:50:33,629][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:50:33,629][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:50:33,629][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:50:33,630][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:50:33,630][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:50:33,630][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:50:33,630][__main__][INFO] - optimal_n_clusters            : 13
[2026-02-26 18:51:49,037][__main__][INFO] - Step 6001/15000, Training loss: 0.0017,  Regularization loss: 0.0018, Validation loss: 0.0012
Output shape from conv layers 2048
[2026-02-26 18:51:51,237][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:51:51,237][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:51:51,237][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:51:51,237][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:51:51,663][__main__][INFO] - Step 6001/15000, Training loss: 0.0015,  Regularization loss: 0.0016, Validation loss: 0.0014
Output shape from conv layers 2048
[2026-02-26 18:51:53,893][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:51:53,893][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:51:53,893][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:51:53,893][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:53:09,329][__main__][INFO] - Step 7001/15000, Training loss: 0.0011,  Regularization loss: 0.0012, Validation loss: 0.0010
Output shape from conv layers 2048
[2026-02-26 18:53:11,585][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:53:11,585][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:53:11,585][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:53:11,585][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:53:12,169][__main__][INFO] - Step 7001/15000, Training loss: 0.0014,  Regularization loss: 0.0015, Validation loss: 0.0012
Output shape from conv layers 2048
[2026-02-26 18:53:14,405][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:53:14,405][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:53:14,405][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:53:14,405][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:54:30,024][__main__][INFO] - Step 8001/15000, Training loss: 0.0010,  Regularization loss: 0.0011, Validation loss: 0.0008
Output shape from conv layers 2048
[2026-02-26 18:54:32,253][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:54:32,253][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:54:32,253][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:54:32,253][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 18:54:32,902][__main__][INFO] - Step 8001/15000, Training loss: 0.0009,  Regularization loss: 0.0010, Validation loss: 0.0025
Output shape from conv layers 2048
[2026-02-26 18:54:35,093][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:54:35,093][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:54:35,093][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:54:35,093][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 18:55:50,479][__main__][INFO] - Step 9001/15000, Training loss: 0.0009,  Regularization loss: 0.0010, Validation loss: 0.0007
Output shape from conv layers 2048
[2026-02-26 18:55:52,647][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:55:52,648][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:55:52,648][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:55:52,648][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 18:55:53,191][__main__][INFO] - Step 9001/15000, Training loss: 0.0012,  Regularization loss: 0.0013, Validation loss: 0.0014
Output shape from conv layers 2048
[2026-02-26 18:55:55,385][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:55:55,386][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:55:55,386][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:55:55,386][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 18:57:10,648][__main__][INFO] - Step 10001/15000, Training loss: 0.0010,  Regularization loss: 0.0011, Validation loss: 0.0006
Output shape from conv layers 2048
[2026-02-26 18:57:12,855][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:57:12,855][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:57:12,855][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:57:12,855][__main__][INFO] - optimal_n_clusters            : 14
[2026-02-26 18:57:13,546][__main__][INFO] - Step 10001/15000, Training loss: 0.0008,  Regularization loss: 0.0009, Validation loss: 0.0008
Output shape from conv layers 2048
[2026-02-26 18:57:15,700][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:57:15,700][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:57:15,701][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:57:15,701][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 18:58:29,893][__main__][INFO] - Step 11001/15000, Training loss: 0.0007,  Regularization loss: 0.0008, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 18:58:31,951][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:58:31,952][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:58:31,952][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:58:31,952][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:58:32,766][__main__][INFO] - Step 11001/15000, Training loss: 0.0006,  Regularization loss: 0.0007, Validation loss: 0.0006
Output shape from conv layers 2048
[2026-02-26 18:58:35,011][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:58:35,011][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:58:35,011][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:58:35,011][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:58:35,011][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:58:35,011][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:58:35,012][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:58:35,012][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:58:35,012][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 18:59:49,035][__main__][INFO] - Step 12001/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0006
Output shape from conv layers 2048
[2026-02-26 18:59:51,177][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:59:51,177][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:59:51,177][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:59:51,177][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:59:51,177][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:59:51,177][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:59:51,177][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:59:51,178][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:59:51,178][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 18:59:52,159][__main__][INFO] - Step 12001/15000, Training loss: 0.0006,  Regularization loss: 0.0007, Validation loss: 0.0010
Output shape from conv layers 2048
[2026-02-26 18:59:54,362][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 18:59:54,362][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 18:59:54,362][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 18:59:54,362][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 19:01:06,664][__main__][INFO] - Step 13001/15000, Training loss: 0.0006,  Regularization loss: 0.0006, Validation loss: 0.0005
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:01:08,533][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:01:08,533][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:01:08,533][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:01:08,533][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 19:01:10,307][__main__][INFO] - Step 13001/15000, Training loss: 0.0006,  Regularization loss: 0.0007, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 19:01:12,236][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:01:12,236][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:01:12,236][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:01:12,236][__main__][INFO] - optimal_n_clusters            : 11
[2026-02-26 19:02:25,416][__main__][INFO] - Step 14001/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0004
Output shape from conv layers 2048
[2026-02-26 19:02:27,387][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:02:27,387][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:02:27,387][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:02:27,387][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:02:27,387][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:02:27,387][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:02:27,387][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:02:27,388][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:02:27,388][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 19:02:29,468][__main__][INFO] - Step 14001/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0005
Output shape from conv layers 2048
[2026-02-26 19:02:31,564][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:02:31,564][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:02:31,564][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:02:31,565][__main__][INFO] - optimal_n_clusters            : 9
[2026-02-26 19:07:54,943][__main__][INFO] - Step 15000/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0004
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:08:41,494][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:08:41,500][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:08:41,500][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:08:41,500][__main__][INFO] - optimal_n_clusters            : 8
Output shape from conv layers 2048
[2026-02-26 19:09:28,932][__main__][INFO] - Step 15000/15000, Training loss: 0.0005,  Regularization loss: 0.0006, Validation loss: 0.0004
Optimal clusters: 8, silhouette scores: [0.22341 0.22119 0.22464 0.25698 0.1823  0.23987 0.31199 0.26884 0.24061 0.2422  0.26464 0.279   0.25941 0.25835 0.2731  0.27234 0.26926 0.26658 0.26833 0.24868 0.24722 0.24517 0.24236]
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
[2026-02-26 19:10:01,847][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:10:01,847][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:10:01,847][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:10:01,847][__main__][INFO] - optimal_n_clusters            : 10
Output shape from conv layers 2048
Cluster 0: orig=0.0004  lesioned=0.0204
Output shape from conv layers 2048
Optimal clusters: 10, silhouette scores: [0.18696 0.2017  0.21577 0.22815 0.22704 0.2416  0.25653 0.26725 0.28753 0.28046 0.27978 0.27764 0.25929 0.26452 0.27635 0.27312 0.27824 0.26977 0.27516 0.23579 0.23293 0.23571 0.2434 ]
Output shape from conv layers 2048
Cluster 1: orig=0.0004  lesioned=0.0148
Output shape from conv layers 2048
Output shape from conv layers 2048
Cluster 2: orig=0.0004  lesioned=0.0074
Output shape from conv layers 2048
Cluster 0: orig=0.0004  lesioned=0.0090
Output shape from conv layers 2048
Cluster 3: orig=0.0004  lesioned=0.0181
Output shape from conv layers 2048
Cluster 1: orig=0.0004  lesioned=0.0088
Output shape from conv layers 2048
Cluster 4: orig=0.0004  lesioned=0.0064
Output shape from conv layers 2048
Cluster 2: orig=0.0004  lesioned=0.0137
Output shape from conv layers 2048
Cluster 5: orig=0.0004  lesioned=0.0035
Output shape from conv layers 2048
Cluster 3: orig=0.0004  lesioned=0.0121
Output shape from conv layers 2048
Cluster 6: orig=0.0004  lesioned=0.0012
Output shape from conv layers 2048
Cluster 4: orig=0.0004  lesioned=0.0048
Output shape from conv layers 2048
Cluster 7: orig=0.0004  lesioned=0.0052
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/neuron_clusters/26-02-26-19-10-27.png
Cluster 5: orig=0.0004  lesioned=0.0004
Output shape from conv layers 2048
Cluster 6: orig=0.0004  lesioned=0.0033
Output shape from conv layers 2048
Cluster 7: orig=0.0004  lesioned=0.0012
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/neuron_clusters_lesion/26-02-26-19-10-31.png
Cluster 8: orig=0.0004  lesioned=0.0071
Output shape from conv layers 2048
Cluster 9: orig=0.0004  lesioned=0.0065
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/neuron_clusters/26-02-26-19-10-35.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/neuron_clusters_lesion_0_[0, 1, 2]/26-02-26-19-10-36.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/neuron_clusters_lesion/26-02-26-19-10-37.png
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/neuron_clusters_lesion_0_[0, 1, 0]/26-02-26-19-10-39.png
Group 0 has [  2   8  17  19  22  25  27  31  34  38  39  42  46  52  57  58  61  74  79  84  87  95  96 101 107 126 127 137 143 149 153 158 178 183 184 189 193 195 198 202 203 213 224 225 234 252 258 261 273
 276 279 283 291 309 316 323 326 335 342 343 346 353 363 375 382 394 410 423 431 432 434 454 466 469 471 477 481 482 488 489 499 500 503 504 509] neurons
Group 1 has [  4   9  54  70  85  88 162 192 226 233 243 245 246 254 282 312 314 329 362 409 422 436 437 448 461 484 495 507 510] neurons
Group 2 has [ 53  94  99 115 125 152 176 180 200 205 215 223 230 236 275 280 284 286 293 443] neurons
Group 3 has [ 47  49  67  69 116 170 175 186 208 251 266 274 320 389 391 399 411 468 502] neurons
Group 4 has [ 20  65  68  86  89 105 106 120 165 167 179 194 247 250 263 352 393 398 470 474 485 505] neurons
Group 5 has [ 76 119 154 166 231 242 259 349 351 355 371 384 418 433 438 442 465 467] neurons
Group 6 has [ 23  51  90 142 156 211 272 297 301 330 367 379 380 385 414 425] neurons
Group 7 has [  5 141 210 257 264 271 277 281 311 313 319 331 370 383 401 417 429 449 486 494] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/context_weights_heatmaps/26-02-26-19-10-40.png
Output shape from conv layers 2048
Output shape from conv layers 2048
Group 0 has [ 23  35  37  40  47  48  49  62  74  78  89  94  95  99 109 112 116 118 119 128 132 133 134 145 148 152 171 175 176 179 180 181 183 184 185 186 197 203 204 213 215 218 220 222 261 266 271 273 277
 287 292 300 326 334 342 345 365 377 388 406 412 415 417 433 436 438 442 445 451 453 454 457 464 468 470 477 484 487 496 498 502 509 510] neurons
Group 1 has [ 16  22  51  60  68  79  81  84 108 135 164 207 250 289 291 316 317 336 347 355 358 360 376 381 399 426 444 460 494] neurons
Group 2 has [  4  32  45  59  85 117 124 150 153 173 174 190 227 242 243 253 257 308 321 330 340 361 368 396 450 459 471] neurons
Group 3 has [ 11  14  18  43  56  80 104 106 138 163 172 191 230 241 249 260 282 346 356 385 408 418 427 429 440 443 490] neurons
Group 4 has [  6  17  39  69  76  90  91  93 105 139 149 189 194 199 208 223 224 225 231 233 236 240 256 259 290 298 312 333 349 363 367 373 395 405 407 428 446 458 463 474 475 499] neurons
Group 5 has [ 13  29  67  96 162 166 211 237 246 247 275 343 364 392 422 430 455 479 503] neurons
Group 6 has [  0  55  61  63  98 143 144 169 188 192 198 239 283 331 335 389] neurons
Group 7 has [ 19  27  57  65 103 157 178 209 210 288 305 310 322 325 332 350 354 366 448] neurons
Group 8 has [ 15  25  72 140 156 232 238 248 268 313 323 348 441 485 504] neurons
Group 9 has [ 20  31  33  50  66  82 228 309 318 319 339 352 384 411 414 423 465 476 480 500 506] neurons
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/context_weights_heatmaps/26-02-26-19-10-42.png
Output shape from conv layers 2048
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/mean_cluster_activity_[[0, 1, 2], [0, 2, 1]]/26-02-26-19-10-45.png
Output shape from conv layers 2048
Output shape from conv layers 2048
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed1
figures_14999/global_cluster_correlation/26-02-26-19-10-46.png
[2026-02-26 19:10:47,344][__main__][INFO] - Saved model at step 14999.
[2026-02-26 19:10:47,444][__main__][INFO] - Training completed after 15000 steps.
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/mean_cluster_activity_[[0, 1, 0], [0, 1, 2]]/26-02-26-19-10-47.png
/home/ghb24/paper_git/er/comp_predictive_learning/revieww/contextual_full_1_0.8/dddshapes_6/256_128/[0]/[0, 1, 2]/[0, 1, 2]/[0]/[0, 1, 2]/[0]/True_False_False_True_False_True_1_1.0_1_1_0.5_0.5/ae_pred/conv_[32, 32, 32]_relu_none_False_512/rnn_512_relu_1_False_eye_0.05/conv_[32, 32, 32]_relu_none/True/15000_0.001_0.0001_0.01_0_0_128/seed0
figures_14999/global_cluster_correlation/26-02-26-19-10-48.png
[2026-02-26 19:10:49,576][__main__][INFO] - Saved model at step 14999.
[2026-02-26 19:10:49,622][__main__][INFO] - Training completed after 15000 steps.
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/paper_git/er/comp_predictive_learning/scripts/train_rnn.py:54: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="configs",config_name="train_rnn")
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train_rnn': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
--- Generating Support: 'full' ---
Total contexts: 27
Train contexts: 21
Validation contexts: 6
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/torch/utils/data/sampler.py:76: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn(
Latents for 3DShapes disentanglement: ['wall_hue', 'object_hue', 'shape'] {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
Output shape from conv layers 2048
Output shape from conv layers 2048
[2026-02-26 19:12:21,100][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 1
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.001
  pretrain_act_decay: 0.0001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 19:12:21,106][__main__][INFO] - -----------------
[2026-02-26 19:12:21,106][__main__][INFO] - Pretraining
[2026-02-26 19:12:21,106][__main__][INFO] - -----------------
[2026-02-26 19:12:21,167][__main__][INFO] - Device: cuda
model:
  _target_: compo_predictive_learning.models.rnn_ae.RNNAE
  encoder_cfg: null
  decoder_cfg: null
  rnn_cfg: null
  type: pred
  name: ae_${.type}
dataset:
  num_train_sequences_per_context: 256
  num_val_sequences_per_context: 128
  contexts:
  - next_shape_offset
  - next_floor_offset
  - next_wall_offset
  - next_object_offset
  - next_scale_offset
  - next_orientation_offset
  next_shape_offset:
  - 0
  - 1
  - 2
  next_scale_offset:
  - 0
  next_orientation_offset:
  - 0
  next_floor_offset:
  - 0
  next_wall_offset:
  - 0
  - 1
  - 2
  next_object_offset:
  - 0
  - 1
  - 2
  one_floor: true
  one_wall: false
  one_object: false
  one_scale: true
  one_shape: false
  one_orientation: true
  subsample_task_abstract: 1.0
  subsample_shape: 1
  subsample_scale: 1.0
  subsample_orientation: 1
  subsample_floor: 1
  subsample_wall: 0.5
  subsample_object: 0.5
  seq_len: 6
  path: /home/ghb24/paper_git/comp_predictive_learning/data/3dshapes.h5
  color_mode: rgb
  name: dddshapes_${.seq_len}/${.num_train_sequences_per_context}_${.num_val_sequences_per_context}/${.next_floor_offset}/${.next_wall_offset}/${.next_object_offset}/${.next_scale_offset}/${.next_shape_offset}/${.next_orientation_offset}/${.one_floor}_${.one_wall}_${.one_object}_${.one_scale}_${.one_shape}_${.one_orientation}_${.subsample_shape}_${.subsample_scale}_${.subsample_orientation}_${.subsample_floor}_${.subsample_wall}_${.subsample_object}
rnn:
  _target_: compo_predictive_learning.models.rnn.JitLeakyRNNLayer
  input_dim: 512
  hidden_dim: 512
  noise: 0.05
  rnn_init: eye
  activation: relu
  leak_alpha: 1
  mlp_dynamics: false
  name: rnn_${.hidden_dim}_${.activation}_${.leak_alpha}_${.mlp_dynamics}_${.rnn_init}_${.noise}
encoder:
  _target_: compo_predictive_learning.models.encoders.ConvEncoder
  input_dim:
  - 3
  - 64
  - 64
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim: 512
  activation: relu
  norm_layer: none
  max_pooling: false
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}_${.max_pooling}_${.output_dim}
decoder:
  _target_: compo_predictive_learning.models.decoders.ConvDecoder
  input_dim: 512
  hidden_dims:
  - 32
  - 32
  - 32
  output_dim:
  - 3
  - 64
  - 64
  activation: relu
  norm_layer: none
  name: conv_${.hidden_dims}_${.activation}_${.norm_layer}
seed: 0
save_models: false
one_context_per_batch: true
no_redo: true
compute_clustering: true
compute_metrics: true
use_whole_drawing: false
one_hot_context: false
support_type: full
subsample_contexts_ratio: 0.8
support_connectivity: 1
make_plots: true
train_loop:
  num_steps: 15000
  eval_every: 1000
  compute_metrics_every: 1000
  make_plots_every: -1
  pretrain_lr: 0.001
  pretrain_decay: 0.001
  pretrain_act_decay: 0.0001
  pretrain_act_l1: 0
  pretrain_weight_l1: 0
  batch_size: 128
  save_model_every: 100000000

RNNAE(
  (encoder): ConvEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
      (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (5): ReLU(inplace=True)
    )
    (fc): Linear(in_features=2048, out_features=512, bias=True)
  )
  (rnn): JitLeakyRNNLayer(
    (input_layer): RecursiveScriptModule(original_name=Linear)
    (weight_hh): RecursiveScriptModule(original_name=Linear)
    (activation): RecursiveScriptModule(original_name=ReLU)
  )
  (decoder): ConvDecoder(
    (fc): Linear(in_features=512, out_features=2048, bias=True)
    (conv_layers): Sequential(
      (0): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (3): ReLU(inplace=True)
    )
    (output_layer): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  )
  (loss_fn): MSELoss()
  (activation): Identity()
)
[2026-02-26 19:12:21,174][__main__][INFO] - -----------------
[2026-02-26 19:12:21,174][__main__][INFO] - Pretraining
[2026-02-26 19:12:21,174][__main__][INFO] - -----------------
[2026-02-26 19:12:27,617][__main__][INFO] - Step 1/15000, Training loss: 0.6951,  Regularization loss: 0.6951, Validation loss: 0.6898
[2026-02-26 19:12:27,940][__main__][INFO] - Step 1/15000, Training loss: 0.6323,  Regularization loss: 0.6323, Validation loss: 0.6242
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:13:19,579][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:13:19,580][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:13:19,580][__main__][INFO] - wall_hue_val_classifier_gener : 0.3333333333333333
[2026-02-26 19:13:19,580][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:13:19,581][__main__][INFO] - object_hue_val_classifier_gener: 0.16666666666666666
[2026-02-26 19:13:19,581][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:13:19,581][__main__][INFO] - shape_val_classifier_gener    : 0.2777777777777778
[2026-02-26 19:13:19,581][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:13:19,582][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 19:13:21,241][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:13:21,241][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:13:21,241][__main__][INFO] - wall_hue_val_classifier_gener : 0.5
[2026-02-26 19:13:21,241][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:13:21,241][__main__][INFO] - object_hue_val_classifier_gener: 0.2
[2026-02-26 19:13:21,241][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:13:21,241][__main__][INFO] - shape_val_classifier_gener    : 0.2222222222222222
[2026-02-26 19:13:21,241][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:13:21,241][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 19:19:52,870][__main__][INFO] - Step 1001/15000, Training loss: 0.0611,  Regularization loss: 0.0613, Validation loss: 0.0181
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:19:53,189][__main__][INFO] - Step 1001/15000, Training loss: 0.0667,  Regularization loss: 0.0669, Validation loss: 0.0190
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:19:55,057][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:19:55,057][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:19:55,057][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:19:55,057][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:19:55,057][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:19:55,057][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:19:55,057][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 19:19:55,057][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:19:55,057][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 19:19:55,749][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:19:55,750][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:19:55,750][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:19:55,750][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:19:55,750][__main__][INFO] - object_hue_val_classifier_gener: 0.6
[2026-02-26 19:19:55,750][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:19:55,750][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 19:19:55,750][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:19:55,750][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 19:24:18,094][__main__][INFO] - Step 2001/15000, Training loss: 0.0121,  Regularization loss: 0.0122, Validation loss: 0.0093
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:24:35,796][__main__][INFO] - Step 2001/15000, Training loss: 0.0113,  Regularization loss: 0.0114, Validation loss: 0.0088
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:25:10,981][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:25:10,986][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:25:10,986][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:25:10,986][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:25:10,986][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:25:10,986][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:25:10,986][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 19:25:10,986][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:25:10,987][__main__][INFO] - optimal_n_clusters            : 4
[2026-02-26 19:25:30,443][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:25:30,444][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:25:30,444][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:25:30,444][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:25:30,444][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:25:30,444][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:25:30,444][__main__][INFO] - shape_val_classifier_gener    : 0.25
[2026-02-26 19:25:30,444][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:25:30,444][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 19:29:31,215][__main__][INFO] - Step 3001/15000, Training loss: 0.0096,  Regularization loss: 0.0096, Validation loss: 0.0077
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:29:31,749][__main__][INFO] - Step 3001/15000, Training loss: 0.0066,  Regularization loss: 0.0067, Validation loss: 0.0036
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:29:33,636][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:29:33,637][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:29:33,637][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:29:33,637][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:29:33,637][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:29:33,637][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:29:33,637][__main__][INFO] - shape_val_classifier_gener    : 0.4444444444444444
[2026-02-26 19:29:33,637][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:29:33,637][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 19:29:34,248][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:29:34,248][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:29:34,248][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:29:34,248][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:29:34,248][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:29:34,248][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:29:34,248][__main__][INFO] - shape_val_classifier_gener    : 0.9166666666666666
[2026-02-26 19:29:34,248][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:29:34,248][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 19:33:13,055][__main__][INFO] - Step 4001/15000, Training loss: 0.0051,  Regularization loss: 0.0051, Validation loss: 0.0029
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:33:26,250][__main__][INFO] - Step 4001/15000, Training loss: 0.0033,  Regularization loss: 0.0033, Validation loss: 0.0020
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:33:51,922][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:33:51,922][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:33:51,922][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:33:51,923][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:33:51,923][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:33:51,923][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:33:51,923][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:33:51,923][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:33:51,923][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 19:34:08,751][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:34:08,751][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:34:08,752][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:34:08,752][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:34:08,752][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:34:08,752][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:34:08,752][__main__][INFO] - shape_val_classifier_gener    : 0.8611111111111112
[2026-02-26 19:34:08,752][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:34:08,752][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 19:49:56,548][__main__][INFO] - Step 5001/15000, Training loss: 0.0026,  Regularization loss: 0.0026, Validation loss: 0.0084
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:50:30,986][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:50:30,987][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 19:50:30,992][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:50:30,993][__main__][INFO] - optimal_n_clusters            : 10
[2026-02-26 19:50:38,074][__main__][INFO] - Step 5001/15000, Training loss: 0.0023,  Regularization loss: 0.0023, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 19:51:11,719][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 19:51:11,720][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 19:51:11,720][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 19:51:11,720][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 19:51:11,720][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 19:51:11,720][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 19:51:11,720][__main__][INFO] - shape_val_classifier_gener    : 0.8611111111111112
[2026-02-26 19:51:11,720][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 19:51:11,720][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 20:06:12,401][__main__][INFO] - Step 6001/15000, Training loss: 0.0022,  Regularization loss: 0.0022, Validation loss: 0.0017
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:06:43,636][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:06:43,636][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:06:43,637][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:06:43,637][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:06:43,637][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:06:43,638][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:06:43,638][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 20:06:43,638][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:06:43,638][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 20:07:03,703][__main__][INFO] - Step 6001/15000, Training loss: 0.0020,  Regularization loss: 0.0020, Validation loss: 0.0012
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:07:37,275][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:07:37,275][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:07:37,275][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:07:37,275][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:07:37,275][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:07:37,275][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:07:37,275][__main__][INFO] - shape_val_classifier_gener    : 0.9444444444444444
[2026-02-26 20:07:37,276][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:07:37,276][__main__][INFO] - optimal_n_clusters            : 5
[2026-02-26 20:22:39,876][__main__][INFO] - Step 7001/15000, Training loss: 0.0015,  Regularization loss: 0.0015, Validation loss: 0.0020
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:23:08,787][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:23:08,787][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 20:23:08,787][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:23:08,787][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 20:23:34,020][__main__][INFO] - Step 7001/15000, Training loss: 0.0011,  Regularization loss: 0.0012, Validation loss: 0.0009
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:24:08,172][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:24:08,173][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:24:08,173][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:24:08,173][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:24:08,173][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:24:08,173][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:24:08,173][__main__][INFO] - shape_val_classifier_gener    : 0.9444444444444444
[2026-02-26 20:24:08,173][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:24:08,173][__main__][INFO] - optimal_n_clusters            : 15
[2026-02-26 20:38:28,447][__main__][INFO] - Step 8001/15000, Training loss: 0.0010,  Regularization loss: 0.0011, Validation loss: 0.0007
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:39:00,672][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:39:00,672][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 20:39:00,672][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:39:00,672][__main__][INFO] - optimal_n_clusters            : 2
[2026-02-26 20:39:45,071][__main__][INFO] - Step 8001/15000, Training loss: 0.0008,  Regularization loss: 0.0008, Validation loss: 0.0037
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:40:17,566][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:40:17,566][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:40:17,566][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:40:17,567][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:40:17,567][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:40:17,567][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:40:17,567][__main__][INFO] - shape_val_classifier_gener    : 0.9444444444444444
[2026-02-26 20:40:17,567][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:40:17,567][__main__][INFO] - optimal_n_clusters            : 6
[2026-02-26 20:54:49,447][__main__][INFO] - Step 9001/15000, Training loss: 0.0011,  Regularization loss: 0.0011, Validation loss: 0.0007
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 20:55:22,634][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:55:22,634][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 20:55:22,634][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:55:22,634][__main__][INFO] - optimal_n_clusters            : 8
[2026-02-26 20:56:15,335][__main__][INFO] - Step 9001/15000, Training loss: 0.0011,  Regularization loss: 0.0011, Validation loss: 0.0016
Output shape from conv layers 2048
[2026-02-26 20:56:51,264][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 20:56:51,264][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 20:56:51,264][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 20:56:51,264][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 20:56:51,264][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 20:56:51,264][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 20:56:51,264][__main__][INFO] - shape_val_classifier_gener    : 0.9444444444444444
[2026-02-26 20:56:51,265][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 20:56:51,265][__main__][INFO] - optimal_n_clusters            : 21
[2026-02-26 21:14:25,951][__main__][INFO] - Step 10001/15000, Training loss: 0.0008,  Regularization loss: 0.0008, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 21:15:02,965][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 21:15:02,966][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 21:15:02,966][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 21:15:02,966][__main__][INFO] - optimal_n_clusters            : 7
[2026-02-26 21:16:32,117][__main__][INFO] - Step 10001/15000, Training loss: 0.0008,  Regularization loss: 0.0008, Validation loss: 0.0006
/home/ghb24/miniconda3/envs/pre/lib/python3.10/site-packages/sklearn/svm/_base.py:1249: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
Output shape from conv layers 2048
[2026-02-26 21:17:20,411][__main__][INFO] - --- Classifier Metrics Metrics ---
[2026-02-26 21:17:20,411][__main__][INFO] - wall_hue_train_classifier_gener: 1.0
[2026-02-26 21:17:20,411][__main__][INFO] - wall_hue_val_classifier_gener : 1.0
[2026-02-26 21:17:20,411][__main__][INFO] - object_hue_train_classifier_gener: 1.0
[2026-02-26 21:17:20,412][__main__][INFO] - object_hue_val_classifier_gener: 1.0
[2026-02-26 21:17:20,412][__main__][INFO] - shape_train_classifier_gener  : 1.0
[2026-02-26 21:17:20,412][__main__][INFO] - shape_val_classifier_gener    : 1.0
[2026-02-26 21:17:20,412][__main__][INFO] - --- Clustering Metrics Metrics ---
[2026-02-26 21:17:20,412][__main__][INFO] - optimal_n_clusters            : 9
