import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data import Dataset
from itertools import product
import random
import math 


class TransitionDSprites(Dataset):
    """
    Sequence dataset built from dSprites for transition prediction.

    Factors:
      shape       : {1=heart, 2=ellipsis, 3=square} (3)
      scale       : uniform [0.5,1.0] (6)
      orientation : uniform [0,2*pi] (40)
      posX        : uniform [0,1] (32)
      posY        : uniform [0,1] (32)
    """
    files = {"train": "../data/raw/dsprites/dsprite_train.npz"}
    urls  = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    n_factors = 5
    factors = ('shape','scale','orientation','posX','posY')
    factor_sizes = np.array([3,6,40,32,32])
    categorical = np.array([1,0,0,0,0])
    img_size = (1,64,64)
    unique_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                       0.12903226, 0.16129032, 0.19354839,
                                       0.22580645, 0.25806452, 0.29032258,
                                       0.32258065, 0.35483871, 0.38709677,
                                       0.41935484, 0.4516129, 0.48387097,
                                       0.51612903, 0.5483871, 0.58064516,
                                       0.61290323, 0.64516129, 0.67741935,
                                       0.70967742, 0.74193548, 0.77419355,
                                       0.80645161, 0.83870968, 0.87096774,
                                       0.90322581, 0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                    0.12903226, 0.16129032, 0.19354839,
                                    0.22580645, 0.25806452, 0.29032258,
                                    0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097,
                                    0.51612903, 0.5483871, 0.58064516,
                                    0.61290323, 0.64516129, 0.67741935,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774,
                                    0.90322581, 0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463,
                                           0.48332195, 0.64442926, 0.80553658,
                                           0.96664389, 1.12775121, 1.28885852,
                                           1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242,
                                           2.41660973, 2.57771705, 2.73882436,
                                           2.89993168, 3.061039, 3.22214631,
                                           3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902,
                                           4.34989752, 4.51100484, 4.67211215,
                                           4.83321947, 4.99432678, 5.1554341,
                                           5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799,
                                           6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  }

    def __init__(self,
                 imgs,
                 factor_values,
                 factor_classes,
                 num_seq,
                 seq_len,
                 next_shape_offset=0,
                 next_scale_offset=0,
                 next_orientation_offset=0,
                 next_posX_offset=0,
                 next_posY_offset=0,
                 one_shape=False,
                 one_scale=False,
                 one_orientation=False,
                 one_posX=False,
                 one_posY=False,
                 target_transform=None,
                 random_data=True,
                 task_abstract=False,
                 task_abstract_latent="shape",
                 task_abstract_train_set=True,
                 subsample_task_abstract=1,
                 subsample_shape=1,
                 subsample_scale=1,
                 subsample_orientation=1,
                 subsample_posX=1,
                 subsample_posY=1):
        self.imgs = imgs
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.target_transform = target_transform

        self.offsets = {
            'shape': next_shape_offset,
            'scale': next_scale_offset,
            'orientation': next_orientation_offset,
            'posX': next_posX_offset,
            'posY': next_posY_offset
        }
        self.context = torch.tensor([off for off in self.offsets.values()]).float()
        self.ones = {
            'shape': one_shape,
            'scale': one_scale,
            'orientation': one_orientation,
            'posX': one_posX,
            'posY': one_posY
        }
        self.subsample_fractions = {
            'shape': subsample_shape,
            'scale': subsample_scale,
            'orientation': subsample_orientation,
            'posX': subsample_posX,
            'posY': subsample_posY
        }

        # Determine possible values per factor
        self.possible_values = {}
        for f, vals in self.unique_values.items():
            # 1. Get fraction (default 1.0)
            fraction = self.subsample_fractions.get(f, 1.0)
            # Ensure valid range
            if not (0 < fraction <= 1.0):
                raise ValueError(f"Subsample fraction for {f} must be in (0, 1]. Got {fraction}")

            step = max(1, int(1 / fraction))
            
            # 3. Slice values
            current_vals = vals[::step]
            
            # 4. Apply One-Value Constraint (Standard)
            if self.ones[f]:
                mid = len(current_vals) // 2
                self.possible_values[f] = np.array([current_vals[mid]])
            else:
                self.possible_values[f] = current_vals
        for f in self.factors:
            assert self.offsets[f] == 0 or not self.ones[f], \
                f"Offset for {f} must be zero if only one value is allowed."

        # Transforms
        image_transforms = [trans.ToTensor(), trans.ConvertImageDtype(torch.float32)]
        self.transform = trans.Compose(image_transforms)

        # Build latent episodes
        if random_data:
            self.episodes_indices = [self.create_episode() for _ in range(num_seq)]
        else:
            self.episodes_indices = self.create_all_abstract_episodes(task_abstract_train_set,test_latent=task_abstract_latent,sampling_factor=subsample_task_abstract)
            self.num_seq = len(self.episodes_indices)

        # Convert indices to actual latent values
        # self.episodes_values = []
        # for ep in self.episodes_indices:
        #     seq = []
        #     for step in ep:
        #         seq.append({f: self.possible_values[f][idx] for f, idx in step.items()})
        #     self.episodes_values.append(seq)
        local_to_global = {}
        for f in self.factors:
            local_vals = self.possible_values[f]
            global_vals = self.unique_values[f]
            # Create a mapping array where map[local_idx] = global_idx
            mapping = []
            for val in local_vals:
                # Find the index of this value in the full unique list
                # (using isclose for safe float comparison, though equality usually works here)
                g_idx = np.where(np.isclose(global_vals, val))[0][0]
                mapping.append(g_idx)
            local_to_global[f] = np.array(mapping)

        # 2. Pre-calculate Strides for converting (shape, scale, ...) -> Flat Image Index
        strides = np.cumprod([1] + list(self.factor_sizes[::-1]))[::-1][1:]

        # 3. Convert all episodes to Global Indices and Flat Image Indices
        self.z_indices = []       # Will hold the latents as integers
        self.flat_img_indices = [] # Will hold the index to slice self.imgs
        
        for ep in self.episodes_indices:
            seq_z = []
            seq_img_idx = []
            
            for step in ep:
                # 'step' contains local indices relative to possible_values
                current_z = []
                current_flat_idx = 0
                
                for k, f in enumerate(self.factors):
                    local_idx = step[f]
                    # Map to global index
                    global_idx = local_to_global[f][local_idx]
                    
                    current_z.append(global_idx)
                    current_flat_idx += global_idx * strides[k]
                
                seq_z.append(current_z)
                seq_img_idx.append(int(current_flat_idx))
                
            self.z_indices.append(seq_z)
            self.flat_img_indices.append(seq_img_idx)

        # Convert to Tensors for fast access in getitem
        self.z_indices = torch.tensor(self.z_indices, dtype=torch.long)
        self.flat_img_indices = torch.tensor(self.flat_img_indices, dtype=torch.long)
    
    def get_output_dims(self):
        return self.factor_sizes.tolist()


    def create_all_abstract_episodes(self, is_train=True,test_latent="shape",sampling_factor=0.1):
        # Split each factor's index range
        index_ranges = {}
        for f, vals in self.possible_values.items():
            n = len(vals)
            if n == 1:
                idxs = [0]
            elif f == test_latent:
                idxs = list(range(n))
            else:
                half = n // 2
                idxs = list(range(half)) if is_train else list(range(half, n))
            index_ranges[f] = idxs

        all_eps = []
        factors = list(self.possible_values.keys())
        all_combis = list(product(*[index_ranges[f] for f in factors]))
        if is_train:
            all_combis = random.sample(all_combis, int(len(all_combis) * sampling_factor))
        for combo in all_combis:
            step = {factors[i]: combo[i] for i in range(len(factors))}
            all_eps.append([step])
        return all_eps

    def create_episode(self):
        # Random start
        start = {f: random.randint(0, len(self.possible_values[f]) - 1)
                 for f in self.factors}
        episode = []
        for t in range(self.seq_len):
            step = {}
            for f in self.factors:
                off = self.offsets[f]
                L = len(self.possible_values[f])
                step[f] = (start[f] + off * t) % L
            episode.append(step)
        return episode

    def latent_indices_to_img_idx(self, latent_vals):
        # Compute strides and index
        strides = np.cumprod([1] + list(self.factor_sizes[::-1]))[::-1][1:]
        idx = 0
        for i, f in enumerate(self.factors):
            val = latent_vals[f]
            pos = self.unique_values[f].tolist().index(val)
            idx += pos * strides[i]
        return int(idx)

    def __getitem__(self, i):
        img_idxs = self.flat_img_indices[i] 
        
        # 2. Get the latents (already global indices, pre-computed)
        latents = self.z_indices[i] 

        # 3. Load and transform images
        # We iterate because transform usually expects single images (C, H, W) 
        # unless you have a batch-ready transform.
        imgs = torch.stack([self.transform(self.imgs[idx]) for idx in img_idxs])
        
        return imgs, latents
        # seq_vals = self.episodes_values[i]
        # imgs = []
        # latents = []
        # for step in seq_vals:
        #     img_idx = self.latent_indices_to_img_idx(step)
        #     imgs.append(self.transform(self.imgs[img_idx]))
        #     latents.append([step[f] for f in self.factors])
        # imgs = torch.stack(imgs)
        # imgs = torch.stack([self.transform(self.imgs[self.latent_indices_to_img_idx(step)]) for step in seq_vals])
        # latents = [[step[f] for f in self.factors] for step in seq_vals]
        
        # latents = torch.tensor(latents, dtype=torch.float32)
        # print(latents)
        # return imgs, latents,self.context

    def __len__(self):
        return self.num_seq


def make_DSprites_collate_fn(device='cpu'):
    def collate_fn(batch):
        imgs, latents,contexts = zip(*batch)
        imgs = torch.stack(imgs, dim=0).transpose(0,1).to(device)
        latents = torch.stack(latents, dim=0).transpose(0,1).to(device)
        contexts = torch.stack(contexts, dim=0).to(device)
        return imgs, latents, contexts
    return collate_fn


def load_raw(path, factor_filter=None):
    data = np.load(path, allow_pickle=True)
    imgs = data['imgs'] * 255
    latents = data['latents_values'][:, 1:]
    classes = data['latents_classes'][:, 1:]
    if factor_filter is not None:
        idx = factor_filter(latents, classes)
        imgs = imgs[idx]
        latents = latents[idx]
        classes = classes[idx]
        if len(imgs) == 0:
            raise ValueError("Filter removed all data")
    return imgs, latents, classes


def make_dataset(num_seq,
                 config,
                 imgs=None,
                 factor_values=None,
                 factor_classes=None,
                 task_abstract=False,
                 task_abstract_latent="shape",
                 task_abstract_train_set=True):
    if imgs is None or factor_values is None or factor_classes is None:
        imgs, factor_values, factor_classes = load_raw(config.dataset.path)
    
    return TransitionDSprites(
        imgs, factor_values, factor_classes,
        num_seq=num_seq,
        seq_len=config.dataset.seq_len,
        next_shape_offset=0,
        next_scale_offset=0,
        next_orientation_offset=0,
        next_posX_offset=0,
        next_posY_offset=0,
        one_shape=config.dataset.one_shape,
        one_scale=config.dataset.one_scale,
        one_orientation=config.dataset.one_orientation,
        one_posX=config.dataset.one_posX,
        one_posY=config.dataset.one_posY,
        random_data=not task_abstract,
        task_abstract=task_abstract,
        task_abstract_latent=task_abstract_latent,
        task_abstract_train_set=task_abstract_train_set,
        subsample_orientation=config.dataset.subsample_orientation,
        subsample_posX=config.dataset.subsample_posX,
        subsample_posY=config.dataset.subsample_posY,
        subsample_scale=config.dataset.subsample_scale,
        subsample_shape=config.dataset.subsample_shape
    )


def make_DSprites_loader(config,
                         contexts,
                         context_vector_size,
                         context_start_idx=0,
                         num_per_context=1000,
                         put_in_dict=False,
                         imgs=None,
                         factor_values=None,
                         factor_classes=None):
    if imgs is None or factor_values is None or factor_classes is None:
        imgs, factor_values, factor_classes = load_raw(config.dataset.path)
    datasets = {} if put_in_dict else []
    context_vectors = []
    
    
    for ctxt_val_idx,ctxt in enumerate(contexts):
        # context_params = {}
        # for ctxt,value in zip(config.dataset.contexts,ctxt_val):
        #     context_params[ctxt] = value
        ctxt_val = list(ctxt.values())
        ds = TransitionDSprites(
            imgs, factor_values, factor_classes,
            num_seq=num_per_context,
            seq_len=config.dataset.seq_len,
            one_shape=config.dataset.one_shape,
            one_scale=config.dataset.one_scale,
            one_orientation=config.dataset.one_orientation,
            one_posX=config.dataset.one_posX,
            one_posY=config.dataset.one_posY,
            subsample_orientation=config.dataset.subsample_orientation,
            subsample_posX=config.dataset.subsample_posX,
            subsample_posY=config.dataset.subsample_posY,
            subsample_scale=config.dataset.subsample_scale,
            subsample_shape=config.dataset.subsample_shape,
            **ctxt
        )

        # if config.one_hot_context:
        #     context_vector = torch.zeros((context_vector_size))
        #     context_vector[ctxt_val_idx+context_start_idx] = 1
        # else:
        context_vector = torch.tensor(ctxt_val, dtype=torch.float32)
        if put_in_dict:
            context_name = "".join([f"{config.dataset.contexts[i]}_{ctxt_val[i]}" for i in range(len(ctxt_val))])
            concat_ds = (ds,context_vector.unsqueeze(0))
            datasets[context_name] = concat_ds
        else:
            datasets.append(ds)
            context_vectors.append(context_vector)

    if put_in_dict:
        return datasets, None
    return datasets, torch.stack(context_vectors)
