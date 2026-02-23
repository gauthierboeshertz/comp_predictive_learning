# --- inside TransitionShapes3D class ---

from itertools import product
import math
import numpy as np
import h5py
import torch
import torchvision.transforms as trans
from skimage.color import rgb2hsv
from torch.utils.data import Dataset
import random


class TransitionShapes3D(Dataset):
    # 1) Fix factor_sizes (must match `factors` order!)
    """
    Disentangled dataset used in Kim and Mnih, (2019)

    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================
    # floor hue:           uniform in range [0.0, 1.0)                      10
    # wall hue:            uniform in range [0.0, 1.0)                      10
    # object hue:          uniform in range [0.0, 1.0)                      10
    # scale:               uniform in range [0.75, 1.25]                     8
    # shape:               0=square, 1=cylinder, 2=sphere, 3=pill            4
    # orientation          uniform in range [-30, 30]                       15
    """
    files = {"train": "../data/raw/shapes3d/3dshapes.h5"}
    n_factors = 6
    factors = ('shape','floor_hue', 'wall_hue', 'object_hue',
               'scale', 'orientation')
    categorical = np.array([0, 0, 0, 0, 1, 0])
    img_size = (3, 64, 64)

    unique_values = {'shape': np.array([0, 1, 2, 3]),
                    'floor_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                            0.5, 0.6, 0.7, 0.8, 0.9]),
                     'wall_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                           0.5, 0.6, 0.7, 0.8, 0.9]),
                     'object_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                             0.5, 0.6, 0.7, 0.8, 0.9]),
                     'scale': np.array([0.75, 0.82142857, 0.89285714, 0.96428571,
                               1.03571429, 1.10714286, 1.17857143, 1.25]),
                     'orientation': np.array([-30., -25.71428571, -21.42857143,
                                     -17.14285714, -12.85714286, -8.57142857,
                                     -4.28571429, 0., 4.28571429, 8.57142857,
                                     12.85714286, 17.14285714, 21.42857143,
                                     25.71428571,  30.])}
    factor_sizes = np.array([4, 10, 10, 10, 8, 15])
    
    
    def __init__(self,
                imgs,
                num_seq,
                seq_len,
                next_floor_offset,
                next_wall_offset,
                next_object_offset,
                next_scale_offset,
                next_shape_offset,
                next_orientation_offset,
                factor_values,
                factor_classes,
                color_mode='rgb',
                one_floor=False,
                one_wall=False,
                one_object=False,
                one_scale=False,
                one_shape=False,
                one_orientation=False,
                target_transform=None,
                random_data=True,
                task_abstract=False,
                task_abstract_latent="shape",
                subsample_task_abstract=1.0,
                task_abstract_train_set=True,
                # OPTIONAL: add subsampling knobs like dsprites (can default to 1)
                subsample_floor=1.0,
                subsample_wall=1.0,
                subsample_object=1.0,
                subsample_scale=1.0,
                subsample_shape=1.0,
                subsample_orientation=1.0):

        self.imgs = imgs
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.target_transform = target_transform

        self.offsets = {
            'shape': next_shape_offset,
            'floor_hue': next_floor_offset,
            'wall_hue': next_wall_offset,
            'object_hue': next_object_offset,
            'scale': next_scale_offset,
            'orientation': next_orientation_offset
        }
        self.ones = {
            'shape': one_shape,
            'floor_hue': one_floor,
            'wall_hue': one_wall,
            'object_hue': one_object,
            'scale': one_scale,
            'orientation': one_orientation
        }
        self.subsample_fractions = {
            'shape': subsample_shape,
            'floor_hue': subsample_floor,
            'wall_hue': subsample_wall,
            'object_hue': subsample_object,
            'scale': subsample_scale,
            'orientation': subsample_orientation
        }

        # transforms (keep your hsv option)
        image_transforms = [trans.ToTensor(), trans.ConvertImageDtype(torch.float32)]
        if color_mode == 'hsv':
            image_transforms.insert(0, trans.Lambda(rgb2hsv))
        self.transform = trans.Compose(image_transforms)

        # 2) Build possible_values (subsample + one-value constraint), like dsprites
        self.possible_values = {}
        for f in self.factors:
            vals = self.unique_values[f]
            fraction = self.subsample_fractions.get(f, 1.0)
            if not (0 < fraction <= 1.0):
                raise ValueError(f"Subsample fraction for {f} must be in (0,1]. Got {fraction}")
            step = max(1, int(1 / fraction))
            current_vals = vals[::step]

            if self.ones[f]:
                mid = len(current_vals) // 2
                self.possible_values[f] = np.array([current_vals[mid]])
            else:
                self.possible_values[f] = current_vals

        for f in self.factors:
            assert self.offsets[f] == 0 or not self.ones[f], \
                f"Offset for {f} must be zero if only one value is allowed."

        # 3) Episodes (same logic as yours; keep abstract option)
        if random_data:
            self.episodes_indices = [self.create_episode() for _ in range(num_seq)]
        else:
            self.episodes_indices = self.create_all_abstract_episodes(
                is_train=task_abstract_train_set,
                test_latent=task_abstract_latent,
                sampling_factor=subsample_task_abstract)
            self.num_seq = len(self.episodes_indices)

        # 4) Local -> Global mapping (like dsprites)
        local_to_global = {}
        for f in self.factors:
            local_vals = self.possible_values[f]
            global_vals = self.unique_values[f]
            mapping = []
            for val in local_vals:
                g_idx = np.where(np.isclose(global_vals, val))[0][0]
                mapping.append(g_idx)
            local_to_global[f] = np.array(mapping)

        # 5) Strides using *corrected* factor_sizes
        strides = np.cumprod([1] + list(self.factor_sizes[::-1]))[::-1][1:]

        # 6) Precompute z_indices and flat_img_indices
        self.z_indices = []
        self.flat_img_indices = []

        for ep in self.episodes_indices:
            seq_z = []
            seq_img = []
            for step in ep:
                z = []
                flat = 0
                for k, f in enumerate(self.factors):
                    local_idx = step[f]
                    global_idx = int(local_to_global[f][local_idx])
                    z.append(global_idx)
                    flat += global_idx * strides[k]
                seq_z.append(z)
                seq_img.append(int(flat))
            self.z_indices.append(seq_z)
            self.flat_img_indices.append(seq_img)

        self.z_indices = torch.tensor(self.z_indices, dtype=torch.long)
        self.flat_img_indices = torch.tensor(self.flat_img_indices, dtype=torch.long)
        self.context = torch.tensor([off for off in self.offsets.values()]).float()

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
        img_idxs = self.flat_img_indices[i]      # [seq_len]
        latents  = self.z_indices[i]             # [seq_len, n_factors] (global indices)

        imgs = torch.stack([self.transform(self.imgs[idx]) for idx in img_idxs])
        return imgs, latents, self.context
    
    def __len__(self):
        return self.num_seq



def load_raw(path, factor_filter=None):
    data_zip = h5py.File(path, 'r')

    imgs = data_zip['images'][()]
    factor_values = data_zip['labels'][()]
    factor_classes = np.asarray(list(product(
        *[range(i) for i in TransitionShapes3D.factor_sizes])))

    if factor_filter is not None:
        idx = factor_filter(factor_values, factor_classes)

        imgs = imgs[idx]
        factor_values = factor_values[idx]
        factor_classes = factor_classes[idx]

        if len(imgs) == 0:
            raise ValueError('Incorrect masking removed all data')

    return imgs, factor_values, factor_classes

def make_DDDShapes_collate_fn(device='cpu'):
    def collate_fn(batch):
        imgs, latents, contexts = zip(*batch)
        imgs = torch.stack(imgs, dim=0).transpose(0,1).to(device)
        latents = torch.stack(latents, dim=0).transpose(0,1).to(device)
        contexts = torch.stack(contexts, dim=0).to(device)
        return imgs, latents, contexts
    return collate_fn


def load(data_filters=(None, None), train=True, color_mode='rgb', path=None):
    train_filter, test_filter = data_filters
    if path is None:
        path = TransitionShapes3D.files['train']
    if train:
        data = TransitionShapes3D(*load_raw(path, train_filter), color_mode=color_mode)
    else:
        data = TransitionShapes3D(*load_raw(path, test_filter), color_mode=color_mode)
    return data


def make_dataset(num_samples,
                 config,
                 imgs=None,
                 factor_values=None,
                 factor_classes=None,
                 task_abstract=False,
                 task_abstract_latent="shape",
                 task_abstract_train_set=True):
    
    ds = TransitionShapes3D(imgs=imgs,
                            factor_classes=factor_classes,
                            factor_values=factor_values,
                            num_seq=num_samples,
                            seq_len=config.dataset.seq_len,
                            next_floor_offset=0,
                            next_wall_offset=0,
                            next_object_offset=0,
                            next_scale_offset=0,
                            next_shape_offset=0,
                            next_orientation_offset=0,
                            one_floor=config.dataset.one_floor,
                            one_wall=config.dataset.one_wall,
                            one_object=config.dataset.one_object,
                            one_scale=config.dataset.one_scale,
                            one_shape=config.dataset.one_shape,
                            one_orientation=config.dataset.one_orientation,
                            color_mode=config.dataset.color_mode,
                            random_data= not task_abstract,
                            task_abstract=task_abstract,
                            task_abstract_latent=task_abstract_latent,
                            subsample_task_abstract= config.dataset.subsample_task_abstract,
                            task_abstract_train_set=task_abstract_train_set,
                            subsample_floor=config.dataset.subsample_floor,
                            subsample_wall=config.dataset.subsample_wall,
                            subsample_object=config.dataset.subsample_object,
                            subsample_scale=config.dataset.subsample_scale,
                            subsample_shape=config.dataset.subsample_shape,
                            subsample_orientation=config.dataset.subsample_orientation)

    return ds


def make_dddshapes_loader(config,
                    context_vals,
                    context_vector_size,
                    context_start_idx=0,
                    num_drawing_per_context=1000,
                    put_in_dict=False,
                    imgs=None,
                    factor_values=None,
                    factor_classes=None):
    
    if not put_in_dict:
        datasets = []
        context_vectors = []
    else:
        datasets = {}
    
    if imgs is None or factor_values is None or factor_classes is None:
        imgs, factor_values, factor_classes = load_raw(config.dataset.path)
    for ctxt_val_idx,ctxt in enumerate(context_vals):
        ctxt_val = list(ctxt.values())
        context_ds = TransitionShapes3D(imgs,
                                        num_seq=num_drawing_per_context,
                                        seq_len=config.dataset.seq_len,
                                        one_floor=config.dataset.one_floor,
                                        one_wall=config.dataset.one_wall,
                                        one_object=config.dataset.one_object,
                                        one_scale=config.dataset.one_scale,
                                        one_shape=config.dataset.one_shape,
                                        one_orientation=config.dataset.one_orientation,
                                        factor_values=factor_values,
                                        factor_classes=factor_classes,
                                        color_mode=config.dataset.color_mode,
                                        subsample_floor=config.dataset.get("subsample_floor", 1.0),
                                        subsample_wall=config.dataset.get("subsample_wall", 1.0),
                                        subsample_object=config.dataset.get("subsample_object", 1.0),
                                        subsample_scale=config.dataset.get("subsample_scale", 1.0),
                                        subsample_shape=config.dataset.get("subsample_shape", 1.0),
                                        subsample_orientation=config.dataset.get("subsample_orientation", 1.0),
                                        random_data=True,
                                        task_abstract=False,
                                        **ctxt)
        if config.one_hot_context:
            context_vector = torch.zeros((context_vector_size))
            context_vector[ctxt_val_idx+context_start_idx] = 1
        else:
            context_vector = torch.tensor(ctxt_val, dtype=torch.float32)
        if put_in_dict:
            context_name = "".join([f"{config.dataset.contexts[i]}_{ctxt_val[i]}" for i in range(len(ctxt_val))])
            concat_ds = (context_ds,context_vector.unsqueeze(0))
            datasets[context_name] = concat_ds
        else:
            datasets.append(context_ds)
            context_vectors.append(context_vector)

    if put_in_dict:
        return datasets, None # actually loaders, ooops, but I am too lazy to change the name, it is not a big deal, right?, I hope so, I am not going to change it, I am too lazy
    
    context_vectors = torch.stack(context_vectors)
    return datasets,context_vectors

