import numpy as np
import random
import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset


class InfiniteSampler(Sampler):
    """
    Sampler that generates infinite random indices with replacement
    """
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.dataset_size = len(dataset)
    
    def __iter__(self):
        while True:
            if self.shuffle:
                yield from torch.randperm(self.dataset_size).tolist()
            else:
                yield from range(self.dataset_size)
    
    def __len__(self):
        return float('inf')  


class ContextualConcatenator(Dataset):
    def __init__(self, 
                datasets,
                context_for_datasets=None):
        self.datasets = datasets
        self.dataset_indices_intervals = []
        cur_len = 0
        for dataset in datasets:
            self.dataset_indices_intervals.append((cur_len,cur_len+len(dataset)))
            cur_len += len(dataset)

        self.context_for_datasets = context_for_datasets if context_for_datasets is not None else torch.eye(len(datasets)).float()
    def __len__(self):
        return sum([len(d) for d in self.datasets])
    
    # def __getitem__(self, idx):
    #     ds_idx = 0
    #     for i, interval in enumerate(self.dataset_indices_intervals):
    #         if idx >= interval[0] and idx < interval[1]:
    #             ds_idx = i
    #             break
    #     ds_item = self.datasets[ds_idx][idx - self.dataset_indices_intervals[ds_idx][0]]
    #     obs = ds_item[0]
    #     targets = ds_item[1]
    #     if len(ds_item) == 3:
    #         obs  = torch.cat([obs,ds_item[2]],dim=1)
    #     if len(ds_item)>3:
    #         return *ds_item, self.context_for_datasets[ds_idx]
    #     return obs,targets, self.context_for_datasets[ds_idx]
    
    def __getitem__(self, idx):
        ds_idx = 0
        for i, interval in enumerate(self.dataset_indices_intervals):
            if idx >= interval[0] and idx < interval[1]:
                ds_idx = i
                break
        ds_item = self.datasets[ds_idx][idx - self.dataset_indices_intervals[ds_idx][0]]
        obs = ds_item[0]
        targets = ds_item[1]
        # if len(ds_item)>3:
        return obs,targets, self.context_for_datasets[ds_idx]

def closest_divisible_num_batches(len_concat_ds, batch_size, n_datasets):
    target_batches = len_concat_ds // batch_size

    if target_batches % n_datasets == 0:
        return target_batches
    else:
        lower = target_batches - (target_batches % n_datasets)
        upper = lower + n_datasets
        if abs(target_batches - lower) <= abs(target_batches - upper):
            return lower
        else:
            return upper


class ConcatExclusiveSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, ds_weights=None,shuffle=True):
        self.intervals = []
        cur_len = 0
        for dataset in concat_dataset.datasets:
            self.intervals.append((cur_len,cur_len+len(dataset)))
            cur_len += len(dataset)
            
        for ds_idx, (start, end) in enumerate(self.intervals):
            length = end - start
            if length % batch_size != 0:
                raise ValueError(
                    f"Dataset {ds_idx} length {length} is not a multiple of batch_size {batch_size}."
                )
                
        self.batch_size = batch_size
        self.num_batches = closest_divisible_num_batches(len(concat_dataset), batch_size, len(concat_dataset.datasets))
        if ds_weights is None:
            self.ds_weights = [1/len(concat_dataset.datasets) for _ in range(len(concat_dataset.datasets))]
        else:
            self.ds_weights = ds_weights
        
        self.shuffle = shuffle
        
        
    def __iter__(self):
        assert self.num_batches % len(self.intervals) == 0, f"Number of batches {self.num_batches},must be divisible by number of datasets {len(self.intervals)}"
        num_ds = len(self.intervals)
        batches_per_ds = self.num_batches // num_ds

        all_intervals = []
        for i in range(num_ds):
            all_intervals += [i] * batches_per_ds
        if self.shuffle:
            random.shuffle(all_intervals)

        if not self.shuffle:
            per_ds_batches = {}
            for ds_idx, (start, end) in enumerate(self.intervals):
                length = end - start
                total_needed = batches_per_ds * self.batch_size

                if total_needed > length:
                    raise ValueError(
                        f"Requested {batches_per_ds} batches of size {self.batch_size} "
                        f"from dataset {ds_idx} (length {length}) — not enough samples."
                    )

                base = torch.arange(start, end, dtype=torch.long)
                seq = base[:total_needed]
                per_ds_batches[ds_idx] = seq.view(batches_per_ds, self.batch_size)

            ds_batch_ptr = {i: 0 for i in range(num_ds)}

        for i in range(self.num_batches):
            ds_idx = all_intervals[i]
            start, end = self.intervals[ds_idx]

            if self.shuffle:
                batch = torch.randint(start, end, (self.batch_size,))
            else:
                j = ds_batch_ptr[ds_idx]
                batch = per_ds_batches[ds_idx][j]
                ds_batch_ptr[ds_idx] += 1

            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class InfiniteConcatExclusiveSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, ds_weights=None):
        super().__init__(concat_dataset)

        self.intervals = concat_dataset.dataset_indices_intervals
        self.num_datasets = len(self.intervals)
        self.batch_size = batch_size

        if ds_weights is None:
            self.ds_weights = [1 / self.num_datasets] * self.num_datasets
        else:
            if len(ds_weights) != self.num_datasets:
                raise ValueError(
                    f"Length of ds_weights ({len(ds_weights)}) must equal the "
                    f"number of datasets ({self.num_datasets})."
                )
            self.ds_weights = ds_weights

        self.dataset_indices = list(range(self.num_datasets))

    def __iter__(self):
        """
        Yields batches of indices indefinitely in a `while True` loop.
        """
        while True:
            chosen_dataset_idx = random.choices(
                population=self.dataset_indices,
                weights=self.ds_weights,
                k=1
            )[0]
            chosen_interval = self.intervals[chosen_dataset_idx]
            batch = torch.randint(
                low=chosen_interval[0],
                high=chosen_interval[1],
                size=(self.batch_size,),
                dtype=torch.int64 
            )

            yield batch.tolist()

    def __len__(self):
        raise TypeError("ConcatExclusiveSampler is an infinite sampler and has no length.")
