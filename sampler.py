from typing import Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


class PackedDistributedBatchSampler(Sampler):
    # experimental and probably a bad idea. Packs sequences of similar length together
    # to increase throughput, but may affect sample efficiency negatively and breaks
    # IID assumptions
    def __init__(
        self,
        batch_max_length: int,
        lengths: np.ndarray,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

    def generate_batches(self):
        sort_indices, sorted_lengths = np.argsort(self.lengths), np.sort(self.lengths)
        bins = [[]]
        # pack some bins
        bins = [[]]
        for index, length in zip(sort_indices, sorted_lengths):
            assert length < self.batch_max_length
            if (
                sum([x[1] for x in bins[-1]]) + length > self.batch_max_length
                and len(bins[-1]) < 64
            ):
                bins.append([(index, length)])
            else:
                bins[-1].append((index, length))
        bins_indexes_only = []
        for bin in bins:
            bins_indexes_only.append([x[0] for x in bin])
        bin_indices = np.random.default_rng(seed=self.seed).permutation(len(bins))
        batches = [bins_indexes_only[i] for i in bin_indices][
            self.rank :: self.num_replicas
        ]
        return batches

    def __iter__(self):
        batches = self.generate_batches()
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def __len__(self):
        return self.num_batches()
