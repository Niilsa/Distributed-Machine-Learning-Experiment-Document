import math
import torch
from torch.utils.data import Dataset, Sampler
import random
import torch.distributed as dist


class RandomSampler(Sampler):
    def __init__(self, dataset:Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas    # number of clients (processes)
        self.num_samples = len(dataset)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed                    # set seed to be the rank of the client, to avoid generating the same indice lists.
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas) 

    def __iter__(self):
        random.seed(self.seed + self.epoch)

        indices = list(range(self.num_samples))

        if self.shuffle:
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    

class BalancedSampler(Sampler):
    def __init__(self, dataset:Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas  # number of clients (processes)
        self.rank = rank  # the rank of this replica
        self.shuffle = shuffle
        self.seed = seed + self.rank  # set seed to be different for each client
        self.epoch = 0
        # calculate number of samples to draw
        self.num_samples = int(math.ceil(len(self.dataset) / float(self.num_replicas)))

    def __iter__(self):
        # set the random seed for shuffling
        random.seed(self.seed + self.epoch)
        # get all the indices
        indices = list(range(len(self.dataset)))
        # shuffle indices if required
        if self.shuffle:
            random.shuffle(indices)
        # only select a subset of the indices corresponding to the size of this replica's share
        indices = indices[self.rank:self.num_samples*self.num_replicas:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch