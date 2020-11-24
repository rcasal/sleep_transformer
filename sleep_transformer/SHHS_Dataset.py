######################################################################
# Dataset class
# -------------
#
# We use torch.utils.data.Dataset to represent the dataset. We override the methods __len__ (len(dataset) return the
# size of the dataset, and __getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
#
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os


class ShhsDataset(Dataset):

    def __init__(self, dbpath, transform=None, loadSaO2=True, loadDB='target2'):
        self.dbpath = dbpath
        self.listdir = os.listdir(self.dbpath)
        self.listdir.sort()
        self.listdir = self.listdir[1:]         # Discard sample 1: sequenceLengths.mat
        self.transform = transform
        self.loadSaO2 = loadSaO2
        self.loadDB = loadDB
        self.lengths = loadmat(os.path.join(self.dbpath, 'sequenceLengths.mat'))
        self.lengths = self.lengths['sequenceLengths'].squeeze()

    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, idx):
        subj = loadmat(os.path.join(self.dbpath, self.listdir[idx]))

        # Input shape: batch size, chanel, Lin
        if self.loadSaO2:
            feats = np.concatenate((subj['HR'].reshape(1, -1), subj['SaO2'].reshape(1, -1)), axis=0)
        else:
            feats = subj['HR'].reshape(1, -1)

        try:
            target = subj[self.loadDB].flatten()
        except:
            print("Insert a valid option for loadDB. Valid options are: target2, target4, target5, targetAASM")

        sample = {'feat': feats, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    # Convert arrays in sample to Tensors

    def __call__(self, sample):
        # Cast in format to LossFunction'
        sample['feat'] = torch.from_numpy(sample['feat'])
        sample['target'] = torch.from_numpy(sample['target'])

        return sample


######################################################################
# Dataloader functions
# --------------------
#
# We create sequence_len(sample) and collate_fn_RC to return the length of each sample and the padded batchs.
#

def sequence_len(sample):
    return len(sample['target'])


def collate_fn_rc(batch):
    """Creates mini-batch tensors from the list of samples

    We should build custom collate_fn rather than using default collate_fn,
    because padding is not supported in default.
    Args:
        batch: list of samples (HR, SaO2, target2).
            - feats: torch tensor of shape (?, 2).
            - target: torch tensor of shape (?).
    Returns:
        feats: torch tensor of shape (batch_size, padded_length).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    batch.sort(key=sequence_len, reverse=True)

    # get the length of each sentence
    lengths = [len(sample['target']) for sample in batch]

    # create an empty matrix with padding tokens
    longest_sent = max(lengths)
    batch_size = len(batch)
    num_feats, _ = batch[0]['feat'].size()
    padded_feats = np.zeros((batch_size, num_feats, longest_sent))-1
    padded_target = np.zeros((batch_size, longest_sent))-1

    # copy over the actual sequences
    for i in range(batch_size):
        feats = batch[i]['feat']
        padded_feats[i, :, 0:lengths[i]] = feats[:lengths[i], :]
        target = batch[i]['target']
        padded_target[i, 0:lengths[i]] = target[:lengths[i]]
        lengths[i] = int(lengths[i])                     # Me quedo con la long del target

    padded_target = padded_target[:, ::30]             # Me quedo con 1 de cada 30

    # ToTensor!
    padded_feats = torch.from_numpy(padded_feats)
    padded_target = torch.from_numpy(padded_target)

    return {'feat': padded_feats, 'target': padded_target, 'lengths': lengths}


# class LoadMiniBatch:
#
#     def __init__(self, sample, batch_size, drop_last=True):
#         self.sample = sample
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#     def get_minibatch(self, idx):
#         idx_i = idx * self.batch_size
#
#         if idx < self.len_mb:
#             minibatch = {'feat': self.sample['feat'][idx_i:idx_i + self.batch_size],
#                          'target': self.sample['target'][idx_i:idx_i + self.batch_size]}  # No hace falta poner el ,:,:]
#
#         if idx == self.len_mb-1 and self.drop_last is False:
#             minibatch = {'feat': self.sample['feat'][idx_i:],
#                          'target': self.sample['target'][idx_i:]}
#
#         return minibatch
#
#     def __len__(self):
#         self.len_mb = len(self.sample['target'])/self.batch_size
#
#         if self.drop_last:
#             self.len_mb = int(np.floor(self.len_mb))
#         else:
#             self.len_mb = int(np.ceil(self.len_mb))
#
#         return self.len_mb
