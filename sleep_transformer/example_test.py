#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
[SimConfig]
Sim_filename = 'Exp_01'
Sim_variables = {'gru': [0, 1], 'h': [64, 128, 256]}
Sim_realizations = {'Exp01': 1}
Sim_name = 'E01'
Sim_path = './'
Sim_hostname = 'nabucodonosor'
Sim_out_filename = 'out'
Sim_eout_filename = 'err'

Slurm_ntasks = 1
Slurm_tasks_per_node = 1
Slurm_cpus_per_task = 4
Slurm_nodes = 1
Slurm_gres = gpu:1
Slurm_email = 'rcasal@gmail.com'
[end]
'''

import os

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader

from sleep_cnn_rnn.Model import ModelGru, ModelLstm

from sleep_cnn_rnn.SHHS_Dataset import ShhsDataset, collate_fn_rc, ToTensor
from sleep_cnn_rnn.Eval_Model import eval_model


# GPU parameters
# cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.device('cuda:0')
torch.cuda.set_device(0)
print('Exp is running in {} No.{}'.format(
    torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

# Exp parameters
interruption = False

Exp = '01_x'
gru = True
h = 128
loadSaO2 = False


# path
path_load = 'models/'
model_name = 'model_ft_pb' + Exp + '.pth'

# Network parameters: d_in is input dimension, H is hidden dimension, d_out is output dimension.
d_in, H, num_layers, batch_size, d_out = 1, h, 2, 1,  2


# Database
datasets = {x: ShhsDataset(dbpath=os.path.abspath(os.path.join('../../db/All_shhsCell_SN/patient/', x)),
                           transform=T.Compose([ToTensor()]), loadSaO2=loadSaO2)
            for x in ['train', 'val', 'test']}


params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True,
          'drop_last': True, 'collate_fn': collate_fn_rc}

dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val', 'test']}


# Load model and set criterion
if gru is True:
    model_ft = ModelGru(d_in, H, num_layers, batch_size, d_out)
else:
    model_ft = ModelLstm(d_in, H, num_layers, batch_size, d_out)

model_ft.load_state_dict(torch.load(os.path.join(path_load, model_name)))
model_ft.to(device=cuda)

# Prints
print('Starting testing:')

print('Parameters:')
print('\tGRU: {}'.format(gru))
print('\td_in: {} \n\tH: {} \n\tnum_layers: {} \n\tbatch_size_ {} \n\td_out: {}'.format(
        d_in, H, num_layers, batch_size, d_out))
print('\n')

######################################################################
# Train and evaluate
# ------------------
#
model_ft = eval_model(model=model_ft, dataloaders=dataloaders, cuda=cuda)

#torch.save(model_ft.state_dict(), os.path.join(path_save, model_name))