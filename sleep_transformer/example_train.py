#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""

import os

import torch
import torch.optim as optim
import torchvision.transforms as t
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from sleep_transformer.Model import OwnTransformerModel, TransformerModel

from sleep_transformer.SHHS_Dataset import ShhsDataset, collate_fn_rc, ToTensor
from sleep_transformer.Train_Model import train_model


# GPU parameters
# cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.device('cuda:0')
torch.cuda.set_device(0)
print('Exp is running in {} No.{}'.format(
    torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

# Exp parameters
interruption = True

Exp = '01_x'
loadSaO2 = True
resnet = False

# path
path_save = 'models/'
model_name = 'model_ft_exp' + Exp + '.pth'

# Network parameters: d_in is input dimension, H is hidden dimension, d_out is output dimension.
d_in, batch_size, d_out, d_model, dropout, nhead, encoder_layers, dim_feedforward, gamma = 2, 16, 2, 512, 0.8, 8, 2, 512, 5

# Database
datasets = {x: ShhsDataset(dbpath=os.path.abspath(os.path.join('../../db/All_shhsCell_N/patient/', x)),
                           transform=t.Compose([ToTensor()]), loadSaO2=loadSaO2)
            for x in ['train', 'val']}


params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True,
          'drop_last': True, 'collate_fn': collate_fn_rc}

dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val']}

max_len = (max(max(datasets['train'].lengths), max(datasets['val'].lengths))+1000)//30
# Load model and set criterion
# model_ft = OwnTransformerModel(d_in=d_in, d_out=d_out, batch_size=batch_size, d_model=512, nhead=8,
#                                num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
#                                dropout=0.2).to(device=cuda)
model_ft = TransformerModel(d_in=d_in, d_out=d_out, batch_size=batch_size, d_model=d_model, nhead=nhead,
                            num_encoder_layers=encoder_layers, dim_feedforward=dim_feedforward, dropout=0.15,
                            max_len=max_len).to(device=cuda)

for p in model_ft.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

learning_rate = 1e-4
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.001)

# criterion = torch.nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

# Prints
print('Parameters:')
print('\td_in: {} \n\tbatch_size: {} \n\td_out: {} \n\td_model: {} \n\tdropout: {} \n\tnhead: {} \n\tencoder_layers: {} '
      '\n\tdim_feedforward: {} \n\tgamma: {} '.format(d_in, batch_size, d_out, d_model, dropout, nhead, encoder_layers, dim_feedforward, gamma))
print('\n')

######################################################################
# Train and evaluate
# ------------------
#
model_ft = train_model(model=model_ft, optimizer=optimizer_ft, dataloaders=dataloaders, num_epochs=1000, cuda=cuda,
                       path_bkp=path_save, checkpoint=interruption, gamma=gamma)

torch.save(model_ft.state_dict(), os.path.join(path_save, model_name))
