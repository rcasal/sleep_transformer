
import os
import torch
import time
from sleep_transformer.Model import ce_loss, ce_loss_multiclass, focal_loss
from sleep_transformer.Model import statistics, statistics_test, statistics_test_multiclass
from scipy.io import savemat
import numpy as np


def eval_model(model, dataloaders, path_save=None, model_name=None, cuda=None, dtype_data=None, dtype_target=None,
               alpha=0.5, gamma=0):

    # init

    if cuda is None:
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype_data is None:
        dtype_data = torch.float32

    if dtype_target is None:
        dtype_target = torch.int64

    if path_save is None:
        save = False
    else:
        save = True
        if model_name is None:
            model_name = "default_name"



    since = time.time()

    print('-' * 111)
    print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format(
        'Phase', 'loss', 'acc', 'se', 'sp', 'pre', 'npv', 'kappa', 'err', 'rel err'))
    print('-' * 111)

    model.eval()  # Set model to evaluate mode

    for phase in ['train', 'val', 'test']:

        running_loss = 0.0
        running_acc = 0.0
        running_se = 0.0
        running_sp = 0.0
        running_pre = 0.0
        running_npv = 0.0
        running_kappa = 0.0
        running_err = 0.0
        running_rel_err = 0.0

        list_specs ={'patient': [], 'acc': [],'se': [], 'sp': [], 'pre': [], 'npv': [], 'kappa': [],
                     'err': [], 'rel_err': [], 'tst': [], 'tst_est': [], 'trt': []}

        # Iterate over data.
        for i_batch, sample in enumerate(dataloaders[phase]):

            # sample to GPU
            sample['feat'] = sample['feat'].type(dtype=dtype_data).to(device=cuda)
            sample['target'] = sample['target'].type(dtype=dtype_target).to(device=cuda)

            outputs = model(src=sample['feat'], has_mask=False)

            # Compute and print loss.
            # loss = ce_loss(outputs, sample['target'])
            loss = focal_loss(outputs, sample['target'], alpha, gamma)
            specs = statistics_test(outputs.cpu(), sample['target'].cpu(), sample['lengths'])

            # del sample
            lb = sample['feat'].size(0)
            del sample, outputs

            #statistics to save
            if save is True:
                list_specs['patient'].append(i_batch)
                list_specs['acc'].append(specs['acc'])
                list_specs['se'].append(specs['se'])
                list_specs['sp'].append(specs['sp'])
                list_specs['pre'].append(specs['pre'])
                list_specs['npv'].append(specs['npv'])
                list_specs['kappa'].append(specs['kappa'])
                list_specs['err'].append(specs['err'])
                list_specs['rel_err'].append(specs['rel_err'])
                list_specs['tst'].append(specs['tst'])
                list_specs['tst_est'].append(specs['tst_est'])
                list_specs['trt'].append(specs['trt'])

            # statistics
            running_loss += loss.item() * lb
            running_acc += specs['acc'] * lb
            running_se += specs['se'] * lb
            running_sp += specs['sp'] * lb
            running_pre += specs['pre'] * lb
            running_npv += specs['npv'] * lb
            running_kappa += specs['kappa'] * lb
            running_err += specs['err'] * lb
            running_rel_err += specs['rel_err'] * lb


        cte_epoch = (i_batch + 1) * lb
        epoch_loss = running_loss / cte_epoch  # Remember drop_last

        epoch_acc = [0.0]
        epoch_se = [0.0]
        epoch_sp = [0.0]
        epoch_pre = [0.0]
        epoch_npv = [0.0]
        epoch_kappa = [0.0]
        epoch_err = [0.0]
        epoch_rel_err = [0.0]

        epoch_acc = running_acc / cte_epoch
        epoch_se = running_se / cte_epoch
        epoch_sp = running_sp / cte_epoch
        epoch_pre = running_pre / cte_epoch
        epoch_npv = running_npv / cte_epoch
        epoch_kappa = running_kappa / cte_epoch
        epoch_err = running_err / cte_epoch
        epoch_rel_err = running_rel_err / cte_epoch

        print('-' * 111)
        print('|{:^10}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|{:^10.4f}|'
              '{:^10.4f}|{:^10.4f}|'.format(phase, epoch_loss, epoch_acc, epoch_se, epoch_sp,
                                            epoch_pre, epoch_npv, epoch_kappa, epoch_err, epoch_rel_err))


        if save is True:
            savemat(os.path.join(path_save, 'stats_' + phase + '_' +  model_name ), list_specs)


    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))




def eval_model_multiclass(model, dataloaders, path_save=None, model_name=None, cuda=None, dtype_data=None,
                dtype_target=None, alpha=None, gamma=0):

    # init

    if cuda is None:
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype_data is None:
        dtype_data = torch.float32

    if dtype_target is None:
        dtype_target = torch.int64

    if path_save is None:
        save = False
    else:
        save = True
        if model_name is None:
            model_name = "default_name"


    since = time.time()

    print('-' * 45)
    print('|{:^10}|{:^10}|{:^10}|{:^10}|'.format(
        'Phase', 'loss', 'acc', 'kappa'))
    print('-' * 45)

    model.eval()  # Set model to evaluate mode

    for phase in ['train', 'val', 'test']:

        running_loss = 0.0
        running_acc = 0.0
        running_kappa = 0.0
        running_confusion_matrix = np.zeros([model.d_out,model.d_out])

        list_specs ={'patient': [], 'acc': [], 'kappa':[], 'confusion_matrix': []} # , 'confusion_matrix_percentage': []}

        # Iterate over data.
        for i_batch, sample in enumerate(dataloaders[phase]):

            # sample to GPU
            sample['feat'] = sample['feat'].type(dtype=dtype_data).to(device=cuda)
            sample['target'] = sample['target'].type(dtype=dtype_target).to(device=cuda)

            outputs = model(src=sample['feat'], has_mask=False)

            # Compute and print loss.
            loss = ce_loss_multiclass(outputs, sample['target'], alpha)

            specs = statistics_test_multiclass(outputs.cpu(), sample['target'].cpu(), sample['lengths'], model.d_out)

            # del sample
            lb = sample['feat'].size(0)
            del sample, outputs

            #statistics to save
            if save is True:
                list_specs['patient'].append(i_batch)
                list_specs['acc'].append(specs['acc'])
                list_specs['kappa'].append(specs['kappa'])
                list_specs['confusion_matrix'].append(specs['confusion_matrix'])
                # list_specs['confusion_matrix_percentage'].append(specs['confusion_matrix_percentage'])

            # statistics
            running_loss += loss.item() * lb

            running_acc += specs['acc'] * lb
            running_kappa += specs['kappa'] * lb
            running_confusion_matrix += specs['confusion_matrix'] * lb

        cte_epoch = (i_batch + 1) * lb
        epoch_loss = running_loss / cte_epoch  # Remember drop_last

        epoch_acc = 0.0
        epoch_kappa = 0.0
        epoch_confusion_matrix  = np.zeros([6,6])

        epoch_acc = running_acc / cte_epoch
        epoch_kappa = running_kappa / cte_epoch
        epoch_confusion_matrix = running_confusion_matrix / cte_epoch

        print('-' * 45)
        print('|{:^10}|{:^10.4f}|{:^10.4f}|{:^10.4f}|'.format(
            phase,  epoch_loss, epoch_acc, epoch_kappa))
        print('Matrix Confusion:')
        print(epoch_confusion_matrix)
        print('-' * 45)

        if save is True:
            savemat(os.path.join(path_save, 'stats_' + phase + '_' +  model_name ), list_specs)


    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))



