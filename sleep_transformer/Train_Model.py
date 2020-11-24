######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
import torch
import time
import copy
import os
from sleep_transformer.Model import ce_loss, statistics, focal_loss, ce_loss_multiclass, statistics_multiclass, \
    mfe_loss_multiclass


def train_model(model, optimizer, dataloaders, scheduler=None, num_epochs=25, cuda=None, dtype_data=None,
                dtype_target=None, path_bkp=None, checkpoint=None, alpha=0.5, gamma=0):

    # init

    if cuda is None:
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype_data is None:
        dtype_data = torch.float32

    if dtype_target is None:
        dtype_target = torch.int64

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_run = 0

    # Checkpoint
    if checkpoint is True:
        cp = torch.load(os.path.join(path_bkp, 'bkp_model_ft.pth'))
        epoch_run = cp['epoch']
        model.load_state_dict(cp['model_state_dict'])          # Load state of the last epoch
        best_model_wts = cp['best_model_wts']
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        best_acc = cp['best_acc']
        print('Resuming script in epoch {}. Best acc = {:4f}'.format(epoch_run, best_acc))

    print('-' * 131)
    print('|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|'.format(
        'Epoch', 'Phase', 'mini_batch', 'loss', 'acc', 'se', 'sp', 'pre', 'npv', 'kappa'))
    print('-' * 131)

    for epoch in range(num_epochs-epoch_run):

        print('-' * 131)

        # since_epoch = time.time()

        # Each epoch has a training and validation phase: by default all the modules are initialized to train
        # mode(self.training = True). Also be aware that some layers have different behavior during train and evaluation
        # (like BatchNorm, Dropout) so setting it matters.

        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_se = 0.0
            running_sp = 0.0
            running_pre = 0.0
            running_npv = 0.0
            running_kappa = 0.0


            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):
                # since_batch = time.time()

                # sample to GPU
                sample['feat'] = sample['feat'].type(dtype=dtype_data).to(device=cuda)
                sample['target'] = sample['target'].type(dtype=dtype_target).to(device=cuda)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(src=sample['feat'], has_mask=False)

                    # Compute and print loss.
                    #loss = ce_loss(outputs, sample['target'], alpha)
                    loss = focal_loss(outputs, sample['target'], alpha, gamma)
                    specs = statistics(outputs, sample['target'], sample['lengths'])

                    # del sample
                    lb = sample['feat'].size(0)
                    del sample, outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # Calling the step function on an Optimizer makes an update to its parameters
                        optimizer.step()

                # statistics
                running_loss += loss.item() * lb
                running_acc += specs['acc'] * lb
                running_se += specs['se'] * lb
                running_sp += specs['sp'] * lb
                running_pre += specs['pre'] * lb
                running_npv += specs['npv'] * lb
                running_kappa += specs['kappa'] * lb

                # print('Batch complete in {:.2f}s'.format((time.time()-since_batch)))

                # prints
                if (i_batch + 1) % 250 == 0 and i_batch != 0 and phase == 'train':
                    cte_batch = (i_batch + 1)*lb
                    print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                        epoch + epoch_run, phase, i_batch, running_loss/cte_batch, running_acc/cte_batch,
                        running_se/cte_batch, running_sp/cte_batch, running_pre/cte_batch, running_npv/cte_batch,
                        running_kappa/cte_batch))

            cte_epoch = (i_batch + 1) * lb
            epoch_loss = running_loss / cte_epoch                   # Remember drop_last
            epoch_acc = running_acc / cte_epoch
            epoch_se = running_se / cte_epoch
            epoch_sp = running_sp / cte_epoch
            epoch_pre = running_pre / cte_epoch
            epoch_npv = running_npv / cte_epoch
            epoch_kappa = running_kappa / cte_epoch


            print('-' * 131)
            print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                epoch + epoch_run, phase, i_batch, epoch_loss, epoch_acc, epoch_se, epoch_sp, epoch_pre,
                epoch_npv, epoch_kappa))

            # if phase == 'val':
            #     time_elapsed_epoch = time.time()-since_epoch
            #     print('Epoch complete in {:.0f}m {:.0f}s'.format(
            #         time_elapsed_epoch // 60, time_elapsed_epoch % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + epoch_run
                best_model_wts = copy.deepcopy(model.state_dict())

        # Save checkpoint
        if path_bkp is not None:
            torch.save({
                'epoch': epoch + epoch_run + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_model_wts': best_model_wts,
                'best_acc': best_acc
            }, os.path.join(path_bkp, 'bkp_model_ft.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('in epoch: {:4f}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




def train_model_multiclass(model, optimizer, dataloaders, scheduler=None, num_epochs=25, cuda=None, dtype_data=None,
                dtype_target=None, path_bkp=None, checkpoint=None, alpha=None, gamma=0):

    # init

    if cuda is None:
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype_data is None:
        dtype_data = torch.float32

    if dtype_target is None:
        dtype_target = torch.int64

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_run = 0

    # Checkpoint
    if checkpoint is True:
        cp = torch.load(os.path.join(path_bkp, 'bkp_model_ft.pth'))
        epoch_run = cp['epoch']
        model.load_state_dict(cp['model_state_dict'])          # Load state of the last epoch
        best_model_wts = cp['best_model_wts']
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        best_acc = cp['best_acc']
        print('Resuming script in epoch {}. Best acc = {:4f}'.format(epoch_run, best_acc))

    print('-' * 143)
    print('|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|'.format(
        'Epoch', 'Phase', 'mini_batch', 'loss', 'acc', 'acc_w', 'acc_n1', 'acc_n2', 'acc_n3', 'acc_rem', 'kappa'))
    print('-' * 143)

    for epoch in range(num_epochs-epoch_run):

        print('-' * 143)

        # since_epoch = time.time()

        # Each epoch has a training and validation phase: by default all the modules are initialized to train
        # mode(self.training = True). Also be aware that some layers have different behavior during train and evaluation
        # (like BatchNorm, Dropout) so setting it matters.

        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_acc_w = 0.0
            running_acc_n1 = 0.0
            running_acc_n2 = 0.0
            running_acc_n3 = 0.0
            running_acc_rem = 0.0
            running_kappa = 0.0


            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):
                # since_batch = time.time()

                # sample to GPU
                sample['feat'] = sample['feat'].type(dtype=dtype_data).to(device=cuda)
                sample['target'] = sample['target'].type(dtype=dtype_target).to(device=cuda)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(src=sample['feat'], has_mask=False)

                    # Compute and print loss.
                    loss = ce_loss_multiclass(outputs, sample['target'], alpha)
                    loss = mfe_loss_multiclass(outputs.cpu(), sample['target'].cpu())
                    # loss = focal_loss(outputs, sample['target'], alpha, gamma)
                    specs = statistics_multiclass(outputs.cpu(), sample['target'].cpu(), sample['lengths'], model.d_out)

                    # del sample
                    lb = sample['feat'].size(0)
                    del sample, outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # Calling the step function on an Optimizer makes an update to its parameters
                        optimizer.step()

                # statistics
                running_loss += loss.item() * lb
                running_acc += specs['acc'] * lb
                running_acc_w += specs['acc_w'] * lb
                running_acc_n1 += specs['acc_n1'] * lb
                running_acc_n2 += specs['acc_n2'] * lb
                running_acc_n3 += specs['acc_n3'] * lb
                running_acc_rem += specs['acc_rem'] * lb
                running_kappa += specs['kappa'] * lb

                # print('Batch complete in {:.2f}s'.format((time.time()-since_batch)))

                # prints
                if (i_batch + 1) % 250 == 0 and i_batch != 0 and phase == 'train':
                    cte_batch = (i_batch + 1)*lb
                    print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                        epoch + epoch_run, phase, i_batch, running_loss/cte_batch, running_acc/cte_batch,
                        running_acc_w/cte_batch, running_acc_n1/cte_batch, running_acc_n2/cte_batch,
                        running_acc_n3/cte_batch, running_acc_rem/cte_batch, running_kappa/cte_batch))

            cte_epoch = (i_batch + 1) * lb
            epoch_loss = running_loss / cte_epoch                   # Remember drop_last
            epoch_acc = running_acc / cte_epoch
            epoch_acc_w = running_acc_w / cte_epoch
            epoch_acc_n1 = running_acc_n1 / cte_epoch
            epoch_acc_n2 = running_acc_n2 / cte_epoch
            epoch_acc_n3 = running_acc_n3 / cte_epoch
            epoch_acc_rem = running_acc_rem / cte_epoch
            epoch_kappa = running_kappa / cte_epoch


            print('-' * 143)
            print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                epoch + epoch_run, phase, i_batch, epoch_loss, epoch_acc, epoch_acc_w, epoch_acc_n1, epoch_acc_n2,
                epoch_acc_n3, epoch_acc_rem, epoch_kappa))

            # if phase == 'val':
            #     time_elapsed_epoch = time.time()-since_epoch
            #     print('Epoch complete in {:.0f}m {:.0f}s'.format(
            #         time_elapsed_epoch // 60, time_elapsed_epoch % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + epoch_run
                best_model_wts = copy.deepcopy(model.state_dict())

        # Save checkpoint
        if path_bkp is not None:
            torch.save({
                'epoch': epoch + epoch_run + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_model_wts': best_model_wts,
                'best_acc': best_acc
            }, os.path.join(path_bkp, 'bkp_model_ft.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('in epoch: {:4f}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model