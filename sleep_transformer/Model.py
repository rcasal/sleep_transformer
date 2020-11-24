######################################################################
# Model CustomLayer
# -----------------
#
# Configure model class
#
import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sleep_transformer.tcn import TemporalConvNet

class OwnTransformerModel(nn.Module):
    "A transformer model as Attention is all you need"
    def __init__(self, d_in, d_out, batch_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.5):

        super(OwnTransformerModel, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.batch_size = batch_size
        self.d_model= d_model

        # self.encoder = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        self.encoder = nn.Linear(in_features=30, out_features=d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        batch_size, _, seq_len = src.size()
        seq_len = int(seq_len / 30)

        src = src.view(-1, 1, 30)    # [batch_size, channels_in, seq_len] -> [batch_size*seq_len/30, channels_in, 30]
        src = self.encoder(src) #* math.sqrt(self.ninp)       # [batch_size*seq_len/30, channels_in, d_model]
        src = src.view(batch_size, seq_len, self.d_model)   # [batch_size, seq_len/30, d_model]

        src = src.transpose(dim0=1, dim1=0)         # [seq_len/30, batch_size, d_model] = [S, N, E]

        output = self.transformer(src=src, tgt=tgt, src_mask=None, tgt_mask=None)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


# MODULE IMPLEMENTED BY PYTORCH
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, d_in, d_out, batch_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.2, max_len=1500):

        super(TransformerModel, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.batch_size = batch_size
        self.d_model = d_model
        self.src_mask = None

        # self.encoder = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_features=30, out_features=d_model),
        #     nn.Dropout(p=dropout),
        #     nn.Tanh(),
        # )
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=self.d_in, out_channels=16, kernel_size=3, stride=5, padding=int((3 - 1) / 2)),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool1d(kernel_size=5, stride=5),
        #
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=int((3 - 1) / 2)),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=int((3 - 1) / 2)),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=3, stride=1, padding=int((3 - 1) / 2)),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(d_model),
        #     nn.ReLU(inplace=True),
        # )

        # if d_in=1
        # self.encoder = TemporalConvNet(num_inputs=d_in, num_channels=[16, 32, 64, d_model], pooling=[5, 3, 2, 1],
        #                                kernel_size=3, dropout=dropout)
        # if d_in=2
        self.encoder = TemporalConvNet(num_inputs=d_in, num_channels=[32, 64, 128, d_model], pooling=[5, 3, 2, 1],
                                       kernel_size=3, dropout=dropout)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers,
                                                         norm=nn.LayerNorm(d_model))

        # self.decoder = nn.Linear(in_features=d_model, out_features=d_out)
        self.decoder = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=d_model, out_features=d_out),
        )

    def masking(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):

        batch_size, _, seq_len = src.size()
        seq_len = int(seq_len / 30)

        # # linear
        # src = src.view(-1, 1, 30)    # [batch_size, channels_in, seq_len] -> [batch_size*seq_len/30, channels_in, 30]
        # src = self.encoder(src) #* math.sqrt(self.ninp)       # [batch_size*seq_len/30, channels_in, d_model]
        # src = src.view(batch_size, seq_len, self.d_model)   # [batch_size, seq_len/30, d_model]
        # src = src.transpose(dim0=1, dim1=0)  # [seq_len/30, batch_size, d_model] = [S, N, E]

        # # conv1d y tcn
        # src = self.encoder(src)
        # src = src.transpose(dim0=2, dim1=0).transpose(dim0=1, dim1=2)  # [seq_len/30, batch_size, d_model] = [S, N, E]

        # tcn
        src = self.encoder(src)  # input should have dimension (N, C, L)
        src = src.transpose(dim0=2, dim1=0).transpose(dim0=1, dim1=2)  # [seq_len/30, batch_size, d_model] = [S, N, E]

        self.masking(src, has_mask)
        src = self.pos_encoder(x=src)
        output = self.transformer_encoder(src=src, mask=self.src_mask)

        output = output.contiguous()
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.transpose(dim0=1, dim1=0)
        return output


def ce_loss(y_hat, y, alpha_i):
    """
    before we calculate the negative log likelihood, we need to mask out the activations
    this means we don't want to take into account padded items in the output vector
    simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    and calculate the loss on that.
    <->  y_hat is log_softmax(output_net)   <->
    """

    # flatten all the labels
    y = y.view(-1)

    _, _, d_out = y_hat.shape
    # flatten all predictions
    y_hat = y_hat.contiguous()
    y_hat = y_hat.view(-1, d_out)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y > tag_pad_token).float()  # .type(dtype=dtype_target)

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).item())

    # mask = mask * ((y == 0).float() + (y == 1).float()*4)  # to include unbalance treatment
    mask = mask * ((y == 0).float() / alpha_i[0] + (y == 1).float() / alpha_i[1])  # 0 despierto, 1 dormido

    # pick the values for the label and zero out the rest with the mask
    y_hat = y_hat[range(y_hat.shape[0]), y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(y_hat) / nb_tokens

    # del variables
    del y_hat, y, tag_pad_token, nb_tokens, mask

    return ce_loss


def focal_loss(y_hat, y, alpha_i, gamma):
    """
    Focal Loss.
    <->  y_hat is log_softmax(output_net)   <->
    """

    # flatten all the labels
    y = y.view(-1)

    _, _, d_out = y_hat.shape
    # flatten all predictions
    y_hat = y_hat.contiguous()
    y_hat = y_hat.view(-1, d_out)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y > tag_pad_token).float()  # .type(dtype=dtype_target)

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).item())

    # pick the values for the label and zero out the rest with the mask
    y_hat = y_hat[range(y_hat.shape[0]), y] * mask

    alpha = mask * ((y == 0).float()/alpha_i[0] + (y == 1).float() / alpha_i[1])        # 0 despierto, 1 dormido

    # compute focal loss loss which ignores all <PAD> tokens
    focal_loss = -torch.sum(alpha*(1-y_hat.exp())**gamma * y_hat) / nb_tokens

    # del variables
    del y_hat, y, tag_pad_token, nb_tokens, mask

    return focal_loss


def mfe_loss_multiclass(y_hat, y, msfe=False):
    """
    before we calculate the negative log likelihood, we need to mask out the activations
    this means we don't want to take into account padded items in the output vector
    simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    and calculate the loss on that.
    <->  y_hat is log_softmax(output_net)   <->
    """
    _, _, d_out = y_hat.shape

    # Classification
    # _, preds = torch.max(y_hat, 2)
    y_hat = 1 / (1 + torch.exp(-y_hat)) # Lo transformo en prob sin perder el grad

    # flatten all the labels and predictions
    y = y.view(-1)
    y_hat = y_hat.contiguous()
    y_hat = y_hat.view(-1, d_out)

    # one-hot encode
    onehot_encoded = list()
    for sample in y:
        _col = [0 for _ in range(d_out)]
        _col[sample] = 1
        onehot_encoded.append(_col)

    y_aux = y
    y = torch.Tensor(onehot_encoded)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y_aux > tag_pad_token).float()  # .type(dtype=dtype_target)

    # create a mask for each class (first index for each class)    W: 0, N1: 1, N2:2, N3:3, rem=4. Tokens are -1, so in this position each mask take value 0.
    mask_classes_aux = torch.zeros(d_out,y.shape[0])
    for i in range(d_out):
        mask_classes_aux[i, :] = (y_aux == i).float()

    mask_classes = torch.zeros(d_out, y.shape[0], d_out)
    for i in range(d_out):
        mask_classes[i, :, :] = torch.repeat_interleave(mask_classes_aux[i,:], d_out).view(-1, d_out)

    # count how many tokens we have for each class
    nb_tokens = int(torch.sum(mask).item())
    nb_tokens_classes = torch.zeros(d_out)
    for i in range(d_out):
        nb_tokens_classes[i] = int(torch.sum(mask_classes[i,:,0]).item())   # Third index is whatever

    # pick the values for the label and zero out the rest with the mask
    # y_hat = y_hat[range(y_hat.shape[0]), y] * mask

    # compute loss for each class
    class_loss = torch.zeros(d_out)
    for i in range(d_out):
        class_loss[i] = ((y_hat - y).pow(2) * mask_classes[i,:,:]).sum() / nb_tokens_classes[i]  #

    mfe_loss = (class_loss.pow(2)).mean()

    # del variables
    del y_hat, y, tag_pad_token, nb_tokens, mask

    return mfe_loss

def ce_loss_multiclass(y_hat, y, alpha_i=None):
    """
    before we calculate the negative log likelihood, we need to mask out the activations
    this means we don't want to take into account padded items in the output vector
    simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    and calculate the loss on that.
    <->  y_hat is log_softmax(output_net)   <->
    """

    # flatten all the labels
    y = y.view(-1)

    _, _, d_out = y_hat.shape
    # flatten all predictions
    y_hat = y_hat.contiguous()
    y_hat = y_hat.view(-1, d_out)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y > tag_pad_token).float()  # .type(dtype=dtype_target)

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).item())

    if alpha_i is not None:     # W: 0, N1: 1, N2:2, N3:3, rem=4
        # mask = mask * ((y == 0).float() + (y == 1).float()*4)  # to include unbalance treatment
        mask = mask * ((y==0).float()/alpha_i[0] + (y==1).float()/alpha_i[1] + (y==2).float()/alpha_i[2]
                       + (y==3).float() / alpha_i[3] + (y==4).float()/alpha_i[4])

    # pick the values for the label and zero out the rest with the mask
    y_hat = y_hat[range(y_hat.shape[0]), y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(y_hat) / nb_tokens

    # del variables
    del y_hat, y, tag_pad_token, nb_tokens, mask

    return ce_loss

def statistics(y_hat, y, y_lengths):
    _, preds = torch.max(y_hat, 2)

    stats = {'acc': 0.0, 'se': 0.0, 'sp': 0.0, 'pre': 0.0, 'npv': 0.0, 'kappa': 0.0}

    for i in range(y.size()[0]):
        # confusion matrix: 1 and 1 (TP); 1 and 0 (FP); 0 and 0 (TN); 0 and 1 (FN)

        aux = preds[i, 0:y_lengths[i]].float() / y.data[i, 0:y_lengths[i]].float()

        tp = torch.sum(aux == 1.0).item() + 1e-8
        fp = torch.sum(aux == float('inf')).item() + 1e-8
        tn = torch.sum(torch.isnan(aux)).item() + 1e-8
        fn = torch.sum(aux == 0).item() + 1e-8

        aux_acc = (tp + tn) / (tp + tn + fp + fn)
        stats['acc'] += aux_acc
        stats['se'] += tp / (tp + fn)
        stats['sp'] += tn / (tn + fp)
        stats['pre'] += tp / (tp + fp)
        stats['npv'] += tn / (tn + fn)
        a_rnd = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn) ** 2
        stats['kappa'] += (aux_acc - a_rnd) / (1 - a_rnd)

        del tp, fp, tn, fn, aux

    stats['acc'] = stats['acc'] / len(y_lengths)
    stats['se'] = stats['se'] / len(y_lengths)
    stats['sp'] = stats['sp'] / len(y_lengths)
    stats['pre'] = stats['pre'] / len(y_lengths)
    stats['npv'] = stats['npv'] / len(y_lengths)
    stats['kappa'] = stats['kappa'] / len(y_lengths)

    return stats

def statistics_test(y_hat, y, y_lengths):

    _, preds = torch.max(y_hat, 2)

    stats = {'acc': 0.0, 'se': 0.0, 'sp': 0.0, 'pre': 0.0, 'npv': 0.0, 'kappa': 0.0, 'err':0.0, 'rel_err':0.0,
             'tst':0.0, 'tst_est':0.0, 'trt':0.0}

    for i in range(y.size()[0]):
        aux = preds[i, 0:y_lengths[i]].float() / y.data[i, 0:y_lengths[i]].float()

        tp = torch.sum(aux == 1.0).item() + 1e-8
        fp = torch.sum(aux == float('inf')).item() + 1e-8
        tn = torch.sum(torch.isnan(aux)).item() + 1e-8
        fn = torch.sum(aux == 0).item() + 1e-8

        tst = torch.sum(y == 1).item() / (2 * 60)
        tst_est = torch.sum(preds == 1).item() / (2 * 60)
        trt = len(y) / (2 * 60)

        aux_acc = (tp + tn) / (tp + tn + fp + fn)
        stats['acc'] += aux_acc
        stats['se'] += tp / (tp + fn)
        stats['sp'] += tn / (tn + fp)
        stats['pre'] += tp / (tp + fp)
        stats['npv'] += tn / (tn + fn)
        a_rnd = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn) ** 2
        stats['kappa'] += (aux_acc - a_rnd) / (1 - a_rnd)

        stats['err'] += abs(tst-tst_est)
        stats['rel_err'] += abs(tst - tst_est) / tst
        stats['tst'] += tst
        stats['tst_est'] += tst_est
        stats['trt'] += trt

        del tp, fp, tn, fn, aux

    stats['acc'] = stats['acc'] / len(y_lengths)
    stats['se'] = stats['se'] / len(y_lengths)
    stats['sp'] = stats['sp'] / len(y_lengths)
    stats['pre'] = stats['pre'] / len(y_lengths)
    stats['npv'] = stats['npv'] / len(y_lengths)
    stats['kappa'] = stats['kappa'] / len(y_lengths)

    stats['err'] = stats['err'] / len(y_lengths)
    stats['rel_err'] = stats['rel_err'] / len(y_lengths)
    stats['tst'] = stats['tst'] / len(y_lengths)
    stats['tst_est'] = stats['tst_est'] / len(y_lengths)
    stats['trt'] = stats['trt'] / len(y_lengths)

    return stats


def statistics_multiclass(y_hat, y, y_lengths, n_class):

    _, preds = torch.max(y_hat, 2)

    stats = {'acc': 0.0, 'acc_w': 0.0, 'acc_n1': 0.0, 'acc_n2': 0.0, 'acc_n3': 0.0, 'acc_rem': 0.0, 'kappa': 0.0 }

    for i in range(y.size()[0]):
        length_i = int(y_lengths[i] / 30)

        y_i =  y.data[i, 0:length_i]
        preds_i = preds[i, 0:length_i]

        CM = confusion_matrix(preds_i, y_i, [0, 1, 2, 3, 4])   # cols: real, rows: pred

        stats['acc'] += (CM.diagonal().sum() + 1e-8 )/ (CM.sum() + 1e-8)
        stats['acc_w'] += (CM[0,0] + 1e-8 )/ (CM[:,0].sum() + 1e-8)
        stats['acc_n1'] += (CM[1, 1] + 1e-8 )/ (CM[:, 1].sum() + 1e-8)
        stats['acc_n2'] += (CM[2, 2] + 1e-8 )/ (CM[:, 2].sum() + 1e-8)
        stats['acc_n3'] += (CM[3, 3] + 1e-8 )/ (CM[:, 3].sum() + 1e-8)
        stats['acc_rem'] += (CM[4, 4] + 1e-8 )/ (CM[:, 4].sum() + 1e-8)

        stats['kappa'] += cohen_kappa_score(preds_i, y_i, [0, 1, 2, 3, 4])


    stats['acc'] = stats['acc'] / len(y_lengths)
    stats['acc_w'] =stats['acc_w'] / len(y_lengths)
    stats['acc_n1'] = stats['acc_n1'] / len(y_lengths)
    stats['acc_n2'] = stats['acc_n2'] / len(y_lengths)
    stats['acc_n3'] = stats['acc_n3'] / len(y_lengths)
    stats['acc_rem'] = stats['acc_rem'] / len(y_lengths)
    stats['kappa'] = stats['kappa'] / len(y_lengths)

    return stats



    # print(stats['confusion_matrix'])
    # plots
    # plt.subplot(2,1,1)
    # plt.plot(y_i.numpy(), 'k')
    # tit = 'Acc:' + str(stats['acc'])
    # plt.title(tit)
    # plt.subplot(2,1,2)
    # plt.plot(preds_i.numpy(), 'k')
    # plt.xlabel('Time [hs]')
    # plt.show()


def statistics_test_multiclass(y_hat, y, y_lengths, n_class):
    "Sólo funciona para n_batch 1!!!"

    if len(y_lengths) is 1:

        length_i = int(y_lengths[0] / 30)
        _, preds = torch.max(y_hat, 2)

        stats = {}
        stats['acc'] = 0.0
        stats['kappa'] = 0.0
        stats['confusion_matrix'] = np.zeros([n_class,n_class])
        stats['confusion_matrix_percentage'] = np.zeros([n_class,n_class])

        y_i = y[0, 0:length_i]
        preds_i = preds[0, 0:length_i]

        coinc = (preds_i.float() == y_i.data.float()).sum()
        stats['acc'] = coinc.item()/preds_i.__len__()

        stats['confusion_matrix'] = confusion_matrix(preds_i, y_i, [0, 1, 2, 3, 4])

        # stats['confusion_matrix_percentage'] = stats['confusion_matrix'] / stats['confusion_matrix'].astype(np.float).sum(axis=0)

        stats['kappa']  = cohen_kappa_score(preds_i, y_i, [0, 1, 2, 3, 4])

        # print(stats['confusion_matrix'])
        # # plots
        # plt.subplot(2,1,1)
        # plt.plot(y_i.numpy(), 'k')
        # tit = 'Acc:' + str(stats['acc'])
        # plt.title(tit)
        # plt.subplot(2,1,2)
        # plt.plot(preds_i.numpy(), 'k')
        # plt.xlabel('Time [hs]')
        # plt.show()
    else:
        print('Error. Minibatch size must be 1!')
        exit()
    return stats
