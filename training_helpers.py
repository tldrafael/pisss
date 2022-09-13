import os
import warnings
import numpy as np
import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class MetricsSet:
    def __init__(self, names=['CE'], semantic_metrics=[]):
        self.names = names
        self.step = {m: [] for m in self.names}
        self.fl_semantic_metrics = len(semantic_metrics) > 0
        self.semantic_metrics = semantic_metrics
        if self.fl_semantic_metrics:
            for m in semantic_metrics:
                self.step[m] = []

        self.iteration = {m: [] for m in self.names}
        self.best_values = {m: np.inf for m in self.names}

    def update_epoch_metrics(self):
        for k, v in self.iteration.items():
            self.step[k].append(np.nanmean(v))
            self.best_values[k] = np.min(self.step[k])
        if self.fl_semantic_metrics:
            for m in self.semantic_metrics:
                self.best_values[m] = np.max(self.step[m])

        self.reset_iteration_metrics()

    def update_iteration_metric(self, name, value):
        self.iteration[name].append(value)

    def reset_iteration_metrics(self):
        self.iteration = {m: [] for m in self.names}

    def get_last_metrics(self):
        last_res = {}
        for k, v in self.step.items():
            if k in self.semantic_metrics:
                last_res[k] = np.round(v[-1], 3)
            else:
                last_res[k] = np.round(v[-1], 2)
        return last_res


def get_CM(pred, label, n_classes):
    # https://github.com/Qualcomm-AI-research/InverseForm/blob/be142136087579d5f7175cbf64c171fc52352fc7/utils/misc.py#L20
    # I ADAPTED THE CODE TO OBEY THE CORRECT RESHAPING ORDER ORDER='F' (IT PROBABLY HAD A BUG BEFORE)
    CM_cur = np.bincount(n_classes * label.flatten() + pred.flatten(), minlength=n_classes ** 2)
    return CM_cur.reshape(n_classes, n_classes, order='F').astype(int)


def get_CM_fromloader_cityscapes(dloader, model, n_classes, ix_nolabel=255):
    # https://github.com/Qualcomm-AI-research/InverseForm/blob/be142136087579d5f7175cbf64c171fc52352fc7/utils/misc.py#L20
    # stretch ground truth labels by num_classes
    # TP at 0 + 0, 1 + 1, 2 + 2 ...  # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    CM_abs = np.zeros((n_classes, n_classes), dtype=int)
    for inp_data, inp_label in dloader:
        test_preds = model(inp_data.cuda()).argmax(1, keepdim=True).cpu()
        for pr_i, y_i in zip(test_preds, inp_label):
            if ix_nolabel is None:
                CM_abs += get_CM(pr_i, y_i, n_classes)
            else:
                if (pr_i == ix_nolabel).sum().item() > 0:
                    warnings.warn('The model has also predicted ix_nolabel value, these pixels are also being ignored '
                                  ' - thus, this metric is not correct!')
                mask = (y_i != ix_nolabel) & (pr_i != ix_nolabel)
                CM_abs += get_CM(pr_i[mask], y_i[mask], n_classes)
    return CM_abs


def get_CM_fromloader_cityscape_frombatch(inp_data, inp_label, model, n_classes):
    CM_abs = np.zeros((n_classes, n_classes), dtype=int)
    test_preds = model.forward(inp_data.cuda()).argmax(1).cpu()
    for i in range(inp_data.shape[0]):
        for pr_i, y_i in zip(test_preds[i], inp_label[i]):
            CM_abs += get_CM(pr_i, y_i, n_classes)
    return CM_abs


def get_miou_f1_fromloader(dloader, model, n_classes, return_all=False, fl_singleimage=False, ix_nolabel=255):
    if not fl_singleimage:
        CM_abs = get_CM_fromloader_cityscapes(dloader, model, n_classes, ix_nolabel)
    else:
        CM_abs = get_CM_fromloader_cityscape_frombatch(*dloader, model, n_classes)

    pred_P = CM_abs.sum(axis=0)
    gt_P = CM_abs.sum(axis=1)
    true_P = np.diag(CM_abs)

    CM_iou = true_P / (pred_P + gt_P - true_P)
    CM_f1 = 2 * true_P / (pred_P + gt_P)

    miou = np.nanmean(CM_iou)
    mf1 = np.nanmean(CM_f1)

    if return_all:
        return CM_abs, CM_iou, miou, CM_f1, mf1
    else:
        return miou, mf1


def compute_miouloss(pred, label, conf_thrs=None, return_mean=True, n_classes=12, ignore_index=255):
    if ignore_index in label:
        label[label == ignore_index] = n_classes
        label_onehot = F.one_hot(label[:, 0], num_classes=n_classes + 1).permute(0, 3, 1, 2)
        label_onehot = label_onehot[:, :n_classes]
    else:
        label_onehot = F.one_hot(label[:, 0], num_classes=n_classes).permute(0, 3, 1, 2)

    assert label_onehot.shape == pred.shape

    if conf_thrs is None:
        denom = (label_onehot + pred - label_onehot * pred).sum([0, 2, 3]).clamp_min(1e-3)
        miou_score = (label_onehot * pred).sum([0, 2, 3]) / denom
    else:
        conf_mask = 1. * (pred.amax(1, keepdim=True) >= conf_thrs)
        denom = ((label_onehot + pred - label_onehot * pred) * conf_mask).sum([0, 2, 3]).clamp_min(1e-3)
        miou_score = (label_onehot * pred * conf_mask).sum([0, 2, 3]) / denom

    if return_mean:
        return 1 - miou_score.mean()
    else:
        return 1 - miou_score


def compute_diceloss(pred, label, conf_thrs=None, return_mean=True, n_classes=12, ignore_index=255):
    if ignore_index in label:
        label[label == ignore_index] = n_classes
        label_onehot = F.one_hot(label[:, 0], num_classes=n_classes + 1).permute(0, 3, 1, 2)
        label_onehot = label_onehot[:, :n_classes]
    else:
        label_onehot = F.one_hot(label[:, 0], num_classes=n_classes).permute(0, 3, 1, 2)

    assert label_onehot.shape == pred.shape

    if conf_thrs is None:
        dice_score = (label_onehot * pred).sum([0, 2, 3]) / (label_onehot + pred).sum([0, 2, 3]).clamp_min(1e-3)
    else:
        conf_mask = 1. * (pred.amax(1, keepdim=True) >= conf_thrs)
        denom = ((label_onehot + pred) * conf_mask).sum([0, 2, 3]).clamp_min(1e-3)
        dice_score = (label_onehot * pred * conf_mask).sum([0, 2, 3]) / denom

    if return_mean:
        return 1 - dice_score.mean()
    else:
        return 1 - dice_score


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma=2, **kwargs):
        kwargs['reduction'] = 'none'
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, input, target):
        input_CE = super().forward(input, target)
        input_probs = torch.exp(-F.cross_entropy(input, target, reduction='none'))

        input_CE *= (1 - input_probs) ** self.gamma
        if self.weight is None:
            return input_CE.mean()
        else:
            return input_CE.sum() / self.weight[target].sum()


def update_setmetrics(set_metrics, dloader, model, n_classes, ix_nolabel=255):
    miou, mf1 = get_miou_f1_fromloader(dloader, model, n_classes=n_classes, ix_nolabel=ix_nolabel)
    set_metrics.step['miou'].append(miou)
    set_metrics.step['mf1'].append(mf1)
    set_metrics.update_epoch_metrics()


def read_dummy_images(fpath, label_colorizer, has_label=True):
    with open(fpath, 'r') as f:
        fnames = f.read().split('\n')[:-1]

    if has_label:
        impaths = [os.path.join('RTK_semanticData/label_data/image', f) for f in fnames]
        dummy_images = torch.stack([tv.io.read_image(p) for p in impaths]) / 255
        lbpaths = [os.path.join('RTK_semanticData/label_data/label', f) for f in fnames]
        dummy_labels = torch.stack([tv.io.read_image(p) for p in lbpaths])
        return dummy_images, label_colorizer(dummy_labels)
    else:
        impaths = [os.path.join('RTK_semanticData/unlabel_data/image', f) for f in fnames]
        dummy_images = torch.stack([tv.io.read_image(p) for p in impaths]) / 255
        return dummy_images


def freeze_batchnorm_layers(model, filtername=None):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if filtername is not None:
                if filtername not in name:
                    continue
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def split_model_params(model):
    encoder, decoder = [], []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            encoder.append(param)
        else:
            decoder.append(param)
    return encoder, decoder


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-4, logger=None, fl_warmup=False,
                 n_warmup_max=5, **kwargs):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        self.logger = logger
        self.write_log('Set PolyLR, max_iters: {}, min_lr: {}'.format(max_iters, min_lr))
        self.fl_warmup = fl_warmup
        if fl_warmup:
            self.write_log('Set Warmup policy, warmup rounds: {}'.format(n_warmup_max))
        self.n_warmup_max = n_warmup_max
        self.n_warmup = 0
        super().__init__(optimizer, last_epoch)

    def write_log(self, msg):
        if self.logger is not None:
            self.logger.log(msg)

    def get_lr(self):
        if self.fl_warmup:
            new_lr = [base_lr * (self.last_epoch + 1) / (self.n_warmup_max + 1) for base_lr in self.base_lrs]
            self.n_warmup += 1.
            if self.last_epoch == (self.n_warmup_max - 1):
                self.write_log('Shut down Warmup policy')
                self.fl_warmup = False
        else:
            attenuate_factor = (1 - (self.last_epoch - self.n_warmup) / (self.max_iters - self.n_warmup)) ** self.power
            new_lr = [max(base_lr * attenuate_factor, self.min_lr) for base_lr in self.base_lrs]

        for i, new_lr_i in enumerate(new_lr):
            self.write_log('lr_{}: {:.2e}'.format(i, new_lr_i))

        return new_lr
