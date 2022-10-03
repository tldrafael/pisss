from functools import partial
import re
import os
import sys
import logging
import shutil
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torch.optim import SGD, Adam
import datapipe as dp
import training_helpers as th
import utils as ut
from deeplabv3 import deeplabv3plus_resnet34, deeplabv3plus_resnet50, deeplabv3plus_resnet101
sys.path.append('segmentation_models_pytorch')
import segmentation_models_pytorch as smp


class Trainer:
    def __init__(self):
        self.logger = ut.Logger()

    def get_default_optimizer(self, optim='adam', sameLR=True, max_lr_sgd=1e-2, min_lr_sgd=None, lr_adam=1e-4,
                              **kwargs):
        optim = optim.lower()
        self.policy = self.policy.lower() if self.policy is not None else self.policy

        self.logger.log('Received args optimizer: {} and policy: {}'.format(optim, self.policy))
        assert optim in ['adam', 'sgd'], 'optim should be "adam" or "sgd"'

        if optim == 'adam':
            self.logger.log('Set optimizer: Adam')
            if sameLR:
                self.optimizer = Adam(self.model.parameters(), lr=lr_adam)
                self.logger.log('Set same LR to encoder and decoder')
            else:
                encoder, decoder = th.split_model_params(self.model)
                self.optimizer = Adam([{'params': encoder, 'lr': lr_adam / 10}, {'params': decoder, 'lr': lr_adam}])
                self.logger.log('Set 1/10 lower LR to encoder')
            self.logger.log('Set Adam LR: {}'.format(str(lr_adam)))
        else:
            self.logger.log('Set optimizer: SGD')
            assert self.nsteps is not None, 'Set the number of iterations to decay the learning rate'
            args_sgd = {'momentum': 0.9, 'weight_decay': 5e-4}

            if sameLR:
                self.optimizer = SGD(self.model.parameters(), lr=max_lr_sgd, **args_sgd)
                self.logger.log('Set same LR to encoder and decoder')
            else:
                encoder, decoder = th.split_model_params(self.model)
                self.optimizer = SGD([{'params': encoder, 'lr': max_lr_sgd / 10, **args_sgd},
                                      {'params': decoder, 'lr': max_lr_sgd, **args_sgd}])
                self.logger.log('Set 1/10 lower LR to encoder')

        if self.policy == 'poly':
            if min_lr_sgd is None:
                min_lr_sgd = max_lr_sgd / 100
            self.scheduler = th.PolyLR(self.optimizer, self.nsteps, min_lr=min_lr_sgd, logger=self.logger,
                                       fl_warmup=self.fl_warmup, **kwargs)
            self.logger.log('Set scheduler: Poly')
        elif self.policy == 'linear':
            self.scheduler = LinearLR(self.optimizer, start_factor=1, end_factor=1e-2, total_steps=self.nsteps)
            self.logger.log('Set scheduler: Linear')
        elif self.policy == 'one_cycle':
            encoder, decoder = th.split_model_params(self.model)
            self.optimizer = Adam([{'params': encoder, 'lr': 1e-4 / 4}, {'params': decoder, 'lr': 1e-4 / 400}],
                                  weight_decay=1e-2)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=1e-4 / 4, pct_start=.9,
                                        total_steps=self.nsteps * self.niters)
            self.logger.log('Set scheduler: OneCycle')
        else:
            self.scheduler = None
            self.logger.log('Set scheduler: None')

        self.logger.log('Set optimizer: {}'.format(self.optimizer))
        self.logger.log('Set scheduler: {}'.format(self.scheduler.state_dict() if self.scheduler else None))

    def save_ckpt(self, path):
        torch.save({
            'cur_step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # 'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.val_metrics.best_values['miou']
        }, path)
        self.logger.log('Model saved at {}'.format(path))

    def set_default_writer(self, fl_log, fl_resume):
        if fl_log:
            self.logdir = os.path.join(self.ckpt_dir, 'logs')
        else:
            self.logdir = 'logs'

        if self.start_epoch == 1 or not fl_resume:
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)
                self.logger.log('Removed log folder: {}'.format(self.logdir))

        self.writer = SummaryWriter(log_dir=self.logdir)
        self.logger.log('Set log writer to: {}'.format(self.logdir))

    def run(self, fl_resume=False, fl_force=False, fl_fasttest=False, fl_save_logimage=False, nexps=1, start_exp=0,
            niters=1000, nsteps=200, model_name='Unet', encoder_name='resnet50', fl_warmup=False,
            policy=None, fl_freeze=False, fl_stemstride=True, fl_richstem=False, fl_parallelstem=False,
            fl_maxpool=True, fl_lfe=False, fl_transpose=False, fl_transpose_odd=False, output_stride=16, p_cutmix=0,
            losses_set=['CE'], **kwargs):

        # *****************************************
        # Fast test
        if fl_fasttest:
            fl_savemodel, fl_log = False, False
            nexps = 1
            niters = 1
            nsteps = 1
        else:
            fl_savemodel, fl_log = True, True

        # *****************************************
        # Start Training
        self.nsteps = nsteps
        self.niters = niters
        self.policy = policy
        self.fl_warmup = fl_warmup

        for self.i_exp in range(start_exp, start_exp + nexps):

            ckpt_bname = '{}-{}'.format(model_name, encoder_name)
            if fl_freeze:
                ckpt_bname += '-freezeBN'

            self.ckpt_dir = os.path.join(self.ckpt_dirbase, ckpt_bname, 'exp_{}'.format(self.i_exp))
            self.logger.log(self.ckpt_dir)
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            elif not fl_resume and not fl_force:
                raise Exception('Either resume the training or erase the directory, or set fl_force to True')
            else:
                pass

            self.logger.add_handler(os.path.join(self.ckpt_dir, 'Trainer.log'), fl_resume)
            self.logger.log_header()

            model_params = {'encoder_name': encoder_name, 'encoder_weights': 'imagenet', 'in_channels': 3,
                            'classes': self.n_classes}

            if model_name == 'DeepLabV3Plus':
                self.logger.log('Set model: DeepLabV3Plus, encoder: {}'.format(encoder_name))
                self.logger.log('Use Resnet Max Pooling: {}'.format(fl_maxpool))
                self.logger.log('Output stride: {}'.format(output_stride))
                self.logger.log('Use Stem Stride: {}'.format(fl_stemstride))
                self.logger.log('Use Rich Stem: {}'.format(fl_richstem))
                self.logger.log('Use Parallel Stem: {}'.format(fl_parallelstem))
                self.logger.log('Use ConvTransposed: {}'.format(fl_transpose))
                self.logger.log('Use ConvTransposed Odd: {}'.format(fl_transpose_odd))
                self.logger.log('Use LFE: {}'.format(fl_lfe))
                model_method = '{}_{}'.format(model_name.lower(), encoder_name.lower())
                self.model = eval(model_method)(num_classes=self.n_classes,
                                                output_stride=output_stride,
                                                fl_maxpool=fl_maxpool,
                                                fl_richstem=fl_richstem,
                                                fl_stemstride=fl_stemstride,
                                                fl_parallelstem=fl_parallelstem,
                                                fl_transpose=fl_transpose,
                                                fl_transpose_odd=fl_transpose_odd,
                                                fl_lfe=fl_lfe).cuda()
            else:
                self.model = eval('smp.' + model_name)(**model_params, fl_maxpool=fl_maxpool, fl_richstem=fl_richstem,
                                                       fl_parallelstem=fl_parallelstem).cuda()
                self.logger.log('Set model: {}'.format(self.model.name))
                # self.model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=12)

            self.get_default_optimizer(**kwargs)

            self.logger.log('Losses set: {}'.format(', '.join(losses_set)))
            semantic_metrics = ['miou', 'mf1']
            other_metrics = ['CE']
            self.train_metrics = th.MetricsSet(names=other_metrics, semantic_metrics=semantic_metrics)
            self.val_metrics = th.MetricsSet(names=other_metrics, semantic_metrics=semantic_metrics)
            logset_metrics = [('train', self.train_metrics), ('val', self.val_metrics)]
            if self.test_loader is not None:
                self.test_metrics = th.MetricsSet(names=[], semantic_metrics=semantic_metrics)
                logset_metrics += [('test', self.test_metrics)]

            self.start_epoch = 1
            if fl_resume:
                self.logger.log('fl_resume: True')
                lastmodel_path = os.path.join(self.ckpt_dir, 'model.last.pth')
                if os.path.exists(lastmodel_path):
                    state_dict = torch.load(lastmodel_path)
                    self.start_epoch += state_dict['cur_step']
                    self.model.load_state_dict(state_dict['model_state'])
                    self.optimizer.load_state_dict(state_dict['optimizer_state'])
                    # if self.scheduler is not None:
                    #    self.scheduler.load_state_dict(state_dict['scheduler_state'])
                    if self.policy == 'poly' and self.fl_warmup:
                        self.scheduler = th.PolyLR(self.optimizer, self.nsteps, logger=self.logger,
                                                   fl_warmup=self.fl_warmup, **kwargs)
                        self.logger.log('Set scheduler: Poly')
                    self.val_metrics.best_values['miou'] = state_dict['best_miou']
                    self.val_metrics.step['miou'].append(state_dict['best_miou'])
                    self.logger.log('Loaded previous model from {}'.format(lastmodel_path))
                    self.logger.log('Training starting from step {}'.format(self.start_epoch))
                else:
                    self.logger.log('There is no previous trained model to start from', logging.WARNING)
            else:
                self.logger.log('fl_resume: False')

            self.set_default_writer(fl_log, fl_resume)

            for self.step in range(self.start_epoch, nsteps + 1):
                # *****************************************
                # Training set cycle
                self.model.train()
                if fl_freeze:
                    th.freeze_batchnorm_layers(self.model)

                i = 1
                while i <= niters:
                    for inp_image, inp_label in self.train_loader:

                        if p_cutmix > 0:
                            if i == 1 and self.step == 1:
                                self.logger.log('Using Cutmix with prob: {}'.format(p_cutmix))
                            inp_image_2, inp_label_2 = next(iter(self.train_loader))
                            inp_image, inp_label, _ = dp.cutmix_images(inp_image, inp_image_2, inp_label, inp_label_2,
                                                                       p=p_cutmix)

                        inp_image = inp_image.cuda()
                        inp_label = inp_label.cuda()

                        out_logits = self.model(inp_image)
                        out_probs = out_logits.softmax(1)

                        self.optimizer.zero_grad()
                        if 'CE' in losses_set:
                            if i == 1 and self.step == 1:
                                self.logger.log('Computing CE loss function')
                            CE = self.CE_criterion(out_logits, inp_label[:, 0])
                            CE.backward(retain_graph=True)
                        else:
                            CE = self.CE_criterion(out_logits, inp_label[:, 0])

                        if 'dice' in losses_set:
                            if i == 1 and self.step == 1:
                                self.logger.log('Computing dice loss function')
                            dice = th.compute_diceloss(out_probs, inp_label, n_classes=self.n_classes)
                            dice.backward(retain_graph=True)

                        if 'miou' in losses_set:
                            if i == 1 and self.step == 1:
                                self.logger.log('Computing miou loss function')
                            miou = th.compute_miouloss(out_probs, inp_label, n_classes=self.n_classes)
                            miou.backward()

                        self.optimizer.step()
                        for name in self.train_metrics.names:
                            self.train_metrics.update_iteration_metric(name, eval(name).item())

                        if self.policy == 'one_cycle':
                            self.scheduler.step()

                        i += 1
                        if i >= niters:
                            break

                self.model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    th.update_setmetrics(self.train_metrics, self.dummy_loader, self.model, n_classes=self.n_classes,
                                         ix_nolabel=self.ix_nolabel)
                    self.logger.log('TRAINSET -- exp:{} - step {}\t {}'.
                                    format(self.i_exp, self.step, self.train_metrics.get_last_metrics()))
                    if self.fl_plot:
                        for i in [0, 2]:
                            dp.display_training_examples_supervised(inp_image, inp_label,
                                                                    out_probs.argmax(1, keepdim=True),
                                                                    self.label_colorizer, i=i, use_display=True)

                    # *****************************************
                    # Validation set cycle
                    for inp_image, inp_label in self.val_loader:
                        inp_image = inp_image.cuda()
                        inp_label = inp_label.cuda()

                        out_logits = self.model(inp_image)
                        out_probs = out_logits.softmax(1)

                        if 'CE' in losses_set:
                            CE = self.CE_criterion(out_logits, inp_label[:, 0])

                        if 'dice' in losses_set:
                            dice = th.compute_diceloss(out_probs, inp_label, n_classes=self.n_classes)

                        if 'miou' in losses_set:
                            miou = th.compute_miouloss(out_probs, inp_label, n_classes=self.n_classes)

                        total = 0
                        for lname in losses_set:
                            total += eval(lname)

                        for name in self.val_metrics.names:
                            self.val_metrics.update_iteration_metric(name, eval(name).item())

                        if fl_fasttest:
                            break

                    th.update_setmetrics(self.val_metrics, self.val_loader, self.model, n_classes=self.n_classes,
                                         ix_nolabel=self.ix_nolabel)
                    self.logger.log('VALSET ---- exp:{} - step {}\t {}'.
                                    format(self.i_exp, self.step, self.val_metrics.get_last_metrics()))
                    if self.fl_plot:
                        for i in [0, 2]:
                            dp.display_training_examples_supervised(inp_image, inp_label,
                                                                    out_probs.argmax(1, keepdim=True),
                                                                    self.label_colorizer, i=i, use_display=True)

                    # Track testset deemed for analysis
                    if self.test_loader is not None:
                        th.update_setmetrics(self.test_metrics, self.test_loader, self.model, n_classes=self.n_classes,
                                             ix_nolabel=self.ix_nolabel)
                        self.logger.log('TESTSET --- exp:{} - step {}\t {}'.
                                        format(self.i_exp, self.step, self.test_metrics.get_last_metrics()))

                    # *****************************************
                    # Log updates
                    for m_name, m_dict in logset_metrics:
                        for m_name_k in m_dict.step.keys():
                            self.writer.add_scalars(m_name_k, {m_name: m_dict.step[m_name_k][-1]}, self.step)
                            self.writer.flush()

                    if fl_savemodel:
                        self.save_ckpt(os.path.join(self.ckpt_dir, 'model.last.pth'))
                        for metric in ['miou']:
                            if (self.val_metrics.step[metric][-1] >= self.val_metrics.best_values[metric]):
                                self.save_ckpt(os.path.join(self.ckpt_dir, 'model.best.pth'))

                    if fl_save_logimage:
                        if self.step == 1:
                            for tag, imgs in self.logset_images + self.logset_masks:
                                self.writer.add_image(tag, tv.utils.make_grid(imgs), self.step)
                                self.writer.flush()

                        if self.step <= 30 or self.step % 10 == 0:
                            for tag, im in self.logset_images:
                                pred = self.model(dp.augs.normalize(im).cuda()).argmax(1, keepdim=True)
                                pred = self.label_colorizer(pred.cpu())
                                grid = tv.utils.make_grid(pred)
                                self.writer.add_image(tag.replace('_image', '_pred'), grid, self.step)
                                self.writer.flush()

                                im_overlay = th.overlay_pred(im, pred, self.label_colorizer)
                                grid = tv.utils.make_grid(im_overlay)
                                self.writer.add_image(tag.replace('_image', '_overlay'), grid, self.step)
                                self.writer.flush()

                self.writer.flush()

                if self.scheduler is not None and self.policy != 'one_cycle':
                    self.scheduler.step()

            self.writer.close()


class TrainerClass(Trainer):
    def __init__(self, ckpt_dirbase=None, void_classes=ut.void_classes['RTK'], n_classes=13, ix_nolabel=255,
                 fl_plot=False, bs_train=4, bs_val=14, bs_test=14, train_path=None, train_eval_path=None,
                 train_dummy_path=None, val_path=None, val_dummy_path=None, test_path=None, fl_classes_weights=False,
                 fl_clsweight_attenuate=False, case_n=None, fl_focal=False, aug_type=None, crop_size=(224, 224),
                 scale_area=(.78, 2), **kwargs):
        super().__init__()

        assert ckpt_dirbase is not None, 'Define ckpt_dirbase before training'
        # The Trainer counts with only the number of tidy class, whereas the `DS` classes counts with the raw number.
        self.n_classes = n_classes - len(void_classes)
        self.ix_nolabel = ix_nolabel
        self.label_colorizer = ut.LabelColorizerWithBg(n_classes=n_classes)
        self.aug_type = aug_type
        self.fl_plot = fl_plot

        # *****************************************
        # Load feeder data
        if len(void_classes) > 0:
            DS = partial(dp.DatasetWithRelabel, void_classes=void_classes, n_classes=n_classes)
        else:
            DS = dp.SimpleDataset
        self.val_ds = DS(annotation_file=val_path)
        self.bs_val = bs_val
        assert len(self.val_ds) % self.bs_val == 0, "batch size of val must be compatible with number of val samples"
        self.val_loader = DataLoader(self.val_ds, batch_size=self.bs_val, pin_memory=True, shuffle=False)

        if test_path is not None:
            self.test_ds = DS(annotation_file=test_path)
            self.bs_test = bs_test
            assert len(self.test_ds) % self.bs_test == 0, \
                "batch size of test must be compatible with number of val samples"
            self.test_loader = DataLoader(self.test_ds, batch_size=self.bs_test, pin_memory=True)
        else:
            self.test_loader = None

        train_eval_path = train_eval_path if train_eval_path is not None else train_path
        self.dummy_ds = DS(annotation_file=train_eval_path)
        self.dummy_loader = DataLoader(self.dummy_ds, batch_size=bs_val, pin_memory=True, shuffle=False, drop_last=True)
        if self.fl_plot:
            # *****************************************
            # Load sample images for logging
            train_dummy_img, train_dummy_mask = th.read_dummy_images(train_dummy_path, self.label_colorizer)
            val_dummy_img, val_dummy_mask = th.read_dummy_images(val_dummy_path, self.label_colorizer)

            dp.plot_grid(train_dummy_img)
            dp.plot_grid(train_dummy_mask)
            dp.plot_grid(val_dummy_img)
            dp.plot_grid(val_dummy_mask)

            self.logset_images = [('#train_image', train_dummy_img), ('#val_image', val_dummy_img)]
            self.logset_masks = [('#train_mask', train_dummy_mask), ('#val_mask', val_dummy_mask)]

        self.logger.dataset_header.append('\n\n\n........ Starting new running .................\n')
        self.logger.dataset_header.append('Batch size - train: {}'.format(bs_train))
        self.logger.dataset_header.append('Batch size -   val: {}'.format(bs_val))

        # *****************************************
        # Start up reusable variables
        if fl_focal:
            self.logger.dataset_header.append('Set Focal loss')
            CE_loss = th.FocalLoss
        else:
            CE_loss = torch.nn.CrossEntropyLoss

        # It is only set for RTK, it would need to generalize code other datasets.
        if fl_classes_weights:
            self.classes_weight = ut.get_classweight_dict()[case_n]
            self.logger.dataset_header.append('classes_weights: {}'.format(self.classes_weight))
            self.classes_weight = torch.Tensor(self.classes_weight).cuda()
            if fl_clsweight_attenuate:
                self.classes_weight = torch.pow(self.classes_weight, 2 / 3)
            self.CE_criterion = CE_loss(reduction='mean', ignore_index=ix_nolabel, weight=self.classes_weight)
        else:
            self.logger.dataset_header.append('classes_weights: False')
            self.CE_criterion = CE_loss(reduction='mean', ignore_index=ix_nolabel)

        # *****************************************
        # Load training dataloader
        self.logger.dataset_header.append('Aug Type: {}'.format(aug_type))
        self.train_ds = DS(annotation_file=train_path)
        if aug_type == 'color':
            self.logger.dataset_header.append('Load dataset with AugType: Color')
            self.train_ds.transform_color = dp.AugColor().input
        elif aug_type == 'crop':
            self.logger.dataset_header.append('Load dataset with AugType: Crop')
            T_crop = T.Compose([T.RandomCrop(size=crop_size), T.RandomHorizontalFlip(p=.5)])
            self.train_ds.transform = T_crop
            self.train_ds.transform_target = T_crop
        elif aug_type == 'crop_color':
            self.logger.dataset_header.append('Load dataset with AugType: Crop&Color')
            T_crop = T.Compose([T.RandomCrop(size=crop_size), T.RandomHorizontalFlip(p=.5)])
            self.train_ds.transform = T_crop
            self.train_ds.transform_target = T_crop
            self.train_ds.transform_color = dp.AugColor().input
        elif aug_type == 'mmseg':
            self.logger.dataset_header.append('Load dataset with AugType: MMSeg')
            aug_mmseg = dp.AugMMSeg(crop_size=crop_size, scales=scale_area)
            self.train_ds.transform = aug_mmseg.input
            self.train_ds.transform_target = aug_mmseg.target
        elif aug_type == 'mmseg_color':
            self.logger.dataset_header.append('Load dataset with AugType: MMSeg&Color')
            aug_mmseg = dp.AugMMSeg(crop_size=crop_size, scales=scale_area)
            self.train_ds.transform = aug_mmseg.input
            self.train_ds.transform_target = aug_mmseg.target
            self.train_ds.transform_color = dp.AugColor().input
        elif aug_type == 'geomRTK':
            self.logger.dataset_header.append('Load dataset with AugType: GeomRTK')
            aug_geom = dp.AugGeometry(p_crop=0, p_affine=0, p_perspective=.5, distortion_scale=.2)
            self.train_ds.transform = aug_geom.input
            self.train_ds.transform_target = aug_geom.target
        elif aug_type == 'geom':
            self.logger.dataset_header.append('Load dataset with AugType: Geometry')
            aug_geom = dp.AugGeometry(**kwargs)
            self.train_ds.transform = aug_geom.input
            self.train_ds.transform_target = aug_geom.target
        else:
            self.logger.dataset_header.append('Load dataset with AugType: None')

        self.bs_train = bs_train
        self.train_loader = DataLoader(self.train_ds, batch_size=self.bs_train, pin_memory=True, shuffle=True,
                                       drop_last=True)
        self.ckpt_dirbase = ckpt_dirbase


def load_model(modelpath, tr_params=None, ds_params=None, unet_encoder='resnet34', use_cpu=False, strict=False):
    n_classes = ds_params['n_classes'] - len(ds_params['void_classes'])
    if 'DeepLab' in modelpath:
        deeplabv3_method = re.search(r'.*(/DeepLab[-a-zA-Z0-9]+/).*', modelpath).group(1)
        deeplabv3_method = deeplabv3_method.replace('/', '').replace('-', '_').lower()
        model = eval(deeplabv3_method)(**tr_params, num_classes=n_classes)
    else:
        model = smp.Unet(encoder_name=unet_encoder, encoder_weights='imagenet', in_channels=3, classes=n_classes,
                         fl_maxpool=tr_params['fl_maxpool'])

    if use_cpu:
        model.load_state_dict(torch.load(modelpath, map_location='cpu')['model_state'], strict=strict)
    else:
        model.load_state_dict(torch.load(modelpath)['model_state'], strict=strict)
        model.cuda()

    model.eval()
    return model


def load_deeplab_model(modelpath, tr_params=None, ds_params=None, encoder='resnet34', use_cpu=False, strict=False):
    n_classes = ds_params['n_classes'] - len(ds_params['void_classes'])
    model = eval('deeplabv3plus_{}'.format(encoder))(**tr_params, num_classes=n_classes)

    if use_cpu:
        model.load_state_dict(torch.load(modelpath, map_location='cpu')['model_state'], strict=strict)
    else:
        model.load_state_dict(torch.load(modelpath)['model_state'], strict=strict)
        model.cuda()

    model.eval()
    return model
