
# Performance Increment Strategy for Semantic Segmentation

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11gFSR4SUHplTYNniChR6EV1SK0__ioQG?usp=sharing)

This the official code repo for the paper `A Performance Increment Strategy for Semantic Segmentation of Low-Resolution Images from Damaged Roads`. 

The code contains many utilities to test different training setups like loss functions, augmentation, and optimizers. Moreover, there are further modifications on the ResNet and DeepLabV3+ architectures on the forked submodules [smp_main](./smp_main) and [deeplabv3](./deeplabv3). Although this code looks like a framework, it was not designed to be so; thus, the code may not be tidy.

A tutorial on Colab is available [here](https://colab.research.google.com/drive/11gFSR4SUHplTYNniChR6EV1SK0__ioQG?usp=sharing).

## Start a training

To start, you just need to fill two dictionaries, one for the dataset options and another for the training options, with the setup information like the example below. 

```
import training as tr

ds_params = {
    'ckpt_dirbase': '/dirsave/'

    'DS': 'RTKDataset',
    'n_classes': 12,
    'ix_nolabel': 255,
    'fl_plot': False,

    'bs_train': 8,
    'bs_val': 8,
    'bs_test': 8,

    'train_path': 'RTK_pisss/labeled_list_classic_n561.txt',
    'train_eval_path': 'RTK_pisss/train_eval.txt',
    'train_dummy_path': 'RTK_pisss/train_dummy.txt',
    'val_path': 'RTK_pisss/labeled_list_classic_val.txt',
    'val_dummy_path': 'RTK_pisss/val_dummy.txt',
    'test_path': None,

    'fl_classes_weights': False,
    'fl_clsweight_attenuate': False,
    'case_n': 'classic_n561',

    'fl_focal': False,
    'aug_type': 'crop',
    'crop_size': (224, 224),
    'scale_area': None
}


tr_params = {
    'fl_resume': False,
    'fl_force': True,
    'fl_fasttest': True,
    'fl_save_logimage': False,

    'nexps': 1,
    'start_exp': 0
    'niters': 1000,
    'nsteps': 200,

    'model_name':  'DeepLabV3Plus',
    'encoder_name': 'resnet50',

    'optim': 'adam',
    'sameLR': True,
    'max_lr_sgd': 1e-2,
    'min_lr_sgd': None,
    'lr_adam': 1e-4,
    'fl_warmup': True,
    'policy': None
    'fl_freeze': False,

    'fl_stemstride': True,
    'fl_richstem': False,
    'fl_parallelstem': False,
    'fl_maxpool': False,
    'fl_lfe': False,
    'fl_transpose': False,
    'fl_transpose_odd': False,

    'output_stride': 16,
    'p_cutmix': 0,
    'losses_set': ['CE', 'dice'],
}


tr.TrainerClass(**ds_params).run(**tr_params)
```

It is not necessary to fill all dictionary options as most of them you will use the default values present at `training.py`. A shorter initiation would be:

```
ds_params = {
    'ckpt_dirbase': './savedir/',
    'bs_train': 8,
    'aug_type': 'geomRTK',
    'train_path': 'RTK_pisss/labeled_list_classic_n561.txt',
    'val_path': 'RTK_pisss/labeled_list_classic_val.txt',
}

tr_params = {
    'nexps': 1,
    'niters': 1000,
    'nsteps': 7,
    'model_name':  'Unet',
    'encoder_name': 'resnet34',
    'optim': 'adam',
    'policy': 'one_cycle',
    'losses_set': ['CE'],
}

tr.TrainerRTK(**ds_params).run(**tr_params)
```

## Training Dictionaries

Both dictionaries present many options; some only work for specific situations, and others have a closed set of options to choose from. The section bellow explains them:

### `ds_params`

+ `ckpt_dirbase`: [str] - the directory path to save all training artifacts.
+ `DS`: [str] - the dataset *Class* to be used, it should be defined in `datapipe.py`; examples [here](./datapipe.py#L212) and [here](./datapipe.py#L96).
+ `n_classes`: [int] - number of classes expected.
+ `ix_nolabel`: [int] - id of the class to be ignored.
+ `fl_plot`: [bool] - to plot on tensorboard log the filepaths listed in the `train_dummy_path` and `val_dummy_path`.
+ `bs_train`: [int] - training batch size.
+ `bs_val`: [int] - validation batch size.
+ `bs_test`: [int] - test batch size, it only needed if there is any `test_path` file.
+ `train_path`: [str] - filepath that lists the images for training.
+ `train_eval_path`: [str] - filepath that lists the images for training evaluation; it matters when you have a large training set and do not want to waste too much time to evaluate `mIoU` on the training set. If no value is assigned for it, it uses the `train_path` file.
+ `train_dummy_path`: [str] - filepath that lists the images to be plotted on the tensorboard log.
+ `val_path`: [str] - filepath that lists the images for validation.
+ `val_dummy_path`: [str] - filepath that lists the images to be plotted on the tensorboard log.
+ `test_path`: [str] - filepath that lists the images for testing.
+ `fl_classes_weights`: [bool] - it should only be True when using WCE, and is only valid for the `RTKDataset` class; see [here](./utils.py#L56).
+ `fl_clsweight_attenuate`: [bool] - it is one workaround to avoid over-represent the underrepresented classes, such exponentially attenuate the weight classes; see [here](./training.py#L432).
+ `case_n`: [str] - this parameter is only valid  for the `RTKDataset` class; see [here](./utils.py#L56).
+ `fl_focal`: [bool] - use this adapted version of cross-entropy.
+ `aug_type`: [str] - the type of augumentation to apply; it should be declared [here](./training.py#L442).
+ `crop_size`: [tuple(int, int)] - crop size only used when `crop` is choice as augmentation.
+ `scale_area`: [tuple(float, float)] - lower and upper boundaries for scaling; it is only used when `resizing` is chosen as augmentation. 

### `tr_params`

+ `fl_resume`: [bool] - set True when you want to start the training from the last checkpoint save in `ckpt_dirbase`.
+ `fl_force`: [bool] - set True when you want to ignore the previous file saved in `ckpt_dirbase` and start from scratch; `WARNING` the new training overwrites the old files.
+ `fl_fasttest`: [bool] - set True to run a single iteration of a single to test if new code changes work.
+ `fl_save_logimage`: [bool] - set True to save the images listed on the dummy files on the tensorboard log.
+ `nexps`: [int] - number of experiments to run.
+ `start_exp`: [int] - the index number to start counting the experiment id.
+ `niters`: [int] - number of iterations between evaluations.
+ `nsteps`: [int] - number of steps.
+ `model_name`:  [str] - any model name available in [Segmentation Models](https://smp.readthedocs.io/en/latest/models.html) or `DeepLabV3Plus`.
+ `encoder_name`: [str] - any model name available in [Segmentation Encoders](https://smp.readthedocs.io/en/latest/encoders.html) or [`resnet34`, `resnet50`, `resnet101`].
+ `optim`: [str] - `adam` or `sgd`.
+ `sameLR`: [bool] - set False when you want to train the encoder with a learning rate 10 times lower than the decoder one.
+ `max_lr_sgd`: [float] - the learning rate used when `optim` is `sgd`.
+ `min_lr_sgd`: [float] - it is only used when `policy` is `poly` and it is optional; if it is not set any value, the `min_lr_sgd` is 100x lower than `max_lr_sgd`.
+ `lr_adam`: [float] - the learning rate used for them `adam` optimizer.
+ `fl_warmup`: [bool] - only used when `policy` is `poly`; set True when you want to start training slowly with a very low learning rate that gradually increases to `max_lr_sgd`.
+ `policy`: [str] - the available policies are [`poly`, `linear`, `one_cycle`, None], see [here](./training.py#L56); Use `None` to train with a constant learning rate.
+ `fl_freeze`: [bool] - set True to do not train the batch normalization parameters.
+ `fl_stemstride`: [bool] - set False to avoid the first ResNet stride; it only works for `DeepLabV3Plus`.
+ `fl_richstem`: [bool] - set True to use a ResNet with a stem attached to a parallel path with ten convolutional layers.
+ `fl_parallelstem`: [bool] - set True to use a ResNet with a stem attached to a parallel path with two convolutional layers.
+ `fl_maxpool`: [bool] - set True to avoid the ResNet max-pooling layer that occurs just after the stem block.
+ `fl_lfe`: [bool] - set True to use the ResNet convolutional blocks with Hybrid Local Feature Extractor (HLFE) rates; it only works for `DeepLabV3Plus`.
+ `fl_transpose`: [bool] - set True to use transposed convolution instead of interpolation upsampling; it only works for `DeepLabV3Plus`.
+ `fl_transpose_odd`: [bool] - set True in case coarse feature map presents a dimension of odd number; it is only needed when tranposed convolution outcomes mismatch the expected dimension size, and it only works for `DeepLabV3Plus`.
+ `output_stride`: [int] - it is a `DeepLabV3Plus` parameters; it only works for the values of `[4, 8, 16, 32]`.
+ `p_cutmix`: [float] - set a value between [0, 1] to apply cutmix during training. 
+ `losses_set`: [list[str]] - a list assigning which losses should be used for training; it allows a permutation combination between `['CE', 'dice', 'miou']`. 


