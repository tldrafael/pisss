import re
import logging
from logging.handlers import RotatingFileHandler
import time
import numpy as np
import seaborn as sns
import torch


rtk_classnames = [(0, 'background'), (1, 'roadAsphalt'), (2, 'roadPaved'), (3, 'roadUnpaved'), (4, 'roadMarking'),
                  (5, 'speedBump'), (6, 'catsEye'), (7, 'stormDrain'), (8, 'manholeCover'), (9, 'patch'),
                  (10, 'waterPuddle'), (11, 'pothole'), (12, 'crack')]

rtk_classnames_tidy = [(0, 'background'), (1, 'roadAsphalt'), (2, 'roadPaved'), (3, 'roadUnpaved'), (4, 'roadMarking'),
                       (5, 'speedBump'), (6, 'catsEye'), (7, 'stormDrain'), (8, 'patch'), (9, 'waterPuddle'),
                       (10, 'pothole'), (11, 'crack')]

void_classes = {'Cityscapes': [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34],
                'RTK': [8],
                'CDU': [5, 9]
               }

class Logger():
    def __init__(self):
        self.root = logging.getLogger()
        self.root.setLevel(logging.DEBUG)
        self.dataset_header = []

        if self.root.hasHandlers():
            self.root.handlers.clear()

        self.handlers = [logging.StreamHandler(),
                         RotatingFileHandler('Trainer.log', mode='a', maxBytes=50e6, backupCount=2)]
        self.formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        for h in self.handlers:
            h.setFormatter(self.formatter)
            h.setLevel(logging.DEBUG)
            self.root.addHandler(h)

    def log(self, msg, level=logging.INFO):
        self.root.log(level, msg)

    def add_handler(self, logpath, fl_resume=False):
        wmode = 'a' if fl_resume else 'w'
        handler = RotatingFileHandler(logpath, mode=wmode, maxBytes=50e6, backupCount=2)
        handler.setFormatter(self.formatter)
        handler.setLevel(logging.DEBUG)
        self.root.addHandler(handler)

    def log_header(self):
        for msg in self.dataset_header:
            self.log(msg)


def get_classweight_fromDS(ds, n_classes=12):
    classes_counts = [0 for _ in range(n_classes)]
    for _, l in ds:
        for c, ct in zip(*np.unique(l, return_counts=True)):
            classes_counts[c] += ct

    classes_counts = np.array(classes_counts)
    classes_weights = classes_counts.max() / (classes_counts + 1e-8)
    return classes_weights.round().astype(int)


def get_classweight_dict():
    dict_classesweight = {'n070': [1, 5, 7, 5, 86,  733, 2116, 5103, 1203,  1446, 17373, 187],
                          'n140': [1, 5, 6, 6, 93,  550, 2943, 6077,  366,  1279, 2108,  181],
                          'n280': [1, 5, 6, 6, 97, 1058, 2799, 4553,  372,  1439, 1308,  186],
                          'n420': [1, 5, 6, 6, 88,  851, 3020, 5105,  256,  1979, 1097,  186],
                          'val' : [1, 4, 6, 7, 84, 8813, 5778, 3245,  273, 16959,  642,  172],
                          'test': [1, 5, 5, 7, 72, 1302, 4567, 2465,  629,  2210, 2022,  319],
                          'all':  [1, 5, 6, 7, 84, 1138, 3615, 3848,  295,  2412, 1045,  200],
                          'rtk':  [1, 5, 6, 7, 75, 1000, 3100, 3300,  270,  2200, 1000,  180],
                          'classic_n561': [1, 5, 6, 7, 87, 1736, 3590, 3793, 292, 2410, 1084, 219],
                          'classic_val':  [1, 5, 7, 7, 75, 1042, 3699, 4065, 308, 2406,  909, 149],
                          }
    # for k, v in dict_classesweight.items():
    #     v.extend([0])
    return dict_classesweight


def keepNB_awake():
    while True:
        time.sleep(5)


def get_tensor_deciles(x):
    return np.quantile(x.cpu().detach(), np.linspace(0, 1, 11))


class LabelColorizer:
    def __init__(self, n_classes=12, ix_nolabel=255):
        assert isinstance(ix_nolabel, (int, type(None)))
        self.ix_nolabel = ix_nolabel
        self.n_classes = n_classes
        self.map = self.get_pallete()

    def get_pallete(self):
        pal = sns.color_palette(palette='gist_rainbow', as_cmap=True)(np.linspace(0, 1, self.n_classes))[..., :3]
        pal = np.vstack([[[0, 0, 0]], pal])
        dict_pal = {}
        for i in range(1, pal.shape[0]):
            dict_pal[i - 1] = torch.tensor(pal[i])

        if self.ix_nolabel is not None:
            dict_pal[self.ix_nolabel] = torch.tensor(pal[0])
        return dict_pal

    def __call__(self, mask):
        fl_single = False
        if len(mask.shape) < 4:
            fl_single = True
            mask = mask[None]

        bs = mask.shape[0]
        cmask = torch.zeros((bs, 3,) + mask.shape[2:]).type(self.map[1].dtype)
        for i in range(bs):
            for k in self.map.keys():
                cmask[i, :, mask[i, 0] == k] = self.map[k][:, None]

        if fl_single:
            cmask = cmask[0]
        return cmask

    def reverse(self, x):
        mask_new = torch.zeros(x.shape[1:])
        for k, v in self.map.items():
            k_pos = torch.all(torch.eq(x.float(), torch.Tensor(v)[:, None, None]), axis=0)
            mask_new[k_pos] = k


class LabelColorizerWithBg(LabelColorizer):
    def __init__(self, n_classes=12, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.map = self.get_pallete()

    def get_pallete(self):
        pal = sns.color_palette(palette='gist_rainbow', as_cmap=True)(np.linspace(0, 1, self.n_classes - 1))[..., :3]
        pal = np.vstack([[[0, 0, 0]], pal])
        dict_pal = {}
        for i in range(pal.shape[0]):
            dict_pal[i] = torch.tensor(pal[i])
        return dict_pal


def map_mask2multidimensional(mask, n_classes):
    mask_new = torch.zeros((n_classes,) + mask.shape[1:], dtype=torch.float)
    for k in np.arange(n_classes):
        mask_new[k] = (mask == k) * 1.
    return mask_new


def turn_image_float2int(x):
    return x.mul(255).to(torch.int)


def set_randomseed(seed=None, return_seed=False):
    if seed is None:
        seed = np.random.randint(2147483647)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if return_seed:
        return seed


def print_all_layers_names(model):
    lnames = []
    for n, m in model.named_modules():
        lnames.append(n)
    return lnames


def sanity_check(ds_params, tr_params, ckpt_dirbase):
    msg = 'Ensure the LOSS is correct in the ckpt_dirbase'
    if re.search('/CE', ckpt_dirbase):
        assert not ds_params['fl_classes_weights'], msg
    elif re.search('/WCE', ckpt_dirbase):
        assert ds_params['fl_classes_weights'], msg
    elif not re.search('/(W)*CE', ckpt_dirbase):
        pass
    else:
        raise Exception(msg)

    msg = 'Ensure the OS is correct in the ckpt_dirbase'
    if re.search('OS', ckpt_dirbase):
        stride = int(re.search(r'OS([0-9]+)', ckpt_dirbase).group(1))
        assert stride == tr_params['output_stride'], msg

    if re.search('Dice', ckpt_dirbase):
        assert 'dice' in tr_params['losses_set'], msg

    if re.search('Miou', ckpt_dirbase):
        assert 'miou' in tr_params['losses_set'], msg

    msg = 'Ensure the LR is correct in the ckpt_dirbase'
    if re.search('encoderLR', ckpt_dirbase):
        assert not tr_params['sameLR'], msg
    else:
        assert tr_params['sameLR'], msg

    msg = 'Ensure the Maxpooling is correct in the ckpt_dirbase'
    if re.search('woMaxPool', ckpt_dirbase):
        assert not tr_params['fl_maxpool'], msg
    else:
        assert tr_params['fl_maxpool'], msg

    msg = 'Ensure the richStem is correct in the ckpt_dirbase'
    if re.search('richStem', ckpt_dirbase):
        assert tr_params['fl_richstem'], msg
    else:
        assert not tr_params['fl_richstem'], msg

    msg = 'Ensure the parallelStem is correct in the ckpt_dirbase'
    if re.search('parallelStem', ckpt_dirbase):
        assert tr_params['fl_parallelstem'], msg
    else:
        assert not tr_params['fl_parallelstem'], msg

    msg = 'Ensure the convT is correct in the ckpt_dirbase'
    if re.search('convT', ckpt_dirbase):
        assert tr_params['fl_transpose'], msg
    else:
        assert not tr_params['fl_transpose'], msg

    aug_name = ckpt_dirbase.split('/')[-2].split('.')[2]
    if 'cutmix' in aug_name or 'p_cutmix' in tr_params.keys():
        assert tr_params['p_cutmix'] > 0, 'Ensure p_cutmix is greater than 0'
        p = int(re.search(r'cutmix([0-9]+)', aug_name).group(1)) / 100
        assert tr_params['p_cutmix'] == p, 'Ensure p_cutmix is equal than the ckpt_dirbase'
        # Remove cutmix from aug_name to follow the next operation
        aug_name = re.search(r'([a-z]+)(_cutmix.*)$', aug_name).group(1)
    aug_name = None if aug_name in ['none'] else aug_name

    if tr_params['optim'] != 'adam':
        optim_name = ckpt_dirbase.split('/')[-2].split('.')[3]
        assert optim_name == tr_params['optim'], 'Ensure optimizer name is correct in the ckpt_dirbase'
