import os
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import Dataset
from types import SimpleNamespace
import matplotlib.pyplot as plt
from IPython import display
import utils as ut


augs = SimpleNamespace()
augs.normalize = T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
augs.normalize_invert = T.Normalize(mean=[-.485/.229, -.456/.224, -.406/.225], std=[1/.229, 1/.224, 1/.225])


class AugMMSeg:
    def __init__(self, crop_size=(512, 1024), scales=(.5, 2)):
        def f_random_resize(x, interpolation=T.InterpolationMode.BILINEAR):
            random_size = torch.rand(1) * (scales[1] - scales[0]) + scales[0]
            shortest_edge = min(x.shape[1:])
            size = torch.floor(shortest_edge * random_size)
            size = size.to(int).item()
            return T.Resize(size=size, interpolation=interpolation)(x)

        self.input = T.Compose([
                            T.Lambda(lambda x: f_random_resize(x)),
                            T.RandomCrop(crop_size),
                            T.RandomHorizontalFlip(p=.5),
                            ])

        self.target = T.Compose([
                            T.Lambda(lambda x: f_random_resize(x, interpolation=T.InterpolationMode.NEAREST)),
                            T.RandomCrop(crop_size),
                            T.RandomHorizontalFlip(p=.5),
                            ])


class AugGeometry:
    def __init__(self, fill=255, p_crop=.2, p_hflip=.5, p_affine=.2, p_perspective=.2, distortion_scale=.15,
                 degrees=(-30, 30), crop_scale=(1/16, 1), crop_ratio=(1., 1.), crop_size=(288, 352), **kwargs):
        self.p_crop = p_crop
        self.p_hflip = p_hflip
        self.p_affine = p_affine
        self.p_perspective = p_perspective
        self.distortion_scale = distortion_scale
        self.degrees = degrees
        self.crop_scale = crop_scale
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.fill = fill

        self.input = T.Compose([
                            T.RandomApply([
                                T.RandomResizedCrop(scale=crop_scale, size=crop_size, ratio=crop_ratio)
                                ], p=p_crop),
                            T.RandomHorizontalFlip(p=p_hflip),
                            T.RandomApply([
                                T.RandomAffine(degrees=degrees)
                            ], p=p_affine),
                            T.RandomPerspective(distortion_scale=distortion_scale, p=p_perspective)
                        ])

        self.target = T.Compose([
                            T.RandomApply([
                                T.RandomResizedCrop(scale=crop_scale, size=crop_size, ratio=crop_ratio,
                                                    interpolation=T.InterpolationMode.NEAREST)
                            ], p=p_crop),
                            T.RandomHorizontalFlip(p=p_hflip),
                            T.RandomApply([
                                T.RandomAffine(degrees=degrees, interpolation=T.InterpolationMode.NEAREST, fill=fill)
                                ], p=p_affine),
                            T.RandomPerspective(distortion_scale=distortion_scale, p=p_perspective,
                                                interpolation=T.InterpolationMode.NEAREST, fill=fill)
                         ])


class AugColor:
    def __init__(self, p_jitter=.2, p_gray=.2, p_contrast=0, p_equalize=0, jitter_default=[.4, .4, .4, .1],
                 jitter_attenuate_factor=1.5, **kwargs):
        self.jitter_attenuate_factor = jitter_attenuate_factor
        self.jitter_params = np.array(jitter_default) / jitter_attenuate_factor
        self.p_jitter = p_jitter
        self.p_gray = p_gray
        self.p_contrast = p_contrast
        self.p_equalize = p_equalize

        self.input = T.Compose([
                           T.RandomApply([T.transforms.ColorJitter(*self.jitter_params)], p=p_jitter),
                           T.RandomGrayscale(p=p_gray),
                           T.RandomAutocontrast(p=p_contrast),
                           T.RandomEqualize(p=p_equalize)
                         ])


class SimpleDataset(Dataset):
    def __init__(self, annotation_file, dirbase=None, has_label=True, transform=None, transform_target=None,
                 transform_color=None, ix_nolabel=255, fl_normalize=True, **kwargs):

        with open(annotation_file, 'r') as f:
            self.impaths = f.read().split('\n')[:-1]

        if dirbase is not None:
            self.impaths = [os.path.join(dirbase, p) for p in self.impaths]

        if has_label:
            self.labelpaths = [p.replace('/image/', '/label/') for p in self.impaths]

        self.dirbase = dirbase
        self.has_label = has_label
        self.transform = transform
        self.transform_color = transform_color
        self.fl_transform_color = transform_color is not None
        self.transform_target = transform_target
        self.ix_nolabel = ix_nolabel
        self.fl_normalize = fl_normalize

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        image = tv.io.read_image(self.impaths[idx])

        if self.fl_transform_color:
            # The color transformation doesn't follow the set seed.
            image = self.transform_color(image)

        if self.transform:
            state = ut.set_randomseed(return_seed=True)
            image = self.transform(image)

        if self.has_label:
            label = tv.io.read_image(self.labelpaths[idx])

            if self.transform:
                ut.set_randomseed(seed=state)
                label = self.transform_target(label).to(torch.uint8)

            label = label.long()
        else:
            label = self.ix_nolabel + torch.zeros(image.shape, dtype=torch.long)

        if self.fl_normalize:
            return augs.normalize(image / 255), label
        else:
            return image, label


def get_classes_remap_dict(ix_nolabel=255, void_classes=[9], n_classes=10):
    dict_labels_remap = {}
    i_new = 0
    for i in range(n_classes):
        if i in void_classes:
            dict_labels_remap[i] = ix_nolabel
        else:
            dict_labels_remap[i] = i_new
            i_new += 1

    array_labels_remap = np.array(list(dict_labels_remap.values()))
    torch_labels_remap = torch.Tensor(array_labels_remap).long()
    return torch_labels_remap


class DatasetWithRelabel(SimpleDataset):
    def __init__(self, void_classes=[], n_classes=10, **kwargs):
        super().__init__(**kwargs)
        assert len(void_classes) > 0, 'Ensure that void_classes has at least one element, or use the class Dataset'
        self.void_classes = void_classes
        self.n_classes = n_classes
        self.labels_remap = get_classes_remap_dict(self.ix_nolabel, void_classes, n_classes)

    def __getitem__(self, idx):
        image = tv.io.read_image(self.impaths[idx])

        if self.fl_transform_color:
            # The color transformation doesn't follow the set seed.
            image = self.transform_color(image)

        if self.transform:
            state = ut.set_randomseed(return_seed=True)
            image = self.transform(image)

        if self.has_label:
            label = tv.io.read_image(self.labelpaths[idx])
            label = self.labels_remap[label.long()]

            if self.transform:
                ut.set_randomseed(seed=state)
                label = self.transform_target(label).to(torch.uint8)

            label = label.long()
        else:
            label = self.ix_nolabel + torch.zeros(image.shape, dtype=torch.long)

        if self.fl_normalize:
            return augs.normalize(image / 255), label
        else:
            return image, label


def get_random_cutmask(im_shape, prop_range=(.5, .5)):
    bs, _, W, H = im_shape
    mask_prop = np.random.uniform(prop_range[0], prop_range[1], size=bs)

    y_prop = np.exp(np.random.uniform(low=0.0, high=1.0, size=bs) * np.log(mask_prop))
    x_prop = mask_prop / y_prop

    sizes = np.array([y_prop * W, x_prop * H])
    positions = np.array([[W], [H]]) - sizes
    positions = positions * np.random.uniform(low=0.0, high=1.0, size=positions.shape)
    rectangles = np.vstack([positions, positions + sizes]).round().astype(int)

    mask = torch.ones((bs, 1, W, H))
    for i in range(bs):
        mask[i, 0, rectangles[0, i]:rectangles[2, i], rectangles[1, i]:rectangles[3, i]] = 0
    return mask


def cutmix_images(images_1, images_2, labels_1=None, labels_2=None, p=.5):
    assert images_1.shape == images_2.shape, 'Ensure batches images have the same shape'

    mix_masks = get_random_cutmask(images_1.shape)
    # Probability to use cutmix, do not apply for all images
    prob_masks = torch.bernoulli((torch.ones((images_1.shape[0], )) * p)).bool()
    mix_masks[~ prob_masks] = 1

    mix_images = mix_masks * images_1 + (1 - mix_masks) * images_2

    mix_labels = None
    if labels_1 is not None and labels_2 is not None:
        assert labels_1.shape == labels_2.shape, 'Ensure batches labels have the same shape'
        mix_labels = mix_masks * labels_1 + (1 - mix_masks) * labels_2

    return mix_images, mix_labels.long(), mix_masks


def plot_grid(grid, use_display=False, is_grid=False, nrows=1, ncols=None, figsize=None):
    if ncols is None:
        if isinstance(grid, list):
            ncols = len(grid)
        else:
            ncols = grid.shape[0]

    if isinstance(grid, np.ndarray):
        grid = torch.Tensor(grid)

    if not is_grid:
        grid = tv.utils.make_grid(grid, nrow=ncols)

    if figsize is None:
        plt.figure(figsize=(4 * ncols, 5 * nrows))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    if use_display:
        display.display(plt.gcf())


def overlay_pred(im, pred, label_colorizer, alpha=.8):
    return alpha * augs.normalize_invert(im) + (1 - alpha) * label_colorizer(pred)


def display_training_examples_supervised(inp_image, inp_label, out_pred, label_colorizer, i=0, **kwargs):
    list_ims = [augs.normalize_invert(inp_image[i]),
                label_colorizer(inp_label[i]),
                label_colorizer(out_pred[i])]
    list_ims = [im.cpu() for im in list_ims]
    plot_grid(list_ims, is_grid=False, **kwargs)


def get_image_from_input_tensor(inp_image, ix=0):
    return augs.normalize_invert(inp_image)[ix].permute(1, 2, 0).detach().cpu().numpy()


def get_input_tensor_from_image(image):
    assert len(image.shape) == 3, 'Image should have 3 axis'
    return augs.normalize(torch.Tensor(image).permute(2, 0, 1)[None])
