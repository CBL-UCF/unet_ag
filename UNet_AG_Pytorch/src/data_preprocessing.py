import re
import glob
import json
import tqdm
import numpy as np
import SimpleITK as sitk

import os
import torch
from PIL import Image
from pathlib import Path
from skimage.transform import resize
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

SEED = 42

DS_PATH = './dataset/training/SCH/'
DS_OUT = './ds/'

NEW_SPACING = [1.5, 1.5]
INPUT_NORM_DIST_PERCENT = 0.95

HEIGHT = 256
WIDTH = 256
N_CHANNELS = 1

N_AUGMENTATION = 10
AUGMENTATION_CONFIG = {
    'rotation': (0, 360),
    'scale': (0.7, 1.3),
    'alpha': (0, 350),
    'sigma': (14, 17),
}

np.random.seed(SEED)

files = glob.glob(DS_PATH + 'images/*.nii')

def input_to_image(input_image):
    input_image = (input_image + 1) / 2
    return np.concatenate([input_image, input_image, input_image], -1)


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:3] + (3,))
    # LVC
    idx = mask == 1
    img[idx] = [1, 0, 0]
    # Healthy LVM
    idx = mask == 2
    img[idx] = [0, 1, 0]
    # RVC
    idx = mask == 3
    img[idx] = [0, 0, 1]
    # # MVO
    # idx = mask == 4
    # img[idx] = [1, 1, 0]
    return img


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def normalize_input(image):
    image = image - image.mean()
    pixels = image.flatten()
    delta_index = int(round(((1 - INPUT_NORM_DIST_PERCENT) / 2) * len(pixels)))
    pixels = np.sort(pixels)
    min = pixels[delta_index]
    max = pixels[-(delta_index + 1)]
    image = 2 * ((image - min) / (max - min)) - 1
    image[image < -1] = -1
    image[image > 1] = 1
    return image


def resize_padding(image, n_channels=None, pad_value=0, expand_dim=False):
    if n_channels is not None:
        data = np.zeros((HEIGHT, WIDTH, n_channels))
    else:
        data = np.zeros((HEIGHT, WIDTH))
    data += pad_value
    h_offest = (HEIGHT - image.shape[1]) // 2
    w_offest = (WIDTH - image.shape[2]) // 2

    t_h_s = max(h_offest, 0)
    t_h_e = t_h_s + min(image.shape[1] + h_offest, image.shape[1]) - max(0, -h_offest)
    t_w_s = max(w_offest, 0)
    t_w_e = t_w_s + min(image.shape[2] + w_offest, image.shape[2]) - max(0, -w_offest)

    s_h_s = max(0, -h_offest)
    s_h_e = s_h_s + t_h_e - t_h_s
    s_w_s = max(0, -w_offest)
    s_w_e = s_w_s + t_w_e - t_w_s

    if expand_dim:
        data[t_h_s:t_h_e, t_w_s:t_w_e] = np.expand_dims(image[s_h_s:s_h_e, s_w_s:s_w_e], axis=-1)
    else:
        data[t_h_s:t_h_e, t_w_s:t_w_e] = image[s_h_s:s_h_e, s_w_s:s_w_e]
    return data


def rotate_image(input_img, label_img):
    rotation = np.random.randint(AUGMENTATION_CONFIG['rotation'][0], AUGMENTATION_CONFIG['rotation'][1] + 1)
    new_input_img = np.zeros(input_img.shape)
    new_label_img = np.zeros(label_img.shape)
    for i in range(input_img.shape[0]):
        new_input_img[i, ] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).rotate(rotation))
        new_label_img[i, ] = np.array(Image.fromarray(label_img[i].astype(np.uint8)).rotate(rotation))
    return new_input_img, new_label_img


def scale_image(input_img, label_img):
    scale = np.random.random() * (AUGMENTATION_CONFIG['scale'][1] - AUGMENTATION_CONFIG['scale'][0]) + AUGMENTATION_CONFIG['scale'][0]
    height = int(scale * input_img.shape[1])
    width = int(scale * input_img.shape[2])
    # print(scale, height, width)
    new_input_img = np.zeros((input_img.shape[0], height, width))
    new_label_img = np.zeros((input_img.shape[0], height, width))
    for i in range(input_img.shape[0]):
        new_input_img[i,] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).resize((width, height), Image.NEAREST))
        new_label_img[i,] = np.array(Image.fromarray(label_img[i].astype(np.uint8)).resize((width, height), Image.NEAREST))
    return input_img, label_img


def elastic_transform(input_img, label_img):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    random_state = np.random.RandomState(None)
    alpha = np.random.randint(AUGMENTATION_CONFIG['alpha'][0], AUGMENTATION_CONFIG['alpha'][1] + 1)
    sigma = np.random.randint(AUGMENTATION_CONFIG['sigma'][0], AUGMENTATION_CONFIG['sigma'][1] + 1)

    shape = input_img.shape[1:]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    for i in range(input_img.shape[0]):
        input_img[i] = map_coordinates(input_img[i], indices, order=1, mode='reflect').reshape(shape)
        label_img[i] = map_coordinates(label_img[i], indices, order=1, mode='reflect').reshape(shape)

    return input_img, label_img


def data_augmentation(X, y):
    augmented_y = []
    for i in range(len(X)):
        X[i] = normalize_input(X[i])

    for i in range(len(y)):
        y_mask = (mask_to_image(y[i])) * 255

        y_mask = np.transpose(y_mask, (0, 3, 1, 2))
        augmented_y.append(y_mask)

    y = np.concatenate(augmented_y, axis=0)

    X_transformed = np.squeeze(X, axis=1)
    X_transformed = np.expand_dims(X_transformed, axis=-1)

    X = input_to_image(X_transformed) * 255

    X = np.transpose(X, (0, 3, 1, 2))

    for i in range(len(X)):
        if np.random.uniform() < 0.5:
            X[i] = np.flip(X[i], 2)
            y[i] = np.flip(y[i], 2)
        if np.random.uniform() < 0.5:
            X[i] = np.flip(X[i], 1)
            y[i] = np.flip(y[i], 1)

    for i in range(len(X)):
        X[i], y[i] = rotate_image(X[i], y[i])
        X[i], y[i] = scale_image(X[i], y[i])

    transformed_X = np.transpose(X, (0, 2, 3, 1))
    transformed_Y = np.transpose(y, (0, 2, 3, 1))

    X, y = elastic_transform(transformed_X, transformed_Y)

    X_new = np.zeros((X.shape[0], HEIGHT, WIDTH, 3), dtype=np.uint8)
    y_new = np.zeros((X.shape[0], HEIGHT, WIDTH, 3), dtype=np.uint8)

    for i in range(len(X)):
        casted_X = np.expand_dims(X[i], axis=0)
        casted_y = np.expand_dims(y[i], axis=0)

        X_new[i] = resize_padding(casted_X, 3)
        y_new[i] = resize_padding(casted_y, 3)

    return X_new, y_new


Path(DS_OUT).mkdir(parents=True, exist_ok=True)

obj_original = {}
obj_augmented = {}

for file in tqdm.tqdm(files):
    pat = re.sub('\D', '', file.split('images')[-1].split('_')[0])
    img_type = file.split('images')[-1].split('_')[-1].split('.')[0]

    if pat not in obj_original:
        obj_original[pat] = []
        obj_augmented[pat] = []

    itk_image = sitk.ReadImage(file)
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    spacing_target = [spacing[0]] + NEW_SPACING
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    image = resize_image(image, spacing, spacing_target, order=3).astype(np.float32)

    itk_label = sitk.ReadImage(file.replace('images', 'labels'))
    label = sitk.GetArrayFromImage(itk_label).astype(float)
    tmp = convert_to_one_hot(label)
    vals = np.unique(label)
    results = []
    for i in range(len(tmp)):
        results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
    label = vals[np.vstack(results).argmax(0)]

    for i, img in enumerate(image):
        is_img_black = np.all(label[i] == 0)
        if (not is_img_black):
            file_name = pat + '_' + img_type + '_' + str(0).zfill(2) + '_' + str(i).zfill(2) + '.png'
            img = normalize_input(img)
            input_img = Image.fromarray((input_to_image(resize_padding(img, N_CHANNELS, -1, True)) * 255).astype(np.uint8))
            input_img.save(DS_OUT + file_name)
            gt_img = Image.fromarray((resize_padding(mask_to_image(label[i]), 3) * 255).astype(np.uint8))
            gt_img.save(DS_OUT + file_name.replace('.png', '_gt.png'))
            obj_original[pat].append({
                'data': file_name,
                'gt': file_name.replace('.png', '_gt.png')
            })
            obj_augmented[pat].append({
                'data': file_name,
                'gt': file_name.replace('.png', '_gt.png')
            })


    for k in range(N_AUGMENTATION):
        inp, gt = data_augmentation(image.copy(), label.copy())
        for i in range(len(inp)):
            is_img_black = np.all(gt[i] == 0)
            if (not is_img_black):
                file_name = pat + '_' + img_type + '_' + str(k + 1).zfill(2) + '_' + str(i).zfill(2) + '.png'
                input_img = Image.fromarray(inp[i])
                input_img.save(DS_OUT + file_name)
                gt_img = Image.fromarray(gt[i])
                gt_img.save(DS_OUT + file_name.replace('.png', '_gt.png'))
                obj_augmented[pat].append({
                    'data': file_name,
                    'gt': file_name.replace('.png', '_gt.png')
                })

with open(DS_OUT + 'ds.json', 'w') as f:
    json.dump(obj_augmented, f)

with open(DS_OUT + 'original-ds.json', 'w') as f:
    json.dump(obj_original, f)

pats = np.array(list(obj_original.keys()))
np.save(DS_OUT + 'pats.npy', pats)
