import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


import os
from PIL import Image

class DataGenerator(Dataset):
    def __init__(self, data_dir='./output/', data_type='train', to_fit=True, batch_size=16,
                 height=320, width=320, n_channels=1, n_classes=2, label_type='LVC',
                 shuffle=True, custom_keys=[], data_file='ds.json', validation_filter=False):
        self.data_dir = data_dir
        self.data_file = data_file
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (height, width)
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.custom_keys = custom_keys
        self.label_type = label_type
        self.dataset = self._get_data(data_type, validation_filter)
        self.epoch_count = 0
        self.on_epoch_end()

        # Transformation to resize and normalize images
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
        #     transforms.Lambda(lambda x: (x * 2) - 1),  # Normalize to [-1, 1]
        #     transforms.Lambda(lambda x: x[:1, :, :])  # Select the first channel
        # ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Generate X data
        X = self._generate_X(index).astype(np.float32)
        X = torch.from_numpy(X)[:, :, :1].permute(2, 0, 1)

        if self.to_fit:
            y = self._generate_y(index).astype(np.float32)
            y = torch.from_numpy(y)[:, :, :].permute(2, 0, 1)

            return { 'inputs': X, 'labels': y }
        else:
            return { 'inputs': X }

    def _generate_y(self, index):
        # Generate data
        img = Image.open(self.dataset[index]['mask'], 'r')
        mask_image = np.array(img) / 255.

        mask = self._mask_image_to_class(mask_image)
        return self._convert_seg_image_to_one_hot_encoding(mask)

    # Generates a batch of X data
    def _generate_X(self, index):
        return (np.array(Image.open(self.dataset[index]['data'], 'r')) / 127.5 - 1)

 # Returns one batch of indexes
    def _get_batch_indexes(self, index):
        indexes = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
        return indexes

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.shuffle:
            np.random.shuffle(self.dataset)

    def _convert_seg_image_to_one_hot_encoding(self, image):
        classes = np.arange(self.n_classes)
        out_image = np.zeros(list(image.shape) + [len(classes)], dtype=np.float32)

        for i, c in enumerate(classes):
            x = np.zeros((len(classes)))
            x[i] = 1
            out_image[image == c] = x

        return out_image
    def _get_data(self, data_type, validation_filter):
        if data_type == 'test':
            return self._get_test_data()
        else:
            return self._get_myosaiq_data(data_type, validation_filter)

    def _get_myosaiq_data(self, data_type, validation_filter):
        dataset = []

        with open(self.data_dir + self.data_file) as json_file:
            ds = json.load(json_file)
        if data_type == 'custom':
            keys = self.custom_keys
        else:
            keys = ds.keys()

        for key in keys:
            for inst in ds[str(key)]:
                if (validation_filter):
                    filtered_data = '_00_' in inst['data']
                    if filtered_data:
                        dataset.append({
                            'data': self.data_dir + inst['data'],
                            'mask': self.data_dir + inst['gt']
                        })
                else:
                    dataset.append({
                        'data': self.data_dir + inst['data'],
                        'mask': self.data_dir + inst['gt']
                    })

        if data_type == 'val':
            return dataset[int(len(dataset) * 0.8):]
        elif data_type == 'train':
            return dataset[:int(len(dataset) * 0.8)]
        else:
            return dataset

    def _get_test_data(self):
        raise NotImplementedError()

    def _mask_image_to_class(self, mask):
        # mask = np.expand_dims(mask, axis=-1)

        classes = np.zeros((mask.shape[0], mask.shape[1]))
        idx = np.sum(np.round(mask), axis=-1) == 2
        classes[idx] = 4

        idx = np.sum(np.round(mask), axis=-1) == 1
        classes[idx] = (np.argmax(mask, axis=-1) + 1)[idx]

        if self.label_type == 'LVC':
            classes[classes == 1] = 0
            classes[classes == 2] = 0
            classes[classes == 3] = 1
        elif self.label_type == 'LVM':
            classes[classes == 1] = 0
            classes[classes == 2] = 1
            classes[classes == 3] = 0
        elif self.label_type == 'RVC':
            classes[classes == 1] = 1
            classes[classes == 2] = 1
            classes[classes == 3] = 1

        return classes

# Example usage:
# dataset = DataGenerator(data_type='train', batch_size=16, to_fit=True)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)