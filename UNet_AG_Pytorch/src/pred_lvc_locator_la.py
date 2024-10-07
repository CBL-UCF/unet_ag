import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from models import UNet
from data_pytorch import DataGenerator
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn

SEED = 10

MODEL_DIR = './lvc-locator-output-la/'
DATA_DIR = './ds/'
# DATA_DIR = './augmented-ds/'
# DATA_DIR = '../prob_mri_segmentation/ds/'
# DATA_DIR = '../acdc/augmented-ds/'
HEIGHT = 256
WIDTH = 256
N_CHANNELS = 1
N_CLASSES = 2
N_CLASSES_GENERATOR = 2
EPOCHS = 300
BATCH_SIZE = 1
CONTOUR_POINTS = 360
CONTROL_POINTS = 20

torch.manual_seed(SEED)
np.random.seed(SEED)

def create_cnn():
    base_model = UNet(
        n_input_channels=N_CHANNELS,
        input_dim=(HEIGHT, WIDTH),
        n_output_classes=N_CLASSES,
    )
    return base_model

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = create_cnn().to(device)
initialize_weights(model)

if 'VAL_SET' in os.environ:
    VAL_SET = os.environ['VAL_SET']
else:
    VAL_SET = '3'

VAL_SET = '1'
def generate_fold(VAL_SET):
    VAL_SET = int(VAL_SET)
    # Ensure VAL_SET is within the valid range
    if VAL_SET < 1 or VAL_SET > 8:
        raise ValueError("VAL_SET must be between 1 and 8")

    # Define the two groups of patients
    patients_group_1 = np.array([14, 15])
    patients_group_2 = np.array([16, 17])

    # Determine validation patients for each fold
    if VAL_SET <= 7:
        # For folds 1 to 7, include one set from each group
        val_patients_1 = np.array([patients_group_1[VAL_SET - 1]])
        val_patients_2 = np.array([patients_group_2[VAL_SET - 1]])

        # Combine validation patients
        val_patients = np.concatenate((val_patients_1, val_patients_2))
    else:
        # For the 8th fold, include two sets from group 1
        val_patients_1 = patients_group_1[-2:]
        val_patients_2 = np.array([])
        
        #Set val_patients to val_patients_1
        val_patients  = val_patients_1
    
    # Determine training patients by removing validation patients from each group
    train_patients_1 = np.setdiff1d(patients_group_1, val_patients_1)
    train_patients_2 = np.setdiff1d(patients_group_2, val_patients_2) if VAL_SET <= 7 else patients_group_2

    # Combine training patients
    train_patients = np.concatenate((train_patients_1, train_patients_2))

    return val_patients, train_patients

for val_set in range(1, 2):
    VAL_SET = str(val_set)
    val_patients, train_patients = generate_fold(VAL_SET)

    weight_path = MODEL_DIR + VAL_SET + '/best-weights.pth'
    model.load_state_dict(torch.load(weight_path)['model_state_dict'])

    for pat in val_patients:
        generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES_GENERATOR,
                                custom_keys=[pat])
        
        validateLoader = DataLoader(generator, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

        Path(MODEL_DIR + VAL_SET + '/' + str(pat).zfill(3)).mkdir(parents=True, exist_ok=True)

        n = generator.__len__()

        X_stack = np.zeros((n, N_CHANNELS, HEIGHT, WIDTH))
        y_stack = np.zeros((n, N_CLASSES_GENERATOR, HEIGHT, WIDTH))
        y_unet = np.zeros((n, N_CLASSES_GENERATOR,HEIGHT, WIDTH))

        for i in range(n):
            val_ds = generator.__getitem__(i)
            X = val_ds['inputs'].unsqueeze(0).to(device)
            y = val_ds['labels'].unsqueeze(0).to(device)

            y_pred = model(X)

            # print('#### y_pred', y_pred.shape)
            # tensor = y_pred.squeeze(0)

            # for k in range(tensor.shape[0]):
            #     # Convert to numpy array and scale to [0, 255]
            #     img_array = tensor[k].cpu().detach().numpy() 
            #     img_array = img_array * 255  
            #     img_array = img_array.astype(np.uint8) 

            #     # Create a PIL Image and save it
            #     img = Image.fromarray(img_array, mode='L')
            #     img.save(f'./PRED/{pat}_channel_{i}_{k}.png')

            X_stack[i] = X[0].cpu()
            y_stack[i] = y[0].cpu()
            y_unet[i] = y_pred[0].detach().cpu()

        np.save(MODEL_DIR + VAL_SET + '/' + str(pat).zfill(3) + '/' + str(pat).zfill(3) + '_X.npy', X_stack)
        np.save(MODEL_DIR + VAL_SET + '/' + str(pat).zfill(3) + '/' + str(pat).zfill(3) + '_y.npy', y_stack)
        np.save(MODEL_DIR + VAL_SET + '/' + str(pat).zfill(3) + '/' + str(pat).zfill(3) + '_y_unet.npy', y_unet)
