import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from models import UNet, cce_dice_loss
import torch
import torch.nn as nn

from data_pytorch import DataGenerator
from torch.utils.data import DataLoader
from PIL import Image

SEED = 10

SRC_MODEL_DIR = './weights/la-weights/locator'
MODEL_DIR = './lvc-locator-output-la/'
DATA_DIR = './ds/'
# DATA_DIR = '../acdc/augmented-ds/'
HEIGHT = 352
WIDTH = 352
N_CHANNELS = 1
N_CLASSES = 2
N_CLASSES_GENERATOR = 2
EPOCHS = 50
BATCH_SIZE = 10
CONTOUR_POINTS = 360
CONTROL_POINTS = 20

torch.manual_seed(SEED)
np.random.seed(SEED)

def create_cnn():
    base_model = UNet(
        n_input_channels=N_CHANNELS,
        input_dim=(HEIGHT, WIDTH),
        n_output_classes=N_CLASSES,
        # seg=True,
        # initializer=initializer
    )
    return base_model

def generate_fold(VAL_SET):
    VAL_SET = int(VAL_SET)
    # Ensure VAL_SET is within the valid range
    if VAL_SET < 1 or VAL_SET > 8:
        raise ValueError("VAL_SET must be between 1 and 8")

    # Define the two groups of patients



    ### convert this to be string with a 0 on the beginning
    patients_group_1 = np.arange(10, 19)  # Patients 10 to 18
    patients_group_2 = np.arange(65, 73)  # Patients 65 to 73
    # patients_group_2 = np.array([1404, 1406, 1434, 1445, 1446, 1492, 1493])  # The numbered patients



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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


model = create_cnn().to(device)
initialize_weights(model)


if 'VAL_SET' in os.environ:
    VAL_SET = os.environ['VAL_SET']
else:
    VAL_SET = '2'

val_patients, train_patients = generate_fold(VAL_SET)

class ModelCheckpoint:
    def __init__(self, monitor='val_loss', filepath='best_model.pth', mode='min', verbose=1):
        self.monitor = monitor
        self.filepath = filepath
        self.mode = mode
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, val_loss):
        if self.mode == 'min':
            if val_loss < self.best:
                if self.verbose:
                    print(f'\nValidation loss decreased ({self.best:.6f} --> {val_loss:.6f}). Saving model.')
                self.best = val_loss
                torch.save(model.state_dict(), self.filepath)
        else:
            if val_loss > self.best:
                if self.verbose:
                    print(f'\nValidation loss increased ({self.best:.6f} --> {val_loss:.6f}). Saving model.')
                self.best = val_loss
                torch.save(model.state_dict(), self.filepath)

class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr=0.0005, decay=0.9985):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay = decay

    def step(self, epoch):
        lr = self.initial_lr * (self.decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

best_loss = float('inf')
def save_checkpoint(model, optimizer, epoch, val_loss, best_loss, checkpoint_path):
    is_best = val_loss < best_loss
    if is_best:
        best_loss = val_loss
        print(f"Saving checkpoint with best val_loss: {val_loss:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        return val_loss
    return best_loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = CustomLRScheduler(optimizer)

checkpoint = ModelCheckpoint(monitor='val_loss', filepath='best_model.pth', mode='min', verbose=1)

train_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES_GENERATOR,
                                custom_keys=['14', '17'])
valid_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES_GENERATOR,
                                custom_keys=['15', '16'])

dataloader = DataLoader(train_generator, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)
validateLoader = DataLoader(valid_generator, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

weight_path = MODEL_DIR + VAL_SET + '/best-weights.pth'

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched['inputs'].to(device), sample_batched['labels'].to(device)
        optimizer.zero_grad()
        # print_input(inputs, i_batch)
        outputs = model(inputs)
        # print_output(outputs, i_batch)
        loss = cce_dice_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader.dataset)
    train_losses.append(epoch_loss)

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, labels_batch in enumerate(validateLoader):
            inputs, labels = sample_batched['inputs'].to(device), sample_batched['labels'].to(device)
            outputs = model(inputs)
            loss = cce_dice_loss(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(validateLoader.dataset)
    val_losses.append(val_loss)
    scheduler.step(epoch)

    best_loss = save_checkpoint(model, optimizer, epoch, val_loss, best_loss, weight_path)

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='s')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Save the plot as a PNG file
plt.savefig('loss_plot.png', dpi=300)  # dpi=300 for high quality
plt.close()  # Close the figure to free memory

os.makedirs(os.path.dirname(weight_path), exist_ok=True)
# Save the final model weights
weights_path = MODEL_DIR + VAL_SET + '/last-weights.pth'
os.makedirs(os.path.dirname(weights_path), exist_ok=True)
torch.save(model.state_dict(), weights_path)