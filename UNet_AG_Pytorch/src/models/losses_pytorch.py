import torch
import torch.nn as nn


from PIL import Image
import os
import numpy as np

def dice_coeff(y_true, y_pred):
    smooth = 1

    # for frame in range(y_true.shape[0]):
    #     slice_data_y_true = y_true[frame, -1:, :, :]
    #     slice_data_y_pred = y_pred[frame, -1:, :, :]

    #     data_img = slice_data_y_true.squeeze(0).detach().cpu().numpy()
    #     data_pred = slice_data_y_pred.squeeze(0).detach().cpu().numpy()

    #     img_converted = ((data_img) * 255.).astype(np.uint8)
    #     img_converted_pred = ((data_pred) * 255.).astype(np.uint8)

    #     numpy_array = img_converted.detach().cpu().numpy().astype(np.uint8)
    #     image = Image.fromarray(img_converted)
    #     image_pred = Image.fromarray(img_converted_pred)

    #     png_file_path = os.path.join('./ResultsHeartLocator/loss', f"y_true_frame{frame}_lvc.png")
    #     png_file_path_pred = os.path.join('./ResultsHeartLocator/loss', f"y_pred_frame{frame}_background.png")

    #     image.save(png_file_path)
    #     image_pred.save(png_file_path_pred)

    # Flatten
    y_true_f = torch.reshape(y_true,(-1,))
    y_pred_f = torch.reshape(y_pred,(-1,))

    intersection = (y_true_f * y_pred_f).sum()
    score = (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def cce_loss(y_true, y_pred):
    # Replaces nn.CrossEntropyLoss
    return (-(y_pred+1e-10).log() * y_true).sum(dim=1).mean()

def cce_dice_loss(y_true, y_pred):
    cce_loss_value = cce_loss(y_true, y_pred)
    dice_loss_value = dice_loss(y_true, y_pred)

    return cce_loss_value + dice_loss_value


def get_unet_ag_loss(batch_size):
    def unet_ag_loss(y_true, y_pred):
        unet_loss = cce_dice_loss(y_true, y_pred)
        spline_loss = dice_loss(y_true, y_pred)
        return unet_loss + spline_loss
    return unet_ag_loss
