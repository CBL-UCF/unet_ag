import torch
import torch.nn as nn


from PIL import Image
import os
import numpy as np

def dice_coeff(y_true, y_pred):
    smooth = 1
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
    def unet_ag_loss(y_true, y_pred, alpha=1.0, beta=1.0):
        unet_loss = cce_dice_loss(y_true, y_pred[:batch_size])
        spline_loss = dice_loss(y_true, y_pred[batch_size:])
        return alpha * unet_loss + beta * spline_loss
    return unet_ag_loss
