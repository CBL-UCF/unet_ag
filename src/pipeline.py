import numpy as np
import tensorflow as tf

from models import get_unet, get_unet_ag
from utils import postprocess_stack, get_stack_center_point

WEIGHT_PATH = '../weights/'

HEIGHT = 352
WIDTH = 352
CROP_HEIGHT = 144
CROP_WIDTH = 144
BATCH_SIZE = 8

LOC_N_CHANNELS = 1
LOC_N_CLASSES = 2

LVC_N_CHANNELS = 1
LVC_N_CLASSES = 2
LVC_CONTOUR_POINTS = 360
LVC_CONTROL_POINTS = 20

LVM_N_CHANNELS = 2
LVM_N_CLASSES = 2
LVM_CONTOUR_POINTS = 360
LVM_CONTROL_POINTS = 20


lvc_locator = get_unet(input_shape=(HEIGHT, WIDTH, LOC_N_CHANNELS), n_class=LOC_N_CLASSES)
lvc_locator.load_weights(WEIGHT_PATH + 'lvc-locator/best-weights')

lvc_unet_ag = get_unet_ag(batch_size=BATCH_SIZE,
                          height=CROP_HEIGHT,
                          width=CROP_WIDTH,
                          n_channels=LVC_N_CHANNELS,
                          n_classes=LVC_N_CLASSES,
                          control_pts=LVC_CONTROL_POINTS,
                          contour_pts=LVC_CONTOUR_POINTS)
lvc_unet_ag.load_weights(WEIGHT_PATH + 'lvc-unet-ag/best-weights')

lvm_unet_ag = get_unet_ag(batch_size=BATCH_SIZE,
                          height=CROP_HEIGHT,
                          width=CROP_WIDTH,
                          n_channels=LVM_N_CHANNELS,
                          n_classes=LVM_N_CLASSES,
                          control_pts=LVM_CONTROL_POINTS,
                          contour_pts=LVM_CONTOUR_POINTS)
lvm_unet_ag.load_weights(WEIGHT_PATH + 'lvm-unet-ag/best-weights')


def prediction_pipeline(inputs):
    """
    LVC Locator
    """
    locator_pred = lvc_locator(inputs)
    xc, yc = get_stack_center_point(postprocess_stack(np.round(locator_pred[:, :, :, 1])))

    x = xc - CROP_WIDTH // 2
    y = yc - CROP_HEIGHT // 2

    cropped_inputs = inputs[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH, :]

    """
    LVC-UNet_AG
    """
    lvc_unet_pred = np.zeros_like(locator_pred)
    lvc_bspline_pred = np.zeros_like(locator_pred)
    lvc_unet_ag_pred = lvc_unet_ag(cropped_inputs)
    lvc_unet_pred[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH] = lvc_unet_ag_pred[:BATCH_SIZE]
    lvc_bspline_pred[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH] = lvc_unet_ag_pred[BATCH_SIZE:]

    """
    LVM-UNet_AG
    """
    lvm_unet_pred = np.zeros_like(locator_pred)
    lvm_bspline_pred = np.zeros_like(locator_pred)
    lvm_unet_ag_pred = lvm_unet_ag(tf.concat([cropped_inputs, lvc_unet_ag_pred[:BATCH_SIZE, :, :, :-1]], axis=-1))
    lvm_unet_pred[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH] = lvm_unet_ag_pred[:BATCH_SIZE]
    lvm_bspline_pred[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH] = lvm_unet_ag_pred[BATCH_SIZE:]

    return locator_pred, lvc_unet_pred, lvc_bspline_pred, lvm_unet_pred, lvm_bspline_pred


if __name__ == '__main__':
    """
    Load your data here.
    """
    data = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, LOC_N_CHANNELS))
    """
    """

    locator_pred, lvc_unet_pred, lvc_bspline_pred, lvm_unet_pred, lvm_bspline_pred = prediction_pipeline(data)
