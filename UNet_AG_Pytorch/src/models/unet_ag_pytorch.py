import os
import torch
import torch.nn as nn

from .unet_pytorch import UNet
from .bspline_head_pytorch import Contour, Spline, FillPolygon
from PIL import Image
import numpy as np

def print_output(outputs, batch):
    for frame in range(outputs.shape[0]):
        slice_data_lvc = outputs[frame, 1, :, :]
        slice_data_bg = outputs[frame, 0, :, :]

        data_img = slice_data_lvc.squeeze(0).detach().cpu().numpy()
        bg_img = slice_data_bg.squeeze(0).detach().cpu().numpy()

        img_converted = ((data_img) * 255.).astype(np.uint8)
        img_bg_converted = ((bg_img) * 255.).astype(np.uint8)

        # numpy_array = img_converted.detach().cpu().numpy().astype(np.uint8)
        image = Image.fromarray(img_converted)
        bgImage = Image.fromarray(img_bg_converted)
        png_file_path = os.path.join('./LVCTraining-unetOutput', f"frame{frame}_lvc.png")
        png_file_path_bg = os.path.join('./LVCTraining-unetOutput', f"frame{frame}_background.png")

        image.save(png_file_path)
        bgImage.save(png_file_path_bg)

class UNetAG(nn.Module):
        def __init__(self, unet, contour, spline, fill_polygon):
            super(UNetAG, self).__init__()
            
            self.unet = unet
            self.contour = contour
            self.spline = spline
            self.fill_polygon = fill_polygon

        def forward(self, inputs):
            out_unet = self.unet(inputs)

            # print_output(out_unet, 1)
            print('saida unet', out_unet.shape)

            x = self.contour(out_unet)
            x = self.spline(x)
            x = self.fill_polygon(x)

            x = x.unsqueeze(1)
            x = torch.cat([1 - x, x], dim=1)

            return x

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def get_unet_ag(batch_size,
                height,
                width,
                n_channels,
                n_classes,
                control_pts,
                contour_pts,
                seed=None):
    input_shape=(n_channels, height, width)
    unet = UNet(n_input_channels=input_shape[0], n_output_classes=n_classes, input_dim=input_shape[1:3])
    initialize_weights(unet)

    contour = Contour(batch_size, height, width, contour_pts)
    spline = Spline(control_pts, contour_pts)

    fill_polygon = FillPolygon(batch_size, height, width)
    model = UNetAG(unet, contour, spline, fill_polygon)
    return model