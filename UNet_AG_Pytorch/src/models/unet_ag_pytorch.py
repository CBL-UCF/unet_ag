import torch
import torch.nn as nn

from .unet_pytorch import UNet
from .bspline_head_pytorch import Contour, Spline, FillPolygon

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class UNetAG(nn.Module):
        def __init__(self, batch_size, height, width, n_channels, n_classes, control_pts, contour_pts, seed=None):
            super(UNetAG, self).__init__()
            self.input_shape=(n_channels, height, width)
            self.unet = UNet(n_input_channels=self.input_shape[0], n_output_classes=n_classes, input_dim=self.input_shape[1:3])
            initialize_weights(self.unet)
            # Custom layers
            self.contour = Contour(batch_size, height, width, contour_pts)
            self.spline = Spline(control_pts, contour_pts)
            self.fill_polygon = FillPolygon(batch_size, height, width)
            


        def forward(self, inputs):
            out_unet = self.unet(inputs)

            x = self.contour(out_unet)
            x = self.spline(x)
            x = self.fill_polygon(x)
           
            x = torch.cat([1 - x, x], dim=1)
            out = torch.cat([out_unet, x], dim=0)

            return out
