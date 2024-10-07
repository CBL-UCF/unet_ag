import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        png_file_path = os.path.join('./LVCTraining-countorOutput', f"frame{frame}_lvc.png")
        png_file_path_bg = os.path.join('./LVCTraining-countorOutput', f"frame{frame}_background.png")

        image.save(png_file_path)
        bgImage.save(png_file_path_bg)

def _get_B_weights(control_points=20, contour_points=200):
    def B(x, k, i, t):
        if k == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        if t[i + k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t)
        if t[i + k + 1] == t[i + 1]:
            c2 = 0.0
        else:
            c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t)
        return c1 + c2

    k = 3
    n = control_points + k
    u = np.linspace(0, 1, contour_points)

    t = np.linspace(0 - k * (1./control_points), 1 + k * (1./control_points), control_points + 2*k + 1)

    weights = np.zeros((n, len(u)))

    for i in range(n):
        for j, x in enumerate(u):
            weights[i, j] = B(x, k, i, t)

    return weights


def pol2cart(rho, phi):
    # x = rho * tf.math.cos(phi)
    x = rho * torch.cos(phi)

    # y = rho * tf.math.sin(phi)
    y = rho * torch.sin(phi)
    
    return x, y


def cart2pol(x, y):
    # rho = tf.math.sqrt(x ** 2 + y ** 2)
    rho = torch.sqrt(x ** 2 + y ** 2)

    # phi = tf.math.atan2(y, x)
    phi = torch.atan2(y, x)

    return rho, phi


def fix_radians(tt):
    tt = -F.relu(-(2 * math.pi - F.relu(-tt)), inplace=False) + F.relu(tt)
    return tt


def get_rbf_points(tt_new, tt1st, r1st):
    tt_n = tt_new
    tt_1 = tt1st

    tt_1 = torch.concat([tt_1 - (2 * math.pi), tt_1, tt_1 + (2 * math.pi)], dim=1)

    r1st = torch.cat([r1st, r1st, r1st], dim=1)

    tt_n_repeated = tt_n.unsqueeze(2).repeat(1, 1, tt_1.shape[1])
    tt_n_repeated_transposed = tt_n_repeated.transpose(1, 2)

    tt_squared_dist = (tt_1.repeat(1, 1, tt_new.shape[1]) - tt_n_repeated_transposed) ** 2

    tt_squared_dist = torch.minimum(tt_squared_dist, torch.ones_like(tt_squared_dist))

    tt_diff = torch.exp(-100.0 * tt_squared_dist)
    num_repeats = tt_1.shape[1]
    tt_diff_sum_over_axis = torch.sum(tt_diff, dim=1, keepdim=True)
    tt_diff_sum = tt_diff_sum_over_axis.repeat(1, num_repeats, 1)

    w = tt_diff / tt_diff_sum
    r_new = torch.matmul(w.transpose(1,2), r1st)
    return r_new


class Contour(nn.Module):
    def __init__(self, batch_size, height, width, contour_points, **kwargs):
        super(Contour, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.contour_points = contour_points
        self.dim_shape = (self.batch_size, self.height, self.width)
        self.half_x = self.width // 2
        self.half_y = self.height / 2
        self.built = True 
        self.type = torch.float32      

    def half_contour(self, pred_norm):
        device = pred_norm.device

        pred1st = torch.round(pred_norm[:, :, :self.half_x])
        x1st = torch.sum(pred1st, dim=2)

        xaux = F.relu(x1st)
        xaux = torch.clamp(xaux, min=0, max=1)
        ythtop = self.half_y - torch.sum(xaux[:, :self.half_x], dim=1)
        ythbottom = self.half_y + torch.sum(xaux[:, self.half_x:], dim=1)

        ythtop = ythtop.unsqueeze(-1).repeat(1, self.width)
        ythbottom = ythbottom.unsqueeze(-1).repeat(1, self.width)

        range_tensor = torch.arange(0, self.height, dtype=torch.float32, device=pred_norm.device)
        range_tensor = range_tensor.repeat(self.batch_size, 1)
        range_tensor = range_tensor.reshape(self.width, self.batch_size).transpose(0, 1)
        y1st = torch.relu(range_tensor - ythtop) + ythtop
        y1st = torch.min(y1st, ythbottom)

        x1st = self.half_x - x1st.unsqueeze(-1)
        y1st = y1st.unsqueeze(-1)

        r1st, tt1st = cart2pol(x1st - self.half_x, y1st - self.half_y)
        r1st.to(device)
        tt1st = fix_radians(tt1st).to(device)
        
        npts = self.contour_points // 2
        tt_new = torch.arange(start=math.pi / 2., end=3. * math.pi / 2., step=math.pi / npts, dtype=self.type)
        tt_new = tt_new.repeat(self.batch_size).reshape(self.batch_size, npts).to(device)

        r_new = get_rbf_points(tt_new, tt1st, r1st)
        x_new, y_new = pol2cart(r_new, tt_new.unsqueeze(-1))

        x_new = x_new + self.half_x
        y_new = y_new + self.half_y

        contour = torch.cat([x_new, y_new], dim=-1)

        return contour

    def forward(self, inputs):
        lvc_pred = inputs[:,-1,:,:]

        print('lvc_pred shape', lvc_pred.shape)
        lvc_pred_min = torch.min(lvc_pred, dim=1)[0]
        lvc_pred_min = torch.min(lvc_pred_min, dim=1)[0]
        print('lvc_pred_min shape', lvc_pred_min.shape)

        lvc_pred_min_repeated = lvc_pred_min.unsqueeze(1).unsqueeze(2)
        print('lvc_pred_min_repeated shape', lvc_pred_min_repeated.shape)

        lvc_pred_min_repeated = lvc_pred_min_repeated.expand(self.batch_size, self.height, self.width)
        lvc_pred_min = lvc_pred_min_repeated.reshape(self.dim_shape)

        lvc_pred_max = torch.min(lvc_pred, dim=1)[0]
        lvc_pred_max = torch.min(lvc_pred_max, dim=1)[0]
        lvc_pred_max_repeated = lvc_pred_max.unsqueeze(1).unsqueeze(2)
        lvc_pred_max_repeated = lvc_pred_max_repeated.expand(self.batch_size, self.height, self.width)
        lvc_pred_max = lvc_pred_max_repeated.reshape(self.dim_shape)

        lvc_pred_norm = (lvc_pred - lvc_pred_min) / (lvc_pred_max - lvc_pred_min)

        contour1st = self.half_contour(lvc_pred_norm)
        contour2nd = self.half_contour(torch.flip(lvc_pred_norm, dims=[2]))
        contour2nd = torch.flip(torch.cat([torch.full_like(contour2nd[:, :, :1], self.height) - contour2nd[:, :, :1], contour2nd[:, :, 1:]], dim=-1), dims=[1])
        contour = torch.cat([contour1st, contour2nd], dim=1)

        return contour

class CircularBSpline(nn.Module):
    def __init__(self, B, **kwargs):
        self.B = torch.tensor(B, dtype=torch.float32, requires_grad=True)
        self.built = True
        super().__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        device = inputs.device
        x = inputs[:, :, 0].detach().cpu()
        y = inputs[:, :, 1].detach().cpu()

        x = torch.cat([x, x[:, :3]], dim=1)
        y = torch.cat([y, y[:, :3]], dim=1)
        B = self.B

        spline_x = torch.matmul(x, B)
        spline_y = torch.matmul(y, B)

        spline = torch.cat([spline_x.unsqueeze(-1), spline_y.unsqueeze(-1)], dim=-1)
        del B
        return spline

class Spline(nn.Module):
    def __init__(self, npts, outpts, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.npts = npts
        self.outpts = outpts
        self.B = _get_B_weights(npts, outpts)
        self.circular_bspline = CircularBSpline(self.B)
    
    def forward(self, inputs, **kwargs):
        totalpts = inputs.shape[1]

        initial_idx = torch.randint(0, self.outpts // self.npts, (1,)).item()
        idx0 = torch.arange(initial_idx, totalpts, step=totalpts // self.npts, dtype=torch.long)
        input0 = inputs[:, idx0]

        output0 = self.circular_bspline(input0)
        return output0

class FillPolygon(nn.Module):
    def __init__(self, batch_size, height, width, **kwargs):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.dim_shape = (self.batch_size, self.height, self.width)
        self.half_x = self.width // 2
        self.half_y = self.height / 2
        self.type = torch.float32
        self.built = True
        super().__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        inputs.detach().cpu()
        x = inputs[:, :, :1]
        y = inputs[:, :, 1:]

        r, tt = cart2pol(x - self.half_x, y - self.half_y)
        r
        tt = fix_radians(tt)

        y_range = torch.arange(start=0, end=self.height, dtype=torch.float32)
        y_matrix = y_range.repeat(self.width).reshape(self.height, self.width)
        y_matrix = y_matrix.repeat(self.batch_size, 1).reshape(self.batch_size, self.height * self.width)
        
        x_range = torch.arange(start=0, end=self.width, dtype=torch.float32)
        x_matrix = x_range.repeat(self.height).reshape(self.width, self.height).transpose(0, 1)
        x_matrix = x_matrix.repeat(self.batch_size, 1).reshape(self.batch_size, self.height * self.width)

        r_matrix, tt_matrix = cart2pol(x_matrix - self.half_x, y_matrix - self.half_y)
        tt_matrix = fix_radians(tt_matrix)

        r_matrix_rbf = get_rbf_points(tt_matrix, tt, r)
        r_matrix_rbf = r_matrix_rbf[:, :, 0]

        flag_matrix = F.relu(1 + r_matrix_rbf - r_matrix, inplace=False)
        flag_matrix = flag_matrix.reshape(self.batch_size, self.height, self.width)

        return flag_matrix.to(device)
