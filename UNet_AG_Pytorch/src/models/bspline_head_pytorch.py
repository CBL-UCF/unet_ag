import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def print_output(outputs):
    for i in range(outputs.shape[0]):
        data_img = outputs[i,:,:].detach().cpu().numpy()
        img_converted = ((data_img) * 255.).astype(np.uint8)
        image = Image.fromarray(img_converted)
        png_file_path = os.path.join('./DEBUG/', f"polygon_pred_{i}.png")
        image.save(png_file_path)

def print_input(outputs, prefix):
    for frame in range(outputs.shape[0]):
        slice_data_bg = outputs[frame, :, :, :]
        bg_img = slice_data_bg.squeeze(0).detach().cpu().numpy()

        img_bg_converted = ((bg_img) * 255.).astype(np.uint8)

        # numpy_array = img_converted.detach().cpu().numpy().astype(np.uint8)
        bgImage = Image.fromarray(img_bg_converted)
        png_file_path_bg = os.path.join('./DEBUG', f"frame{frame}_input_{prefix}.png")
        bgImage.save(png_file_path_bg)

def print_points(output, filename):
    print('printing')
    for i in range(output.shape[0]):
        points = output[i,:,:].detach().cpu().numpy()
        plt.figure(figsize=(144, 144))

        plt.scatter(points[:, 1], points[:, 0], c='blue', marker='o', s=1000)

        # plt.xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
        # plt.ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
        plt.xlabel('X')
        plt.xticks(fontsize=250)
        plt.ylabel('Y')
        plt.yticks(fontsize=250)
        plt.title('Pontos no Plano Cartesiano')
        plt.grid()
        # Salva a imagem
        plt.savefig(f'./DEBUG/{filename}_lvc_{i}.png')
        plt.close()

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
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return rho, phi


def custom_relu(tensor, threshold=0):
    return torch.where(tensor > threshold, tensor, torch.tensor(0.0, device=tensor.device))

## Maybe I should implement my on Relu
def fix_radians(tt):
    threshold=-2 * math.pi
    relu = custom_relu(-(2 * math.pi - F.relu(-tt)), threshold)
    tt = -relu + F.relu(tt)
    return tt

# def get_rbf_points(tt_new, tt1st, r1st):
#     tt_1 = torch.cat([tt1st - (2 * math.pi), tt1st, tt1st + (2 * math.pi)], dim=1)
#     r1st = r1st.unsqueeze(1).expand(-1, tt_1.shape[1], -1)

#     v1 = tt_new[:, :, None].repeat(1, 1, tt_1.shape[1])
#     v2 = v1.transpose(1, 2)
#     v3 = tt_1.repeat(1, 1, tt_new.shape[1])
#     tt_squared_dist = (v3 -v2) ** 2
#     tt_squared_dist = torch.minimum(tt_squared_dist, torch.ones_like(tt_squared_dist, device=tt_new.device))
#     tt_diff = torch.exp(-100.0 * tt_squared_dist)
#     tt_diff_sum = tt_diff.sum(dim=1)[:, None, :].repeat(1, tt_1.shape[1], 1)
#     w = tt_diff / tt_diff_sum
#     r_new = torch.matmul(w.transpose(1,2), r1st)
#     return r_new


def get_rbf_points(tt_new, tt1st, r1st):
    tt_n = tt_new
    tt_1 = tt1st
    tt_1 = torch.concat([tt_1 - (2 * math.pi), tt_1, tt_1 + (2 * math.pi)], dim=1)
    r1st = torch.cat([r1st, r1st, r1st], dim=1)


    v1 = tt_n[:, :, None].repeat(1, 1, tt_1.shape[1])
    v2 = v1.transpose(1, 2)
    v3 = tt_1.repeat(1, 1, tt_new.shape[1])
    tt_squared_dist = (v3 -v2) ** 2
    tt_squared_dist = torch.minimum(tt_squared_dist, torch.ones_like(tt_squared_dist))
    tt_diff = torch.exp(-100.0 * tt_squared_dist)
    tt_diff_sum = tt_diff.sum(dim=1)[:, None, :].repeat(1, tt_1.shape[1], 1)
    w = tt_diff / tt_diff_sum
    r_new = torch.matmul(w.transpose(1,2), r1st)
    return r_new


class Contour(nn.Module):
    def __init__(self, batch_size, height, width, contour_points, **kwargs):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.contour_points = contour_points
        self.dim_shape = (self.batch_size, self.height, self.width)
        self.half_x = self.width // 2
        self.half_y = self.height / 2
        self.built = True 
        self.type = torch.float32
        super(Contour, self).__init__(**kwargs)

    def half_contour(self, pred_norm):
        pred1st = torch.round(pred_norm[:, :, :self.half_x])
        x1st = torch.sum(pred1st, dim=2)

        xaux = F.relu(x1st)
        xaux = torch.clamp(xaux, min=0, max=1)

        ythtop = self.half_y - torch.sum(xaux[:, :self.half_x], dim=1, keepdim=True)
        ythbottom = self.half_y + torch.sum(xaux[:, self.half_x:], dim=1, keepdim=True)
        ythtop = ythtop.expand(-1, self.width)
        ythbottom = ythbottom.expand(-1, self.width)

        range_tensor = torch.arange(start=0, end=self.height, dtype=self.type, device=device)
        range_tensor = range_tensor.unsqueeze(0).expand(self.batch_size, -1)

        y1st = torch.relu(range_tensor - ythtop) + ythtop
        y1st = torch.min(y1st, ythbottom)

        x1st = self.half_x - x1st.unsqueeze(-1)
        y1st = y1st.unsqueeze(-1)

        r1st, tt1st = cart2pol(x1st - self.half_x, y1st - self.half_y)
        tt1st = fix_radians(tt1st)
        
        npts = self.contour_points // 2
        tt_new = torch.arange(start=math.pi / 2., end=3. * math.pi / 2., step=math.pi / npts, dtype=self.type, device=device)
        tt_new = tt_new.unsqueeze(0).expand(self.batch_size, -1)

        r_new = get_rbf_points(tt_new, tt1st, r1st)
        x_new, y_new = pol2cart(r_new, tt_new.unsqueeze(-1))

        x_new = x_new + self.half_x
        y_new = y_new + self.half_y

        contour = torch.cat([x_new, y_new], dim=-1)

        return contour
    
    # def half_contour(self, pred_norm, debug=False):
    #     pred1st = torch.round(pred_norm[:, :, :self.half_x].clone())
    #     x1st = torch.sum(pred1st, dim=2)

    #     xaux = F.relu(x1st)
    #     xaux = torch.clamp(xaux, min=0, max=1)

    #     ythtop = self.half_y - torch.sum(xaux[:, :self.half_x], dim=1)
    #     ythbottom = self.half_y + torch.sum(xaux[:, self.half_x:], dim=1)
    #     ythtop = ythtop.repeat(self.width).reshape(self.batch_size, self.width)
    #     ythbottom = ythbottom.repeat(self.width).reshape(self.batch_size, self.width)

    #     range_tensor = torch.arange(start=0, end=self.height, dtype=self.type, device=device)
    #     range_tensor = range_tensor.repeat(self.batch_size)
    #     range_tensor = range_tensor.reshape(self.width, self.batch_size).T

    #     y1st = torch.relu(range_tensor - ythtop) + ythtop
    #     y1st = torch.min(y1st, ythbottom)

    #     x1st = self.half_x - x1st.unsqueeze(-1)
    #     y1st = y1st.unsqueeze(-1)

    #     if (debug):
    #         test = torch.cat([x1st - self.half_x, y1st - self.half_y], dim=-1)
    #         print_points(test, 'before_get rbf')

    #     r1st, tt1st = cart2pol(x1st - self.half_x, y1st - self.half_y)
    #     tt1st = fix_radians(tt1st)
        
    #     npts = self.contour_points // 2
    #     tt_new = torch.arange(start=math.pi / 2., end=3. * math.pi / 2., step=math.pi / npts, dtype=self.type, device=device)
    #     tt_new = (tt_new.repeat(self.batch_size).reshape(npts, self.batch_size)).T

    #     r_new = get_rbf_points(tt_new, tt1st, r1st)
    #     x_new, y_new = pol2cart(r_new, tt_new.unsqueeze(-1))

    #     x_new = x_new + self.half_x
    #     y_new = y_new + self.half_y

    #     contour = torch.cat([x_new, y_new], dim=-1)

    #     return contour


    def forward(self, inputs):
        lvc_pred = inputs[:,-1,:,:]
        
        lvc_pred_min = (lvc_pred.min(dim=2)[0].min(dim=1)[0]).repeat(self.height * self.width).reshape(self.dim_shape)
        lvc_pred_max = (lvc_pred.max(dim=2)[0].max(dim=1)[0]).repeat(self.height * self.width).reshape(self.dim_shape)
        lvc_pred_norm = (lvc_pred - lvc_pred_min) / (lvc_pred_max - lvc_pred_min)

        # print('lvc_pred_norm', lvc_pred_norm.shape)

        contour1st = self.half_contour(lvc_pred_norm)
        # print_points(contour1st, 'half1_contour')
        # print('lvc_pred_norm', lvc_pred_norm.shape)
        contour2nd = self.half_contour(torch.flip(lvc_pred_norm, dims=[2]))

        # print_points(contour2nd, 'half2_contour')

        t1 = torch.ones_like(contour2nd[:, :, :1])
        t2 = torch.concat([t1 * self.height - contour2nd[:, :, :1], contour2nd[:, :, 1:]], dim=-1)
        contour2nd = torch.flip(t2, dims=[1])
        # print_points(contour2nd, 'half2_contour_flipped')


        contour = torch.cat([contour1st, contour2nd], dim=1)

        # print_points(contour, 'contour')

        return contour

class CircularBSpline(nn.Module):
    def __init__(self, B, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.built = True
        super().__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        x = inputs[:, :, 0]
        y = inputs[:, :, 1]

        x = torch.cat([x, x[:, :3]], dim=1)
        y = torch.cat([y, y[:, :3]], dim=1)

        spline_x = torch.matmul(x, self.B)
        spline_y = torch.matmul(y, self.B)

        spline = torch.cat([spline_x.unsqueeze(-1), spline_y.unsqueeze(-1)], dim=-1)
        # print_points(spline, 'spline')
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
        device = inputs.device
        initial_idx = torch.randint(low=0, high=self.outpts // self.npts, size=(1,), dtype=torch.int32, device=device).item()
        idx0 = torch.arange(initial_idx, totalpts, step=totalpts // self.npts, dtype=torch.int32, device=device)
        input0 = inputs[:, idx0]

        output0 = self.circular_bspline(input0)
        # print_points(output0, 'spline')

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
        device = inputs.device
        x = inputs[:, :, 1:]
        y = inputs[:, :, :1]

        r, tt = cart2pol(x - self.half_x, y - self.half_y)
        tt = fix_radians(tt)

        y_matrix = torch.arange(start=0, end=self.height, dtype=self.type, device=device)
        v1 = y_matrix.repeat(self.width)
        v2 = v1.reshape(self.height, self.width)
        v3 = torch.ones(self.dim_shape, dtype=self.type, device=device)
        v4 = (v3 * v2).to(self.type)
        y_matrix = v4.reshape(self.batch_size, self.height * self.width)
 
        x_matrix = torch.arange(start=0, end=self.width, dtype=self.type, device=device)
        x1 = x_matrix.repeat(self.height)
        x2 = x1.reshape(self.width, self.height)
        x3 = torch.ones(self.dim_shape, dtype=self.type, device=device)
        x4 = (x3 * x2.t()).to(self.type)
        x_matrix = x4.reshape(self.batch_size, self.height * self.width)

        r_matrix, tt_matrix = cart2pol(x_matrix - self.half_x, y_matrix - self.half_y)
        tt_matrix = fix_radians(tt_matrix)
        r_matrix_rbf = get_rbf_points(tt_matrix, tt, r)

        flag_matrix = F.relu(1 + r_matrix_rbf[:,:,0] - r_matrix)
        flag_matrix = torch.clamp(flag_matrix, min=0, max=1)
        flag_matrix = flag_matrix.reshape(self.batch_size, self.height, self.width)

        # print_output(flag_matrix)
        flag_matrix = flag_matrix.unsqueeze(1)
        return flag_matrix