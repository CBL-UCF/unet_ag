import math
import numpy as np
import tensorflow as tf


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
    x = rho * tf.math.cos(phi)
    y = rho * tf.math.sin(phi)
    return x, y


def cart2pol(x, y):
    rho = tf.math.sqrt(x ** 2 + y ** 2)
    phi = tf.math.atan2(y, x)
    return rho, phi


def fix_radians(tt):
    tt = -tf.keras.activations.relu(-(2 * math.pi - tf.keras.activations.relu(-tt)),
                                    threshold=-2 * math.pi) + tf.keras.activations.relu(tt)
    return tt


def get_rbf_points(tt_new, tt1st, r1st):
    tt_n = tt_new
    tt_1 = tt1st
    tt_1 = tf.concat([tt_1 - (2 * math.pi), tt_1, tt_1 + (2 * math.pi)], axis=1)
    r1st = tf.concat([r1st, r1st, r1st], axis=1)
    tt_squared_dist = (tf.repeat(tt_1, tt_new.shape[1], axis=2) - tf.transpose(
        tf.repeat(tt_n[:, :, tf.newaxis], tt_1.shape[1], axis=2), perm=[0, 2, 1])) ** 2
    tt_squared_dist = tf.minimum(tt_squared_dist, tf.ones_like(tt_squared_dist))
    tt_diff = tf.math.exp(-100.0 * tt_squared_dist)
    tt_diff_sum = tf.repeat(tf.reduce_sum(tt_diff, axis=1)[:, tf.newaxis, :], tt_1.shape[1], axis=1)
    w = tt_diff / tt_diff_sum
    r_new = tf.matmul(tf.transpose(w, perm=[0, 2, 1]), r1st)
    return r_new


class Contour(tf.keras.layers.Layer):
    def __init__(self, batch_size, height, width, contour_points, **kwargs):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.contour_points = contour_points
        self.dim_shape = (self.batch_size, self.height, self.width)
        self.half_x = self.width // 2
        self.half_y = self.height / 2
        self.type = tf.float32
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        self.built = True

    def half_contour(self, pred_norm):
        pred1st = tf.math.round(pred_norm[:, :, :self.half_x])
        x1st = tf.math.reduce_sum(pred1st, axis=2)

        xaux = tf.keras.activations.relu(x1st, threshold=0, max_value=1)
        ythtop = self.half_y - tf.math.reduce_sum(xaux[:, :self.half_x], axis=1)
        ythbottom = self.half_y + tf.math.reduce_sum(xaux[:, self.half_x:], axis=1)

        ythtop = tf.reshape(tf.repeat(ythtop, self.width), (self.batch_size, self.width))
        ythbottom = tf.reshape(tf.repeat(ythbottom, self.width), (self.batch_size, self.width))

        y1st = tf.keras.activations.relu(tf.transpose(
            tf.reshape(tf.repeat(tf.range(start=0, limit=self.height, dtype=self.type), self.batch_size),
                       (self.width, self.batch_size))) - ythtop) + ythtop
        y1st = tf.minimum(y1st, ythbottom)

        x1st = self.half_x - x1st[:, :, tf.newaxis]
        y1st = y1st[:, :, tf.newaxis]

        r1st, tt1st = cart2pol(x1st - self.half_x, y1st - self.half_y)
        tt1st = fix_radians(tt1st)

        npts = self.contour_points // 2
        tt_new = tf.range(start=math.pi / 2., limit=3. * math.pi / 2., delta=math.pi / npts, dtype=self.type)
        tt_new = tf.transpose(tf.reshape(tf.repeat(tt_new, self.batch_size), (npts, self.batch_size)))

        r_new = get_rbf_points(tt_new, tt1st, r1st)
        x_new, y_new = pol2cart(r_new, tt_new[:, :, tf.newaxis])

        x_new = x_new + self.half_x
        y_new = y_new + self.half_y

        contour = tf.concat([x_new, y_new], axis=-1)

        return contour

    def call(self, inputs, **kwargs):
        lvc_pred = inputs[:, :, :, -1]
        lvc_pred_min = tf.reshape(tf.repeat(tf.math.reduce_min(lvc_pred, axis=[1, 2]), self.height * self.width),
                                  self.dim_shape)
        lvc_pred_max = tf.reshape(tf.repeat(tf.math.reduce_max(lvc_pred, axis=[1, 2]), self.height * self.width),
                                  self.dim_shape)
        lvc_pred_norm = (lvc_pred - lvc_pred_min) / (lvc_pred_max - lvc_pred_min)

        contour1st = self.half_contour(lvc_pred_norm)
        contour2nd = self.half_contour(tf.reverse(lvc_pred_norm, axis=[2]))
        contour2nd = tf.reverse(
            tf.concat([tf.ones_like(contour2nd[:, :, :1]) * self.height - contour2nd[:, :, :1], contour2nd[:, :, 1:]], axis=-1),
            axis=[1])

        contour = tf.concat([contour1st, contour2nd], axis=1)
        return contour


class CircularBSpline(tf.keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs[:, :, 0]
        y = inputs[:, :, 1]

        x = tf.concat([x, x[:, :3]], 1)
        y = tf.concat([y, y[:, :3]], 1)

        spline_x = tf.matmul(x, self.B)
        spline_y = tf.matmul(y, self.B)

        spline = tf.concat([spline_x[:, :, tf.newaxis], spline_y[:, :, tf.newaxis]], -1)

        return spline


class Spline(tf.keras.layers.Layer):
    def __init__(self, npts, outpts, **kwargs):
        self.npts = npts
        self.outpts = outpts
        self.B = _get_B_weights(npts, outpts)
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        self.circular_bspline = CircularBSpline(self.B)
        self.built = True

    def call(self, inputs, **kwargs):
        totalpts = inputs.shape[1]

        initial_idx = tf.random.uniform(shape=[], maxval=self.outpts // self.npts, dtype=tf.int32)

        idx0 = tf.range(initial_idx, totalpts, delta=totalpts // self.npts)
        input0 = tf.gather(inputs, idx0, axis=1)
        output0 = self.circular_bspline(input0)

        return output0


class FillPolygon(tf.keras.layers.Layer):
    def __init__(self, batch_size, height, width, **kwargs):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.dim_shape = (self.batch_size, self.height, self.width)
        self.half_x = self.width // 2
        self.half_y = self.height / 2
        self.type = tf.float32
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs[:, :, :1]
        y = inputs[:, :, 1:]

        r, tt = cart2pol(x - self.half_x, y - self.half_y)
        tt = fix_radians(tt)

        y_matrix = tf.range(start=0, limit=self.height, dtype=self.type)
        y_matrix = tf.reshape(tf.cast(
            tf.ones(self.dim_shape, dtype=self.type) * tf.reshape(tf.repeat(y_matrix, self.width),
                                                                  (self.height, self.width)), dtype=self.type),
            (self.batch_size, self.height * self.width))
        x_matrix = tf.range(start=0, limit=self.width, dtype=self.type)
        x_matrix = tf.reshape(tf.cast(tf.ones(self.dim_shape, dtype=self.type) * tf.transpose(
            tf.reshape(tf.repeat(x_matrix, self.height), (self.width, self.height))), dtype=self.type),
                              (self.batch_size, self.height * self.width))

        r_matrix, tt_matrix = cart2pol(x_matrix - self.half_x, y_matrix - self.half_y)
        tt_matrix = fix_radians(tt_matrix)
        r_matrix_rbf = get_rbf_points(tt_matrix, tt, r)

        flag_matrix = tf.reshape(tf.keras.activations.relu(1 + r_matrix_rbf[:, :, 0] - r_matrix, threshold=0, max_value=1),
                                 (self.batch_size, self.height, self.width))

        return flag_matrix
