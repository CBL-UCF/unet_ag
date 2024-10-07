import tensorflow as tf

from .unet import get_unet
from .bspline_head import Contour, Spline, FillPolygon


def get_unet_ag(batch_size,
                height,
                width,
                n_channels,
                n_classes,
                control_pts,
                contour_pts,
                seed=None):

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    unet = get_unet(
        input_shape=(height, width, n_channels),
        n_class=n_classes,
        initializer=initializer
    )

    inputs = tf.keras.layers.Input((height, width, n_channels))

    out_unet = unet(inputs)
    x = Contour(batch_size, height, width, contour_pts)(out_unet)
    x = Spline(control_pts, contour_pts)(x)
    x = FillPolygon(batch_size, height, width)(x)

    x = x[:, :, :, tf.newaxis]
    x = tf.concat([1 - x, x], axis=-1)

    out = tf.concat([out_unet, x], axis=0)

    model = tf.keras.Model(inputs, out)

    return model
