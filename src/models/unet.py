from collections import OrderedDict
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Add, Reshape, Softmax, Activation, concatenate
from tensorflow.keras import activations
from tensorflow.keras import models


def _build_unet(n_input_channels=1, n_output_classes=4, input_dim=(352, 352), base_n_filters=48, dropout=0.3, pad='same', kernel_size=3, seg=False, initializer='glorot_uniform'):
    net = OrderedDict()
    name = 'input'
    net[name] = Input((input_dim[0], input_dim[1], n_input_channels), name=name)

    prev = name
    name = 'contr_1_1'
    net[name] = Conv2D(base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_1_2'
    net[name] = Conv2D(base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool1'
    net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net[prev])

    prev = name
    name = 'contr_2_1'
    net[name] = Conv2D(base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_2_2'
    net[name] = Conv2D(base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool2'
    x = net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop2')(x)

    name = 'contr_3_1'
    net[name] = Conv2D(base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_3_2'
    net[name] = Conv2D(base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool3'
    x = net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop3')(x)

    name = 'contr_4_1'
    net[name] = Conv2D(base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_4_2'
    net[name] = Conv2D(base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool4'
    x = net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop4')(x)

    name = 'encode_1'
    net[name] = Conv2D(base_n_filters*16, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'encode_2'
    net[name] = Conv2D(base_n_filters*16, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale1'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net[prev])

    name = 'concat1'
    x = net[name] = concatenate([net['upscale1'], net['contr_4_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop5')(x)

    name = 'expand_1_1'
    net[name] = Conv2D(base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_1_2'
    net[name] = Conv2D(base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale2'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net[prev])

    name = 'concat2'
    x = net[name] = concatenate([net['upscale2'], net['contr_3_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop6')(x)

    name = 'expand_2_1'
    net[name] = Conv2D(base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_2_2'
    net[name] = Conv2D(base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    ds2 = net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale3'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net[prev])

    name = 'concat3'
    x = net[name] = concatenate([net['upscale3'], net['contr_2_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop7')(x)

    name = 'expand_3_1'
    net[name] = Conv2D(base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_3_2'
    net[name] = Conv2D(base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale4'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net[prev])

    name = 'concat4'
    net[name] = concatenate([net['upscale4'], net['contr_1_2']], axis=-1, name=name)

    prev = name
    name = 'expand_4_1'
    net[name] = Conv2D(base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_4_2'
    net[name] = Conv2D(base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'output_segmentation'
    net[name] = Conv2D(n_output_classes, 1, name=name)(net[prev])

    ds2_1x1_conv = Conv2D(n_output_classes, 1, padding='same', kernel_initializer=initializer, name='ds2_1x1_conv')(ds2)
    ds1_ds2_sum_upscale = UpSampling2D(size=(2, 2), name='ds1_ds2_sum_upscale')(ds2_1x1_conv)
    ds3_1x1_conv = Conv2D(n_output_classes, 1, padding='same', kernel_initializer=initializer, name='ds3_1x1_conv')(net['expand_3_2'])
    ds1_ds2_sum_upscale_ds3_sum = Add(name='ds1_ds2_sum_upscale_ds3_sum')([ds1_ds2_sum_upscale, ds3_1x1_conv])
    ds1_ds2_sum_upscale_ds3_sum_upscale = UpSampling2D(size=(2, 2), name='ds1_ds2_sum_upscale_ds3_sum_upscale')(ds1_ds2_sum_upscale_ds3_sum)

    seg_layer = Add(name='seg')([net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale])

    net['reshapeSeg'] = Reshape((input_dim[0], input_dim[1], n_output_classes))(seg_layer)

    if not seg:
        net['output'] = Softmax()(net['reshapeSeg'])
    else:
        net['output'] = Activation(activations.sigmoid)(net['reshapeSeg'])

    return net, net['output']


def get_unet(
        input_shape=(352, 352, 1),
        num_filters=48,
        n_class=1,
        dropout=0.3,
        initializer='glorot_uniform',
        seg=False
):

    net, out = _build_unet(n_input_channels=input_shape[2], n_output_classes=n_class, input_dim=input_shape[0:2], base_n_filters=num_filters, dropout=dropout, initializer=initializer, seg=seg)

    model = models.Model(inputs=[net['input']], outputs=[out])

    return model
