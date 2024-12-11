import torch
import torch.nn as nn
import torch.nn.functional as F

def double_convolutional_layer(n_input_channels, base_n_filters, kernel_size, padding, kernel_initializer):
    return nn.Sequential(
        nn.Conv2d(in_channels=n_input_channels,out_channels=base_n_filters, kernel_size=kernel_size,padding=padding, bias=False),
        nn.BatchNorm2d(base_n_filters),
        nn.Conv2d(in_channels=base_n_filters,out_channels=base_n_filters, kernel_size=kernel_size,padding=padding, bias=False),
        nn.BatchNorm2d(base_n_filters),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    # Not sure if I should update the padding or not, Docs says it can accept the 'same' or 'valid' values, same on Keras
    def __init__(self, n_input_channels=1, n_output_classes=4, input_dim=(352, 352), base_n_filters=48, dropout=0.3, pad='same', kernel_size=3, seg=False, initializer='glorot_uniform'):
        super(UNet, self).__init__()
        
        self.input_dim = input_dim
        self.n_output_classes = n_output_classes
        self.seg = seg

        self.maxPool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropoutValue = dropout
        self.dropout =  nn.Dropout(dropout)
        self.upscaling = nn.Upsample(scale_factor=(2,2))

        self.contr_1 = double_convolutional_layer(n_input_channels, base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer)
        self.contr_2 = double_convolutional_layer(base_n_filters, base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer)
        self.contr_3 = double_convolutional_layer(base_n_filters*2, base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer)
        self.contr_4 = double_convolutional_layer(base_n_filters*4, base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer)
        self.encode = double_convolutional_layer(base_n_filters*8, base_n_filters*16, kernel_size, padding=pad, kernel_initializer=initializer)
        self.expand_1 = double_convolutional_layer((base_n_filters*16 + base_n_filters*8), base_n_filters*8, kernel_size, padding=pad, kernel_initializer=initializer)
        self.expand_2 = double_convolutional_layer((base_n_filters*8 + base_n_filters*4), base_n_filters*4, kernel_size, padding=pad, kernel_initializer=initializer)
        self.expand_3 = double_convolutional_layer((base_n_filters*4 + base_n_filters*2), base_n_filters*2, kernel_size, padding=pad, kernel_initializer=initializer)
        self.expand_4 = double_convolutional_layer((base_n_filters*2 + base_n_filters), base_n_filters, kernel_size, padding=pad, kernel_initializer=initializer)
        self.output_segmentation = nn.Conv2d(base_n_filters, n_output_classes, 1)
        self.output_segmentation2 = nn.Conv2d(n_output_classes, n_output_classes, 1)


        self.ds2_1x1_conv = nn.Conv2d(base_n_filters*4, n_output_classes, 1, padding='same')
        self.ds3_1x1_conv = nn.Conv2d(base_n_filters*2, n_output_classes, 1, padding='same')
    
    def forward(self, input):
        contr1 = self.contr_1(input)
        x = self.maxPool(contr1)

        contr2 = self.contr_2(x)
        x = self.maxPool(contr2)

        if (self.dropoutValue):
            x = self.dropout(x)

        contr3 = self.contr_3(x)
        x = self.maxPool(contr3)

        if (self.dropoutValue):
            x = self.dropout(x)

        contr4 = self.contr_4(x)
        x = self.maxPool(contr4)

        if (self.dropoutValue):
            x = self.dropout(x)

        x = self.encode(x)
        x = self.upscaling(x)

        x = torch.cat([x, contr4], dim=1)

        if (self.dropoutValue):
            x = self.dropout(x)

        x = self.expand_1(x)
        x = self.upscaling(x)
        x = torch.cat([x, contr3], dim=1)

        if (self.dropoutValue):
            x = self.dropout(x)

        ds2 = x = self.expand_2(x)
        x = self.upscaling(x)
        x = torch.cat([x, contr2], dim=1)

        if (self.dropoutValue):
            x = self.dropout(x)

        expand3 = self.expand_3(x)
        x = self.upscaling(expand3)
        x = torch.cat([x, contr1], dim=1)

        x = self.expand_4(x)
        x = self.output_segmentation(x)

        ds2_1x1_conv = self.ds2_1x1_conv(ds2)
        ds1_ds2_sum_upscale = self.upscaling(ds2_1x1_conv)

        ds3_1x1_conv = self.ds3_1x1_conv(expand3)

        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv

        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upscaling(ds1_ds2_sum_upscale_ds3_sum)

        
        seg_layer = self.output_segmentation2(ds1_ds2_sum_upscale_ds3_sum_upscale)

        if not self.seg:
            output = F.softmax(seg_layer, dim=1)
        else:
            output = torch.sigmoid(seg_layer)

        return output