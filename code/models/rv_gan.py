import torch
import torch.nn as nn
import torch.nn.functional as F


def encoder_block(X, down_filters, i):
    X = nn.Conv2d(down_filters, down_filters, kernel_size=(4, 4), stride=(2, 2))(X)
    X = nn.BatchNorm2d(down_filters)(X)
    X = nn.LeakyReLU()(X)
    return X


def decoder_block(X, up_filters, i):
    X = nn.ConvTranspose2d(up_filters, up_filters, kernel_size=(4, 4), stride=(2, 2))(X)
    X = nn.BatchNorm2d(up_filters)(X)
    X = nn.LeakyReLU()(X)
    return X


def novel_residual_block(X_input, in_filter, filters, base):
    X = X_input
    X = nn.ReflectionPad2d(1)(X)
    X = SeparableConv2d(in_filter, filters, (3, 3)).forward(X)
    X = nn.BatchNorm2d(filters)(X)
    X = nn.LeakyReLU()(X)

    # Branch 1 ext1
    X_branch_1 = nn.ReflectionPad2d(1)(X)
    X_branch_1 = SeparableConv2d(in_filter, filters, (3, 3)).forward(X_branch_1)
    X_branch_1 = nn.BatchNorm2d(filters)(X_branch_1)
    X_branch_1 = nn.LeakyReLU()(X_branch_1)

    # Branch 2
    X_branch_2 = nn.ReflectionPad2d(2)(X)
    X_branch_2 = SeparableConv2d(in_filter, filters, (3, 3)).forward(X_branch_2)
    X_branch_2 = nn.BatchNorm2d(filters)(X_branch_2)
    X_branch_2 = nn.LeakyReLU()(X_branch_2)
    X_add_branch_1_2 = torch.add(X_branch_2, X_branch_1)
    X = torch.add(X_input, X_add_branch_1_2)
    return X


def SFA(X, filters, i):
    X_input = X
    X = SeparableConv2d(filters, filters, (3, 3)).forward(X)
    X = nn.BatchNorm2d(filters)(X)
    X = nn.LeakyReLU()(X)
    X = torch.add(X_input, X)
    X = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1))(X)
    X = nn.BatchNorm2d(filters)(X)
    X = nn.LeakyReLU()(X)
    X = torch.add(X_input, X)
    return X


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def coarse_generator(self, x_input, x_mask, x_coarse, x_coarse_shape=(256, 256, 64), input_shape=(512, 512, 3),
                     mask_shape=(512, 512, 1), ncf=64, n_downsampling=2, n_blocks=9, n_channels=1):
    X = torch.cat([x_input, x_mask])
    X = nn.ReflectionPad2d(3)(X)
    X = nn.Conv2d(3, ncf, (7, 7), stride=(1, 1))(X)
    X = nn.BatchNorm2d(ncf)(X)
    X_pre_down = nn.LeakyReLU()(X)

    down_filters = ncf * pow(2, 0) * 2
    X_down1 = encoder_block(X, down_filters, 0)
    down_filters = ncf * pow(2, 1) * 2
    X_down2 = encoder_block(X_down1, down_filters, 1)
    X = X_down2

    # Novel Residual Blocks
    res_filters = pow(2, n_downsampling)
    for i in range(n_blocks):
        X = novel_residual_block(X, down_filters, ncf * res_filters, base="block_" + str(i + 1))

    # Upsampling layers
    up_filters = int(ncf * pow(2, (n_downsampling - 0)) / 2)
    X_up1 = decoder_block(X, up_filters, 0)
    X_up1_att = SFA(X_down1, ncf * 2, 0)
    X_up1_add = torch.add(X_up1_att, X_up1)
    up_filters = int(ncf * pow(2, (n_downsampling - 1)) / 2)
    X_up2 = decoder_block(X_up1_add, up_filters, 1)
    X_up2_att = SFA(X_pre_down, ncf, 1)
    X_up2_add = torch.add(X_up2_att, X_up2)
    feature_out = X_up2_add
    X = nn.ReflectionPad2d(3)(X_up2_add)
    X = nn.Conv2d(ncf, n_channels, (7, 7), stride=(1, 1))(X)
    X = nn.Tanh()(X)
    return X


def fine_generator(x_input, x_mask, x_coarse, x_coarse_shape=(256, 256, 64), input_shape=(512, 512, 3),
                   mask_shape=(512, 512, 1), nff=64, n_blocks=3, n_coarse_gen=1, n_channels=1):
    for i in range(1, n_coarse_gen + 1):
        # Downsampling layers
        down_filters = nff * (2 ** (n_coarse_gen - i))
        X = torch.cat([x_input, x_mask])
        X = nn.ReflectionPad2d(3)(X)
        X = nn.Conv2d(3, down_filters, (7, 7), stride=(1, 1), padding='valid')(X)
        X = nn.BatchNorm2d(down_filters)(X)
        X_pre_down = nn.LeakyReLU()(X)

        X_down1 = encoder_block(X, down_filters, i - 1)
        # Connection from coarse generator
        X = torch.add(x_coarse, X_down1)
        X = SeparableConv2d(down_filters, down_filters * 2, (3, 3)).forward(X)
        X = nn.BatchNorm2d(down_filters * 2)(X)
        X = nn.LeakyReLU()(X)
        for j in range(n_blocks - 1):
            res_filters = nff * (2 ** (n_coarse_gen - i)) * 2
            X = novel_residual_block(X, down_filters * 2, res_filters, base="block_" + str(j + 1))

        # Upsampling layers
        up_filters = nff * (2 ** (n_coarse_gen - i))
        X_up1 = decoder_block(X, up_filters, i - 1)
        X_up1_att = SFA(X_pre_down, up_filters, i - 1)
        X_up1_add = torch.add(X_up1_att, X_up1)

    X = nn.ReflectionPad2d(3)(X_up1_add)
    X = nn.Conv2d(up_filters, n_channels, (7, 7), stride=(1, 1), padding='valid')(X)
    X = nn.Tanh()(X)
    return X
