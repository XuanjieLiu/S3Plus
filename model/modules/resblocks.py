import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride, padding):
        super().__init__()

        n_channels_in = int(n_channels_in)
        n_channels_out = int(n_channels_out)

        self.conv_0 = nn.Conv2d(
            n_channels_in, n_channels_out, kernel_size, stride, padding
        )
        self.bn_0 = nn.BatchNorm2d(n_channels_out)
        self.conv_1 = nn.Conv2d(
            n_channels_out, n_channels_out, kernel_size, (1, 1), padding
        )  # only the first layer has stride
        self.bn_1 = nn.BatchNorm2d(n_channels_out)
        self.act = nn.GELU()

        self.is_bottleneck = False
        if n_channels_in != n_channels_out:
            # shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    n_channels_in, n_channels_out, (1, 1), stride, (0, 0)
                ),  # no activation
                nn.BatchNorm2d(n_channels_out),
            )
            self.is_bottleneck = True

    def forward(self, x):
        if self.is_bottleneck:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = x + residual
        x = self.act(x)

        return x


class ResBlockTranspose(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride, padding):
        super().__init__()

        n_channels_in = int(n_channels_in)
        n_channels_out = int(n_channels_out)

        self.conv_0 = nn.ConvTranspose2d(
            n_channels_in, n_channels_in, kernel_size, stride, padding
        )
        self.bn_0 = nn.BatchNorm2d(n_channels_in)
        self.conv_1 = nn.ConvTranspose2d(
            n_channels_in, n_channels_out, kernel_size, stride, padding
        )
        self.bn_1 = nn.BatchNorm2d(n_channels_out)
        self.act = nn.GELU()

        self.is_bottleneck = False
        if n_channels_in != n_channels_out:
            # shortcut
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    n_channels_in, n_channels_out, (1, 1), 1, (0, 0)
                ),  # no activation
                nn.BatchNorm2d(n_channels_out),
            )
            self.is_bottleneck = True

    def forward(self, x):
        if self.is_bottleneck:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = x + residual
        x = self.act(x)

        return x
