import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class SimpleNet(nn.Module):
    def __init__(self, output_size: int):
        super(SimpleNet, self).__init__()

        # in (64, 5, 3, 224, 224)
        # in (64, 32, 3, 112, 112) conv
        # in (64, 64, 3, 56, 56) conv2
        # in (64, 128 * 3 * 28 * 28) flat
        # in (64, 128, -) lineal (1024)
        # relu
        # in (64, 128, -) lineal (512)s
        # relu
        # in (64, 128, -) lineal (num_classes)
        # softmax

        nn.NLLLoss()

        # in (64, 32) maxpool

        self.conv3d = self._conv_layer_set(5, 32)
        self.conv3d_2 = self._conv_layer_set(32, 64)
        self.flat = nn.Flatten(start_dim=2)
        self.fc1 = nn.Linear(32, 64)

        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm3d(5)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=3, padding=1, stride=2
            ),  # mover a parametro
            nn.LeakyReLU(),
            # nn.MaxPool3d((2)),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv3d(x)
        out = self.conv3d_2(out)

        out = self.relu(out)
        out = self.batch(out)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.fc2(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(5, 100)
        self.conv_layer2 = self._conv_layer_set(100, 64)
        self.fc1 = nn.Linear(2**3 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 32, 32), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        # out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (
        np.floor(
            (img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1
        ).astype(int),
        np.floor(
            (img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1
        ).astype(int),
        np.floor(
            (img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1
        ).astype(int),
    )
    return outshape


class CNN3D(nn.Module):
    def __init__(
        self,
        t_dim=120,
        img_x=90,
        img_y=120,
        drop_p=0.2,
        fc_hidden1=256,
        fc_hidden2=128,
        num_classes=50,
    ):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1
        )
        self.conv2_outshape = conv3D_output_size(
            self.conv1_outshape, self.pd2, self.k2, self.s2
        )

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.ch1,
            kernel_size=self.k1,
            stride=self.s1,
            padding=self.pd1,
        )
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(
            in_channels=self.ch1,
            out_channels=self.ch2,
            kernel_size=self.k2,
            stride=self.s2,
            padding=self.pd2,
        )
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(
            self.ch2
            * self.conv2_outshape[0]
            * self.conv2_outshape[1]
            * self.conv2_outshape[2],
            self.fc_hidden1,
        )  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(
            self.fc_hidden2, self.num_classes
        )  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x


# class Net(nn.Module):
#     def __init__(self, num_classes, size: Tuple[int, int, int]):
#         super(Net, self).__init__()

#         batch_size, n_images, image_size = size

#         self.conv1 = nn.Conv3d(n_images, batch_size, kernel_size=image_size)
#         self.conv2 = nn.Conv3d(batch_size, 20, kernel_size=image_size)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool3d(self.conv1(x), 2))
#         x = F.relu(F.max_pool3d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# Simple Net that recieves vector of images using pytorch module.
# class Net(nn.Module):
#     def __init__(self, num_classes, size: Tuple[int, int, int]):
#         super(Net, self).__init__()

#         batch_size, n_images, image_size = size

#         self.conv1 = nn.Conv3d(n_images, batch_size, kernel_size=3)
#         self.conv2 = nn.Conv3d(batch_size, batch_size, kernel_size=image_size)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool3d(self.conv1(x), 2))
#         x = F.relu(F.max_pool3d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
    ):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.temporal_spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
    the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the block.
        kernel_size (int or tuple): Size of the convolving kernels.
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(
                in_channels, out_channels, 1, stride=2
            )
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(
                in_channels, out_channels, kernel_size, padding=padding, stride=2
            )
        else:
            self.conv1 = SpatioTemporalConv(
                in_channels, out_channels, kernel_size, padding=padding
            )

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(
            out_channels, out_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layer_size,
        block_type=SpatioTemporalResBlock,
        downsample=False,
    ):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(
            5, 64, [3, 32, 32], stride=[1, 2, 2], padding=[1, 3, 3]
        )
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(
            64, 64, 3, layer_sizes[0], block_type=block_type
        )
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(
            64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True
        )
        self.conv4 = SpatioTemporalResLayer(
            128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True
        )
        self.conv5 = SpatioTemporalResLayer(
            256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True
        )

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class R3DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(
        self,
        num_classes,
        layer_sizes,
        block_type=SpatioTemporalResBlock,
        pretrained=False,
    ):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    import torch

    inputs = torch.rand(1, 3, 16, 112, 112)
    net = R3DClassifier(101, (2, 2, 2, 2), pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())
