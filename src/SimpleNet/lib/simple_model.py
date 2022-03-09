from this import d
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, output_size: int):
        super(SimpleNet, self).__init__()

        # in (64, 5, 3, 224, 224)
        # in (64, 32, 3, 112, 112) conv
        # in (64, 64, 3, 56, 56) conv2
        # in (64, 128 * 3 * 28 * 28) flat
        # in (64, 128, -) lineal (1024)
        # relu
        # in (64, 128, -) lineal (512)
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
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=2),#mover a parametro
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
