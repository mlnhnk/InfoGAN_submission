# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def fullycon(in_channels, out_channels, batch_norm=True, init_zero_weights=False):
    """Creates a fully connected layer, with optional batch normalization.
    """
    layers = []
    fc_layer = nn.Linear(in_channels, out_channels)
    if init_zero_weights:
        fc_layer.weight.data = torch.randn(out_channels, in_channels) * 0.001
    layers.append(fc_layer)

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        
        self.fc1 = fullycon(noise_size, 1024, batch_norm=True)
#        self.fc1 = nn.Linear(noise_size, 1024)
        self.fc2 = fullycon(1024, 7 * 7 * 128, batch_norm=True)
#        self.fc2 = nn.Linear(1024, 7 * 7 * 128)
        self.deconv1 = deconv(128, 64, 4, batch_norm=True, stride=2, padding=1)
        self.deconv2 = deconv(64, 1, 4, batch_norm=False, stride=2, padding=1)
        

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(-1, 128, 7, 7)
        out = F.relu(self.deconv1(out))
        out = self.deconv2(out)
        return F.sigmoid(out)


class SharedPartDQ(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self):
        super(SharedPartDQ, self).__init__()        
#        input 28x28x1 Gray image
#        4x4 conv layer 64 Leaky ReLU stride 2
        self.conv1 = conv(1, 64, 4, stride=2, padding=1, batch_norm=False)
#        4 × 4 conv. 128 lRELU. stride 2. batchnorm 
        self.conv2 = conv(64, 128, 4, stride=2, padding=1, batch_norm=True)
#        FC. 1024 lRELU. batchnorm         
        self.fc1 = fullycon(128 * 7 * 7, 1024, batch_norm=True)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.1)
        out = out.view(-1, 128*7*7)
        out = F.leaky_relu(self.fc1(out), negative_slope=0.1)
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # FC. output layer for D,
        self.discr_fc2 = fullycon(1024, 1, batch_norm=False)

        
    def forward(self, x):
        out = self.discr_fc2(x)
        return F.sigmoid(out)
    
    
class Recognition(nn.Module):
    def __init__(self, categorical_dims, continuous_dims):
        super(Recognition, self).__init__()
        
        self.cat_dims = categorical_dims
        self.cont_dims = continuous_dims
        
        self.recog_fc2 = fullycon(1024, 128, batch_norm=True)
        
        # Layers to get categorical and continous values
        self.cat_fc = fullycon(128, self.cat_dims, batch_norm=False)
        self.cont_mu_fc = fullycon(128, self.cont_dims, batch_norm=False)
        self.cont_sigma_fc = fullycon(128, self.cont_dims, batch_norm=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.recog_fc2(x), negative_slope=0.1)
        
        #TODO MISSCHEIN SOFTMAX TEOVOEGEN?
        cat_out = self.cat_fc(out)
        
        cont_mu_out = self.cont_mu_fc(out)
        cont_sigma_out = self.cont_sigma_fc(out)
        
        return cat_out.squeeze(), cont_mu_out.squeeze(), cont_sigma_out.squeeze().exp()