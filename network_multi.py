import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet101
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torch.nn import ReLU, init

#use both the difference map and the consective two tactile images for force estimation


# Define the custom CNN for initial processing

import torch.nn.init as init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

class InitialCNN(nn.Module):
    def __init__(self, is_rgb=True):
        super(InitialCNN, self).__init__()
        if is_rgb ==True:
            self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height*width).permute(0, 2, 1)  # B X N X C
        key = self.key_conv(x).view(batch_size, -1, height*width)  # B X C x N
        value = self.value_conv(x).view(batch_size, -1, height*width)  # B X C x N

        attention = torch.bmm(query, key)  # Batch matrix multiplication, B x N x N
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out
# Define the combined model
class Net(nn.Module):
    def __init__(self,is_rgb = False,is_normal = True):
        super(Net, self).__init__()
        self.initial_cnn1 = InitialCNN(is_rgb=is_rgb)
        self.initial_cnn2 = InitialCNN(is_rgb = is_rgb)



        self.resnet = resnet101(pretrained=False)

        # Modify the first convolutional layer of ResNet to accept the combined input channels
        self.resnet.conv1 = nn.Conv2d(512, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        

        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()



        if is_normal==True:
            self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                # nn.Dropout(0.3),  # Dropout for regularization
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                # nn.Sigmoid()  # Added Sigmoid activation for constrained output range
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),  # Dropout for regularization
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )


        self.attention1 =  GAM_Attention(in_channels=256,out_channels=256)
        self.attention2 =  GAM_Attention(in_channels=256,out_channels=256)
        self.attention3 =  GAM_Attention(in_channels=512,out_channels=512)


    def forward(self, img1_1, img1_2, img2_1, img2_2):
        # Process each image through the initial CNN
        combined_features1 = torch.cat((img1_1, img1_2), dim=1)
        combined_features2 = torch.cat((img2_1, img2_2), dim=1)
        features1 = self.initial_cnn1(combined_features1)
        features2= self.initial_cnn2(combined_features2)

        features1 = self.attention1(features1)
        features2 = self.attention2(features2)


        # Combine the features
        combined_features = torch.cat((features1, features2), dim=1)
        combined_features = self.attention3(combined_features)

        # Pass through ResNet-50
        resnet_features = self.resnet(combined_features)


        # Pass through the final fully connected layer
        output = self.fc(resnet_features)

        return output

    def get_embedding(self, img1_1, img1_2):
        combined_features1 = torch.cat((img1_1, img1_2), dim=1)
        features1 = self.initial_cnn1(combined_features1)
        return features1.view(features1.size(0), -1)


