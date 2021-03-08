import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import *
from layers import *


class StyleTransfer(nn.Module):

    def __init__(self, style_img, content_layers, style_layers, alpha):
        super(StyleTransfer, self).__init__()
        self.alpha = alpha
        self.transform_net = TransformNet()
        self.loss_net = LossNet(style_img, content_layers, style_layers)
        self.optimizer = torch.optim.Adam(self.transform_net.parameters(), lr=1e-3)
    def forward(self, x):
        y_hat = self.transform_net(x)
        y = x
        content_loss, style_loss = self.loss_net(y_hat, y)
        total_loss = content_loss + self.alpha * style_loss
        total_loss.backward()
        return total_loss

    def train_batch(self, x):
        self.optimizer.zero_grad()
        loss = self.forward(x)
        loss.backward()
        self.optimizer.step()
        return loss

class ContentLoss(nn.Module):

    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target_feature = target_feature

    def forward(self, input_feature):
        self.loss = F.mse_loss(input_feature, self.target_feature)
        return input_feature


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_gramm = get_gramm_matrix(target_feature)

    def forward(self, input_feature):
        input_gramm = get_gramm_matrix(input_feature)
        self.loss = F.mse_loss(input_gramm, self.target_gramm)
        return input_feature


class LossNet(nn.Module):

    def __init__(self, style_img,
                 content_layers, style_layers):
        super(LossNet, self).__init__()
        self.content_layers = content_layers

        self.content_losses = []
        self.style_losses = []

        self.model = nn.Sequential()
        vgg = models.vgg19(pretrained=True).features.eval()

        i, j = 1, 1
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv{i}_{j}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{i}_{j}'
                layer = nn.ReLU(False)
                j += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{i}'
                i += 1
                j = 1
            else:
                raise RuntimeError('Unrecognized layer')

            self.model.add_module(name, layer)

            if name in style_layers:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module(f'sl{i}_{j-1}', style_loss)
                self.style_losses.append(style_loss)

    def forward(self, y_hat, y):
        content_loss, style_loss = 0, 0
        for name, module in self.model.named_children():
            y_hat = module(y_hat)
            y = module(y)

            if name in self.content_layers:
                content_loss += F.mse_loss(y_hat, y)

            if isinstance(module, StyleLoss):
                style_loss += module.loss

        return content_loss, style_loss


class TransformNet(nn.Module):

    def __init__(self):
        super(TransformNet, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module('conv1', ConvBlock(3, 32, 9, 1, 'relu'))
        self.seq.add_module('conv2', ConvBlock(32, 64, 3, 2, 'relu'))
        self.seq.add_module('conv3', ConvBlock(64, 128, 3, 2, 'relu'))
        self.seq.add_module('res1', ResBlock(128, 128))
        self.seq.add_module('res2', ResBlock(128, 128))
        self.seq.add_module('res3', ResBlock(128, 128))
        self.seq.add_module('res4', ResBlock(128, 128))
        self.seq.add_module('res5', ResBlock(128, 128))
        self.seq.add_module('up1', Upsample(128, 64))
        self.seq.add_module('up2', Upsample(64, 32))
        self.seq.add_module('conv_last', ConvBlock(32, 3, 9, 1, 'sigmoid'))

    def forward(self, x):
        return self.seq(x)
