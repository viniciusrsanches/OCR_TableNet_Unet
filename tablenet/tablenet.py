"""TableNet Module."""

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
#from torchvision.models import vgg19, vgg19_bn
from collections import OrderedDict

EPSILON = 1e-15


class TableNetModule(pl.LightningModule):
    """Pytorch Lightning Module for TableNet."""

    def __init__(self, num_class: int = 1, batch_norm: bool = False):

        super().__init__()
        self.model = TableNet(num_class, batch_norm)
        self.num_class = num_class
        self.dice_loss = DiceLoss()

    def forward(self, batch):

        return self.model(batch)

    def training_step(self, batch, batch_idx):

        samples, labels_table,labels_column = batch
        #samples, labels_table = batch
        output_table, output_column = self.forward(samples)
        #output = self.forward(samples)
        #output_table, output_column = output[:,0,::].unsqueeze(dim=1) , output[:,1,::].unsqueeze(dim=1)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)
        #loss_table = self.dice_loss(output, labels_table)
        #loss_table = self.dice_loss(output[:,1,:,:], labels_table[])

        self.log('train_loss_table', loss_table)
        self.log('train_loss_column', loss_column)
        self.log('train_loss', loss_column + loss_table)
        self.log('train_loss', loss_table)
        
        
        return loss_table + loss_column

    def validation_step(self, batch, batch_idx):

        samples, labels_table, labels_column = batch
        #samples, labels_table = batch
        output_table, output_column = self.forward(samples)
        #output = self.forward(samples)
        #output_table, output_column = output[:,0,::].unsqueeze(dim=1),output[:,1,::].unsqueeze(dim=1)
        #loss_table = self.dice_loss(output_table, labels_table)
        loss_table = self.dice_loss(output_table, labels_table)
        #loss_column = self.dice_loss(output_column, labels_column)
        loss_column = self.dice_loss(output_column, labels_column)
        #loss = self.dice_loss(output, labels_table)

        if batch_idx == 0:
            self._log_images("validation", samples, labels_table, labels_column, output_table, output_column)
            #self._log_images("validation", samples, labels_table, output)

        self.log('valid_loss_table', loss_table, on_epoch=True)
        self.log('valid_loss_column', loss_column, on_epoch=True)
        self.log('validation_loss', loss_column + loss_table, on_epoch=True)
        self.log('validation_loss', loss_table, on_epoch=True)
        self.log('validation_iou_table', binary_mean_iou(output_table, labels_table), on_epoch=True)
        self.log('validation_iou_column', binary_mean_iou(output_column, labels_column), on_epoch=True)
        return loss_table + loss_column        
        

    def test_step(self, batch, batch_idx):

        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)
        #output = self.forward(samples)
        #output_table, output_column = output[:,0,::].unsqueeze(dim=1) , output[:,1,::].unsqueeze(dim=1)


        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        if batch_idx == 0:
            self._log_images("test", samples, labels_table, labels_column, output_table, output_column)
            #self._log_images("test", samples, labels_table, labels_column, output_table, output_column)

        self.log('test_loss_table', loss_table, on_epoch=True)
        self.log('test_loss_column', loss_column, on_epoch=True)
        self.log('test_loss', loss_column + loss_table, on_epoch=True)
        self.log('test_iou_table', binary_mean_iou(output_table, labels_table), on_epoch=True)
        self.log('test_iou_column', binary_mean_iou(output_column, labels_column), on_epoch=True)
        return loss_table + loss_column

    def configure_optimizers(self):

        optimizer = optim.SGD(self.parameters(), lr=0.0001)
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=0.001, 
                                                       steps_per_epoch=204, 
                                                       epochs=500, 
                                                       pct_start=0.1),
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    def _log_images(self, mode, samples, labels_table, labels_column, output_table, output_column):
        self.logger.experiment.add_images(f'{mode}_generated_images', samples[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_labels_table', labels_table[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_labels_column', labels_column[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_output_table', output_table[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_output_column', output_column[0:4], self.current_epoch)


class TableNet(nn.Module):
    """TableNet. Unet from mateuszbuda github baseline code"""

    def __init__(self, num_class: int = 1, batch_norm: bool = False, in_channels: int = 3, init_features: int = 32):
        super(TableNet, self).__init__()
        out_channels = num_class
        features = init_features
        self.encoder1 = TableNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = TableNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = TableNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = TableNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = TableNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.upconv4_ = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = TableNet._block((features * 8) * 2, features * 8, name="dec4")
        self.decoder4_ = TableNet._block((features * 8) * 2, features * 8, name="dec4_")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.upconv3_ = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = TableNet._block((features * 4) * 2, features * 4, name="dec3")
        self.decoder3_ = TableNet._block((features * 4) * 2, features * 4, name="dec3_")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.upconv2_ = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = TableNet._block((features * 2) * 2, features * 2, name="dec2")
        self.decoder2_ = TableNet._block((features * 2) * 2, features * 2, name="dec2_")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.upconv1_ = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = TableNet._block(features * 2, features, name="dec1")
        self.decoder1_ = TableNet._block(features * 2, features, name="dec1_")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.conv_ = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4_ = self.upconv4_(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4_ = torch.cat((dec4_, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4_ = self.decoder4_(dec4_)
        dec3 = self.upconv3(dec4)
        dec3_ = self.upconv3_(dec4_)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3_ = torch.cat((dec3_, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3_ = self.decoder3_(dec3_)
        dec2 = self.upconv2(dec3)
        dec2_ = self.upconv2_(dec3_)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2_ = torch.cat((dec2_, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2_ = self.decoder2_(dec2_)
        dec1 = self.upconv1(dec2)
        dec1_ = self.upconv1_(dec2_)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1_ = torch.cat((dec1_, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1_ = self.decoder1_(dec1_)
        #conv1 = torch.sigmoid(self.conv(dec1)[:,0,::]).unsqueeze(dim=1)
        #conv1_ = torch.sigmoid(self.conv(dec1)[:,1,::]).unsqueeze(dim=1)
        return torch.sigmoid(self.conv(dec1)), torch.sigmoid(self.conv_(dec1_))
        #print(conv1[:,0,::].unsqueeze(dim=1).shape)
        #print (conv1_.shape)
        #return conv1 , conv1_
        #return torch.sigmoid(self.conv(dec1)), torch.sigmoid(self.conv(dec1_))
        #return conv1 , conv1_

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class DiceLoss(nn.Module):
    """Dice loss."""

    def __init__(self):
        """Dice Loss."""
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """Calculate loss.

        Args:
            inputs (tensor): Output from the forward pass.
            targets (tensor): Labels.
            smooth (float): Value to smooth the loss.

        Returns (tensor): Dice loss.

        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def binary_mean_iou(inputs, targets):
    """Calculate binary mean intersection over union.

    Args:
        inputs (tensor): Output from the forward pass.
        targets (tensor): Labels.

    Returns (tensor): Intersection over union value.
    """
    output = (inputs > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result
