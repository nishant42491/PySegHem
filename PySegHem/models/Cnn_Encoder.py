import torch
from torch import Tensor
import torch.nn as nn

import pytorch_lightning as pl


class Cnn_Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.seq1 = self._create_block(input_channels=1, output_channels=64, max_pool=False)
        self.dropout_after_seq1 = self._create_dropout(channel_size=64, dropout_prob=0.5)

        self.seq2 = self._create_block(input_channels=64, output_channels=128, max_pool=True)
        self.dropout_after_seq2 = self._create_dropout(channel_size=128, dropout_prob=0.5)

        self.seq3 = self._create_block(input_channels=128, output_channels=256, max_pool=True)
        self.seq3 = nn.Sequential(self.seq3,
                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),)
        self.dropout_after_seq3=self._create_dropout(channel_size=256, dropout_prob=0.5)

        self.seq4 = self._create_block(input_channels=256, output_channels=512)
        self.seq4 = nn.Sequential(self.seq4,
                                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(),)
        self.dropout_after_seq4=self._create_dropout(channel_size=512,dropout_prob=0.5)

        self.final_conv=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1),
                                      nn.Sigmoid(),)

        self.dropout_after_seq4 = self._create_dropout(channel_size=512, dropout_prob=0.5)

        self.upsample_seq2 = self._create_upsample(scale_factor=2)

        self.upsample_seq3 = self._create_upsample(scale_factor=4)

        self.upsample_seq4 = self._create_upsample(scale_factor=8)

    def forward(self, x:Tensor) -> Tensor:
        x = self.seq1(x)
        x_after_dropout1 = self.dropout_after_seq1(x)

        x = self.seq2(x)
        x_after_dropout2 = self.dropout_after_seq2(x)

        x = self.seq3(x)
        x_after_dropout3 = self.dropout_after_seq3(x)

        x = self.seq4(x)
        x_after_dropout4 = self.dropout_after_seq4(x)

        upsampled_2 = self.upsample_seq2(x_after_dropout2)
        upsampled_3 = self.upsample_seq3(x_after_dropout3)
        upsampled_4 = self.upsample_seq4(x_after_dropout4)

        concat_output=torch.cat([x_after_dropout1,upsampled_2,upsampled_3,upsampled_4], dim=1)

        out = self.final_conv(concat_output)

        return out






    @classmethod
    def _create_block(cls, input_channels: int, output_channels: int, max_pool: bool = True):

        if max_pool:
            return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    @classmethod
    def _create_dropout(cls, channel_size: int, dropout_prob: float):
        return nn.Sequential(nn.Conv2d(channel_size, 16, kernel_size=3, padding=1),
                             nn.Dropout2d(dropout_prob),)

    @classmethod
    def _create_upsample(cls, scale_factor: int):
        return nn.Upsample(scale_factor=scale_factor, mode='nearest')




