import torch.nn as nn

class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()
        self.seq1 = self._create_block(1, 16, dropout_prob=0.3)
        self.seq2 = self._create_block(32, 16, dropout_prob=0.1)
        self.seq3 = self._create_block(32, 16, dropout_prob=0.1)
        self.seq4 = self._create_block(32, 16, dropout_prob=0)
        self.final_seq = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                     nn.Sigmoid())



    def forward(self,x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x=self.final_seq(x)
        return x




    @classmethod
    def _create_block(cls, input_channels, output_channels, skip_connections, dropout_prob):
        return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                       nn.Dropout2d(dropout_prob),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(),
                                       nn.Upsample(scale_factor=2),
                                       )
