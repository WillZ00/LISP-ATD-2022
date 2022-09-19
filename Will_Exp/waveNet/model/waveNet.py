import torch
from torch import nn
import torch.nn.init as init

class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=hyperparams['nb_filters'],
                                             kernel_size=hyperparams['kernel_size'],
                                             dilation=dilation_factor,
                                             )
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=hyperparams['nb_filters'],
                                         kernel_size=1,
                                         padding=0)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_causal_conv(x))
        #print(x1.shape)
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return x1 + x2


class WaveNet(nn.Module):
    def __init__(self, hyperparams: dict, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
             range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=5200,
                                      kernel_size=1)
        self.output_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.fc = nn.Linear(5200,5200)
        self.batch_norm1d = nn.BatchNorm1d(num_features=35)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.batch_norm1d(x)
        x=x.permute(0,2,1)
        for dilated_causal_conv in self.dilated_causal_convs:
            #print(x.shape)
            x = dilated_causal_conv(x)
            x = self.dropout(x)
            #print(x.shape)
        x = self.leaky_relu(self.output_layer(x))
        x = x.permute(0,2,1)

        x = self.fc(x)
        return x