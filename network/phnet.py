from torch import nn
from .mlpp import MLPP
from .basic_block import Up_sum, conv_layer


class PHNet(nn.Module):
    def __init__(self, res_ratio, layers, in_channels, out_channels, embed_dims, segment_dim, mlp_ratio, dropout_rate):
        super(PHNet,self).__init__()
        dim = (in_channels,embed_dims[0],embed_dims[1],embed_dims[2])
        self.conv = conv_layer(dim, res_ratio)
        self.mlpp = MLPP(res_ratio,layers,in_channels=embed_dims[-3],embed_dims=embed_dims[-2:],segment_dim=segment_dim,
                                    mlp_ratios=mlp_ratio,dropout_rate=dropout_rate)
        self.decoder_1 = Up_sum(in_chns=embed_dims[1], out_chns=embed_dims[0] ,kernel=(2,2,1),stride=(2,2,1),dropout=dropout_rate,halves=True)
        self.decoder_2 = Up_sum(in_chns=embed_dims[2], out_chns=embed_dims[1], kernel=(2,2,1), stride=(2,2,1), dropout=dropout_rate,
                                halves=True)
        self.decoder_3 = Up_sum(in_chns=embed_dims[3], out_chns=embed_dims[2], kernel=(2,2,1), stride=(2,2,1), dropout=dropout_rate,
                                halves=False)
        self.decoder_4 = Up_sum(in_chns=embed_dims[4], out_chns=embed_dims[3], kernel=2, stride=2, dropout=dropout_rate,
                                halves=True)
        self.final_conv = nn.Conv3d(embed_dims[0],out_channels=out_channels,kernel_size=1)

    def forward(self, x):
        x0, x1, x2 = self.conv(x)
        x3, x4 = self.mlpp(x2)
        u3 = self.decoder_4(x4,x3)
        u2 = self.decoder_3(u3,x2)
        u1 = self.decoder_2(u2,x1)
        u0 = self.decoder_1(u1,x0)
        logits = self.final_conv(u0)
        return logits
