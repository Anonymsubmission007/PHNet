from monai.networks.layers import DropPath, trunc_normal_
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            hidden_size_2: int,
            dropout_rate: float = 0.0,
    ):
        super(MLP, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size_2)
        self.fn = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size: tuple, hidden_size, dropout_rate=0.0):
        super(PatchEmbedding, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_embeddings = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_dim = hidden_size

    def forward(self, x):
        x2 = self.patch_embeddings(x)
        x_shape = x2.size()
        x2 = x2.flatten(2).transpose(1, 2)
        x2 = self.norm(x2)
        d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
        x2 = x2.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        x2 = self.dropout(x2)
        return x2


class PatchMerging3d(nn.Module):
    def __init__(self, dim, kernel_size=2, double=True):
        super(PatchMerging3d, self).__init__()
        if double:
            self.pool = nn.Conv3d(dim, 2 * dim, kernel_size=kernel_size, stride=kernel_size)
        else:
            self.pool = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x


class IP_MLP(nn.Module):
    def __init__(self, dim, segment_dim=14, qkv_bias=False, proj_drop=0.0):
        super(IP_MLP, self).__init__()
        self.segment_dim = segment_dim
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)  # dim = h*s = segment_dim * s
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = MLP(dim, dim // 4, dim * 3)
        self.attention_reweight = nn.Linear(segment_dim*segment_dim,segment_dim*segment_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D, C = x.shape
        S = C // self.segment_dim  # S is the number of segment
        
        m = x.reshape(B,H//self.segment_dim, self.segment_dim, W//self.segment_dim,self.segment_dim,D,C).permute(0,1,3,5,6,2,4).reshape(B,H//self.segment_dim,W//self.segment_dim,D,C,self.segment_dim*self.segment_dim)
        m = self.attention_reweight(m).reshape(B,H//self.segment_dim,W//self.segment_dim,D,C,self.segment_dim,self.segment_dim).permute(0,1,5,2,6,3,4).reshape(B, H, W, D, C) 

        h = x.transpose(2,1).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H*W//self.segment_dim, self.segment_dim, D,
                                                                                          self.segment_dim* S)
        h = self.mlp_h(h).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, W, H, D, C).transpose(2,1)

        w = x.reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H*W//self.segment_dim, self.segment_dim, D,
                                                                                         self.segment_dim * S)
        w = self.mlp_w(w).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H, W, D, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        x = x+m*x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TP_MLP(nn.Module):
    def __init__(self, dim, segment_dim=14, qkv_bias=False, proj_drop=0.0):
        super(TP_MLP, self).__init__()
        self.segment_dim = segment_dim
        self.mlp_d = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D, C = x.shape
        S = C // self.segment_dim
        d = x.reshape(B, H, W, D//self.segment_dim,self.segment_dim, self.segment_dim, S).permute(0, 1, 2, 3, 5,4,6).reshape(B, H, W, D//self.segment_dim, self.segment_dim, 
                                                                                      self.segment_dim * S)
        x = self.mlp_d(d).reshape(B, H, W, D//self.segment_dim, self.segment_dim,self.segment_dim , S).permute(0, 1, 2, 3, 5,4,6).reshape(B, H, W, D, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim, mlp_ratio=3.0, qkv_bias=False,
                 drop_path=0.0, skip_lam=1.0):
        super(PermutatorBlock, self).__init__()
        self.s_norm = nn.LayerNorm(dim)
        self.t_norm = nn.LayerNorm(dim)
        self.attn1 = IP_MLP(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, proj_drop=drop_path)
        self.attn2 = TP_MLP(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, proj_drop=drop_path)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(hidden_size=dim, mlp_dim=mlp_hidden_dim, hidden_size_2=dim, dropout_rate=drop_path)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = x + self.attn1(self.s_norm(x))/self.skip_lam
        x = x + self.drop_path(self.attn2(self.t_norm(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        x = x.permute(0, 4, 1, 2, 3)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3.0, qkv_bias=False, drop_path_rate=0.0, skip_lam=1.0):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop_path=block_dpr, skip_lam=skip_lam).to(device))
    blocks = nn.Sequential(*blocks)
    return blocks


class MLPP(nn.Module):
    def __init__(self, res_ratio, layers, in_channels=1,
                 embed_dims=None, segment_dim=None, mlp_ratios=3.0, skip_lam=1.0,
                 qkv_bias=False, dropout_rate=0.2,
                 ):
        super(MLPP, self).__init__()

        if res_ratio > 6 : # after two 2D downsample still anisotropic
            patch_size=(2, 2, 1)
        else:
            patch_size=(2, 2, 2)
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, hidden_size=embed_dims[0],
                                          dropout_rate=dropout_rate)

        self.network = []
        for i in range(len(layers)):
            self.network.append(
                basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                             drop_path_rate=dropout_rate, skip_lam=skip_lam))

            if i >= len(layers) - 1:
                break
            elif embed_dims[i + 1] == 2 * embed_dims[i]:
                self.network.append(PatchMerging3d(embed_dims[i]).to(device))
            else:
                self.network.append(PatchMerging3d(embed_dims[i], double=False).to(device))
        self.network = nn.Sequential(*self.network)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        hidden_states_out = []
        x1 = self.patch_embed(x)
        for i in range(len(self.network) // 2 + 1):
            if i >= len(self.network) // 2:
                x1 = self.network[2 * i](x1)
            else:
                hidden = self.network[2 * i](x1)  # MLPP
                hidden_states_out.append(hidden)
                x1 = self.network[2 * i + 1](x1)  # patch merging

        hidden_states_out.append(x1)
        return hidden_states_out
