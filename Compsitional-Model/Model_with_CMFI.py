import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import einops
Base = 32
class Mlp_2d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

## Original efficient additive attention code from https://github.com/Amshaker/SwiftFormer/blob/main/models/swiftformer.py ##
class X_EfficientAdditiveAttnetion(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        ### For Image ###
        self.to_query_I = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key_I = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g_I = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor_I = token_dim ** -0.5
        self.Proj_I1 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_I2 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_I3 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_I4 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final_I = nn.Linear(token_dim * num_heads, token_dim)
        ### For MetaData ###
        self.to_query_M = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key_M = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g_M = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor_M = token_dim ** -0.5
    def forward(self, x_I, x_M):
        query_I = self.to_query_I(x_I)
        key_I = self.to_key_I(x_I)
        query_M = self.to_query_M(x_M)
        key_M = self.to_key_M(x_M)
        query_I = torch.nn.functional.normalize(query_I, dim=-1)
        key_I = torch.nn.functional.normalize(key_I, dim=-1)
        query_M = torch.nn.functional.normalize(query_M, dim=-1)
        key_M = torch.nn.functional.normalize(key_M, dim=-1)
        ### For Image Features ###
        query_weight_I = query_I @ self.w_g_I
        A_I = query_weight_I * self.scale_factor_I
        A_I = torch.nn.functional.normalize(A_I, dim=1)
        G_I = torch.sum(A_I * query_I, dim=1)
        G_I = einops.repeat(G_I, "b d -> b repeat d", repeat=key_M.shape[1])
        ### For MetaData Features ###
        query_weight_M = query_M @ self.w_g_M
        A_M = query_weight_M * self.scale_factor_M
        A_M = torch.nn.functional.normalize(A_M, dim=1)
        G_M = torch.sum(A_M * query_M, dim=1)
        G_M = einops.repeat(G_M, "b d -> b repeat d", repeat=key_M.shape[1])
        out_I = self.Proj_I1(G_I * key_I) + self.Proj_I2(G_I * key_M) + self.Proj_I3(G_M * key_M) + self.Proj_I4(G_M * key_I) + query_I + query_M
        out_I = self.final_I(out_I)
        return out_I
class CMFI_Module(nn.Module):
    def __init__(self, out=16):
        super(CMFI_Module, self).__init__()
        self.attn_X = X_EfficientAdditiveAttnetion(in_dims=out, token_dim=out, num_heads=2)
        drop_path = 0.
        layer_scale_init_value = 1e-5
        self.drop_path_I = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_M = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1_I = nn.Parameter(layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_2_I = nn.Parameter(layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_1_M = nn.Parameter(layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_2_M = nn.Parameter(layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.linear_2d_I = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)
        self.linear_2d_M = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_I, x_M):
        B1, C1, H1, W1 = x_I.shape
        B2, C2, H2, W2 = x_M.shape
        x1_I = self.attn_X(x_I.permute(0, 2, 3, 1).reshape(B1, H1 * W1, C1), x_M.permute(0, 2, 3, 1).reshape(B2, H2 * W2, C2))
        x_I = x_I + self.drop_path_I(self.layer_scale_1_I * x1_I.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2))
        x_I = x_I + self.drop_path_I(self.layer_scale_2_I * self.linear_2d_I(x_I))
        return x_I

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
class Conv_11(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DoubleConv_s2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.conv_s2 = nn.Sequential(
            DoubleConv_s2(in_channels, out_channels,stride)
        )
    def forward(self, x):
        return self.conv_s2(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels,stride,increase_channels = True):
        super().__init__()
        if increase_channels:
            if (in_channels%out_channels) ==0:
                mid_channel = in_channels
            else:
                mid_channel = in_channels + Base*3
        else:
            mid_channel = in_channels + in_channels//2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=stride, stride=stride)
        self.conv = DoubleConv(mid_channel , out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class SingleLinear(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        return self.linear_1(x)
    
class MetaDataMLP1(nn.Module):  ## This MLP is for MnM2 data having all classification nodes
    def __init__(self):
        super().__init__()
        self.m_inc1 = SingleLinear(4, Base , 0.1)
        self.m_inc2 = SingleLinear(Base , 2*Base , 0.1)
        self.m_inc3 = SingleLinear(2*Base , 4*Base , 0.1)
        self.m_inc4 = SingleLinear(4*Base , 8*Base , 0.1)
        self.m_inc5 = SingleLinear(8*Base , 10*Base , 0.1)
        self.m_inc6 = SingleLinear(10*Base , 10*Base , 0.1)
        self.linear1 = nn.Linear(10*Base , 128)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.act_R = nn.ReLU(inplace=True)
        self.linear_v = nn.Linear(128, 3)
        self.linear_s = nn.Linear(128, 9)
        self.linear_d = nn.Linear(128, 6)
        self.linear_f = nn.Linear(128, 2)
    def forward(self, Meta_Data):
        m1 = self.m_inc1(Meta_Data)
        m2 = self.m_inc2(m1)
        m3 = self.m_inc3(m2)
        m4 = self.m_inc4(m3)
        m5 = self.m_inc5(m4)
        m6 = self.m_inc6(m5)
        m = self.act(self.dropout(self.linear1(m6)))
        logits_v = self.linear_v(m)
        logits_s = self.linear_s(m)
        logits_d = self.linear_d(m)
        logits_f = self.linear_f(m)
        return logits_v, logits_s, logits_d, logits_f, m1, m2, m3, m4, m5, m6
class MetaDataMLP2(nn.Module): ## This MLP is for CAMUS data having classification + Regression nodes
    def __init__(self):
        super().__init__()
        self.m_inc1 = SingleLinear(7, Base , 0.1)
        self.m_inc2 = SingleLinear(Base , 2*Base , 0.1)
        self.m_inc3 = SingleLinear(2*Base , 4*Base , 0.1)
        self.m_inc4 = SingleLinear(4*Base , 8*Base , 0.1)
        self.m_inc5 = SingleLinear(8*Base , 16*Base , 0.1)
        self.m_inc6 = SingleLinear(16*Base , 20*Base , 0.1)
        self.linear1 = nn.Linear(20*Base , 128)
        self.linear_sex = nn.Linear(128, 2)
        self.linear_img_Q = nn.Linear(128, 3)
        self.linear_es = nn.Linear(128, 1)
        self.linear_nb_frame = nn.Linear(128, 1)
        self.linear_age = nn.Linear(128, 1)
        self.linear_ef = nn.Linear(128, 1)
        self.linear_frame_rate = nn.Linear(128, 1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.act_R = nn.ReLU(inplace=True)

    def forward(self, Meta_Data):
        m1 = self.m_inc1(Meta_Data)
        m2 = self.m_inc2(m1)
        m3 = self.m_inc3(m2)
        m4 = self.m_inc4(m3)
        m5 = self.m_inc5(m4)
        m6 = self.m_inc6(m5)
        m = self.act(self.dropout(self.linear1(m6)))
        logits_sex = self.linear_sex(m)
        logits_img_Q  = self.linear_img_Q(m)
        logits_es = self.act_R(self.linear_es(m))
        logits_nb_frame = self.act_R(self.linear_nb_frame(m))
        logits_nb_age = self.act_R(self.linear_age(m))
        logits_ef  = self.act_R(self.linear_ef(m))
        logits_frame_rate  = self.act_R(self.linear_frame_rate(m))

        return logits_sex,logits_img_Q,logits_es,logits_nb_frame,logits_nb_age,logits_ef,logits_frame_rate, m1, m2, m3, m4, m5, m6
    

class OutConv(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class CompSegNet(nn.Module):   ##   CompSegNet --> Compositional Segmentaiton Network
    def __init__(self, n_channels = 1):
        super(CompSegNet, self).__init__()
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, Base)
        self.down1 = Down(Base, 2*Base,2)
        self.down2 = Down(2*Base, 4*Base,2)
        self.down3 = Down(4*Base, 8*Base,2)
        self.down4 = Down(8*Base, 10*Base,2)
        self.down5 = Down(10*Base, 10*Base,2)
        self.up0 = Up(10*Base, 10*Base,2, False)
        self.up1 = Up(10*Base, 8*Base,2)
        self.up2 = Up(8*Base, 4*Base,2)
        self.up3 = Up(4*Base, 2*Base,2)
        self.up4 = Up(2*Base, Base,2)
        self.up0_ = Up(10*Base, 10*Base,2,False)
        self.up1_ = Up(10*Base, 8*Base,2)
        self.up2_ = Up(8*Base, 4*Base,2)
        self.up3_ = Up(4*Base, 2*Base,2)
        self.up4_ = Up(2*Base, Base,2)
        self.outc = OutConv(Base,2)
        self.outc4 = OutConv(Base,4)
        self.dropout6E = nn.Dropout3d(p=0.30) 
        self.X_1 = CMFI_Module(out=Base)
        self.X_2 = CMFI_Module(out=2*Base)
        self.X_3 = CMFI_Module(out=4*Base)
        self.X_4 = CMFI_Module(out=8*Base)
        self.X_5 = CMFI_Module(out=10*Base)
        self.X_6 = CMFI_Module(out=10*Base)
        self.D_1 = CMFI_Module(out=Base)
        self.D_2 = CMFI_Module(out=2*Base)
        self.D_3 = CMFI_Module(out=4*Base)
        self.D_4 = CMFI_Module(out=8*Base)
        self.D_5 = CMFI_Module(out=10*Base)
    def forward(self, x, m1, m2, m3, m4, m5, m6):
        ## Expanding MetaData Dimenssions ##
        m1 = m1.unsqueeze(-1).unsqueeze(-1)
        m2 = m2.unsqueeze(-1).unsqueeze(-1)
        m3 = m3.unsqueeze(-1).unsqueeze(-1)
        m4 = m4.unsqueeze(-1).unsqueeze(-1)
        m5 = m5.unsqueeze(-1).unsqueeze(-1)
        m6 = m6.unsqueeze(-1).unsqueeze(-1)
        ## Image Encoder ###
        x1 = self.inc(x)
        m1 = m1.expand((m1.size()[:-2]  + x1.size()[-2:] ))
        x1 = self.X_1(x1,m1)
        x2 = self.down1(x1)
        m2 = m2.expand((m2.size()[:-2]  + x2.size()[-2:] ))
        x2 = self.X_2(x2,m2)
        x3 = self.down2(x2)
        m3 = m3.expand((m3.size()[:-2]  + x3.size()[-2:] ))
        x3 = self.X_3(x3,m3)
        x4 = self.down3(x3)
        m4 = m4.expand((m4.size()[:-2]  + x4.size()[-2:] ))
        x4 = self.X_4(x4,m4)
        x5 = self.down4(x4) 
        m5 = m5.expand((m5.size()[:-2]  + x5.size()[-2:] ))
        x5 = self.X_5(x5,m5)
        x6 = self.down5(x5) 
        m6 = m6.expand((m6.size()[:-2]  + x6.size()[-2:] ))
        x6 = self.X_6(x6,m6)
        x6 = self.dropout6E(x6)
         ## Super-Segmentation Decoder ##
        z1 = self.up0(x6, x5)
        z1 = self.D_5(z1,m5)
        z2 = self.up1(z1, x4)
        z2 = self.D_4(z2,m4)
        z3 = self.up2(z2, x3)
        z3 = self.D_3(z3,m3)
        z4 = self.up3(z3, x2)
        z4 = self.D_2(z4,m2)
        z5 = self.up4(z4, x1)
        z5 = self.D_1(z5,m1)
        logits2 = self.outc(z5)
        ## Sub-Segmentation Decoder ##
        y1 = self.up0_(x6, z1)
        y2 = self.up1_(y1, z2)
        y3 = self.up2_(y2, z3)
        y4 = self.up3_(y3, z4)
        y5 = self.up4_(y4, z5)
        logits4 = self.outc4(y5)
        return logits2,logits4
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.SegNetwork = CompSegNet()
        self.ClassNetwork = MetaDataMLP1() # MetaDataMLP2() 
    def forward(self, x, Meta_Data):
        logits_v, logits_s, logits_d, logits_f, m1, m2, m3, m4, m5, m6 = self.ClassNetwork(Meta_Data)
        logits2, logits4 = self.SegNetwork(x, m1, m2, m3, m4, m5, m6)
        return logits2, logits4,logits_v, logits_s, logits_d, logits_f
    
    

from torchsummary import summary
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    model.to(device=DEVICE, dtype=torch.float)
    summary(model, [(1, 256, 256), (4,)])
    
    image = torch.randn(10, 1, 256, 256).to(device=DEVICE, dtype=torch.float)
    meta = torch.randn(10, 4).to(device=DEVICE, dtype=torch.float)
    output = model(image, meta)
