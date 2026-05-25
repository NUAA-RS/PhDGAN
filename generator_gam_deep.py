import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from Resnetblock import ResnetBlock
import functools


class DeepUnfoldedGammaLayer(nn.Module):
    """深度展开的Gamma参数估计模块"""

    def __init__(self, window_size=7, init_weights=True):
        super().__init__()
        self.window_size = window_size
        self.eps = 1e-6

        # 均值估计卷积层
        self.conv_mu = nn.Conv2d(1, 1, window_size,
                                 padding=window_size // 2, bias=False)
        # 方差估计卷积层
        self.conv_var = nn.Conv2d(1, 1, window_size,
                                  padding=window_size // 2, bias=False)

        # 初始化权重为均值滤波器
        if init_weights:
            val = 1.0 / (window_size ** 2)
            nn.init.constant_(self.conv_mu.weight, val)
            nn.init.constant_(self.conv_var.weight, val)

        # 可学习参数增强
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 均值增强系数
        self.beta = nn.Parameter(torch.tensor(1.0))  # 方差增强系数

    def forward(self, x):
        # x: [B,1,H,W] 输入单通道SAR图像

        # 计算局部均值 (可学习滑动平均)
        mu = self.conv_mu(x) * self.alpha  # [B,1,H,W]

        # 计算局部方差 (可学习滑动方差)
        x_sq = x ** 2
        mu_sq = self.conv_var(x_sq) * self.beta  # [B,1,H,W]
        var = torch.clamp(mu_sq - mu ** 2, min=self.eps)  # 确保方差非负

        # 计算Gamma参数
        k = (mu ** 2) / (var + self.eps)  # 形状参数
        theta = var / (mu + self.eps)  # 尺度参数
        k = torch.clamp(k, min=1e-3)
        theta = torch.clamp(theta, min=1e-3)

        return torch.cat([k, theta], dim=1)  # 输出[B,2,H,W]







class EnhancedGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        if type(nn.BatchNorm2d) == functools.partial:
            use_bias = nn.BatchNorm2d.func == nn.InstanceNorm2d
        else:
            use_bias = nn.BatchNorm2d == nn.InstanceNorm2d
        # 散射特征提取模块
        # self.scattering_extractor = ScatteringFeatureExtractor(in_channels)
        self.gamma_layer = DeepUnfoldedGammaLayer()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # 特征融合模块
        self.fusion_conv = nn.Conv2d(514, 512, 3, padding=1)

        model2 = []
        ngf=64

        for i in range(3):  # add ResNet blocks

            model2 += [ResnetBlock(ngf * 8, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False,
                                   use_bias=use_bias)]
        self.model2 = nn.Sequential(*model2)
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )


        # 散射特征注意力
        self.attention = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取散射特征
        gamma_params = self.gamma_layer(x)
        # scattering_feat = self.scattering_extractor(x)  # [B,128,H,W]

        # 编码器特征
        enc_feat = self.encoder(x)  # [B,256,H/8,W/8]

        enc_feat = self.model2(enc_feat)
        # 特征融合
        gamma_feat = F.adaptive_avg_pool2d(gamma_params, enc_feat.shape[2:])
        #
        fused = torch.cat([enc_feat, gamma_feat], dim=1)
        fused = self.fusion_conv(fused)

        # 解码过程
        output = self.decoder(fused)
        # output = self.apply_gamma_constraint(dec, gamma_feat)

        # 散射特征注意力增强
        # attn = self.attention(dec)
        # attn = F.interpolate(attn, scale_factor=32, mode='bilinear')
        # output = dec * (1 + attn)  # 残差注意力

        return output

