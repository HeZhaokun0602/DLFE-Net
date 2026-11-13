from einops import rearrange
from net.transformer_utils import *
import torch

class LDR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 base_dilation_rates=None,
                 kernel_size=3):
        super(LDR, self).__init__()

        self.input_conv = nn.Conv2d(inp_channels, dim, kernel_size=1, bias=False)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=rate, dilation=rate, bias=False)
            for rate in base_dilation_rates
        ])
        self.combine_weights = nn.Parameter(torch.randn(len(base_dilation_rates)), requires_grad=True)
        self.output_conv = nn.Conv2d(dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.input_conv(x)
        conv_outputs = torch.stack([conv(x) for conv in self.conv_layers], dim=2)
        combine_weights = self.combine_weights.softmax(dim=0).view(1, 1, -1, 1, 1)
        combined_output = torch.sum(conv_outputs * combine_weights, dim=2)
        x = self.output_conv(combined_output)
        return x

class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)       # [b,1,c]
        y = self.conv(y).view(b, c, 1, 1)        # 1D卷积替代FC
        return x * self.sigmoid(y)

class NoiseEstimator(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48):
        super(NoiseEstimator, self).__init__()
        self.level1 = LDR(inp_channels=inp_channels, out_channels=dim // 2, dim=dim, kernel_size=3,
                          base_dilation_rates=[1, 2, 3])
        self.level2 = LDR(inp_channels=inp_channels, out_channels=dim, dim=dim, kernel_size=3,
                          base_dilation_rates=[4, 5, 6])
        self.level3 = LDR(inp_channels=inp_channels, out_channels=dim, dim=dim, kernel_size=3,
                          base_dilation_rates=[7, 8, 9])
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2 + dim // 2, dim * 2, kernel_size=1),
            ECALayer(dim * 2),
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        l1 = self.level1(x)
        l2 = self.level2(x)
        l3 = self.level3(x)
        noise_map = self.fuse(torch.cat([l1, l2, l3], dim=1))
        output = self.out_conv(noise_map)
        return output

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, sigma=0.2, low_ratio=0.5):
        super(FourierUnit, self).__init__()
        self.low_ratio = low_ratio
        self.sigma = sigma

        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False),
            LayerNorm(out_channels),
            nn.GELU()
        )

        self.conv_high = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

        self.mask_gen = nn.Parameter(torch.randn(1))  # 控制衰减强度

    def forward(self, x):
        batch, c, h, w = x.size()

        # 生成低频高斯掩码
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(0, 1, w // 2 + 1, device=x.device),
            indexing='ij'  # 明确指定索引顺序
        )
        dist = torch.sqrt(yy ** 2 + xx ** 2)
        mask = torch.exp(-(dist / (self.sigma * self.mask_gen.abs())))

        # FFT处理
        ffted = torch.fft.rfft2(x, norm='ortho')

        # 低频增强路径
        ffted_low = ffted * mask
        x_low = self._process_complex(ffted_low, self.conv_low)

        # 高频残差路径
        x_high = self.conv_high(x)  # 空间域直接处理

        # 合并结果
        output = torch.fft.irfft2(x_low, s=(h, w), norm='ortho') + x_high * 0.1  # 弱化高频影响
        return output

    def _process_complex(self, ffted, conv):
        """处理复数信号通用方法"""
        real = torch.unsqueeze(torch.real(ffted), -1)
        imag = torch.unsqueeze(torch.imag(ffted), -1)
        ffted = torch.cat([real, imag], -1).permute(0, 1, 4, 2, 3).flatten(1, 2)
        return conv(ffted)

class Freq_Fusion(nn.Module):
    def __init__(self, dim, sigma):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.FFC = FourierUnit(self.dim*2, self.dim*2, sigma)
        self.norm = LayerNorm(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),  # 保持通道数一致
            nn.Sigmoid()  #输出0到1区间
        )

    def forward(self, x, y):
        x0 = torch.cat([x, y], dim=1)
        gate = self.gate(x0)
        x = self.FFC(x0) + gate * x0
        x = self.relu(self.norm(x))
        return x

class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(self, dim, sigma):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer = Freq_Fusion(dim, sigma)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = LayerNorm(dim)
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2, groups=self.dim, padding_mode='reflect'),
            #nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2, groups=self.dim, padding_mode='reflect'),
            #nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_init(x)  #放大通道数*2
        x = list(torch.split(x, self.dim, dim=1))   #将x平分通道
        x_local_1 = self.dw_conv_1(x[0])   #对于每个x通过不同的核卷积
        x_local_2 = self.dw_conv_2(x[1])

        x_gloal = self.mixer(x_local_1, x_local_2)   #特征融合
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x
        return x

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias, sigma, flag_noise=False, flag_enhancement=False):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.flag_enhancement = flag_enhancement
        self.flag_noise = flag_noise

        if self.flag_noise:
            self.ldr_qkv = NoiseEstimator(inp_channels=dim, out_channels=dim, dim=dim // 4)
            self.gate = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid()
            )
            self.residual_path = nn.Conv2d(dim, dim, kernel_size=1)  # 特征变换

        elif self.flag_enhancement:
            self.Fused_Fourier_Conv_Mixer = Fused_Fourier_Conv_Mixer(dim=dim, sigma=sigma)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        if self.flag_noise:
            noise = self.ldr_qkv(x)
            gate = self.gate(x)
            modulated = self.residual_path(x) - noise
            x = x + gate * modulated
            q = self.q_dwconv(self.q(x))

        elif self.flag_enhancement:
            x = x + self.Fused_Fourier_Conv_Mixer(x)
            q = self.q_dwconv(self.q(x))

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, sigma, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias, sigma=sigma, flag_noise=True)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, sigma, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias, sigma=sigma, flag_enhancement=True)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x
