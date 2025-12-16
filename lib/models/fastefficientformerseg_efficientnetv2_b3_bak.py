import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BatchNorm2d
from .efficientnetv2_b3 import EfficientNetV2_B3


# -------------------------
# 基础模块
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Lite Context Encoder
# 替代 SPPM（更快、更稳）
# -------------------------
class LiteContextEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.pwconv = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.ctx = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        y = self.dwconv(x)
        y = self.pwconv(y)
        y = self.bn(y)
        y = self.act(y)

        ctx = y.mean((2, 3), keepdim=True)
        ctx = self.ctx(ctx)
        return y * torch.sigmoid(ctx)


# -------------------------
# Token Fusion（替代 UAFM）
# -------------------------
class TokenFusion(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim):
        super().__init__()
        self.proj_h = nn.Conv2d(high_dim, out_dim, 1, bias=False)
        self.proj_l = nn.Conv2d(low_dim, out_dim, 1, bias=False)

        self.dwconv = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim, bias=False)
        self.pwconv = nn.Conv2d(out_dim, out_dim, 1, bias=False)
        self.bn = BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l):
        x_h = self.proj_h(x_h)
        x_h = F.interpolate(x_h, size=x_l.shape[2:], mode='nearest')
        x_l = self.proj_l(x_l)

        fuse = x_h + x_l
        fuse = self.dwconv(fuse)
        fuse = self.pwconv(fuse)
        return self.act(self.bn(fuse))


# -------------------------
# DDR 风格双分辨率解码
# -------------------------
class DualResDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.low_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.high_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, low, high):
        low_up = F.interpolate(low, size=high.shape[2:], mode='nearest')
        out = self.low_proj(low_up) + self.high_proj(high)
        return self.act(self.bn(out))


# -------------------------
# 轻量 Seg Head
# -------------------------
class LightSegHead(nn.Module):
    def __init__(self, dim, n_classes, scale_factor=8):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.bn = BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(dim, n_classes, 1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.act(self.bn(self.dwconv(x)))
        x = self.out(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


# =========================================================
# 主模型
# =========================================================
class FastEfficientFormerSeg_EfficientNetV2_B3(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        super().__init__()
        self.aux_mode = aux_mode
        self.use_fp16 = use_fp16
        self.img_size = img_size

        # 固定归一化（TRT 友好）
        self.register_buffer('mean', torch.tensor([120.0, 114.0, 104.0]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([70.0, 69.0, 73.0]).view(1, 3, 1, 1))

        # Backbone
        self.backbone = EfficientNetV2_B3()

        # Backbone 输出通道（与你原模型一致）
        self.c3_chan = 56    # stride 8
        self.c4_chan = 136   # stride 16
        self.c5_chan = 232   # stride 32

        # 投影
        self.proj_c3 = ConvBNReLU(self.c3_chan, 128, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, 128, ks=1, padding=0)
        self.proj_c5 = ConvBNReLU(self.c5_chan, 128, ks=1, padding=0)

        # Context Encoder
        self.context = LiteContextEncoder(128)

        # Fusion
        self.fuse_c5_c4 = TokenFusion(128, 128, 128)

        # DDR Decoder
        self.decoder = DualResDecoder(128)

        # Head
        self.head = LightSegHead(128, n_classes, scale_factor=8)

        # Aux heads（训练用）
        if self.aux_mode == 'train':
            self.aux_head_c4 = LightSegHead(128, n_classes, scale_factor=16)
            self.aux_head_c5 = LightSegHead(128, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # if self.aux_mode == 'pred':
            #     x = x.float()
            #     x = torch.flip(x, dims=[1])
            #     x = (x - self.mean) / self.std

            feat8, feat16, feat32 = self.backbone(x)

            c3 = self.proj_c3(feat8)
            c4 = self.proj_c4(feat16)
            c5 = self.proj_c5(feat32)

            c5 = self.context(c5)
            c4 = self.fuse_c5_c4(c5, c4)
            feat = self.decoder(c4, c3)

            logits = self.head(feat)

            if self.aux_mode == 'train':
                aux1 = self.aux_head_c4(c4)
                aux2 = self.aux_head_c5(c5)
                return logits, aux1, aux2
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                out = torch.argmax(logits, dim=1).float()
                return out
            else:
                raise NotImplementedError


# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    model = FastEfficientFormerSeg_EfficientNetV2_B3(
        n_classes=19,
        aux_mode='train',
        img_size=(640, 640)
    )

    x = torch.randn(2, 3, 640, 640)
    y = model(x)

    print("Main:", y[0].shape)
    print("Aux1:", y[1].shape)
    print("Aux2:", y[2].shape)
