"""
FastEfficientBiSeNet with Structural Reparameterization
支持训练时多分支，推理时自动融合为单分支，兼容 timm.utils.reparameterize_model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .cpubone_nano import CPUBone_Nano


def fuse_conv_bn(conv, bn):
    """
    将 Conv2d 和 BatchNorm2d 融合为单个 Conv2d
    用于重参数化时的 BN 融合
    """
    # 获取 Conv 参数
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

    # 获取 BN 参数
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_gamma = bn.weight
    bn_beta = bn.bias
    bn_eps = bn.eps

    # 计算融合后的参数
    bn_std = torch.sqrt(bn_var + bn_eps)
    fused_weight = conv_w * (bn_gamma / bn_std).reshape(-1, 1, 1, 1)
    fused_bias = bn_beta + (conv_b - bn_mean) * bn_gamma / bn_std

    # 创建新的 Conv
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias

    return fused_conv


class RepConvBNReLU(nn.Module):
    """
    重参数化卷积模块：训练时多分支（3x3 + 1x1 + identity），推理时单分支
    兼容 timm.utils.reparameterize_model
    """
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1,
                 use_identity=True, use_1x1=True):
        super(RepConvBNReLU, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_identity = use_identity and (in_chan == out_chan) and (stride == 1)
        self.use_1x1 = use_1x1

        # 主分支：ks x ks 卷积
        self.conv_main = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                                   padding=padding, groups=groups, bias=False)
        self.bn_main = BatchNorm2d(out_chan)

        # 1x1 分支
        if self.use_1x1:
            self.conv_1x1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride,
                                      padding=0, groups=groups, bias=False)
            self.bn_1x1 = BatchNorm2d(out_chan)

        # Identity 分支（仅当输入输出通道相同且 stride=1）
        if self.use_identity:
            self.bn_identity = BatchNorm2d(out_chan)

        self.relu = nn.ReLU(inplace=True)

        # 标记是否已融合
        self.is_fused = False
        self.fused_conv = None

    def forward(self, x):
        if self.is_fused:
            # 推理模式：使用融合后的单分支
            return self.relu(self.fused_conv(x))

        # 训练模式：多分支
        out = self.bn_main(self.conv_main(x))

        if self.use_1x1:
            out += self.bn_1x1(self.conv_1x1(x))

        if self.use_identity:
            out += self.bn_identity(x)

        return self.relu(out)

    def _pad_kernel_to_target_size(self, kernel, target_size):
        """将小的 kernel 填充到目标大小"""
        if kernel.size(2) == target_size and kernel.size(3) == target_size:
            return kernel

        pad_size = (target_size - kernel.size(2)) // 2
        return F.pad(kernel, [pad_size, pad_size, pad_size, pad_size])

    def _get_equivalent_kernel_bias(self):
        """获取等价的融合 kernel 和 bias"""
        # 1. 融合主分支 (ks x ks)
        kernel_main, bias_main = self._fuse_conv_bn(self.conv_main, self.bn_main)

        # 2. 融合 1x1 分支
        if self.use_1x1:
            kernel_1x1, bias_1x1 = self._fuse_conv_bn(self.conv_1x1, self.bn_1x1)
            # 将 1x1 kernel 填充到 ks x ks
            kernel_1x1 = self._pad_kernel_to_target_size(kernel_1x1, self.ks)
            kernel_main += kernel_1x1
            bias_main += bias_1x1

        # 3. 融合 identity 分支
        if self.use_identity:
            kernel_identity, bias_identity = self._get_identity_kernel_bias()
            kernel_main += kernel_identity
            bias_main += bias_identity

        return kernel_main, bias_main

    def _fuse_conv_bn(self, conv, bn):
        """融合 Conv 和 BN 层"""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias

    def _get_identity_kernel_bias(self):
        """获取 identity 分支的等价 kernel 和 bias"""
        # Identity 相当于一个对角矩阵
        input_dim = self.in_chan // self.groups
        kernel_value = torch.zeros((self.in_chan, input_dim, self.ks, self.ks),
                                   dtype=self.conv_main.weight.dtype,
                                   device=self.conv_main.weight.device)

        # 在中心位置设置 1
        center = self.ks // 2
        for i in range(self.in_chan):
            kernel_value[i, i % input_dim, center, center] = 1

        # 融合 BN
        running_mean = self.bn_identity.running_mean
        running_var = self.bn_identity.running_var
        gamma = self.bn_identity.weight
        beta = self.bn_identity.bias
        eps = self.bn_identity.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel_value * t
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias

    def fuse(self):
        """
        融合多分支为单分支
        此方法会被 timm.utils.reparameterize_model 调用
        """
        if self.is_fused:
            return self

        # 获取融合后的 kernel 和 bias
        kernel, bias = self._get_equivalent_kernel_bias()

        # 创建融合后的卷积层
        self.fused_conv = nn.Conv2d(
            self.in_chan,
            self.out_chan,
            kernel_size=self.ks,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias

        # 删除训练时的分支以节省内存
        self.__delattr__('conv_main')
        self.__delattr__('bn_main')
        if self.use_1x1:
            self.__delattr__('conv_1x1')
            self.__delattr__('bn_1x1')
        if self.use_identity:
            self.__delattr__('bn_identity')

        self.is_fused = True
        return self


class ConvBNReLU(nn.Module):
    """标准的卷积-BN-激活模块（保留用于不需要重参数化的地方）"""
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RepSPPF(nn.Module):
    """
    SPPF 模块（YOLOv5 风格）
    用一个 k=5 的 MaxPool 串联 3 次替代多尺度池化，尺寸不变，
    全程 stride=1，无需静态池化参数，对 TensorRT 非常友好。

    结构: Conv1x1(降维) -> [x, mp(x), mp(mp(x)), mp(mp(mp(x)))] -> Concat -> Conv1x1(融合)

    注：两个 1x1 卷积输入输出通道不同，无法启用 identity 分支，
    也未启用 1x1 并联分支，本质是单分支，故使用普通 ConvBNReLU
    而非重参数化模块。
    """
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_dim = in_channels // 2
        self.cv1 = ConvBNReLU(in_channels, hidden_dim, ks=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNReLU(hidden_dim * 4, out_channels, ks=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class RepUAFM(nn.Module):
    """
    Unified Attention Fusion Module（含重参数化输出卷积）

    conv_high/conv_low 是 1x1 投影，输入输出通道往往不同（high_chan/low_chan
    与 out_chan），无法启用 identity 分支，也未启用 1x1 并联分支，本质是单分支，
    故使用普通 ConvBNReLU。真正的多分支重参数化只在 conv_out（3x3, out_chan==out_chan）
    上生效。
    """
    def __init__(self, high_chan, low_chan, out_chan):
        super(RepUAFM, self).__init__()
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        # 注意力模块保持不变（通道数较小，重参数化收益不大）
        self.atten_conv = nn.Sequential(
            nn.Conv2d(out_chan, out_chan // 2, kernel_size=1, bias=False),
            BatchNorm2d(out_chan // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 2, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 输出卷积使用重参数化（3x3 卷积）
        self.conv_out = RepConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    def forward(self, x_high, x_low):
        high_feat = self.conv_high(x_high)
        low_feat = self.conv_low(x_low)

        # 上采样高层特征
        high_feat_up = F.interpolate(high_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=False)

        fuse = high_feat_up + low_feat
        atten = torch.mean(fuse, dim=(2, 3), keepdim=True)
        atten = self.atten_conv(atten)

        out = fuse * atten + low_feat
        return self.conv_out(out)


class RepSegmentationHead(nn.Module):
    """带重参数化的分割头"""
    def __init__(self, in_chan, n_classes, scale_factor=8):
        super(RepSegmentationHead, self).__init__()
        # 使用重参数化卷积
        self.conv = RepConvBNReLU(in_chan, 128, ks=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(128, n_classes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class FastEfficientBiSeNet_CPUBone_Nano_Reparam(nn.Module):
    """
    带结构重参数化的 FastEfficientBiSeNet

    特性：
    1. 仅在 in_chan==out_chan 的 3x3 卷积（RepUAFM.conv_out、
       RepSegmentationHead.conv）上启用真正的多分支重参数化
       （3x3 + 1x1 + identity）；1x1 投影/降维层通道不等，
       无法构成多分支，统一使用普通 ConvBNReLU。
    2. 推理时多分支自动融合为单分支，速度基本不变
    3. 兼容 timm.utils.reparameterize_model

    使用方法：
    ```python
    # 训练
    model = FastEfficientBiSeNet_FasterNet_T0_Reparam(n_classes=19)
    model.train()
    # ... training code ...

    # 推理前转换
    from timm.utils import reparameterize_model
    model.eval()
    model = reparameterize_model(model)  # 自动调用所有 RepConv 的 fuse() 方法
    # ... inference code ...
    ```
    """
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        """
        img_size: 用于计算 SPPM 静态池化参数，务必与实际输入一致。
        """
        super(FastEfficientBiSeNet_CPUBone_Nano_Reparam, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络
        self.backbone = CPUBone_Nano()

        # 通道定义 (需根据实际 Backbone 输出调整)
        self.c3_chan = 48  # Stride 8
        self.c4_chan = 96  # Stride 16
        self.c5_chan = 192  # Stride 32

        # 投影层（1x1，通道不等，无法启用 Rep 多分支，故用普通 ConvBNReLU）
        self.proj_c5 = ConvBNReLU(self.c5_chan, 128, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, 128, ks=1, padding=0)
        self.proj_c3 = ConvBNReLU(self.c3_chan, 128, ks=1, padding=0)

        # 2. SPPF (重参数化，尺寸无关，无需静态池化参数)
        self.sppf = RepSPPF(in_channels=128, out_channels=128, k=5)

        # 3. 融合模块 - 使用重参数化
        self.fuse_context = RepUAFM(high_chan=128, low_chan=128, out_chan=128)
        self.fuse_final = RepUAFM(high_chan=128, low_chan=128, out_chan=128)

        # 4. 输出头 - 使用重参数化
        self.head = RepSegmentationHead(128, n_classes, scale_factor=8)

        # 5. 辅助头
        if self.aux_mode == 'train':
            self.aux_head_c4 = RepSegmentationHead(128, n_classes, scale_factor=16)
            self.aux_head_c5 = RepSegmentationHead(128, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            H, W = x.size()[2:]

            # Encoder
            feat8, feat16, feat32 = self.backbone(x)

            # Projections
            c5 = self.proj_c5(feat32)
            c4 = self.proj_c4(feat16)
            c3 = self.proj_c3(feat8)

            # SPPF
            c5_sppf = self.sppf(c5)

            # Context Fusion（内部 conv_out 为重参数化 3x3 卷积）
            feat_context = self.fuse_context(c5_sppf, c4)

            # Spatial Fusion（内部 conv_out 为重参数化 3x3 卷积）
            feat_final = self.fuse_final(feat_context, c3)

            # Output Head（内部 conv 为重参数化 3x3 卷积）
            logits = self.head(feat_final)

            if self.aux_mode == 'train':
                aux_out1 = self.aux_head_c4(c4)
                aux_out2 = self.aux_head_c5(c5_sppf)
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                pred=pred[:, None, :, :]
                return pred
                # return pred.float()
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 测试重参数化功能
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print("=" * 80)
        print("测试带重参数化的模型")
        print("=" * 80)

        print(f"\n初始化模型 (image size: {img_height}x{img_width})...")
        net = FastEfficientBiSeNet_CPUBone_Nano_Reparam(
            n_classes=n_classes,
            aux_mode='train',
            img_size=(img_height, img_width)
        )
        net.eval()  # 切换到评估模式

        # 模拟输入
        if torch.cuda.is_available():
            net.cuda()
            in_ten = torch.randn(2, 3, img_height, img_width).cuda()
            print("使用 CUDA 设备")
            # 关闭 TF32：TF32 会在"多分支各自卷积再相加"与"融合后单次卷积"之间
            # 引入约 1e-3 量级的舍入路径差异（并非融合公式错误），
            # 关闭后可验证融合数学上的精确等价性（应降至 ~1e-5~1e-6）。
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            in_ten = torch.randn(2, 3, img_height, img_width)
            print("使用 CPU 设备")

        # 测试训练模式（多分支）
        print("\n" + "=" * 80)
        print("阶段 1: 训练模式（多分支结构）")
        print("=" * 80)

        # 统计重参数化模块数量
        rep_count = sum(1 for m in net.modules() if isinstance(m, RepConvBNReLU))
        print(f"模型中的重参数化模块数量: {rep_count}")

        with torch.no_grad():
            out_train = net(in_ten)
            if isinstance(out_train, tuple):
                out_train = out_train[0]

        print(f"训练模式输出形状: {out_train.shape}")

        # 测试推理模式（单分支融合）
        print("\n" + "=" * 80)
        print("阶段 2: 推理模式（融合为单分支）")
        print("=" * 80)

        # 方法1: 使用 timm.utils.reparameterize_model（推荐）
        try:
            from timm.utils import reparameterize_model
            print("使用 timm.utils.reparameterize_model 进行融合...")
            net_fused = reparameterize_model(net)
            use_timm = True
        except ImportError:
            print("timm 未安装，使用手动融合...")
            # 方法2: 手动调用 fuse()
            for module in net.modules():
                if hasattr(module, 'fuse'):
                    module.fuse()
            net_fused = net
            use_timm = False

        # 检查融合状态
        fused_count = sum(1 for m in net_fused.modules()
                         if isinstance(m, RepConvBNReLU) and m.is_fused)
        print(f"已融合的重参数化模块数量: {fused_count}/{rep_count}")

        with torch.no_grad():
            out_fused = net_fused(in_ten)
            if isinstance(out_fused, tuple):
                out_fused = out_fused[0]

        print(f"推理模式输出形状: {out_fused.shape}")

        # 验证输出一致性
        print("\n" + "=" * 80)
        print("阶段 3: 验证结果")
        print("=" * 80)

        diff = torch.abs(out_train - out_fused).max().item()
        print(f"训练模式 vs 推理模式最大差异: {diff:.6f}")

        # 注：验证时已关闭 TF32，此时融合公式在数学上应精确等价，
        # 误差应在 fp32 舍入噪声量级（~1e-5）。若仍 >1e-4 才需要排查融合公式。
        if diff < 1e-4:
            print("✓ 融合成功！输出一致性验证通过")
        else:
            print("⚠ 警告：输出差异较大，可能存在问题")

        # 形状验证
        assert out_fused.shape[2:] == (img_height, img_width), "输出形状不匹配！"
        print(f"✓ 输出形状验证通过: {out_fused.shape}")

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print("\n使用说明:")
        print("1. 训练时正常使用，模型会自动使用多分支结构")
        print("2. 推理前调用: model = reparameterize_model(model)")
        print("3. 推理速度与原始单分支模型基本相同")
        print("4. 精度通常会有小幅提升")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")
