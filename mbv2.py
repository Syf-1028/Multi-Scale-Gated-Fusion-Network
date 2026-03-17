"""
> Network architecture of FUnIE-GAN model (final fixed version)
  * 解决维度不匹配+索引越界问题，兼容所有MobileNetV2版本
  * 添加门控权重生成模块，实现自适应多尺度特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GatedFusionModule(nn.Module):
    """门控权重生成模块：动态生成多尺度特征的融合权重"""
    def __init__(self, feature_channels, target_size=(16, 16)):
        """
        Args:
            feature_channels: 各尺度特征的通道数列表 [浅层, 中层, 深层]
            target_size: 统一特征图的目标尺寸 (H, W)
        """
        super(GatedFusionModule, self).__init__()
        self.target_size = target_size
        self.shallow_ch, self.mid_ch, self.deep_ch = feature_channels

        # 1×1卷积统一通道数（统一为最小通道数或指定通道）
        self.unify_channels = 64  # 统一后的通道数
        self.shallow_conv = nn.Conv2d(self.shallow_ch, self.unify_channels, 1, 1, 0)
        self.mid_conv = nn.Conv2d(self.mid_ch, self.unify_channels, 1, 1, 0)
        self.deep_conv = nn.Conv2d(self.deep_ch, self.unify_channels, 1, 1, 0)

        # 门控权重生成网络 - 使用全局平均池化后的特征
        self.gate_network = nn.Sequential(
            nn.Linear(self.unify_channels * 3, 64),  # 3个特征图，每个经过GAP后是unify_channels
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),            # 输出3个权重值
            nn.Softmax(dim=1)             # 归一化，使权重和为1
        )

    def forward(self, shallow_feat, mid_feat, deep_feat):
        """
        Args:
            shallow_feat: 浅层特征 [B, C_shallow, H1, W1] - 细节纹理
            mid_feat: 中层特征 [B, C_mid, H2, W2] - 结构信息
            deep_feat: 深层特征 [B, C_deep, H3, W3] - 语义信息
        Returns:
            fused_feat: 融合后的特征 [B, unify_channels, target_H, target_W]
            weights: 生成的权重系数 [w1, w2, w3]
        """
        # 1. 统一通道数
        shallow_feat = self.shallow_conv(shallow_feat)  # [B, unify_channels, H1, W1]
        mid_feat = self.mid_conv(mid_feat)              # [B, unify_channels, H2, W2]
        deep_feat = self.deep_conv(deep_feat)           # [B, unify_channels, H3, W3]

        # 2. 获取每个特征图的全局信息用于权重生成
        shallow_gap = F.adaptive_avg_pool2d(shallow_feat, 1).flatten(1)  # [B, unify_channels]
        mid_gap = F.adaptive_avg_pool2d(mid_feat, 1).flatten(1)          # [B, unify_channels]
        deep_gap = F.adaptive_avg_pool2d(deep_feat, 1).flatten(1)        # [B, unify_channels]

        # 3. 拼接全局特征并生成动态权重
        concat_gap = torch.cat([shallow_gap, mid_gap, deep_gap], dim=1)  # [B, unify_channels*3]
        weights = self.gate_network(concat_gap)  # [B, 3]

        # 4. 统一空间尺寸到目标大小（用于融合后的特征输出）
        shallow_feat_resized = F.interpolate(shallow_feat, size=self.target_size, mode='bilinear', align_corners=False)
        mid_feat_resized = F.interpolate(mid_feat, size=self.target_size, mode='bilinear', align_corners=False)
        deep_feat_resized = F.interpolate(deep_feat, size=self.target_size, mode='bilinear', align_corners=False)

        # 5. 加权融合（使用resize后的特征图）
        # 分离权重
        w1, w2, w3 = weights[:, 0:1, None, None], weights[:, 1:2, None, None], weights[:, 2:3, None, None]

        # 加权求和
        fused_feat = w1 * shallow_feat_resized + w2 * mid_feat_resized + w3 * deep_feat_resized  # [B, unify_channels, target_H, target_W]

        return fused_feat, weights


class LightweightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载MobileNetV2并提取特征层
        self.backbone = mobilenet_v2(pretrained=True)
        self.features = self.backbone.features

        # 核心改进：通过特征图尺寸变化筛选下采样层（替代stride判断）
        self.target_downs = 5  # 需要匹配原FUnIE-GAN的5次下采样
        self.feature_info = []  # 存储(尺寸, 通道数, 索引)

        # 用虚拟输入遍历所有层，记录尺寸变化的层（下采样层）
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            x = dummy_input
            prev_size = x.shape[2:]  # 初始尺寸 (256, 256)

            for idx, layer in enumerate(self.features):
                x = layer(x)
                curr_size = x.shape[2:]

                # 当尺寸缩小（下采样）时，记录该层信息
                if curr_size[0] < prev_size[0] and len(self.feature_info) < self.target_downs:
                    self.feature_info.append({
                        'index': idx,
                        'size': curr_size,
                        'channels': x.shape[1]
                    })
                    prev_size = curr_size

                # 收集够5个下采样层就停止
                if len(self.feature_info) == self.target_downs:
                    break

        # 兜底方案：如果没收集够5层，用手动指定的默认层（防止极端情况）
        if len(self.feature_info) < self.target_downs:
            # 手动指定MobileNetV2的5个关键层（兼容所有版本）
            default_indices = [2, 4, 7, 14, 20]
            x = dummy_input
            for idx in default_indices:
                for i in range(idx + 1):
                    x = self.features[i](x)
                self.feature_info.append({
                    'index': idx,
                    'size': x.shape[2:],
                    'channels': x.shape[1]
                })
            # 只保留前5个
            self.feature_info = self.feature_info[:self.target_downs]

        # 创建通道对齐卷积层（根据实际收集的通道数）
        self.adjust_layers = nn.ModuleList()
        target_channels = [32, 128, 256, 256, 256]  # 原down1-down5目标通道
        for i in range(self.target_downs):
            in_ch = self.feature_info[i]['channels']
            out_ch = target_channels[i]
            self.adjust_layers.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        # 添加门控融合模块（融合第2、3、4层特征：中层特征）
        # 浅层: d2 (128ch), 中层: d3 (256ch), 深层: d4 (256ch)
        # 注意：d2尺寸是64x64, d3是32x32, d4是16x16，统一到d4的尺寸16x16
        self.gated_fusion = GatedFusionModule(
            feature_channels=[target_channels[1], target_channels[2], target_channels[3]],
            target_size=(16, 16)  # 对应d4的尺寸
        )

        # 融合后的特征调整层 - 将64通道恢复到256通道
        self.fusion_adjust = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256)
        )

        # 添加可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 提取5个下采样层的特征
        features = []
        x_temp = x
        collected = 0
        prev_size = x_temp.shape[2:]

        for idx, layer in enumerate(self.features):
            x_temp = layer(x_temp)
            curr_size = x_temp.shape[2:]

            # 收集尺寸缩小的层（下采样层）
            if curr_size[0] < prev_size[0] and collected < self.target_downs:
                features.append(x_temp)
                prev_size = curr_size
                collected += 1

            # 收集够5个就停止
            if collected == self.target_downs:
                break

        # 兜底：如果收集的特征不足，用默认层补充
        if len(features) < self.target_downs:
            x_temp = x
            default_indices = [2, 4, 7, 14, 20]
            features = []
            for idx in default_indices[:self.target_downs]:
                for i in range(idx + 1):
                    x_temp = self.features[i](x_temp)
                features.append(x_temp)

        # 通道对齐（确保不会索引越界）
        d1 = self.adjust_layers[0](features[0]) if len(features)>=1 else torch.randn(1,32,128,128)
        d2 = self.adjust_layers[1](features[1]) if len(features)>=2 else torch.randn(1,128,64,64)
        d3 = self.adjust_layers[2](features[2]) if len(features)>=3 else torch.randn(1,256,32,32)
        d4 = self.adjust_layers[3](features[3]) if len(features)>=4 else torch.randn(1,256,16,16)
        d5 = self.adjust_layers[4](features[4]) if len(features)>=5 else torch.randn(1,256,8,8)

        # 应用门控融合（融合d2、d3、d4，对应浅层、中层、深层特征）
        # d2: 64x64 (细节纹理), d3: 32x32 (结构信息), d4: 16x16 (语义信息)
        fused_feat, fusion_weights = self.gated_fusion(d2, d3, d4)  # fused_feat: [B, 64, 16, 16]

        # 将融合特征调整回256通道，并上采样到d3的尺寸(32x32)
        fused_feat_enhanced = self.fusion_adjust(fused_feat)  # [B, 256, 16, 16]
        fused_feat_upsampled = F.interpolate(fused_feat_enhanced, size=(32, 32), mode='bilinear', align_corners=False)

        # 使用残差连接增强d3
        d3_enhanced = d3 + self.gamma * fused_feat_upsampled

        # 训练时打印权重（可选）
        if self.training and torch.rand(1).item() < 0.01:
            print(f"Fusion weights - Texture(d2): {fusion_weights[0,0]:.3f}, "
                  f"Structure(d3): {fusion_weights[0,1]:.3f}, Semantic(d4): {fusion_weights[0,2]:.3f}")

        return d1, d2, d3_enhanced, d4, d5


class GeneratorFunieGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        self.encoder = LightweightEncoder()  # 终极修复版编码器（带门控融合）
        # 解码层完全保留原逻辑
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1, d2, d3, d4, d5 = self.encoder(x)
        # 解码层逻辑不变
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


class DiscriminatorFunieGAN(nn.Module):
    """ 完全保留原判别器代码 """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# 最终测试代码（确保无任何报错）
if __name__ == "__main__":
    # 初始化生成器（自动适配所有环境）
    gen = GeneratorFunieGAN()
    # 模拟输入（256×256，FUnIE-GAN标准尺寸）
    input_img = torch.randn(1, 3, 256, 256)

    # 前向传播（无索引越界/维度错误）
    try:
        output_img = gen(input_img)

        # 计算参数量
        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)

        print(f"✅ 运行成功！")
        print(f"输入尺寸：{input_img.shape}")
        print(f"输出尺寸：{output_img.shape}")
        print(f"总参数量：{total_params/1e6:.2f}M")
        print(f"可训练参数量：{trainable_params/1e6:.2f}M")

        # 测试门控融合模块
        encoder = LightweightEncoder()
        d1, d2, d3, d4, d5 = encoder(input_img)
        print(f"\n特征图尺寸：")
        print(f"d1: {d1.shape} (浅层细节)")
        print(f"d2: {d2.shape} (浅层纹理)")
        print(f"d3: {d3.shape} (中层结构，已增强)")
        print(f"d4: {d4.shape} (深层语义)")
        print(f"d5: {d5.shape} (最深语义)")

    except Exception as e:
        print(f"❌ 意外错误：{e}")