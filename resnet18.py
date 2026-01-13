# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiModalFlameNet(nn.Module):
    """
    图像 + GAF 多模态火焰检测网络（带门控融合）
    - 图像分支：ResNet18
    - GAF 分支：ResNet18
    - 融合方式：根据 feat_img & feat_gaf 学一个 gate \in [0,1]，做加权和
    """
    def __init__(self, num_classes=3):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT  # 预训练权重（等价于 pretrained=True）

        # 图像分支 backbone
        self.img_backbone = resnet18(weights=weights)
        self.img_backbone.fc = nn.Identity()   # 去掉最后一层分类，输出 512 维

        # GAF 分支 backbone
        self.gaf_backbone = resnet18(weights=weights)
        self.gaf_backbone.fc = nn.Identity()

        # 门控网络：输入 1024 维（feat_img + feat_gaf），输出一个 gate ∈ [0,1]
        self.gate_net = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()                   # 输出 (B,1) 的 gate
        )

        # 分类头：输入融合后的 512 维特征
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, gaf):
        """
        img: (B, 3, H, W)
        gaf: (B, 1 or 3, H, W)
        """
        # GAF 单通道 → 三通道，适配 ResNet
        if gaf.shape[1] == 1:
            gaf = gaf.repeat(1, 3, 1, 1)

        # 各自 backbone 提取特征
        feat_img = self.img_backbone(img)   # (B, 512)
        feat_gaf = self.gaf_backbone(gaf)   # (B, 512)

        # 拼接后输入 gate 网络
        feat_cat = torch.cat([feat_img, feat_gaf], dim=1)   # (B, 1024)
        gate = self.gate_net(feat_cat)                      # (B, 1), 每个样本一个 gate

        # 融合：gate 越大越偏向图像，越小越偏向 GAF
        fused_feat = gate * feat_img + (1.0 - gate) * feat_gaf   # (B, 512)

        # 分类
        logits = self.classifier(fused_feat)  # (B, num_classes)

        # 训练/推理时只用 logits 就够了；你想分析 gate 再改 forward 返回
        return logits
