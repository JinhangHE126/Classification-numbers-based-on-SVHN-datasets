# SVHN 数字分类 —— 深度学习大作业

[![中文文档](https://img.shields.io/badge/📖-中文文档-blue?style=flat-square)](./README_CN.md)
[![English](https://img.shields.io/badge/📖-English-blue?style=flat-square)](./README.md)

基于 PyTorch 实现的街景门牌号（SVHN）数字分类项目。本项目对比了 MLP、改进 MLP、CNN 三种架构，并围绕数据增强、学习率调度、架构变体、正则化策略展开了系统性的四阶段消融实验。

## 数据集

- [Street View House Numbers (SVHN) - Format 2 (Cropped Digits)](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers/data)
- **训练集**：604,388 张（train + extra 合并）
- **测试集**：26,032 张
- **图像尺寸**：32 × 32 像素，RGB 三通道
- **类别数**：10 类（数字 0 ~ 9）

## 模型

### 模型 1 —— 基线 MLP

| 组件 | 详情 |
|------|------|
| 网络结构 | Flatten → FC(512, ReLU) → FC(10) |
| 优化器 | Adam（lr = 0.001） |
| Batch Size | 64 |
| Epochs | 15 |
| 损失函数 | Cross-Entropy |

| 指标 | 数值 |
|------|------|
| 训练 Loss | 0.2647 |
| 训练准确率 | 92.97% |
| **测试准确率** | **86.45%** |

### 模型 2 —— 改进 MLP

相较模型 1 做了以下四项改进：

1. **加深网络**：从单隐藏层扩展为三层（1024 → 512 → 256）
2. **激活函数**：ReLU → LeakyReLU（负斜率 0.01）
3. **优化器**：Adam → AdamW（引入权重衰减正则化）
4. **正则化**：每层后加入 Dropout（0.5）

| 组件 | 详情 |
|------|------|
| 网络结构 | FC(1024) → LeakyReLU → Dropout → FC(512) → LeakyReLU → Dropout → FC(256) → LeakyReLU → Dropout → FC(10) |
| 优化器 | AdamW（lr = 0.0005） |
| Batch Size | 64 |
| Epochs | 20 |
| 损失函数 | Cross-Entropy |

| 指标 | 数值 |
|------|------|
| 训练 Loss | 0.4980 |
| 训练准确率 | 85.91% |
| **测试准确率** | **84.96%** |

> **分析：** 虽然模型 2 参数更多、加入了 Dropout 和权重衰减等正则化手段，但测试准确率反而下降了约 1.5%。根本原因在于 MLP 将图像展平为一维向量，完全丢弃了像素间的空间位置关系——这在图像任务中是致命短板，再多的全连接层也无法弥补。

### 模型 3 —— CNN

利用卷积神经网络捕获图像中的空间结构信息。

| 组件 | 详情 |
|------|------|
| 卷积块 ×3 | Conv → BatchNorm → ReLU → MaxPool(2)（通道数：3 → 32 → 64 → 128） |
| 分类器 | Flatten(128×4×4=2048) → FC(512, ReLU) → Dropout(0.5) → FC(10) |
| 参数量 | ~115 万 |
| 优化器 | Adam（lr = 0.001） |
| Epochs | 15 |

| 指标 | 数值 |
|------|------|
| 训练 Loss | 0.0482 |
| 训练准确率 | 98.65% |
| **测试准确率** | **96.06%**（最佳轮次：96.17%） |

> CNN 的测试准确率比模型 1 高出近 **10 个百分点**，且收敛 Loss 仅为 MLP 的 1/5。卷积核的局部感知和权重共享机制，使其天然适合提取图像的边缘、纹理等空间特征，这是 MLP 无法比拟的。

---

## 消融实验

以基线 CNN 为起点，分四个阶段逐一消融关键设计因素，定位最优配置。

### 阶段一 —— 数据增强

| 实验 | 测试准确率 | 泛化差距 |
|------|:--------:|:------:|
| 基线 CNN（无增强） | 96.06% | 2.59% |
| 轻度增强（RandomCrop + RandomRotation） | 96.07% | **0.74%** |
| 色彩增强（ColorJitter） | — | — |

> 轻度数据增强几乎不改变测试准确率，却将泛化差距从 2.59% 压缩至 0.74%，说明过拟合得到了有效抑制。

### 阶段二 —— 学习率调度器

| 实验 | 测试准确率 |
|------|:--------:|
| 固定学习率（0.001） | 96.13% |
| StepLR（每 5 轮衰减 0.5） | 96.36% |
| CosineAnnealingLR（15 轮） | 96.54% |
| CosineAnnealingLR（30 轮） | **96.61%** 🔥 |

> CosineAnnealingLR 在 30 轮训练中取得了**全部实验的最高准确率 96.61%**。余弦退火让学习率在训练后期平滑下降，有助于模型收敛到更优的局部极小点。

### 阶段三 —— 架构变体

| 实验 | 参数量 | 测试准确率 |
|------|:------:|:--------:|
| 基线（3 个卷积块，kernel 3×3） | 115 万 | 96.06% |
| 加深 CNN（4 个卷积块，新增 256 通道层） | 92 万 | 96.09% |
| **首层 kernel 改为 5×5** | 115 万 | **96.39%** ✅ |
| 全局平均池化（GAP 替代全连接） | 9 万 | 95.48% |

> 将首层卷积核从 3×3 扩至 5×5 是**最有效的单点优化**——准确率提升 0.33%，且参数量近乎不变。更大的感受野有助于捕获数字的笔画轮廓。GAP 将参数量压缩至 9 万（仅为基线的 8%），适合部署场景，但会牺牲约 0.6% 的精度。

### 阶段四 —— 正则化

| 实验 | 测试准确率 | 说明 |
|------|:--------:|------|
| Dropout 0.3 | 96.25% | 比基线（0.5）略高，欠正则化风险 |
| Dropout 0.7 | — | 过度丢弃，收敛缓慢 |
| Weight Decay 1e-4 | — | 与 Adam 联合作用，效果待进一步验证 |

> Dropout 在 0.3~0.5 区间内表现稳定，0.5 作为默认值在收敛速度和泛化能力之间取得了较好平衡。

---

## 项目结构

```
├── Code.ipynb                     # 主实验：数据加载 → 三模型训练 → 评估对比
├── CNNBaseline_Improve.ipynb      # CNN 消融实验（四阶段，14 组实验）
├── Test_SavedModel.ipynb          # 加载已保存模型 & 训练曲线可视化
├── Augmentation_Visualization.ipynb # 数据增强效果可视化（12 张原图 vs 增强图）
├── saved_models/
│   ├── training_history.pkl       # 三模型训练指标（pickle）
│   ├── training_history.csv       # 三模型训练指标（CSV）
│   └── cnn_ablation_results.pkl   # 14 组消融实验结果
├── README.md                      # 英文文档
└── README_CN.md                   # 中文文档（本文件）
```

## 运行环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.6.0（CUDA 12.4） |
| torchvision | 0.21.0 |
| 其他依赖 | scipy, matplotlib, pandas, tqdm |
| 训练平台 | AWS EC2（GPU 实例） |

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision torchaudio scipy tqdm matplotlib pandas

# 2. 从 Kaggle 下载 SVHN 数据集（Format 2），将 .mat 文件放入 data/ 目录
#    https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers/data

# 3. 按顺序运行 Jupyter Notebook
#    Code.ipynb               → 数据准备 → 模型训练 → 评估
#    CNNBaseline_Improve.ipynb → 消融实验
#    Test_SavedModel.ipynb     → 可视化训练曲线
```

## 关键结论

1. **CNN 是图像分类的基石**：同样的训练条件下，CNN（96.06%）比最优 MLP（86.45%）高出近 10%，且 Loss 收敛更快、更彻底。
2. **轻度数据增强是性价比最高的正则化**：泛化差距从 2.59% 降至 0.74%，几乎零成本抑制过拟合。
3. **CosineAnnealingLR 是调参利器**：30 轮训练下达到 96.61% 的全场最佳，优于固定学习率和 StepLR。
4. **首层大卷积核行之有效**：将第一层卷积核从 3×3 改为 5×5，准确率提升至 96.39%，建议作为默认配置。
