# HAM-ViT: Hybrid Attention Mechanism with Vision Transformer

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个结合了混合注意力机制(HAM)和视觉变换器(ViT)的深度学习模型，用于图像分类任务。该项目实现了创新的注意力机制，并提供了丰富的可视化功能。

## 🌟 特性

- **混合注意力机制(HAM)**: 结合通道注意力和空间注意力的创新机制
- **轻量级ViT**: 简化版的Vision Transformer实现
- **深度可分离卷积**: 高效的卷积操作，减少参数量
- **丰富的可视化**: 包括注意力图、3D特征可视化等
- **完整的训练流程**: 包含数据增强、学习率调度、早停等

## 📋 目录

- [安装](#安装)
- [数据准备](#数据准备)
- [模型架构](#模型架构)
- [使用方法](#使用方法)
- [可视化功能](#可视化功能)
- [训练配置](#训练配置)
- [结果](#结果)
- [贡献](#贡献)
- [许可证](#许可证)

## 🚀 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/HAM-ViT.git
cd HAM-ViT

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.8.0
torchvision>=0.9.0
matplotlib>=3.3.0
seaborn>=0.11.0
numpy>=1.19.0
einops>=0.3.0
scikit-image>=0.18.0
```

## 📁 数据准备

项目期望以下数据结构：

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   └── ...
    ├── class2/
    │   └── ...
    └── ...
```

支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`

## 🏗️ 模型架构

### 整体架构

```
输入图像 (3×224×224)
    ↓
深度可分离卷积层 (3层)
    ↓
HAM注意力机制
    ↓
简化版ViT
    ↓
分类器 (3层全连接)
    ↓
输出 (num_classes)
```

### 核心组件

#### 1. 深度可分离卷积 (DepthwiseSeparableConv)
- 减少参数量和计算复杂度
- 保持特征提取能力

#### 2. HAM注意力机制
- **通道注意力**: 自适应学习通道重要性
- **空间注意力**: 基于重要通道分离的空间注意力
- **残差连接**: 保证梯度流动

#### 3. 简化版ViT
- Patch嵌入层
- Transformer编码器
- 全局平均池化

## 🎯 使用方法

### 基本训练

```bash
python model_train_visualize.py
```

### 自定义配置

```python
# 修改模型参数
model = CompleteModel(num_classes=10)  # 根据数据集类别数调整

# 修改训练参数
num_epochs = 50
batch_size = 32
learning_rate = 0.0001
```

### 仅使用HAM注意力机制

```python
from HAM_attention import HAM

# 创建HAM模块
ham = HAM(in_channels=128)

# 在你的模型中使用
x = ham(x)  # x shape: (B, L, C)
```

## 📊 可视化功能

项目提供多种可视化功能：

### 1. 注意力可视化
- 空间注意力热力图
- 通道注意力分布
- 原始图像对比

### 2. 3D特征可视化
- 特征图的3D表面图
- 多通道特征叠加
- 论文级别的可视化效果

### 3. 训练过程可视化
- 实时损失曲线
- 准确率变化
- 学习率调度

## ⚙️ 训练配置

### 数据增强
```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 优化器设置
- **优化器**: Adam
- **学习率**: 0.0001
- **学习率调度**: ReduceLROnPlateau
- **早停**: 25个epoch的patience

### 正则化
- Batch Normalization
- Dropout (0.3)
- 权重衰减

## 📈 结果

### 模型性能
- 参数量: ~2M (相比标准ViT大幅减少)
- 训练时间: ~30分钟 (GTX 1080Ti)
- 内存占用: ~4GB

### 可视化输出
训练过程中会在 `result/` 目录下生成：
- 注意力可视化图像
- 3D特征可视化
- 模型检查点
- 训练日志

## 🔧 高级用法

### 自定义HAM参数

```python
class CustomHAM(HAM):
    def __init__(self, in_channels, lambda_val=0.5):
        super().__init__(in_channels)
        self.SpatialAttention.Lambda = lambda_val
```

### 模型集成

```python
# 加载预训练模型
model = CompleteModel(num_classes=10)
model.load_state_dict(torch.load('result/best_model.pth'))

# 特征提取
features = model.features(images)
attention_features = model.ham(features)
```

## 📝 代码结构

```
HAM-ViT/
├── model_train_visualize.py    # 主训练脚本
├── HAM_attention.py           # HAM注意力机制实现
├── README.md                  # 项目说明
├── requirements.txt           # 依赖列表
├── data/                      # 数据目录
│   ├── train/
│   └── test/
└── result/                    # 结果输出目录
    └── timestamp/
        ├── best_model.pth
        ├── final_model.pth
        └── visualizations/
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- Vision Transformer的原始论文作者
- 注意力机制相关研究的贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: your.email@example.com
- GitHub Issues: [提交问题](https://github.com/yourusername/HAM-ViT/issues)

## 🔗 相关链接

- [Vision Transformer论文](https://arxiv.org/abs/2010.11929)
- [注意力机制综述](https://arxiv.org/abs/1706.03762)
- [PyTorch官方文档](https://pytorch.org/docs/)

---

⭐ 如果这个项目对你有帮助，请给个星标！ 