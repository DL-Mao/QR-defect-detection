# DHAM-ViT: A Hybrid Attention Visualization Method Based on Vision Transformer


## 🌟 特性

- **双通道协同注意力机制(DHAM)**: 结合通道注意力和空间注意力的创新机制
- **轻量级ViT**: 简化版的Vision Transformer实现

## 🚀 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/DHAM-ViT.git
cd DHAM-ViT
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


### 基本训练

```bash
python model_train_visualize.py
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


### 可视化输出
训练过程中会在 `result/` 目录下生成：
- 注意力可视化图像
- 3D特征可视化
