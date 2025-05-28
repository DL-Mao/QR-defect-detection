# DHAM-Visual: A Hybrid Attention Visualization Method


## 🌟 特性

- **双通道协同注意力机制(DHAM)**: 结合通道注意力和空间注意力的创新机制

## 🚀 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/DHAM-Visual.git
cd DHAM-ViT
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

### 3. 可视化输出
训练过程中会在 `result/` 目录下生成：
- 注意力可视化图像
- 3D特征可视化
