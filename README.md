# 光伏板沾灰程度分类系统

基于卷积神经网络（CNN）的光伏板污渍程度自动识别系统，支持无灰、轻度沾灰、重度沾灰三类分类，并提供 Gradio 交互式 Web 界面。

---

## 项目简介

光伏板表面积灰会显著降低发电效率。本系统利用深度学习技术，自动识别光伏板图像的沾污程度，为智能清洗决策提供依据。项目实现了多种 CNN 模型结构（基准 CNN、深层 CNN、残差 CNN），并通过不同超参数配置进行对比实验，最终提供可视化的预测与评估平台。

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.x（CUDA 12.x/13.x） |
| Gradio | 6.x |
| 其他 | 见 `requirements_lock.txt` |

### 安装步骤

```bash
# 克隆仓库
git clone <仓库地址>
cd PV_dust_classification

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements_lock.txt
```

---

## 数据集结构

数据集按以下方式组织在 `data/` 目录下：

```
data/
├── train/
│   ├── 0_assless/          # 无灰
│   ├── 1_little_ashes/     # 轻度沾灰
│   └── 2_all_ashes/        # 重度沾灰
└── val/
    ├── 0_assless/
    ├── 1_little_ashes/
    └── 2_all_ashes/
```

---

## 模型训练

提供多个训练入口脚本，分别对应不同的模型结构和超参数配置：

```bash
# 基准模型（lr=0.001, Adam）
python main.py

# 深层模型
python main_v2.py

# 残差模型
python main_v3.py

# 低学习率实验（lr=0.0001, Adam）
python main_config_b.py

# SGD 优化器实验（lr=0.01, SGD+momentum）
python main_config_c.py
```

训练完成后，权重文件保存在 `checkpoints/` 目录下，同时自动生成训练曲线图、混淆矩阵图和 ROC 曲线图。

---

## 启动 UI

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`，即可使用交互式界面。

### 界面功能

| 功能 | 说明 |
|------|------|
| 图像预测 | 单张预测显示类别、置信度和柱状图；批量预测返回表格结果 |
| 模型评估 | 查看各模型的准确率、精确率、召回率、F1 条形图及 ROC 曲线 |
| 模型对比 | 一键生成所有模型的四项指标分组对比图 |

---

## 可用模型

| 模型名 | 结构 | 优化器 |
|--------|------|--------|
| `model` | 基准 CNN（3层卷积） | Adam, lr=0.001 |
| `model_v2` | 深层 CNN（5层卷积） | Adam, lr=0.001 |
| `model_v3` | 残差 CNN | Adam, lr=0.001 |
| `config_a` | 同 model | 同 model |
| `config_b` | 同 model | Adam, lr=0.0001 |
| `config_c` | 同 model_v3 | SGD+momentum, lr=0.01 |

---

## 项目文件结构

```
PV_dust_classification/
├── data/                          # 数据集
├── checkpoints/                   # 模型权重和图表
├── dataset.py                     # 数据加载模块
├── model.py / model_v2.py / model_v3.py   # 模型定义
├── train.py                       # 训练模块
├── evaluate.py                    # 评估模块
├── main.py / main_v2.py / main_v3.py      # 训练入口
├── main_config_b.py / main_config_c.py    # 超参数实验入口
├── app.py                         # Gradio UI
├── requirements.txt               # 核心依赖
├── requirements_lock.txt          # 锁定版依赖
└── README.md
```



