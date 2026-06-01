"""
光伏板沾灰程度分类系统 - Gradio 交互应用 (美观版)
- 单张/批量预测（带置信度分布提示）
- 评估指标条形图 + ROC 曲线（Y轴0.85-1.0放大差异）
- 一键模型性能对比图（Y轴0.85-1.0）
- 支持 model, model_v2, model_v3, config_a, config_b, config_c
"""

import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------- 中文字体与风格 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 使用 Soft 主题
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

CLASS_NAMES = ['0_assless (无灰)', '1_little_ashes (轻度沾灰)', '2_all_ashes (重度沾灰)']
NUM_CLASSES = len(CLASS_NAMES)

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- 导入模型构建函数 ----------
model_builders = {}
try:
    from model import build_model as build_baseline
    model_builders['model'] = build_baseline
    print("✅ 导入 model 成功")
except Exception as e:
    model_builders['model'] = None
    print(f"⚠️ 导入 model 失败: {e}")

try:
    from model_v2 import build_model as build_v2
    model_builders['model_v2'] = build_v2
    print("✅ 导入 model_v2 成功")
except Exception as e:
    model_builders['model_v2'] = None
    print(f"⚠️ 导入 model_v2 失败: {e}")

try:
    from model_v3 import build_model as build_v3
    model_builders['model_v3'] = build_v3
    print("✅ 导入 model_v3 成功")
except Exception as e:
    model_builders['model_v3'] = None
    print(f"⚠️ 导入 model_v3 失败: {e}")

# 配置映射
model_builders['config_a'] = model_builders.get('model')
model_builders['config_b'] = model_builders.get('model')
model_builders['config_c'] = model_builders.get('model_v3')
print("✅ config_a/b 复用 model 结构，config_c 复用 model_v3 结构")

WEIGHT_PATHS = {
    'model': os.path.join('checkpoints', 'model.pth'),
    'model_v2': os.path.join('checkpoints', 'model_v2.pth'),
    'model_v3': os.path.join('checkpoints', 'model_v3.pth'),
    'config_a': os.path.join('checkpoints', 'model.pth'),
    'config_b': os.path.join('checkpoints', 'config_b.pth'),
    'config_c': os.path.join('checkpoints', 'config_c.pth'),
}

# ---------- 加载模型 ----------
models = {}
for name in ['model', 'model_v2', 'model_v3', 'config_a', 'config_b', 'config_c']:
    path = WEIGHT_PATHS[name]
    build_fn = model_builders.get(name)
    if build_fn and os.path.exists(path):
        m = build_fn(num_classes=NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models[name] = m
        print(f"✅ 已加载 {name}")
    else:
        models[name] = None
        print(f"⚠️ {name} 不可用")

# ---------- 预计算评估指标 ----------
EVAL_METRICS = {}

def compute_metrics():
    global EVAL_METRICS
    try:
        from dataset import get_dataloaders
        _, _, test_loader = get_dataloaders(data_dir='./data', batch_size=32)
    except Exception as e:
        print(f"无法加载测试集: {e}")
        return
    for name, model in models.items():
        if model is None:
            continue
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0)
        EVAL_METRICS[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
        print(f"📊 {name}: Acc={acc:.4f}, F1={f1:.4f}")

compute_metrics()

# ---------- 辅助绘图函数 ----------
def get_roc_image(model_name):
    path = os.path.join('checkpoints', f'{model_name}_roc_curves.png')
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB')
        except:
            pass
    img = Image.new('RGB', (400, 300), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((50, 130), "暂无 ROC 曲线图", fill=(100, 100, 100), font=font)
    return img

def make_placeholder_image(text="暂无数据"):
    """生成占位图"""
    img = Image.new('RGB', (400, 300), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((50, 130), text, fill=(100, 100, 100), font=font)
    return img

# ---------- 预测函数 ----------
def predict(image, model_name):
    if image is None:
        return "请先上传一张图片", make_placeholder_image("请上传图片")
    if models.get(model_name) is None:
        return f"模型 {model_name} 不可用", make_placeholder_image("模型不可用")

    model = models[model_name]
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1).cpu().numpy().flatten()
    pred_idx = int(np.argmax(probs))
    text = f"预测类别: {CLASS_NAMES[pred_idx]}\n置信度: {probs[pred_idx]:.2%}\n\n各类别概率:\n"
    for i, cls in enumerate(CLASS_NAMES):
        text += f"  {cls}: {probs[i]:.2%}\n"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, probs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('概率')
    ax.set_title('各类别置信度')
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f'{p:.2%}', ha='center')
    plt.tight_layout()
    return text, fig

def batch_predict(files, model_name):
    if not files:
        return "请上传至少一张图片", make_placeholder_image("请上传图片")
    if models.get(model_name) is None:
        return f"模型 {model_name} 不可用", make_placeholder_image("模型不可用")

    model = models[model_name]
    lines = [f"{'文件名':<30} {'预测类别':<30} {'置信度':<10}", "-"*70]
    for fp in files:
        try:
            img = Image.open(fp).convert('RGB')
            img_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1).cpu().numpy().flatten()
            pred_idx = int(np.argmax(probs))
            lines.append(f"{os.path.basename(fp):<30} {CLASS_NAMES[pred_idx]:<30} {probs[pred_idx]:<10.2%}")
        except:
            lines.append(f"{os.path.basename(fp):<30} 预测失败")

    # 批量预测不显示置信度分布，返回一个提示图
    prompt_fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, '批量预测不显示置信度分布\n请使用单张预测查看详细图表',
            ha='center', va='center', fontsize=14, color='gray')
    ax.axis('off')
    plt.tight_layout()
    return "\n".join(lines), prompt_fig

# ---------- 评估指标图（Y轴0.85-1.0放大差异） ----------
def plot_metrics(model_name):
    if model_name not in EVAL_METRICS:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '暂无数据', ha='center')
        ax.axis('off')
        return fig
    m = EVAL_METRICS[model_name]
    names = ['准确率', '精确率', '召回率', 'F1分数']
    vals = [m['accuracy'], m['precision'], m['recall'], m['f1_score']]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color='#5B9BD5')
    ax.set_ylim(0.85, 1.0)           # ← 放大差异，更直观
    ax.set_ylabel('分数')
    ax.set_title(f'{model_name} 测试集评估指标')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    return fig

# ---------- 模型对比图（Y轴0.85-1.0） ----------
def compare_models():
    available = [name for name in EVAL_METRICS.keys()]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '暂无可用模型数据', ha='center', va='center')
        ax.axis('off')
        return fig

    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    x = np.arange(len(metrics_names))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(available):
        vals = [EVAL_METRICS[model_name][k] for k in ['accuracy', 'precision', 'recall', 'f1_score']]
        ax.bar(x + i * width, vals, width, label=model_name)

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.85, 1.0)           # ← 放大差异
    ax.set_ylabel('分数')
    ax.set_title('模型性能对比')
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig

# ---------- UI 界面 ----------
with gr.Blocks(title="光伏板沾灰程度分类系统") as demo:
    # 自定义标题
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="margin: 0; color: #2c3e50;">🌞 光伏板沾灰程度分类系统</h1>
        <p style="color: #7f8c8d; font-size: 16px;">基于卷积神经网络的智能污渍检测与模型评估平台</p>
    </div>
    """)

    # ========== 预测区域 ==========
    gr.Markdown("## 📸 图像预测")
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=list(models.keys()),
                value='model_v2' if 'model_v2' in models else list(models.keys())[0],
                label="选择模型"
            )
            img_input = gr.Image(type="numpy", label="上传单张图片")
            btn_pred = gr.Button("开始预测", variant="primary")
            gr.Markdown("---")
            files_input = gr.File(file_count="multiple", label="批量上传多张图片")
            btn_batch = gr.Button("批量预测", variant="primary")
        with gr.Column(scale=2):
            result_output = gr.Textbox(label="预测结果", lines=10)
            conf_plot = gr.Plot(label="置信度分布")

    # ========== 评估区域 ==========
    gr.Markdown("## 📊 模型评估")
    with gr.Row():
        model_eval = gr.Dropdown(
            choices=list(models.keys()),
            value='model_v2' if 'model_v2' in models else list(models.keys())[0],
            label="选择模型查看详情"
        )
    with gr.Row():
        metrics_plot = gr.Plot(label="评估指标条形图")
        roc_image = gr.Image(label="ROC 曲线", type="pil")

    # 模型对比功能
    with gr.Row():
        btn_compare = gr.Button("生成模型对比图", variant="secondary")
        compare_plot = gr.Plot(label="所有模型性能对比")

    # ========== 事件绑定 ==========
    btn_pred.click(
        fn=predict,
        inputs=[img_input, model_choice],
        outputs=[result_output, conf_plot]
    )
    btn_batch.click(
        fn=batch_predict,
        inputs=[files_input, model_choice],
        outputs=[result_output, conf_plot]
    )
    model_eval.change(
        fn=lambda n: (plot_metrics(n), get_roc_image(n)),
        inputs=model_eval,
        outputs=[metrics_plot, roc_image]
    )
    btn_compare.click(
        fn=compare_models,
        outputs=compare_plot
    )
    # 页面加载默认显示
    demo.load(
        fn=lambda: (plot_metrics('model_v2'), get_roc_image('model_v2')),
        outputs=[metrics_plot, roc_image]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme   # Gradio 6.0 需要将 theme 传给 launch()
    )