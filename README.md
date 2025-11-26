# MouseGNN

Minimal tooling to annotate single-mouse behaviors (行走、转圈、嗅探、直立、蜷缩) and train a Spatial-Temporal Graph Convolutional Network (ST-GCN) on pose sequences.

## 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. 标注老鼠行为
`mousegnn/annotation/annotate_video.py` 提供键盘驱动的视频标注器：

```bash
python -m mousegnn.annotation.annotate_video /path/to/video.mp4 annotations/mouse1.csv
```

快捷键：
- `w` 行走 (walk)
- `c` 转圈 (circle)
- `s` 嗅探 (sniff)
- `r` 直立 (rear)
- `k` 蜷缩 (curl)
- `space` 清除当前片段；`q` 保存后退出

程序会在切换标签时记录片段 `(start_frame, end_frame, behavior, behavior_index)` 并写入 CSV。可按以下流程获得 ST-GCN 训练数据：
1. 使用 OpenPose/AlphaPose/DeepLabCut 等获得关键点，存为 `(T, V, C)` 的 `keypoints` 数组。
2. 根据 CSV 中的时间段切分对应帧，保存为 `*.npz`，包含：
   - `keypoints`: 形状 `(T, V, C)` 的 `numpy` 数组（C=2 表示 x/y，或 C=3 包含置信度）。
   - `label`: 字符串，取值 `walk|circle|sniff|rear|curl`。

## 2. 使用 ST-GCN 进行行为识别
示例配置位于 `configs/example_stgcn.yaml`：
- `data.train_dir`/`data.val_dir`: 存放 `*.npz` 片段的目录。
- `data.num_points`: 每帧关键点数量。
- `data.in_channels`: 坐标维度（2 或 3）。
- `data.clip_len`: 每个片段的时间长度（帧）。

### 训练
```bash
python -m mousegnn.training.train_stgcn configs/example_stgcn.yaml --device cuda
```
会输出最佳模型到 `training.output`（默认 `runs/stgcn/best.pt`）。

### 推理
```bash
python -m mousegnn.training.predict_stgcn runs/stgcn/best.pt path/to/sample.npz --device cpu
```
输出预测标签及每个行为的概率分布。

## 目录结构
- `mousegnn/data/behaviors.py`: 行为标签常量。
- `mousegnn/annotation/annotate_video.py`: 视频标注工具。
- `mousegnn/datasets/skeleton_sequences.py`: 关键点片段数据集与预处理。
- `mousegnn/models/stgcn.py`: 简化版 ST-GCN 模型。
- `mousegnn/training/train_stgcn.py`: 训练脚本。
- `mousegnn/training/predict_stgcn.py`: 推理脚本。
