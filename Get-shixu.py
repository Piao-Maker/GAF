import cv2
import numpy as np
import os

def get_flame_area_from_label(label_path, img_shape):
    """根据YOLO格式标签计算火焰区域面积"""
    h, w = img_shape[:2]
    flame_area = 0.0
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                # YOLO标签格式：class x_center y_center w_ratio h_ratio
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w_r, h_r = map(float, parts)
                # 转换为像素级宽高
                flame_w = w_r * w
                flame_h = h_r * h
                flame_area += flame_w * flame_h  # 多火焰区域累加
    return flame_area

# 数据集路径配置
img_dir = r"D:\BaiduNetdiskDownload\fire\data\yolo\yolo\val\images"
label_dir = r"D:\BaiduNetdiskDownload\fire\data\yolo\yolo\val\labels"
ts_save_dir = r"D:\BaiduNetdiskDownload\fire\data\yolo\yolo\val\time_series"
os.makedirs(ts_save_dir, exist_ok=True)

# 读取连续帧（按文件名排序，确保时序正确）
img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
seq_length = 256
print(f"找到 {len(img_names)} 张图片，开始生成时序数据...")

generated_count = 0  # 统计成功生成的文件数
skipped_count = 0  # 统计跳过的文件数

for idx, img_name in enumerate(img_names):
    # 1. 计算帧索引
    start_idx = max(0, idx - seq_length // 2)
    end_idx = start_idx + seq_length
    if end_idx > len(img_names):
        end_idx = len(img_names)
        start_idx = max(0, end_idx - seq_length)
    seq_img_names = img_names[start_idx:end_idx]

    # 2. 检查是否提取到帧
    if len(seq_img_names) == 0:
        print(f"跳过 {img_name}：未提取到任何帧")
        skipped_count += 1
        continue

    # 3. 计算时序数据
    time_series = []
    valid_frame_count = 0  # 统计有效帧数量
    for s_img_name in seq_img_names:
        img_path = os.path.join(img_dir, s_img_name)
        label_path = os.path.join(label_dir, s_img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 新增：打印真实读取的路径
        print(f"尝试读取标签：{label_path}")  # 关键！看是否和实际标签文件路径一致

        # 检查图片
        if not os.path.exists(img_path):
            print(f"警告：{s_img_name} 图片不存在")
            time_series.append(0.0)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：{s_img_name} 图片损坏")
            time_series.append(0.0)
            continue

        # 检查标签
        if not os.path.exists(label_path):
            print(f"警告：{s_img_name} 标签不存在")
            flame_area = 0.0
        else:
            if os.path.getsize(label_path) == 0:
                print(f"警告：{s_img_name} 标签为空")
                flame_area = 0.0
            else:
                flame_area = get_flame_area_from_label(label_path, img.shape)
                if flame_area > 0:
                    valid_frame_count += 1

        time_series.append(flame_area)

    # 4. 补全时序长度
    while len(time_series) < seq_length:
        time_series.append(0.0)

    # 5. 只有存在有效帧才保存（避免全0文件）
    if valid_frame_count > 0:
        ts_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
        np.save(os.path.join(ts_save_dir, ts_name), np.array(time_series))
        generated_count += 1
    else:
        print(f"跳过 {img_name}：无有效火焰面积数据")
        skipped_count += 1

print(f"\n时序数据生成完成！成功生成 {generated_count} 个文件，跳过 {skipped_count} 个文件")