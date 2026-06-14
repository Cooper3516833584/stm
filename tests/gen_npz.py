import cv2
import numpy as np
import glob
import os

# ================= 配置区 =================
# 1. 树木校准图片的本地实际路径 (使用 r 前缀防止转义字符报错)
image_folder = r"D:\drone2\adjustment\trees" 

# 2. 输出的 npz 文件名 (遵循诊断报告中的命名约定)
output_npz = "tree_furniture_calibration.npz"

# 3. 输入节点名称 (通常与同架构的道路模型一致为 "images"，若后续平台提示找不到输入，请再次用 Netron 确认)
onnx_input_name = "images" 

# 4. 目标尺寸和需要处理的图片数量 (依据报告，树木模型截取 133 张)
target_size = (416, 416)
max_images = 133 
# ==========================================

image_list = glob.glob(os.path.join(image_folder, "*.jpg"))
if not image_list:
    print(f"错误：在 {image_folder} 中没有找到 .jpg 文件！")
    exit()

# 限制图片数量：确保只处理 133 张
image_list = image_list[:max_images]
calibration_data = []

print(f"正在处理 {len(image_list)} 张图片...")

for img_path in image_list:
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}")
        continue
        
    # 2. BGR 转 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 缩放到 416x416
    img = cv2.resize(img, target_size)
    
    # 4. 归一化到 0.0 - 1.0 之间，并转换为 32 位浮点数
    img = img.astype(np.float32) / 255.0
    
    # 5. 调整维度顺序：(HWC) -> (CHW)
    img = np.transpose(img, (2, 0, 1))
    
    calibration_data.append(img)

# 将列表转换为统一的 NumPy 数组
# 最终形状为 (133, 3, 416, 416)
calibration_data_array = np.array(calibration_data)

# 保存为 ST Edge AI Cloud 所需的格式
np.savez(output_npz, **{onnx_input_name: calibration_data_array})

print(f"成功！已生成 {output_npz}")
print(f"数据形状为: {calibration_data_array.shape}")