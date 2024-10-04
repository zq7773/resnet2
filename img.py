import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import uwb_dataset as uwb_dataset

# 假设数据已经从 uwb_dataset 导入
columns, data = uwb_dataset.import_from_files()

for item in data:
    item[15:] = item[15:] / float(item[2])

df_uwb = pd.DataFrame(data=data, columns=columns)

# 检查数据情况
print("Data shape:", df_uwb.shape)
print("Null/NaN Data Count:", df_uwb.isna().sum().sum())
los_count = df_uwb.query("NLOS == 0")["NLOS"].count()
nlos_count = df_uwb.query("NLOS == 1")["NLOS"].count()
print("Line of Sight Count:", los_count)
print("Non Line of Sight Count:", nlos_count)

# 创建保存图像的文件夹
output_dir = 'CIR_images3'
os.makedirs(output_dir, exist_ok=True)

# 遍历每一行数据
for index, row in df_uwb.iterrows():
    # 提取第740列到890列的数据
    CIR_data = row[755:905].to_numpy()  # 注意：列索引是左闭右开的
    label = row['NLOS']  # 标签位于第一列

    # 1. 数据归一化到[0, 1]区间
    normalized_CIR = (CIR_data - np.min(CIR_data)) / (np.max(CIR_data) - np.min(CIR_data))

    # 2. 将归一化后的数据转换为极坐标
    phi = np.arccos(normalized_CIR)  # 角度
    r = np.arange(len(normalized_CIR)) / len(normalized_CIR)  # 半径

    # 3. 生成Gramian Angular Field (GAF)
    GAF = np.zeros((len(normalized_CIR), len(normalized_CIR)))

    for i in range(len(normalized_CIR)):
        for j in range(len(normalized_CIR)):
            GAF[i, j] = np.cos(phi[i] + phi[j])

    # 4. 绘制GAF图像
    plt.figure(figsize=(2.24, 2.24))
    # img = plt.imshow(GAF, cmap='cividis_r', interpolation='nearest', vmin=np.min(GAF), vmax=np.max(GAF))
    img = plt.imshow(GAF, cmap='YlGnBu', interpolation='nearest', vmin=np.min(GAF), vmax=np.max(GAF))
    # plt.colorbar(img)
    # plt.title(f'GAF of UWB CIR Data - Row {index} - Label {label}')
    # plt.xlabel('CIR Index')
    # plt.ylabel('CIR Index')

    # 保存图像
    # image_path = os.path.join(output_dir, f'CIR_image_row{index}_label{label}.png')
    image_path = os.path.join(output_dir, f'CIR_image_{index}.jpg')

    # 保存图像标签到txt文件中
    with open('CIR_labels3.txt', 'a') as f:
        f.write(f'CIR_image_{index}.jpg {label}\n')


    plt.savefig(image_path)
    plt.close()

    # 打印进度
    print(f"Saved image: {image_path}")

print("Image generation completed.")
