import numpy as np
import pandas as pd

# 加载npz文件
npz_file = np.load("/home/kemove/output/drjohnson-1/point_cloud/iteration_35000/point_cloud.npz")

# 创建一个Excel writer对象
excel_writer = pd.ExcelWriter('/home/kemove/output/drjohnson-1/point_cloud/iteration_35000/drjohnson-1.xlsx', engine='xlsxwriter')

# 创建一个字典来存储每个数组的长度
array_lengths = {}

# 遍历npz文件中的数组
for array_name in npz_file.files:
    array_data = npz_file[array_name]
    
    # 判断数组的维度并处理长度
    if array_data.ndim == 0:  # 如果是标量
        array_lengths[array_name] = 1  # 标量长度为1
        df = pd.DataFrame([array_data], columns=[array_name])  # 将标量转换为单列的DataFrame
    else:
        # 保存数组的总长度
        array_lengths[array_name] = array_data.shape[0]
        
        # 对于一维及以上的数据，只保存1/100的内容
        sample_size = max(1, array_data.shape[0] // 100)  # 保证至少保存1行
        sampled_data = array_data[:sample_size]  # 提取前1/100的数据
        
        # 根据数据维度将其转换为DataFrame
        if array_data.ndim == 1:  # 一维数组
            df = pd.DataFrame(sampled_data, columns=[array_name])
        elif array_data.ndim == 2:  # 二维数组
            df = pd.DataFrame(sampled_data)
        else:  # 多维数组展平处理
            df = pd.DataFrame(sampled_data.reshape(-1, sampled_data.shape[-1]))
    
        # 将采样后的DataFrame写入Excel文件
        df.to_excel(excel_writer, sheet_name=array_name, index=False)

# 保存每个数组的长度信息到最后一个sheet
lengths_df = pd.DataFrame(list(array_lengths.items()), columns=['Array Name', 'Length'])
lengths_df.to_excel(excel_writer, sheet_name='Array_Lengths', index=False)

# 关闭Excel writer对象，保存文件
excel_writer.close()

print("文件已成功保存为 '/home/kemove/output/drjohnson-1/point_cloud/iteration_35000/drjohnson-1.xlsx'")
