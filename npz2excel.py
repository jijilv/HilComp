import numpy as np
import pandas as pd

# 加载 npz 文件
npz_file = np.load('/home/kemove/github/c3dgs/output/bonsai/point_cloud/iteration_35000/point_cloud.npz')

# 加载 'features_dc' 和 'features_rest'
features_dc = npz_file['features_dc']  # (110509, 1, 3)
features_rest = npz_file['features_rest']  # (110509, 15, 3)

# 将 'features_dc' 和 'features_rest' 转换为 2D 数据
features_dc_reshaped = features_dc.reshape(110509, 3)  # (110509, 3)
features_rest_reshaped = features_rest.reshape(110509, 45)  # (110509, 45)

# 将两个数组合并为一个大的 DataFrame
combined_features = np.hstack([features_dc_reshaped, features_rest_reshaped])  # (110509, 48)
df_combined = pd.DataFrame(combined_features)

# 定义Excel表格的最大行数
max_rows_per_sheet = 1048576

# 将每列数据拆分为 (110509, 1) 大小的数据块
with pd.ExcelWriter('features_dc_rest_splitted.xlsx') as writer:
    for i in range(df_combined.shape[1]):
        column_data = df_combined.iloc[:, i]
        
        # 检查列的行数，如果超过Excel限制则拆分
        if len(column_data) > max_rows_per_sheet:
            num_chunks = (len(column_data) // max_rows_per_sheet) + 1
            for chunk_idx in range(num_chunks):
                start_row = chunk_idx * max_rows_per_sheet
                end_row = (chunk_idx + 1) * max_rows_per_sheet
                chunk_data = column_data[start_row:end_row]
                
                # 写入到不同的sheet中
                new_sheet_name = f"feature_{i+1}_part{chunk_idx+1}"
                chunk_data.to_excel(writer, sheet_name=new_sheet_name, index=False)
        else:
            # 行数未超过限制，直接写入单个sheet
            sheet_name = f"feature_{i+1}"
            column_data.to_excel(writer, sheet_name=sheet_name, index=False)

print("文件已成功转换为 Excel 格式，并拆分为多个表格")