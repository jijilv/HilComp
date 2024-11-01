import numpy as np

# 加载npz文件
npz_file = np.load("/home/kemove/output/drjohnson-gvq/point_cloud/iteration_30000/point_cloud.npz")

# 创建一个字典来存储每个数组的长度、维度和大小
array_info = {}
total_size = 0  # 用于累加所有数组的大小

# 遍历npz文件中的数组
for array_name in npz_file.files:
    array_data = npz_file[array_name]
    
    # 获取数组的大小（字节数）并累加到总大小
    array_size_bytes = array_data.nbytes
    total_size += array_size_bytes
    
    # 判断数组的维度并处理长度、大小
    if array_data.ndim == 0:  # 如果是标量
        array_info[array_name] = {'length': 1, 'shape': 'scalar', 'size_bytes': array_size_bytes}  # 标量长度为1，维度为scalar
    else:
        # 保存数组的总长度、维度和大小（字节数）
        array_info[array_name] = {
            'length': array_data.shape[0],  # 第一维长度
            'shape': array_data.shape,  # 维度信息
            'size_bytes': array_size_bytes  # 数组的总大小（字节数）
        }

# 计算文件总大小（MB）
total_size_MB = total_size / (1024**2)

# 将数组信息按照大小从大到小排序
sorted_array_info = sorted(array_info.items(), key=lambda x: x[1]['size_bytes'], reverse=True)

# 将数组的长度、维度和大小信息按大小顺序保存到一个txt文件
with open('/home/kemove/github/c3dgs/size/drjohnson-gvq-array_info.txt', 'w') as f:
    for array_name, info in sorted_array_info:
        size_bytes = info['size_bytes']
        if size_bytes >= 1024**2:
            size = f"{size_bytes / (1024**2):.2f} MB"  # 大于或等于1MB，用MB表示
        else:
            size = f"{size_bytes / 1024:.2f} KB"  # 小于1MB，用KB表示
        
        f.write(f"{array_name} - Shape: {info['shape']}, Length: {info['length']}, Size: {size}\n")
    
    # 输出文件的总大小
    f.write(f"\nTotal size of all arrays: {total_size_MB:.2f} MB\n")

print("数组信息已成功保存，且按照大小顺序排列，并根据大小选择MB或KB表示")
