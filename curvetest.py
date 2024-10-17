import torch
from hilbertcurve.hilbertcurve import HilbertCurve

# 获取当前设备（CUDA 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hilbert_compress(data: torch.Tensor):
    # 将数据移动到相同设备上（GPU 或 CPU）
    data = data.to(device)

    N = data.shape[0]  # 样本数量
    dim1 = data.shape[1]

    # 定义一个归一化函数，使用 PyTorch 操作
    def normalize_data(segment: torch.Tensor):
        min_val, _ = torch.min(segment, dim=0, keepdim=True)  # 获取每个维度的最小值
        max_val, _ = torch.max(segment, dim=0, keepdim=True)  # 获取每个维度的最大值
        return (segment - min_val) / (max_val - min_val + 1e-6)  # 归一化到 [0, 1]

    # 如果数据是6维的，压缩为5维
    if dim1 == 6:
        p_3 = 5  # 用于 3 维数据的 Hilbert 曲线阶数
        n_3 = 3  # 每段 3 维度的压缩
        hilbert_curve_3 = HilbertCurve(p_3, n_3)

        p_2 = 5  # 用于 2 维数据的 Hilbert 曲线阶数
        n_2 = 2  # 每段 2 维度的压缩
        hilbert_curve_2 = HilbertCurve(p_2, n_2)

        # 定义一个函数将 segment 数据通过 Hilbert 曲线压缩为 1 维
        def compress_segment(segment: torch.Tensor):
            normalized_segment = normalize_data(segment)
            max_value = 2 ** p_2 - 1
            # 将归一化后的数据缩放到 [0, 2^p - 1]，并转换为整数类型
            scaled_segment = (normalized_segment * max_value).to(torch.int64)

            # 转换为 NumPy 数组进行 Hilbert 曲线压缩（在 CPU 上计算 Hilbert 曲线距离）
            hilbert_indices = [hilbert_curve_2.distance_from_point(point.cpu().numpy()) for point in scaled_segment]
            
            # 将结果转换回 CUDA Tensor
            return torch.tensor(hilbert_indices, dtype=torch.int64, device=device)

        # 压缩前两个维度
        first_dim = compress_segment(data[:, :2])  # 对应 1, 2 维
        # 后面4维不变
        remaining_dims = data[:, 2:]  # 对应 3, 4, 5, 6 维

        # 将压缩后的维度和未变的维度组合
        compressed_data = torch.cat([first_dim.unsqueeze(1), remaining_dims], dim=1)
        
    # 如果数据是48维的，压缩为5维
    elif dim1 == 48:
        p_5 = 5  # Hilbert 曲线的阶数
        
        hilbert_curve = HilbertCurve(p_5, 1)  # 用于将多个维度压缩为1维的Hilbert曲线

        # 将数据分成指定的维度段
        split_data_1_16 = data[:, :16]     # 1-16维
        split_data_17_32 = data[:, 16:32]  # 17-32维
        split_data_33_48 = data[:, 32:48]  # 33-48维
        split_data_31_40 = data[:, 30:40]   # 31-40维
        split_data_41_48 = data[:, 40:48]   # 41-48维

        # 定义一个函数将每个段通过Hilbert曲线压缩为1维
        def compress_segment(segment: torch.Tensor):
            normalized_segment = normalize_data(segment)
            max_value = 2 ** p_5 - 1
            scaled_segment = (normalized_segment * max_value).to(torch.int64)

            # 在CPU上计算Hilbert曲线距离
            hilbert_indices = [hilbert_curve.distance_from_point(point.cpu().numpy()) for point in scaled_segment]
            
            # 将结果转换回CUDA Tensor
            return torch.tensor(hilbert_indices, dtype=torch.int64, device=device)

        # 压缩各个段
        compressed_1_16 = compress_segment(split_data_1_16)
        compressed_17_32 = compress_segment(split_data_17_32)
        compressed_33_48 = compress_segment(split_data_33_48)
        compressed_31_40 = compress_segment(split_data_31_40)
        compressed_41_48 = compress_segment(split_data_41_48)

        # 将压缩结果组合为(N, 5)
        compressed_data = torch.cat([compressed_1_16.unsqueeze(1), 
                                      compressed_17_32.unsqueeze(1), 
                                      compressed_33_48.unsqueeze(1), 
                                      compressed_31_40.unsqueeze(1), 
                                      compressed_41_48.unsqueeze(1)], dim=1)

    else:
        raise ValueError("只支持 6维 或 48维 数据的压缩")

    return compressed_data
