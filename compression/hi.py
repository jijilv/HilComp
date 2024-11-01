from __future__ import annotations
from dataclasses import dataclass
from functools import partial, reduce
from contextlib import nullcontext
import gc
import operator
import time
from typing import List, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from torch_scatter import scatter
from torch.nn import Module
from torch import Tensor, int64
from torch.amp import autocast
from hilbertcurve.hilbertcurve import HilbertCurve


from einops import rearrange
from tqdm import tqdm, trange

from curvetest import hilbert_compress
from scene.gaussian_model import GaussianModel
from utils.splats import extract_rot_scale, to_full_cov

import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

class LSHCluster:
    def __init__(self, n_planes, feature_dim, seed=None):
        """
        LSH initialization for clustering data into classes.
        
        Parameters:
        - n_planes (int): The number of random planes for hashing.
        - feature_dim (int): The dimension of input features.
        - seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.planes = np.random.randn(n_planes, feature_dim)

    def hash(self, data):
        """
        Hash the data using LSH and return the class labels.
        
        Parameters:
        - data (Tensor): Data tensor of shape (B, N).
        
        Returns:
        - hashes (Tensor): Tensor of class labels of shape (B,).
        """
        data_np = data.cpu().numpy()
        projections = np.dot(data_np, self.planes.T)
        binary_hash = (projections >= 0).astype(int)
        class_labels = np.packbits(binary_hash, axis=1).view(np.uint16)
        return torch.tensor(class_labels[:, 0] % (2 ** binary_hash.shape[1]), device=data.device)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

# main class

class Hilbert(Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int = 2**12,
        decay: float = 0.5,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size=codebook_size
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5

    def uniform_init_hi(self, x: torch.Tensor, tmp: torch.Tensor):
        """
        初始化码本，并将所有数据均匀地分配到不同的**大索引**中。
        """
        B, N = tmp.shape

        total_bins = self.codebook_size 

        # 设置希尔伯特曲线的阶数
        p = 8  # 希尔伯特曲线的阶数，根据数据的分辨率需要调整
        hilbert_curve = HilbertCurve(p, N)

        print("Compressing Gaussian splats with Hilbert curve...")
        start_time = time.time()

        # 对数据进行归一化处理，使得所有数据点为非负整数
        tmp_min = tmp.min()
        tmp_max = tmp.max()
        scaled_tmp = ((tmp - tmp_min) / (tmp_max - tmp_min) * (2**p - 1)).round().long()

        # 计算每个数据点的希尔伯特索引并显示进度
        hilbert_indices = []
        for i in tqdm(range(B), desc="Calculating Hilbert indices"):
            point = scaled_tmp[i].tolist()  # 将每个数据点转换为整数列表
            hilbert_index = hilbert_curve.distance_from_point(point)
            hilbert_indices.append(hilbert_index)

        # 将 hilbert_indices 转换为张量并进行排序
        hilbert_indices = torch.tensor(hilbert_indices, device=tmp.device)
        sorted_indices = torch.argsort(hilbert_indices)

        # 每个大索引对应的数据量
        bin_size = B // total_bins
        remainder = B % total_bins

        # 初始化大索引张量
        big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

        # 均匀分配大索引
        current_start = 0
        for i in range(total_bins):
            current_end = current_start + bin_size + (1 if i < remainder else 0)
            big_index[sorted_indices[current_start:current_end]] = i
            current_start = current_end

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"Total compression time with Hilbert curve: {total_hilbert_time:.2f} seconds")

        # 初始化码本的数据
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

        return big_index

    def uniform_init_hi2(self, x: torch.Tensor, tmp: torch.Tensor, importance: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        初始化码本，并将所有数据均匀地分配到不同的**大索引**中。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size 

        # 设置希尔伯特曲线的阶数
        p = 8  # 希尔伯特曲线的阶数，根据数据的分辨率需要调整
        hilbert_curve = HilbertCurve(p, N)

        print("Compressing Gaussian splats with Hilbert curve...")
        start_time = time.time()

        # 对数据进行归一化处理，使得所有数据点为非负整数
        tmp_min = tmp.min()
        tmp_max = tmp.max()
        scaled_tmp = ((tmp - tmp_min) / (tmp_max - tmp_min) * (2**p - 1)).round().long()

        # 计算每个数据点的希尔伯特索引
        hilbert_indices = []
        for i in tqdm(range(B), desc="Calculating Hilbert indices"):
            point = scaled_tmp[i].tolist()  # 将每个数据点转换为整数列表
            hilbert_index = hilbert_curve.distance_from_point(point)
            hilbert_indices.append(hilbert_index)

        # 将 hilbert_indices 转换为张量并进行排序
        hilbert_indices = torch.tensor(hilbert_indices, device=tmp.device)
        sorted_indices = torch.argsort(hilbert_indices)

        # 每个大索引对应的数据量
        bin_size = B // total_bins
        remainder = B % total_bins

        # 初始化大索引张量
        big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

        # 均匀分配大索引
        current_start = 0
        for i in range(total_bins):
            current_end = current_start + bin_size + (1 if i < remainder else 0)
            big_index[sorted_indices[current_start:current_end]] = i
            current_start = current_end

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"Total compression time with Hilbert curve: {total_hilbert_time:.2f} seconds")

        # 使用 weighted average 初始化码本
        with torch.no_grad():
            idx = big_index.long()

            # 计算每个大索引的累积重要性
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            # 按权重（重要性）计算码本值的初始加权平均
            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            ) / (acc_importance[:, None] + self.eps)

            # 设置码本数据为加权平均值
            self.codebook.data = codebook

            # 迹归一化步骤
            if scale_normalize:
                # 计算协方差矩阵的迹
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                # 使用迹进行归一化
                self.codebook /= tr[:, None]

        return big_index

    def uniform_init_hi3(self, x: torch.Tensor, tmp: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        初始化码本，并将所有数据均匀地分配到不同的**大索引**中。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size 

        # 设置希尔伯特曲线的阶数
        p = 8  # 希尔伯特曲线的阶数，根据数据的分辨率需要调整
        hilbert_curve = HilbertCurve(p, N)

        print("Compressing Gaussian splats with Hilbert curve...")
        start_time = time.time()

        # 对数据进行归一化处理，使得所有数据点为非负整数
        tmp_min = tmp.min()
        tmp_max = tmp.max()
        scaled_tmp = ((tmp - tmp_min) / (tmp_max - tmp_min) * (2**p - 1)).round().long()

        # 计算每个数据点的希尔伯特索引
        hilbert_indices = []
        for i in tqdm(range(B), desc="Calculating Hilbert indices"):
            point = scaled_tmp[i].tolist()  # 将每个数据点转换为整数列表
            hilbert_index = hilbert_curve.distance_from_point(point)
            hilbert_indices.append(hilbert_index)

        # 将 hilbert_indices 转换为张量并进行排序
        hilbert_indices = torch.tensor(hilbert_indices, device=tmp.device)
        sorted_indices = torch.argsort(hilbert_indices)

        # 每个大索引对应的数据量
        bin_size = B // total_bins
        remainder = B % total_bins

        # 初始化大索引张量
        big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

        # 均匀分配大索引
        current_start = 0
        for i in range(total_bins):
            current_end = current_start + bin_size + (1 if i < remainder else 0)
            big_index[sorted_indices[current_start:current_end]] = i
            current_start = current_end

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"Total compression time with Hilbert curve: {total_hilbert_time:.2f} seconds")

        # 使用普通平均初始化码本
        with torch.no_grad():
            idx = big_index.long()

            # 按索引对 `x` 求平均，初始化 `codebook`
            codebook = scatter(
                x,
                idx,
                0,
                reduce="mean",
                dim_size=self.codebook.shape[0],
            )

            # 设置码本数据为普通平均值
            self.codebook.data = codebook

            # 迹归一化步骤
            if scale_normalize:
                # 计算协方差矩阵的迹
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                # 使用迹进行归一化
                self.codebook /= tr[:, None]

        return big_index
    
    def uniform_init_hi4(self, x: torch.Tensor, tmp: torch.Tensor, importance: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        初始化码本，并根据希尔伯特索引将数据分配到不同的大索引中。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size 

        # 设置希尔伯特曲线的阶数
        p = 8  # 希尔伯特曲线的阶数，根据数据的分辨率需要调整
        hilbert_curve = HilbertCurve(p, N)

        print("Compressing Gaussian splats with Hilbert curve...")
        start_time = time.time()

        # 对数据进行归一化处理，使得所有数据点为非负整数
        tmp_min = tmp.min()
        tmp_max = tmp.max()
        scaled_tmp = ((tmp - tmp_min) / (tmp_max - tmp_min) * (2**p - 1)).round().long()

        # 计算每个数据点的希尔伯特索引
        hilbert_indices = []
        for i in tqdm(range(B), desc="Calculating Hilbert indices"):
            point = scaled_tmp[i].tolist()  # 将每个数据点转换为整数列表
            hilbert_index = hilbert_curve.distance_from_point(point)
            hilbert_indices.append(hilbert_index)

        # 将 hilbert_indices 转换为张量并归一化
        hilbert_indices = torch.tensor(hilbert_indices, device=tmp.device).float()
        hilbert_indices = (hilbert_indices - hilbert_indices.min()) / (hilbert_indices.max() - hilbert_indices.min())
        hilbert_indices = hilbert_indices * (total_bins - 1)

        # 将归一化后的索引分配到大索引中
        big_index = hilbert_indices.floor().long()

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"Total compression time with Hilbert curve: {total_hilbert_time:.2f} seconds")

        # 使用 weighted average 初始化码本
        with torch.no_grad():
            idx = big_index.long()

            # 计算每个大索引的累积重要性
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            # 按权重（重要性）计算码本值的初始加权平均
            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            ) / (acc_importance[:, None] + self.eps)

            # 设置码本数据为加权平均值
            self.codebook.data = codebook

            # 迹归一化步骤
            if scale_normalize:
                # 计算协方差矩阵的迹
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                # 使用迹进行归一化
                self.codebook /= tr[:, None]

        return big_index
    
    from tqdm import tqdm

    def uniform_init_hi5(self, x: torch.Tensor, tmp: torch.Tensor, importance: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        初始化码本，并将所有数据均匀地分配到不同的**大索引**中，然后合并相邻块。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size 

        # 设置希尔伯特曲线的阶数
        p = 8  # 希尔伯特曲线的阶数，根据数据的分辨率需要调整
        hilbert_curve = HilbertCurve(p, N)

        print("Compressing Gaussian splats with Hilbert curve...")
        start_time = time.time()

        # 对数据进行归一化处理，使得所有数据点为非负整数
        tmp_min = tmp.min()
        tmp_max = tmp.max()
        scaled_tmp = ((tmp - tmp_min) / (tmp_max - tmp_min) * (2**p - 1)).round().long()

        # 计算每个数据点的希尔伯特索引
        hilbert_indices = []
        for i in tqdm(range(B), desc="Calculating Hilbert indices"):
            point = scaled_tmp[i].tolist()  # 将每个数据点转换为整数列表
            hilbert_index = hilbert_curve.distance_from_point(point)
            hilbert_indices.append(hilbert_index)

        # 将 hilbert_indices 转换为张量并进行排序
        hilbert_indices = torch.tensor(hilbert_indices, device=tmp.device)
        sorted_indices = torch.argsort(hilbert_indices)

        # 每个大索引对应的数据量
        bin_size = B // total_bins
        remainder = B % total_bins

        # 初始化大索引张量
        big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

        # 均匀分配大索引
        block_ranges = []
        current_start = 0
        for i in range(total_bins):
            current_end = current_start + bin_size + (1 if i < remainder else 0)
            big_index[sorted_indices[current_start:current_end]] = i
            block_ranges.append((current_start, current_end))  # 存储每个块的范围
            current_start = current_end

        # 手动设置合并阈值
        threshold = 1e-10 if N == 3 else 20000  # 根据需要调整阈值

        # 使用 tqdm 显示合并进度
        with tqdm(total=len(block_ranges), desc="Merging blocks") as pbar:
            merged_bin_id = 0
            # 循环直到无法再合并块
            while True:
                merged = False
                i = 0
                while i < len(block_ranges) - 1:
                    current_start, current_end = block_ranges[i]
                    next_start, next_end = block_ranges[i + 1]

                    # 比较当前块和下一个块的索引差值，决定是否合并
                    if abs(hilbert_indices[sorted_indices[next_start]] - hilbert_indices[sorted_indices[current_start]]) < threshold:
                        # 合并当前块和下一个块
                        big_index[sorted_indices[current_start:next_end]] = merged_bin_id
                        block_ranges[i] = (current_start, next_end)  # 更新当前块的范围
                        block_ranges.pop(i + 1)  # 移除合并的下一个块
                        merged = True  # 标记合并发生
                    else:
                        # 如果不合并，移动到下一个块
                        big_index[sorted_indices[current_start:current_end]] = merged_bin_id
                        merged_bin_id += 1
                        i += 1

                    # 更新进度条
                    pbar.update(1)

                # 确保最后一个块也被标记
                if i == len(block_ranges) - 1:
                    current_start, current_end = block_ranges[i]
                    big_index[sorted_indices[current_start:current_end]] = merged_bin_id

                # 如果在此轮合并中没有发生任何合并，跳出循环
                if not merged:
                    break

                # 重置进度条和 bin_id 以进行下一轮合并
                pbar.reset(total=len(block_ranges))
                merged_bin_id = 0

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"Total compression time with Hilbert curve: {total_hilbert_time:.2f} seconds")

        # 使用 weighted average 初始化码本
        with torch.no_grad():
            idx = big_index.long()

            # 计算每个大索引的累积重要性
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            # 按权重（重要性）计算码本值的初始加权平均
            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            ) / (acc_importance[:, None] + self.eps)

            # 设置码本数据为加权平均值
            self.codebook.data = codebook

            # 迹归一化步骤
            if scale_normalize:
                # 计算协方差矩阵的迹
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                # 使用迹进行归一化
                self.codebook /= tr[:, None]

        return big_index
    
    def uniform_init_lsh(self, x: torch.Tensor, tmp: torch.Tensor, importance: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        使用 LSH 聚类初始化码本，并将所有数据均匀地分配到不同的**大索引**中。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size
        n_planes = int(np.log2(total_bins))

        # 初始化 LSH 聚类器
        lsh_cluster = LSHCluster(n_planes=n_planes, feature_dim=N, seed=42)

        print("Compressing data with LSH...")
        start_time = time.time()

        # 计算 LSH 类标签
        lsh_labels = lsh_cluster.hash(tmp)

        # 对 LSH 类标签进行排序
        sorted_indices = torch.argsort(lsh_labels.to(torch.int32))


        # 每个大索引对应的数据量
        bin_size = B // total_bins
        remainder = B % total_bins

        # 初始化大索引张量
        big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

        # 均匀分配大索引
        current_start = 0
        for i in range(total_bins):
            current_end = current_start + bin_size + (1 if i < remainder else 0)
            big_index[sorted_indices[current_start:current_end]] = i
            current_start = current_end

        end_time = time.time()
        total_lsh_time = end_time - start_time
        print(f"Total compression time with LSH: {total_lsh_time:.2f} seconds")

        # 使用 weighted average 初始化码本
        with torch.no_grad():
            idx = big_index.long()

            # 计算每个大索引的累积重要性
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            # 按权重（重要性）计算码本值的初始加权平均
            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            ) / (acc_importance[:, None] + self.eps)

            # 设置码本数据为加权平均值
            self.codebook.data = codebook

            # 迹归一化步骤
            if scale_normalize:
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                self.codebook /= tr[:, None]

        return big_index

    def uniform_init_lsh_or_hilbert(self, x: torch.Tensor, tmp: torch.Tensor, importance: torch.Tensor, scale_normalize: bool = False) -> torch.Tensor:
        """
        初始化码本，根据数据维度选择不同的索引分配方法。
        - 如果维度为 3：直接对每个维度排序，平均分成 self.codebook_size 立方根个小索引，最后拼接成大索引。
        - 如果维度为 6：分别对前 3 维和后 3 维使用 Hilbert 曲线，平均分成 self.codebook_size 平方根个小索引，最后拼接成大索引。
        """
        B, N = tmp.shape
        total_bins = self.codebook_size

        print("Compressing data...")
        start_time = time.time()

        if N == 3:
            # 如果维度为 3，对每个维度排序，分配索引并拼接
            sorted_indices = torch.argsort(tmp, dim=0)
            bins_per_dim = int(np.cbrt(total_bins))  # Cube root for 3D
            big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

            for dim in range(N):
                current_start = 0
                for bin_id in range(bins_per_dim):
                    bin_size = B // bins_per_dim + (1 if bin_id < B % bins_per_dim else 0)
                    big_index[sorted_indices[current_start:current_start + bin_size, dim]] += bin_id * (bins_per_dim ** dim)
                    current_start += bin_size

        elif N == 6:
            # 如果维度为 6，分别对前 3 维和后 3 维使用 Hilbert 曲线
            p = int(np.sqrt(total_bins))  # Use square root for 6D case
            p = min(8, p)
            def calculate_hilbert_indices(data, p, N):
                hilbert_curve = HilbertCurve(p, N)
                hilbert_indices = []
                for i in tqdm(range(data.shape[0]), desc="Calculating Hilbert indices"):
                    point = data[i].tolist()
                    hilbert_index = hilbert_curve.distance_from_point(point)
                    hilbert_indices.append(hilbert_index)
                return torch.tensor(hilbert_indices, device=data.device)

            front_3_scaled = ((tmp[:, :3] - tmp[:, :3].min()) / (tmp[:, :3].max() - tmp[:, :3].min()) * (2**p - 1)).round().long()
            back_3_scaled = ((tmp[:, 3:] - tmp[:, 3:].min()) / (tmp[:, 3:].max() - tmp[:, 3:].min()) * (2**p - 1)).round().long()

            front_hilbert_indices = calculate_hilbert_indices(front_3_scaled, p, 3)
            back_hilbert_indices = calculate_hilbert_indices(back_3_scaled, p, 3)

            combined_indices = front_hilbert_indices * (2**p) + back_hilbert_indices
            sorted_indices = torch.argsort(combined_indices)

            bin_size = B // total_bins
            remainder = B % total_bins

            big_index = torch.zeros(B, dtype=torch.int64, device=tmp.device)

            current_start = 0
            for i in range(total_bins):
                current_end = current_start + bin_size + (1 if i < remainder else 0)
                big_index[sorted_indices[current_start:current_end]] = i
                current_start = current_end

        else:
            raise ValueError("Dimension not supported. Only 3 or 6 dimensions are allowed.")

        end_time = time.time()
        print(f"Total compression time: {end_time - start_time:.2f} seconds")

        # 使用 weighted average 初始化码本
        with torch.no_grad():
            idx = big_index.long()

            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            ) / (acc_importance[:, None] + self.eps)

            self.codebook.data = codebook

            if scale_normalize:
                tr = self.codebook[:, [0, 3, 5]].sum(-1)
                self.codebook /= tr[:, None]

        return big_index

    def update(self,idx: torch.Tensor, tmp: torch.Tensor, hi_modal: Hilbert, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            idx = idx.long()
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            ema_inplace(self.entry_importance, acc_importance, self.decay)

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            )

            ema_inplace(
                self.codebook,
                codebook / (acc_importance[:, None] + self.eps),
                self.decay,
            )

    def forward(self, z):

        indices = self.quantize(z)

        return self.codebook.data[indices], indices

def hi_features(
    compressed_features: torch.Tensor,
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance/importance.max()
    hi_modal = Hilbert(
        channels=features.shape[-1],
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    
    # bigindices = hi_modal.uniform_init_hi(features,compressed_features)

    bigindices = hi_modal.uniform_init_hi2(features,compressed_features,importance_n,scale_normalize)

    # bigindices = hi_modal.uniform_init_hi3(features,compressed_features,scale_normalize)
    start_time = time.time()

    errors = []
    # for i in trange(steps):
        
    #     batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])

    #     qq_feature = features[batch]
    #     indices = bigindices[batch]
    #     compressed_feature = compressed_features[batch]

    #     # qq_modal.update(compressed_feature, qq_modal, qq_feature, importance=importance_n[batch])

    #     hi_modal.update(indices,compressed_feature, hi_modal, qq_feature, importance=importance_n[batch])
    #     if scale_normalize:
    #         # this computes the trace of the codebook covariance matrices
    #         # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
    #         tr = hi_modal.codebook[:, [0, 3, 5]].sum(-1)
    #         hi_modal.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    # _, qq_indices = qq_modal(compressed_features)
    hi_indices=bigindices

    print_fsq_indices_distribution(hi_indices,codebook_size)

    torch.cuda.synchronize(device=hi_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return hi_modal.codebook.data.detach(), hi_indices.detach()

def print_fsq_indices_distribution(fsq_indices, codebook_size):
    """
    打印 fsq_indices 的分布信息，包括每个索引的出现次数和码本利用率。
    
    参数：
    - fsq_indices: 输入的张量
    - codebook_size: 码本的大小
    """
    # 如果 fsq_indices 是 PyTorch 张量，将其转换为 NumPy 数组
    if isinstance(fsq_indices, torch.Tensor):
        fsq_indices = fsq_indices.cpu().numpy()

    # 将 fsq_indices 扁平化以计算总体分布
    fsq_indices_flat = fsq_indices.flatten()

    # 生成从 0 到 codebook_size - 1 的完整索引范围
    full_range = np.arange(0, codebook_size)

    # 计算唯一值的分布
    unique, counts = np.unique(fsq_indices_flat, return_counts=True)

    # 创建包含完整范围的字典以存储索引出现次数
    index_count_dict = {index: 0 for index in full_range}

    # 更新字典中的出现次数
    for u, c in zip(unique, counts):
        index_count_dict[u] = c

    # 打印每个索引的出现次数
    # print("FSQ Indices 分布 (从 Index 0 开始):")
    # for index in full_range:
    #     count = index_count_dict.get(index, 0)
    #     print(f"Index {index}: {count} 次")

    # 计算码本利用率
    utilized_codebook = len([index for index in index_count_dict if index_count_dict[index] > 0])
    utilization_rate = (utilized_codebook / codebook_size) * 100

    # 打印码本利用率
    print(f"\n码本利用率: {utilization_rate:.2f}%")

    # 计算并打印索引超过使用 10 次的几率
    exceeding_indices_10 = len([index for index in index_count_dict if index_count_dict[index] > 10])
    exceeding_rate_10 = (exceeding_indices_10 / codebook_size) * 100
    print(f"索引超过使用 10 次的几率: {exceeding_rate_10:.2f}%")

    # 计算并打印索引超过使用 100 次的几率
    exceeding_indices_100 = len([index for index in index_count_dict if index_count_dict[index] > 100])
    exceeding_rate_100 = (exceeding_indices_100 / codebook_size) * 100
    print(f"索引超过使用 100 次的几率: {exceeding_rate_100:.2f}%")

    # 计算并打印索引超过使用 1000 次的几率
    exceeding_indices_1000 = len([index for index in index_count_dict if index_count_dict[index] > 1000])
    exceeding_rate_1000 = (exceeding_indices_1000 / codebook_size) * 100
    print(f"索引超过使用 1000 次的几率: {exceeding_rate_1000:.2f}%")

    # 计算并打印索引超过使用 10000 次的几率
    exceeding_indices_10000 = len([index for index in index_count_dict if index_count_dict[index] > 10000])
    exceeding_rate_10000 = (exceeding_indices_10000 / codebook_size) * 100
    print(f"索引超过使用 10000 次的几率: {exceeding_rate_10000:.2f}%")

def join_features(
    all_features: torch.Tensor,
    keep_mask: torch.Tensor,
    codebook: torch.Tensor,
    codebook_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_features = all_features[keep_mask]
    compressed_features = torch.cat([codebook, keep_features], 0)

    indices = torch.zeros(
        len(all_features), dtype=torch.long, device=all_features.device
    )
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(
        codebook
    )

    return compressed_features, indices

@dataclass
class CompressionSettings:
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int

def transform_data(input_data):

    # 第1-3维直接对应
    transformed_data = input_data[:, :3]
    
    return transformed_data

def compress_color(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    color_comp: CompressionSettings,
    color_compress_non_dir: bool,
):
    # keep_mask = color_importance > color_comp.importance_include

    top_k_indices = torch.argsort(color_importance, descending=True)[:10000]

    # 创建一个与 color_importance 相同形状的布尔 mask
    keep_mask = torch.zeros_like(color_importance, dtype=torch.bool)

    # 将前 10000 个索引的位置设置为 True
    keep_mask[top_k_indices] = True

    print(
        f"color keep: {keep_mask.float().mean()*100:.2f}%"
    )

    qq_mask_c = ~keep_mask

    # remove zero sh component
    if color_compress_non_dir:
        n_sh_coefs = gaussians.get_features.shape[1]
        color_features = gaussians.get_features.detach().flatten(-2)
    else:
        n_sh_coefs = gaussians.get_features.shape[1] - 1
        color_features = gaussians.get_features[:, 1:].detach().flatten(-2)

    compressed_features = transform_data(color_features)

    # compressed_features = symmetric_shift(compressed_features)

    if qq_mask_c.any():

        print("compressing color...")

        color_codebook, color_vq_indices = hi_features(
            compressed_features[qq_mask_c],
            color_features[qq_mask_c],
            color_importance[qq_mask_c],
            color_comp.codebook_size,
            color_comp.batch_size,
            color_comp.steps,
        )
    else:
        color_codebook = torch.empty(
            (0, color_features.shape[-1]), device=color_features.device
        )
        color_vq_indices = torch.empty(
            (0,), device=color_features.device, dtype=torch.long
        )

    all_features = color_features
    compressed_features, indices = join_features(
        all_features, keep_mask, color_codebook, color_vq_indices
    )

    print(f"color码本的大小：{compressed_features.shape[0]}")

    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)

def compress_covariance(
    gaussians: GaussianModel,
    gaussian_importance: torch.Tensor,
    gaussian_comp: CompressionSettings,
):

    # keep_mask_g = gaussian_importance > gaussian_comp.importance_include

    # 对 gaussian_importance 进行排序并获取前 20000 个索引
    top_k_indices_g = torch.argsort(gaussian_importance, descending=True)[:20000]

    # 创建一个与 gaussian_importance 相同形状的布尔 mask
    keep_mask_g = torch.zeros_like(gaussian_importance, dtype=torch.bool)

    # 将前 20000 个索引的位置设置为 True
    keep_mask_g[top_k_indices_g] = True

    qq_mask_g = ~keep_mask_g

    print(f"gaussians keep: {keep_mask_g.float().mean()*100:.2f}%")

    covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()

    
    if qq_mask_g.any():

        print("compressing gaussian splats...")
        # start_time = time.time()
        compressed_features = covariance

        # compressed_features = normalize_to_range2(compressed_features)
        # compressed_features = random_adjustment(compressed_features)

        # end_time = time.time()
        # total_hilbert_time = end_time - start_time
        # print(f"cov Hilbert 曲线总压缩时间: {total_hilbert_time:.2f} 秒")
        cov_codebook, cov_vq_indices = hi_features(
            compressed_features[qq_mask_g],
            covariance[qq_mask_g],
            gaussian_importance[qq_mask_g],
            gaussian_comp.codebook_size,
            gaussian_comp.batch_size,
            gaussian_comp.steps,
            scale_normalize=True,
        )
    else:
        cov_codebook = torch.empty(
            (0, covariance.shape[1], 1), device=covariance.device
        )
        cov_vq_indices = torch.empty((0,), device=covariance.device, dtype=torch.long)

    compressed_cov, cov_indices = join_features(
        covariance,
        keep_mask_g,
        cov_codebook,
        cov_vq_indices,
    )

    print(f"cov码本的大小：{compressed_cov.shape[0]}")

    rot_vq, scale_vq = extract_rot_scale(to_full_cov(compressed_cov))

    gaussians.set_gaussian_indexed(
        rot_vq.to(compressed_cov.device),
        scale_vq.to(compressed_cov.device),
        cov_indices,
    )

def compress_gaussians(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    gaussian_importance: torch.Tensor,
    color_comp: Optional[CompressionSettings],
    gaussian_comp: Optional[CompressionSettings],
    color_compress_non_dir: bool,
    prune_threshold:float=0.,
):
    with torch.no_grad():
        if prune_threshold >= 0:

            percentile = 0.65
            prune_threshold = torch.quantile(color_importance, percentile)

            non_prune_mask = color_importance > prune_threshold
            print(f"prune: {(1-non_prune_mask.float().mean())*100:.2f}%")
            gaussians.mask_splats(non_prune_mask)
            gaussian_importance = gaussian_importance[non_prune_mask]
            color_importance = color_importance[non_prune_mask]

            print(f"最后高斯球的个数：{color_importance.shape[0]}")
        
        if color_comp is not None:
            compress_color(
                gaussians,
                color_importance,
                color_comp,
                color_compress_non_dir,
            )
        if gaussian_comp is not None:
            compress_covariance(
                gaussians,
                gaussian_importance,
                gaussian_comp,
            )

