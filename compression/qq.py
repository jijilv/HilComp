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

def verify_uniform_distribution(indices: torch.Tensor, levels: list):
    """
    验证每个区间分配的样本数是否相同。
    参数:
    - indices: 分配后的索引张量 (B, N)，其中每一列是一个维度的区间索引。
    - levels: 每个维度的区间数量列表。

    输出:
    - 打印每个维度中各个区间的样本数量。
    """
    B, N = indices.shape
    for i in range(N):
        num_bins = levels[i]
        counts = torch.zeros(num_bins, dtype=torch.int64)

        # 统计每个区间中的样本数
        for j in range(num_bins):
            counts[j] = torch.sum(indices[:, i] == j)

        # 输出结果
        print(f"Dimension {i}:")
        for j in range(num_bins):
            print(f"  Bin {j}: {counts[j].item()} samples")
        
        # 检查均匀性
        if torch.all(counts == counts[0]):
            print(f"  All bins in Dimension {i} have an equal number of samples.")
        else:
            print(f"  Warning: Bins in Dimension {i} have different sample counts.")


def normalize_to_range(data, min_val=-6.0, max_val=6.0):

    # 计算每个维度的最小值和最大值
    min_data = data.min(dim=0, keepdim=True).values
    max_data = data.max(dim=0, keepdim=True).values
    
    # 防止除以0的情况
    range_data = max_data - min_data
    range_data[range_data == 0] = 1  # 如果某个维度的范围为0，避免除以0

    # 将数据归一化到 [0, 1]
    norm_data = (data - min_data) / range_data

    # 将归一化后的数据映射到 [min_val, max_val]
    scaled_data = norm_data * (max_val - min_val) + min_val

    return scaled_data

def normalize_to_range2(data, min_val=-5.0, max_val=5.0):

    # 将数据转换为 numpy 数组进行排序
    data_np = data.cpu().numpy()

    # 计算分位数
    ranks = np.argsort(np.argsort(data_np, axis=0), axis=0) / (data_np.shape[0] - 1)

    # 将分位数映射到目标范围
    scaled_data = ranks * (max_val - min_val) + min_val

    # 转换回 PyTorch 张量
    scaled_data = torch.tensor(scaled_data, dtype=data.dtype, device=data.device)

    return scaled_data

def random_adjustment(data, max_multiplier=10):
    """
    随机对输入数据加上或减去 1e-10 及其倍数，或不进行调整。

    参数：
    - data: 输入数据，形状为 (B, N)
    - max_multiplier: 1e-10 的最大倍数，默认为 10

    返回：
    - adjusted_data: 调整后的数据
    """
    B, N = data.shape

    device = data.device  # 获取输入数据的设备

    # 生成随机选择的加减量或保持不变，并确保张量在相同设备上
    adjustments = (torch.randint(-1, 2, (B, N), dtype=torch.float32, device=device) *
                   1e-4 * torch.randint(1, max_multiplier + 1, (B, N), dtype=torch.float32, device=device))

    # 调整后的数据
    adjusted_data = data + adjustments

    is_same = torch.equal(data, adjusted_data)

    return adjusted_data

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

# main class

class QQ(Module):
    def __init__(
        self,
        channels: int,
        levels: List[int],
        decay: float = 0.5,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int64)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int64)
        self.register_buffer("_basis", _basis, persistent = False)

        self.codebook_dim = len(levels)

        max_level = self._levels.max().item()
        self.percentiles = torch.zeros((max_level, self.codebook_dim), dtype=torch.float32)

        self.decay = decay

        self.codebook_size = self._levels.prod().item()
        self.codebook = nn.Parameter(
            torch.empty(self.codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(self.codebook_size), requires_grad=False
        )
        self.eps = 1e-5


    def uniform_init(self, x: torch.Tensor,tmp: torch.Tensor):
        """
        初始化码本，并提前计算每个维度的分位数，以便快速量化。
        根据 levels 动态计算分位数的划分。
        """
        B, N = tmp.shape

        # 按每个维度计算对应 levels 的分位数并存储
        for i in range(N):
            num_bins = self._levels[i]  # 获取当前维度的分位数数目
            self.percentiles[:num_bins, i] = torch.quantile(
                tmp[:, i], torch.linspace(0, 1, num_bins + 1, device=x.device)[:-1]
            )

        # 初始化码本的数据
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def uniform_init2(self, x: torch.Tensor, tmp: torch.Tensor):
        """
        初始化码本，并按每个维度计算等频区间的边界。
        """
        B, N = tmp.shape

        # 计算等频区间
        for i in range(N):
            num_bins = self._levels[i]  # 获取当前维度的区间数量
            sorted_data, _ = tmp[:, i].sort()

            # 确保每个区间的数据量相同
            bin_size = B // num_bins

            # 计算等频区间的边界
            for j in range(num_bins):
                if j < num_bins - 1:
                    self.percentiles[j, i] = sorted_data[(j + 1) * bin_size - 1]
                else:
                    self.percentiles[j, i] = sorted_data[-1]  # 最后一个区间包含剩余的数据

        # 初始化码本的数据
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def uniform_init3(self, x: torch.Tensor, tmp: torch.Tensor):
        """
        初始化码本，并将所有数据均匀地分配到不同的索引中（按每个维度的数值大小逐步分配）。
        """
        B, N = tmp.shape

        # 初始化索引张量
        indices = torch.zeros(B, N, dtype=torch.int64, device=tmp.device)

        # 按每个维度的数值大小进行分配
        for i in range(N):
            num_bins = self._levels[i]  # 获取当前维度的区间数量
            sorted_indices = torch.argsort(tmp[:, i])  # 对当前维度的数据进行排序

            # 每个区间的大小
            bin_size = B // num_bins
            remainder = B % num_bins  # 计算剩余样本数

            # 分配索引，确保尽量均匀
            current_start = 0
            for j in range(num_bins):
                # 确定当前区间的结束索引
                current_end = current_start + bin_size + (1 if j < remainder else 0)  # 如果有剩余，前 remainder 个区间多分一个样本
                indices[sorted_indices[current_start:current_end], i] = j
                current_start = current_end

        # verify_uniform_distribution(indices,self._levels)

        # 计算大索引
        big_index = (indices * self._basis).sum(dim=-1)

        # 初始化码本的数据
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

        # 返回大索引
        return big_index
    
    def uniform_init4(self, x: torch.Tensor, tmp: torch.Tensor):
        """
        初始化码本，并将所有数据均匀地分配到不同的**大索引**中。
        """
        B, N = tmp.shape

        # 计算大索引的总数量
        codebook_size = 262144
        total_bins = codebook_size  # 即所有小索引组合的总数

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

    
    def quantize2(self, z):
        """
        根据等频区间对 z 进行量化，并返回大索引。
        """
        B, N = z.shape
        indices = torch.zeros(B, N, dtype=torch.int64, device=z.device)

        # 根据等频区间进行量化
        for i in range(N):
            num_bins = self._levels[i]
            for j in range(num_bins):
                # 判断 z 中每个样本是否落在当前等频区间
                if j == 0:
                    mask = (z[:, i] <= self.percentiles[j, i])
                else:
                    mask = (z[:, i] > self.percentiles[j - 1, i]) & (z[:, i] <= self.percentiles[j, i])
                indices[mask, i] = j

        # 计算大索引
        big_index = (indices * self._basis).sum(dim=-1)

        return big_index

    def update(self,tmp: torch.Tensor, qq_modal: QQ, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            #xtmp = self.linear(x)
            xhat, idx = qq_modal(tmp)
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

    def update2(self,idx: torch.Tensor, tmp: torch.Tensor, qq_modal: QQ, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            #xtmp = self.linear(x)
            # xhat, idx = qq_modal(tmp)
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

    def quantize(self, z):
        """
        量化 z，并返回每个样本在各个维度上的分位数索引。
        分位数的划分由 levels 决定。
        """
        B, N = z.shape
        indices = torch.zeros(B, N, dtype=torch.int64, device=z.device)

        # 快速确定每个维度数据所在的分位数区间
        for i in range(N):
            num_bins = self._levels[i]
            for j in range(num_bins):
                mask = (z[:, i] >= self.percentiles[j, i])
                indices[mask, i] = j
                
        big_index = (indices * self._basis).sum(dim=-1)

        return big_index

    def forward(self, z):

        indices = self.quantize(z)

        # print_distribution(codes)

        return self.codebook.data[indices], indices
    
def print_distribution(codes):
    """
    打印 codes 的分布信息，包括按维度统计和总体统计。
    """
    # 如果 codes 是 PyTorch 张量，将其转换为 NumPy 数组
    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()

    # 获取维度信息
    B, N = codes.shape

    print("Code Distribution by Dimension:")
    
    # 逐个维度计算分布
    for dim in range(N):
        codes_dim = codes[:, dim]
        unique, counts = np.unique(codes_dim, return_counts=True)
        print(f"\n维度 {dim + 1} 的分布:")
        for u, c in zip(unique, counts):
            print(f"Code {u}: {c} 次")

    # 总体分布统计
    codes_flat = codes.flatten()
    unique, counts = np.unique(codes_flat, return_counts=True)

    print("\n总体 Code 分布:")
    for u, c in zip(unique, counts):
        print(f"Code {u}: {c} 次")

def qq_features(
    compressed_features: torch.Tensor,
    features: torch.Tensor,
    importance: torch.Tensor,
    levels: List[int],
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance/importance.max()
    qq_modal = QQ(
        channels=features.shape[-1],
        levels=levels,
        decay=decay,
    ).to(device=features.device)

    codebook_size = reduce(operator.mul, levels)

    # qq_modal.uniform_init(features,compressed_features)
    bigindices = qq_modal.uniform_init4(features,compressed_features)

    # print_fsq_indices_distribution(bigindices,codebook_size)
    start_time = time.time()

    errors = []
    for i in trange(steps):
        
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])

        qq_feature = features[batch]
        indices = bigindices[batch]
        compressed_feature = compressed_features[batch]

        # qq_modal.update(compressed_feature, qq_modal, qq_feature, importance=importance_n[batch])

        qq_modal.update2(indices,compressed_feature, qq_modal, qq_feature, importance=importance_n[batch])
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = qq_modal.codebook[:, [0, 3, 5]].sum(-1)
            qq_modal.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    # _, qq_indices = qq_modal(compressed_features)
    qq_indices=bigindices

    codebook_size = reduce(operator.mul, levels)

    print_fsq_indices_distribution(qq_indices,codebook_size)

    torch.cuda.synchronize(device=qq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return qq_modal.codebook.data.detach(), qq_indices.detach()

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
    print("FSQ Indices 分布 (从 Index 0 开始):")
    for index in full_range:
        count = index_count_dict.get(index, 0)
        print(f"Index {index}: {count} 次")

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
    # codebook_size: int
    levels: List[int]
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int

def transform_data2(input_data):

    # 第1-3维直接对应
    transformed_data = input_data[:, :3]
    
    return transformed_data

def transform_data3(input_data):

    # 取第0、1、3、4维数据
    transformed_data = input_data[:, [0, 1, 3, 4]]
    
    return transformed_data

def transform_data4(input_data):

    transformed_data = input_data[:, :4]
    
    return transformed_data

def transform_data5(input_data):

    # 取第0、1、3、4维数据
    transformed_data = input_data[:, [0, 1, 3]]
    
    return transformed_data

def compress_color(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    color_comp: CompressionSettings,
    color_compress_non_dir: bool,
):
    keep_mask = color_importance > color_comp.importance_include

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

    compressed_features = transform_data2(color_features)

    # compressed_features = symmetric_shift(compressed_features)

    if qq_mask_c.any():

        print("compressing color...")

        color_codebook, color_vq_indices = qq_features(
            compressed_features[qq_mask_c],
            color_features[qq_mask_c],
            color_importance[qq_mask_c],
            color_comp.levels,
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

    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)

def compress_covariance(
    gaussians: GaussianModel,
    gaussian_importance: torch.Tensor,
    gaussian_comp: CompressionSettings,
):

    keep_mask_g = gaussian_importance > gaussian_comp.importance_include

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
        cov_codebook, cov_vq_indices = qq_features(
            compressed_features[qq_mask_g],
            covariance[qq_mask_g],
            gaussian_importance[qq_mask_g],
            gaussian_comp.levels,
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
            non_prune_mask = color_importance > prune_threshold
            print(f"prune: {(1-non_prune_mask.float().mean())*100:.2f}%")
            gaussians.mask_splats(non_prune_mask)
            gaussian_importance = gaussian_importance[non_prune_mask]
            color_importance = color_importance[non_prune_mask]
        
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

