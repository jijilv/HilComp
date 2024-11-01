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

from einops import rearrange
from tqdm import trange

from curvetest import hilbert_compress
from scene.gaussian_model3 import GaussianModel
from utils.splats import extract_rot_scale, to_full_cov

import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

def normalize_to_range2(data, min_val=-2.0, max_val=2.0):
    """
    使用分位数归一化将输入数据映射到 (-1.6, 1.6) 范围内，以便数据更加分散地分布在目标范围内。
    
    参数：
    - data: 输入的 (B, N) 张量
    - min_val: 归一化的最小值
    - max_val: 归一化的最大值
    
    返回：
    - 归一化后的 (B, N) 张量
    """
    # 将数据转换为 numpy 数组进行排序
    data_np = data.cpu().numpy()

    # 对每个维度进行排序
    sorted_data = np.sort(data_np, axis=0)

    # 计算分位数
    ranks = np.argsort(np.argsort(data_np, axis=0), axis=0) / (data_np.shape[0] - 1)

    # 将分位数映射到目标范围
    scaled_data = ranks * (max_val - min_val) + min_val

    # 转换回 PyTorch 张量
    scaled_data = torch.tensor(scaled_data, dtype=data.dtype, device=data.device)

    return scaled_data

def exists(v):
    return v is not None

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

# main class

class FSQ(Module):
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

        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)


    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self,tmp: torch.Tensor, fsq_modal: FSQ, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            #xtmp = self.linear(x)
            xhat, idx = fsq_modal(tmp)
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

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        # z = z.to(shift.device)
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int64)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        codes = self._indices_to_codes(indices)

        return codes

    def forward(self, z):

        orig_dtype = z.dtype

        codes = self.quantize(z)

        indices = None

        indices = self.codes_to_indices(codes)

        codes = codes.type(orig_dtype)

        return self.codebook.data[indices], indices

def fsq_features(
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
    fsq_modal = FSQ(
        channels=features.shape[-1],
        levels=levels,
        decay=decay,
    ).to(device=features.device)

    fsq_modal.uniform_init(features)

    start_time = time.time()

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        fsq_feature = features[batch]
        compressed_feature = compressed_features[batch]

        fsq_modal.update(compressed_feature, fsq_modal, fsq_feature, importance=importance_n[batch])
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = fsq_modal.codebook[:, [0, 3, 5]].sum(-1)
            fsq_modal.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    _, fsq_indices = fsq_modal(compressed_features)

    codebook_size = codebook_size = reduce(operator.mul, levels)

    print_fsq_indices_distribution(fsq_indices,codebook_size)

    torch.cuda.synchronize(device=fsq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return fsq_modal.codebook.data.detach(), fsq_indices.detach()

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

    exceeding_indices = len([index for index in index_count_dict if index_count_dict[index] > 10])
    exceeding_rate = (exceeding_indices / codebook_size) * 100

    # 打印超过使用 10 次的几率
    print(f"索引超过使用 10 次的几率: {exceeding_rate:.2f}%")

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

    # 统计合并后每个码本条目的出现次数
    total_codebook_count = torch.bincount(indices, minlength=compressed_features.size(0))

    # 转换为概率分布
    codebook_prob = total_codebook_count.float() / total_codebook_count.sum()

    codebook_prob = torch.clamp(codebook_prob, min=1e-10)

    return compressed_features, indices, codebook_prob

@dataclass
class CompressionSettings:
    # codebook_size: int
    levels: List[int]
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int

def transform_data(input_data):
    """
    将 (N, 48) 维数据转换为 (N, 6) 维数据。
    
    第 1 维对应第 1 维，
    第 2 维对应第 2 维，
    第 3 维对应第 3 维，
    第 4 维为第 4-12 维的平均值，
    第 5 维为第 13-27 维的平均值，
    第 6 维为第 28-48 维的平均值。
    """
    # 第1-3维直接对应
    new_data = input_data[:, :3]

    # 计算第4-6维
    dim4 = torch.mean(input_data[:, 3:12], dim=1, keepdim=True)
    dim5 = torch.mean(input_data[:, 12:27], dim=1, keepdim=True)
    dim6 = torch.mean(input_data[:, 27:48], dim=1, keepdim=True)

    # 拼接结果
    transformed_data = torch.cat((new_data, dim4, dim5, dim6), dim=1)
    
    return transformed_data

def transform_data2(input_data):

    # 第1-3维直接对应
    transformed_data = input_data[:, :3]
    
    return transformed_data

def transform_data3(input_data):

    # 第1-3维直接对应
    new_data = input_data[:, :3]

    dim4 = torch.mean(input_data[:, 3:48], dim=1, keepdim=True)

    transformed_data = torch.cat((new_data, dim4), dim=1)
    
    return transformed_data

def transform_data4(input_data):
    """
    将 (N, 48) 维数据转换为 (N, 6) 维数据。
    
    第 1 维对应第 1 维，
    第 2 维对应第 2 维，
    第 3 维对应第 3 维，
    第 4 维为第 4, 7, 10, 13, ..., 46 维的平均值，
    第 5 维为第 5, 8, 11, 14, ..., 47 维的平均值，
    第 6 维为第 6, 9, 12, 15, ..., 48 维的平均值。
    """
    # 第1-3维直接对应
    new_data = input_data[:, :3]

    # 计算第4-6维
    dim4 = torch.mean(input_data[:, 3:46:3], dim=1, keepdim=True)  # 第 4, 7, 10, ..., 46 维
    dim5 = torch.mean(input_data[:, 4:47:3], dim=1, keepdim=True)  # 第 5, 8, 11, ..., 47 维
    dim6 = torch.mean(input_data[:, 5:48:3], dim=1, keepdim=True)  # 第 6, 9, 12, ..., 48 维

    # 拼接结果
    transformed_data = torch.cat((new_data, dim4, dim5, dim6), dim=1)
    
    return transformed_data

def transform_data5(input_data):

    # 第1-6维直接对应
    transformed_data = input_data[:, :6]
    
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

    fsq_mask_c = ~keep_mask

    # remove zero sh component
    if color_compress_non_dir:
        n_sh_coefs = gaussians.get_features.shape[1]
        color_features = gaussians.get_features.detach().flatten(-2)
    else:
        n_sh_coefs = gaussians.get_features.shape[1] - 1
        color_features = gaussians.get_features[:, 1:].detach().flatten(-2)

    compressed_features=transform_data2(color_features)

    if fsq_mask_c.any():

        print("compressing color...")

        color_codebook, color_vq_indices = fsq_features(
            compressed_features[fsq_mask_c],
            color_features[fsq_mask_c],
            color_importance[fsq_mask_c],
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
    compressed_features, indices, codebook_prob = join_features(
        all_features, keep_mask, color_codebook, color_vq_indices
    )

    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices, codebook_prob)

def compress_covariance(
    gaussians: GaussianModel,
    gaussian_importance: torch.Tensor,
    gaussian_comp: CompressionSettings,
):

    keep_mask_g = gaussian_importance > gaussian_comp.importance_include

    fsq_mask_g = ~keep_mask_g

    print(f"gaussians keep: {keep_mask_g.float().mean()*100:.2f}%")

    covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()

    
    if fsq_mask_g.any():

        print("compressing gaussian splats...")
        start_time = time.time()
        compressed_features = covariance

        compressed_features = normalize_to_range2(compressed_features)

        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"cov Hilbert 曲线总压缩时间: {total_hilbert_time:.2f} 秒")
        cov_codebook, cov_vq_indices = fsq_features(
            compressed_features[fsq_mask_g],
            covariance[fsq_mask_g],
            gaussian_importance[fsq_mask_g],
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

    compressed_cov, cov_indices, codebook_prob = join_features(
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
        codebook_prob
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

