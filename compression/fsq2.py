from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from contextlib import nullcontext
import gc
import time
from typing import List, Optional, Tuple
import torch.nn as nn
import torch
from torch_scatter import scatter
from torch.nn import Module
from torch import Tensor, int64
from torch.amp import autocast

from einops import rearrange
from tqdm import trange

from curvetest import hilbert_compress
from scene.gaussian_model import GaussianModel
from utils.splats import extract_rot_scale, to_full_cov

def exists(v):
    return v is not None

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    # 使用非 in-place 操作
    updated_moving_avg = moving_avg * decay + new * (1 - decay)
    # 返回更新后的值
    return updated_moving_avg


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
        self.linear1=nn.Linear(48, self.codebook_dim)
        self.linear2=nn.Linear(6, self.codebook_dim)

        self.decay = decay

        self.codebook_size = self._levels.prod().item()
        self.codebook = nn.Parameter(
            torch.empty(self.codebook_size, channels), requires_grad=True
        )
        nn.init.kaiming_uniform_(self.codebook)
        # 设置 requires_grad=True
        self.entry_importance = nn.Parameter(
            torch.zeros(self.codebook_size), requires_grad=True
        )
        self.eps = 1e-5

        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)


    
    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, tmp: torch.Tensor, fsq_modal: FSQ, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        # 不再使用 torch.no_grad()，因为我们需要计算梯度
        # 使用 tmp 作为输入通过 fsq_modal 得到量化的 xhat 和索引 idx
        xhat, idx = fsq_modal(tmp)
        idx = idx.long()  # 确保 idx 是 long 类型
        
        # 聚合 importance 通过 scatter 加总
        acc_importance = scatter(
            importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
        )

        # 更新 entry_importance，使用 EMA (指数移动平均)
        ema_inplace(self.entry_importance, acc_importance, self.decay)

        # 根据索引 idx，将 x 和 importance 聚合到 codebook 上
        codebook = scatter(
            x * importance[:, None],
            idx,
            0,
            reduce="sum",
            dim_size=self.codebook.shape[0],
        )

        # 使用 EMA 更新 codebook
        ema_inplace(
            self.codebook,
            codebook / (acc_importance[:, None] + self.eps),
            self.decay,
        )
        
        # 计算损失部分
        # 将 x 和 codebook 中对应 idx 的向量进行欧氏距离的计算
        codebook_values = self.codebook[idx]  # 获取 codebook 中对应 idx 的值
        diff = x - codebook_values  # 计算 x 和 codebook 对应值的差
        loss = torch.sum(importance * torch.norm(diff, dim=1))  # 计算欧氏距离并累加

        return loss  # 返回损失，允许它在反向传播中回传



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

        # if z.shape[-1] != 5 :
        #     z=hilbert_compress(z)

        codes = self.quantize(z)

        indices = None

        indices = self.codes_to_indices(codes)

        codes = codes.type(orig_dtype)

        return self.codebook.data[indices], indices

def fsq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    levels: List[int],
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance / importance.max()
    fsq_modal = FSQ(
        channels=features.shape[-1],
        levels=levels,
        decay=decay,
    ).to(device=features.device)

    fsq_modal.uniform_init(features)

    # 冻结所有参数
    for param in fsq_modal.parameters():
        param.requires_grad = False

    # 只解冻 linear1 和 linear2
    for param in fsq_modal.linear1.parameters():
        param.requires_grad = True
    for param in fsq_modal.linear2.parameters():
        param.requires_grad = True

    # 定义优化器，只优化 linear1 和 linear2
    optimizer = torch.optim.Adam([
        {'params': fsq_modal.linear1.parameters()},
        {'params': fsq_modal.linear2.parameters()},
    ], lr=1e-3)

    start_time = time.time()

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        fsq_feature = features[batch]
        fsq_feature.requires_grad_(True)  # 确保 fsq_feature 需要梯度

        # 根据输入的最后一维度来选择是用 linear1 还是 linear2
        if fsq_feature.shape[-1] == 48:
            compressed_feature = fsq_modal.linear1(fsq_feature)
        elif fsq_feature.shape[-1] == 6:
            compressed_feature = fsq_modal.linear2(fsq_feature)
        compressed_feature.requires_grad_(True)  # 确保需要梯度

        # 计算 error
        error = fsq_modal.update(compressed_feature, fsq_modal, fsq_feature, importance=importance_n[batch])

        optimizer.zero_grad()  # 清空梯度
        error.backward()       # 反向传播，计算梯度
        optimizer.step()       # 更新权重

        if scale_normalize:
            tr = fsq_modal.codebook[:, [0, 3, 5]].sum(-1)
            with torch.no_grad():
                fsq_modal.codebook.copy_(fsq_modal.codebook / tr[:, None])

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    if features.shape[-1] == 48:
        compressed_features = fsq_modal.linear1(features)
    elif fsq_feature.shape[-1] == 6:
        compressed_features = fsq_modal.linear2(features)
    _, fsq_indices = fsq_modal(compressed_features)  # 计算最终的 indices
    torch.cuda.synchronize(device=fsq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")

    return fsq_modal.codebook.data.detach(), fsq_indices.detach()

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
    # linear1=nn.Linear(48, 5)
    if fsq_mask_c.any():
        # linear1=linear1.to(color_features.device)
        # compressed_features = linear1(color_features)

        # print("compressing color...")
        # start_time = time.time()
        # compressed_features = hilbert_compress(color_features)  # 一次性压缩所有数据
        # end_time = time.time()
        # total_hilbert_time = end_time - start_time
        # print(f"颜色 Hilbert 曲线总压缩时间: {total_hilbert_time:.2f} 秒")
        color_codebook, color_vq_indices = fsq_features(
            # compressed_features[fsq_mask_c],
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

    fsq_mask_g = ~keep_mask_g

    print(f"gaussians keep: {keep_mask_g.float().mean()*100:.2f}%")

    covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()

    # linear2=nn.Linear(6, 5)
    if fsq_mask_g.any():
        # linear2=linear2.to(covariance.device)
        # compressed_features = linear2(covariance)

        print("compressing gaussian splats...")
        # start_time = time.time()
        # compressed_features = hilbert_compress(covariance)  # 一次性压缩所有数据
        # end_time = time.time()
        # total_hilbert_time = end_time - start_time
        # print(f"cov Hilbert 曲线总压缩时间: {total_hilbert_time:.2f} 秒")
        cov_codebook, cov_vq_indices = fsq_features(
            # compressed_features[fsq_mask_g],
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

