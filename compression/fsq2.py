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

    def update(self,features: torch.Tensor, fsq_modal: FSQ, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            #xtmp = self.linear(x)
            xhat, idx = fsq_modal(features)
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

        # if z.shape[-1] != 5 :
        #     z=hilbert_compress(z)

        codes = self.quantize(z)

        indices = None

        indices = self.codes_to_indices(codes)

        codes = codes.type(orig_dtype)

        return self.codebook.data[indices], indices

def fsq_features(
    # compressed_features: torch.Tensor, 
    features: torch.Tensor, 
    importance: torch.Tensor, 
    levels1: List[int],  # 用于第一个FSQ模型的level
    levels2: List[int],  # 用于第二个FSQ模型的level
    vq_chunk: int = 2**16, 
    steps: int = 1000, 
    decay: float = 0.8, 
    scale_normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 对48维的特征分为6个8维的小块，使用fsq_model1编码每个8维数据
    feature_splits = torch.split(features, 8, dim=-1)
    
    fsq_modal1_list = []
    indices1_list = []
    
    for feature_split in feature_splits:
        # 对每个8维特征应用一个FSQ模型 (fsq_model1)
        fsq_modal1 = FSQ(
            channels=feature_split.shape[-1],
            levels=levels1,
            decay=decay
        ).to(device=feature_split.device)
        
        fsq_modal1.uniform_init(feature_split)

        _, indices1 = fsq_modal1(feature_split)
        fsq_modal1_list.append(fsq_modal1.codebook.data.detach())  # 保存第一个FSQ模型的codebook
        indices1_list.append(indices1.detach())  # 保存第一个FSQ模型的indices

    # 将所有 indices1 组合成一个大的 indices 矩阵
    indices1_combined = torch.stack(indices1_list, dim=-1)

    # 将 indices1 作为新的特征输入给第二个 FSQ 模型 (fsq_model2)
    fsq_modal2 = FSQ(
        channels=features.shape[-1],  
        levels=levels2,
        decay=decay
    ).to(device=features.device)

    fsq_modal2.uniform_init(indices1_combined)

    # 训练 fsq_model2
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        feature = features[batch]
        indice1_combined = indices1_combined[batch]
        
        fsq_modal2.update(indice1_combined, fsq_modal2, feature, importance=importance[batch])

        if scale_normalize:
            # 计算codebook协方差矩阵的迹，保证矩阵有归一化的特征值/尺度
            tr = fsq_modal2.codebook[:, [0, 3, 5]].sum(-1)
            fsq_modal2.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    # 生成最终的 indices 和 codebook
    _, final_indices = fsq_modal2(indices1_combined)
    torch.cuda.synchronize(device=final_indices.device)

    return fsq_modal2.codebook.data.detach(), final_indices.detach()


def join_features(
    all_features: torch.Tensor,
    keep_mask: torch.Tensor,
    codebook: torch.Tensor,
    codebook_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_features = all_features[keep_mask]
    features = torch.cat([codebook, keep_features], 0)

    indices = torch.zeros(
        len(all_features), dtype=torch.long, device=all_features.device
    )
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(
        codebook
    )

    return features, indices

@dataclass
class CompressionSettings:
    # codebook_size: int
    levels1: List[int]
    levels2: List[int]
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

    if fsq_mask_c.any():
        

        print("compressing color...")
        
        color_codebook, color_vq_indices = fsq_features(
            # compressed_features[fsq_mask_c],
            color_features[fsq_mask_c],
            color_importance[fsq_mask_c],
            color_comp.levels1,
            color_comp.levels2,
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
    color_features, indices = join_features(
        all_features, keep_mask, color_codebook, color_vq_indices
    )

    gaussians.set_color_indexed(color_features.reshape(-1, n_sh_coefs, 3), indices)

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
        # linear2=linear2.to(covariance.device)
        # compressed_features = linear2(covariance)

        print("compressing gaussian splats...")
        start_time = time.time()
        compressed_features = covariance
        end_time = time.time()
        total_hilbert_time = end_time - start_time
        print(f"cov Hilbert 曲线总压缩时间: {total_hilbert_time:.2f} 秒")
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

