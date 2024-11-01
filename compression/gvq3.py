from dataclasses import dataclass
import math
import time
import torch
from torch import nn
from torch_scatter import scatter
from typing import Tuple, Optional
from tqdm import trange
import gc
from scene.gaussian_model import GaussianModel
from utils.splats import to_full_cov, extract_rot_scale
from weighted_distance._C import weightedDistance


class VectorQuantize(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int = 2**10,
        decay: float = 0.5,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5

    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_dists, idx = weightedDistance(x.detach(), self.codebook.detach())
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

            return min_dists

    def forward(
        self,
        x: torch.Tensor,
        return_dists: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        min_dists, idx = weightedDistance(x.detach(), self.codebook.detach())
        if return_dists:
            return self.codebook[idx], idx, min_dists
        else:
            return self.codebook[idx], idx


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def vq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance/importance.max()
    vq_model = VectorQuantize(
        channels=features.shape[-1],
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    vq_model.uniform_init(features)

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        vq_feature = features[batch]
        error = vq_model.update(vq_feature, importance=importance_n[batch]).mean().item()
        errors.append(error)
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
            vq_model.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    _, vq_indices = vq_model(features)
    torch.cuda.synchronize(device=vq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return vq_model.codebook.data.detach(), vq_indices.detach()

def group_vq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 将48维特征划分为3组，每组16维
    features_1 = features[:, :16]
    features_2 = features[:, 16:32]
    features_3 = features[:, 32:]

    # 归一化importance
    importance_n = importance / importance.max()

    # 为每个16维组创建一个单独的VQ模型
    vq_model_1 = VectorQuantize(
        channels=16,
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    vq_model_2 = VectorQuantize(
        channels=16,
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    vq_model_3 = VectorQuantize(
        channels=16,
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)

    # 初始化码本
    vq_model_1.uniform_init(features_1)
    vq_model_2.uniform_init(features_2)
    vq_model_3.uniform_init(features_3)

    # 训练每个码本
    for i in trange(steps, desc="Group VQ training"):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        
        error_1 = vq_model_1.update(features_1[batch], importance=importance_n[batch])
        error_2 = vq_model_2.update(features_2[batch], importance=importance_n[batch])
        error_3 = vq_model_3.update(features_3[batch], importance=importance_n[batch])

    # 获取VQ索引
    _, indices_1 = vq_model_1(features_1)
    _, indices_2 = vq_model_2(features_2)
    _, indices_3 = vq_model_3(features_3)

    return (
        vq_model_1.codebook.data.detach(),
        vq_model_2.codebook.data.detach(),
        vq_model_3.codebook.data.detach(),
        torch.stack([indices_1, indices_2, indices_3], dim=-1).detach(),
    )


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

def join_features_group_vq(
    all_features: torch.Tensor,
    keep_mask: torch.Tensor,
    codebooks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    codebook_indices: torch.Tensor,
    batch_size: int = 1024  # 批次大小，用于逐批生成
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将多个码本合并为一个大码本，并返回单一索引。
    
    :param all_features: 原始颜色特征，形状为 (N, 48)
    :param keep_mask: 保留特征的掩码，形状为 (N,)
    :param codebooks: 包含三个码本，每个码本的形状为 (M, 16)
    :param codebook_indices: 压缩后的索引，形状为 (N', 3)
    :param batch_size: 每次生成的码本组合批次大小
    :return: (压缩后的特征, 最终的联合索引)
    """
    # 将原始特征拆分为 3 部分
    features_1 = all_features[:, :16]
    features_2 = all_features[:, 16:32]
    features_3 = all_features[:, 32:]

    # 获取保留的特征
    keep_features_1 = features_1[keep_mask]
    keep_features_2 = features_2[keep_mask]
    keep_features_3 = features_3[keep_mask]

    # 码本拆分
    codebook_1, codebook_2, codebook_3 = codebooks
    M = codebook_1.shape[0]  # 假设每个码本的大小相同

    # 计算联合索引
    combined_indices = (
        codebook_indices[:, 0] * (M * M) +
        codebook_indices[:, 1] * M +
        codebook_indices[:, 2]
    )

    # 逐批生成大码本
    combined_codebook_list = []
    for i in range(0, M, batch_size):
        end_i = min(i + batch_size, M)
        for j in range(0, M, batch_size):
            end_j = min(j + batch_size, M)
            for k in range(0, M, batch_size):
                end_k = min(k + batch_size, M)

                # 生成当前批次的组合
                batch_combined = torch.cat([
                    codebook_1[i:end_i].unsqueeze(1).repeat(1, end_j - j, end_k - k).view(-1, 16),
                    codebook_2[j:end_j].unsqueeze(0).repeat(end_i - i, 1, end_k - k).view(-1, 16),
                    codebook_3[k:end_k].unsqueeze(0).repeat(end_i - i, end_j - j, 1).view(-1, 16)
                ], dim=-1)

                combined_codebook_list.append(batch_combined)

    # 合并所有批次的结果
    combined_codebook = torch.cat(combined_codebook_list, dim=0)

    # 将保留的特征合并到大码本中
    keep_features = torch.cat([keep_features_1, keep_features_2, keep_features_3], dim=-1)
    compressed_features = torch.cat([combined_codebook, keep_features], dim=0)

    # 生成最终的索引
    indices = torch.zeros(len(all_features), dtype=torch.long, device=all_features.device)
    indices[~keep_mask] = combined_indices
    keep_indices = torch.arange(len(keep_features), device=all_features.device) + combined_codebook.shape[0]
    indices[keep_mask] = keep_indices

    return compressed_features, indices


@dataclass
class CompressionSettings:
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int

def compress_color_group_vq(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    color_comp: CompressionSettings,
    color_compress_non_dir: bool,
):
    keep_mask = color_importance > color_comp.importance_include
    print(f"color keep: {keep_mask.float().mean() * 100:.2f}%")

    vq_mask_c = ~keep_mask

    # 将颜色特征按通道展平
    if color_compress_non_dir:
        n_sh_coefs = gaussians.get_features.shape[1]
        
        # 提取 R、G、B 通道特征
        color_features = gaussians.get_features.detach().flatten(-2)
        features_r = color_features[:, ::3]  # R 通道 (0, 3, 6,...)
        features_g = color_features[:, 1::3]  # G 通道 (1, 4, 7,...)
        features_b = color_features[:, 2::3]  # B 通道 (2, 5, 8,...)

        # 合并通道特征并保持 (N, 48)
        color_features = torch.cat((features_r, features_g, features_b), dim=-1)  # 形状为 (N, 48)
    else:
        n_sh_coefs = gaussians.get_features.shape[1] - 1
        
        # 对于非方向性情况，去掉第一个特征
        color_features = gaussians.get_features.detach().flatten(-2)
        features_r = color_features[:, ::3]  # R 通道
        features_g = color_features[:, 1::3]  # G 通道
        features_b = color_features[:, 2::3]  # B 通道

        # 合并通道特征并保持 (N, 48)
        color_features = torch.cat((features_r, features_g, features_b), dim=-1)  # 形状为 (N, 48)

    if vq_mask_c.any():
        print("compressing color with group VQ...")
        codebook_1, codebook_2, codebook_3, color_vq_indices = group_vq_features(
            color_features[vq_mask_c],
            color_importance[vq_mask_c],
            2**6,
            color_comp.batch_size,
            color_comp.steps,
        )
    else:
        codebook_1 = torch.empty((0, 16), device=color_features.device)
        codebook_2 = torch.empty((0, 16), device=color_features.device)
        codebook_3 = torch.empty((0, 16), device=color_features.device)
        color_vq_indices = torch.empty((0, 3), device=color_features.device, dtype=torch.long)

    all_features = color_features
    compressed_features, indices = join_features_group_vq(
        all_features,
        keep_mask,
        (codebook_1, codebook_2, codebook_3),  # 直接传入三个码本作为元组
        color_vq_indices  # 直接传入原始形状的索引张量，不需要展平
    )

    # 这里的假设是 compressed_features 是 (N, 48)
    # 重塑为 (N, 16, 3)，确保按 RGB 通道顺序排列
    compressed_features = compressed_features.reshape(-1, 3, 16)  # 先调整为 (N, 3, 16)
    
    # 重新排列为 (N, 16, 3) 确保每个样本的顺序为 R, G, B
    compressed_features = compressed_features.permute(0, 2, 1)  # 改变维度顺序到 (N, 16, 3)

    gaussians.set_color_indexed(compressed_features, indices)  # 传入调整后的特征和索引

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

    vq_mask_c = ~keep_mask

    # remove zero sh component
    if color_compress_non_dir:
        n_sh_coefs = gaussians.get_features.shape[1]
        color_features = gaussians.get_features.detach().flatten(-2)
    else:
        n_sh_coefs = gaussians.get_features.shape[1] - 1
        color_features = gaussians.get_features[:, 1:].detach().flatten(-2)
    if vq_mask_c.any():
        print("compressing color...")
        color_codebook, color_vq_indices = vq_features(
            color_features[vq_mask_c],
            color_importance[vq_mask_c],
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

    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)

def compress_covariance(
    gaussians: GaussianModel,
    gaussian_importance: torch.Tensor,
    gaussian_comp: CompressionSettings,
):

    keep_mask_g = gaussian_importance > gaussian_comp.importance_include

    vq_mask_g = ~keep_mask_g

    print(f"gaussians keep: {keep_mask_g.float().mean()*100:.2f}%")

    covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()

    if vq_mask_g.any():
        print("compressing gaussian splats...")
        cov_codebook, cov_vq_indices = vq_features(
            covariance[vq_mask_g],
            gaussian_importance[vq_mask_g],
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
            compress_color_group_vq(
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

