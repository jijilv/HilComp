# %%
import gc
import json
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import path
from shutil import copyfile
from typing import Dict, Tuple
import pynvml
import pyRAPL

import numpy as np
import torch
from tqdm import tqdm

# %%
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from tool.energy_meter import EnergyMeter
from compression.hi import CompressionSettings, compress_gaussians, compress_gaussians2
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from finetune import finetune
from utils.image_utils import psnr
from utils.loss_utils import ssim


def unique_output_folder():
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    return os.path.join("./output_vq/", unique_str[0:10])


def calc_importance(
    gaussians: GaussianModel, scene, pipeline_params
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling = gaussians.scaling_qa(
        gaussians.scaling_activation(gaussians._scaling.detach())
    )
    cov3d = gaussians.covariance_activation(
        scaling, 1.0, gaussians.get_rotation.detach(), True
    ).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(
        gaussians.scaling_factor_qa(gaussians._scaling_factor.detach())
    )

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0
    for camera in tqdm(scene.getTrainCameras(), desc="Calculating sensitivity"):
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = render(
            camera,
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d_scaled,
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]

    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels

    # importance = torch.cat(
    #     [gaussians._features_rest.grad],
    #     1,
    # ).flatten(-2)/num_pixels

    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach(), cov_grad.detach()

def calc_importance2(
    gaussians: GaussianModel, scene, pipeline_params
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling = gaussians.scaling_qa(
        gaussians.scaling_activation(gaussians._scaling.detach())
    )
    cov3d = gaussians.covariance_activation(
        scaling, 1.0, gaussians.get_rotation.detach(), True
    ).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(
        gaussians.scaling_factor_qa(gaussians._scaling_factor.detach())
    )

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    h4 = gaussians._opacity.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None

    gaussians._opacity.grad = None

    num_pixels = 0
    for camera in tqdm(scene.getTrainCameras(), desc="Calculating sensitivity"):
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = render(
            camera,
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d_scaled,
        )["render"]
        loss = rendering.sum()

        # gt_image = camera.original_image.cuda()
        # loss = torch.abs(gt_image.sum()-rendering.sum())

        # Ll1 = l1_loss(rendering, rendering)
        # loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(rendering, gt_image))
        # loss = torch.abs((rendering - gt_image)).sum()
        
        loss.backward()

        # loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]

    color_importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels

    dc_importance = torch.cat(
        [gaussians._features_dc.grad],
        1,
    ).flatten(-2)/num_pixels

    sh_importance = torch.cat(
        [gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels

    opacity_contribution = gaussians._opacity.grad / num_pixels
    opacity_contribution = opacity_contribution.squeeze(1)


    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    torch.cuda.empty_cache()
    return color_importance.detach(), dc_importance.detach(), sh_importance.detach(), cov_grad.detach(), opacity_contribution.detach()



# def render_and_eval(
#     gaussians: GaussianModel,
#     scene: Scene,
#     model_params: ModelParams,
#     pipeline_params: PipelineParams,
# ) -> Dict[str, float]:
#     with torch.no_grad():
#         ssims = []
#         psnrs = []
#         lpipss = []

#         views = scene.getTestCameras()

#         bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         for view in tqdm(views, desc="Rendering progress"):
#             rendering = render(view, gaussians, pipeline_params, background)[
#                 "render"
#             ].unsqueeze(0)
#             gt = view.original_image[0:3, :, :].unsqueeze(0)

#             ssims.append(ssim(rendering, gt))
#             psnrs.append(psnr(rendering, gt))
#             lpipss.append(lpips(rendering, gt, net_type="vgg"))
#             gc.collect()
#             torch.cuda.empty_cache()

#         return {
#             "SSIM": torch.tensor(ssims).mean().item(),
#             "PSNR": torch.tensor(psnrs).mean().item(),
#             "LPIPS": torch.tensor(lpipss).mean().item(),
#         }


def render_and_eval(
    gaussians: GaussianModel,
    scene: Scene,
    model_params: ModelParams,
    pipeline_params: PipelineParams,
) -> Dict[str, float]:
    # Initialize pynvml for GPU power measurement
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU (index 0)

    # Initialize pyRAPL for CPU power measurement
    pyRAPL.setup()
    cpu_meter = pyRAPL.Measurement("cpu_measurement")

    with torch.no_grad():
        ssims = []
        psnrs = []
        lpipss = []
        render_time_list = []  # 用于存储每次渲染的时间
        total_render_time_seconds = 0
        gpu_power_samples = []
        cpu_power_samples = []

        views = scene.getTestCameras()
        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for view in tqdm(views, desc="Rendering progress"):
            # Start CPU power measurement
            cpu_meter.begin()

            # Measure GPU power usage and start timing
            torch.cuda.synchronize()
            start_time = time.time()
            rendering = render(view, gaussians, pipeline_params, background)["render"].unsqueeze(0)
            torch.cuda.synchronize()
            end_time = time.time()

            # Stop CPU power measurement
            cpu_meter.end()

            render_time_seconds = end_time - start_time
            total_render_time_seconds += render_time_seconds
            render_time_ms = render_time_seconds * 1000  # 转换为毫秒
            render_time_list.append(render_time_ms)

            # Measure GPU power usage (in Watts)
            gpu_power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert milliwatts to watts
            gpu_power_samples.append(gpu_power_draw)

            # Record CPU power usage from pyRAPL
            cpu_energy_microjoules = cpu_meter.result.pkg[0]  # Total package energy in microjoules
            cpu_power = cpu_energy_microjoules / (render_time_seconds * 1e6)  # Convert to watts
            cpu_power_samples.append(cpu_power)

            gt = view.original_image[0:3, :, :].unsqueeze(0)
            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type="vgg"))
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate average power consumption
        avg_gpu_power = sum(gpu_power_samples) / len(gpu_power_samples) if gpu_power_samples else 0
        avg_cpu_power = sum(cpu_power_samples) / len(cpu_power_samples) if cpu_power_samples else 0

        # Calculate total energy consumption (Energy = Power * Time)
        total_gpu_energy = avg_gpu_power * total_render_time_seconds  # in Joules
        total_cpu_energy = avg_cpu_power * total_render_time_seconds  # in Joules
        total_energy = total_gpu_energy + total_cpu_energy

        # Calculate average energy per view
        avg_gpu_energy = total_gpu_energy / len(views) if len(views) > 0 else 0
        avg_cpu_energy = total_cpu_energy / len(views) if len(views) > 0 else 0
        avg_total_energy = total_energy / len(views) if len(views) > 0 else 0

        # Calculate average render time and FPS
        mean_render_time_ms = sum(render_time_list) / len(render_time_list) if render_time_list else 0
        mean_render_time_seconds = mean_render_time_ms / 1000
        fps = 1.0 / mean_render_time_seconds if mean_render_time_seconds > 0 else 0

        # Print render time and energy consumption statistics
        print(f"Mean render time: {mean_render_time_ms:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"Average GPU power: {avg_gpu_power:.2f} watts")
        print(f"Average CPU power: {avg_cpu_power:.2f} watts")
        print(f"Total GPU energy consumption: {total_gpu_energy:.2f} joules")
        print(f"Total CPU energy consumption: {total_cpu_energy:.2f} joules")
        print(f"Total energy consumption: {total_energy:.2f} joules")
        print(f"Average GPU energy consumption per view: {avg_gpu_energy:.2f} joules")
        print(f"Average CPU energy consumption per view: {avg_cpu_energy:.2f} joules")
        print(f"Average total energy consumption per view: {avg_total_energy:.2f} joules")

        # Clean up pynvml
        pynvml.nvmlShutdown()

        return {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }

# def render_and_eval(
#     gaussians: GaussianModel,
#     scene: Scene,
#     model_params: ModelParams,
#     pipeline_params: PipelineParams,
# ) -> Dict[str, float]:
#     with torch.no_grad():
#         ssims = []
#         psnrs = []
#         lpipss = []
#         render_time_list = []  # 用于存储每次渲染的时间

#         views = scene.getTestCameras()

#         bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         # 初始化能耗计量器
#         em = EnergyMeter(
#             disk_avg_speed=1600 * 1e6,  # Example disk speed (adjust as per your specs)
#             disk_active_power=6,       # Example active power (adjust as per your specs)
#             disk_idle_power=1.42,      # Example idle power (adjust as per your specs)
#             label="Rendering Energy Meter",
#             include_idle=False
#         )
#         em.begin()  # 开始测量能耗

#         for view in tqdm(views, desc="Rendering progress"):
#             torch.cuda.synchronize()
#             start_time = time.time()
#             rendering = render(view, gaussians, pipeline_params, background)["render"].unsqueeze(0)
#             torch.cuda.synchronize()
#             end_time = time.time()

#             render_time_ms = (end_time - start_time) * 1000  # 渲染时间（毫秒）
#             render_time_list.append(render_time_ms)

#             gt = view.original_image[0:3, :, :].unsqueeze(0)

#             ssims.append(ssim(rendering, gt))
#             psnrs.append(psnr(rendering, gt))
#             lpipss.append(lpips(rendering, gt, net_type="vgg"))
#             gc.collect()
#             torch.cuda.empty_cache()

#         em.end()  # 结束能耗测量

#         # 计算总能耗和平均能耗
#         energy_consumption = em.get_total_joules_per_component()
#         # 将可能的numpy数组转换为标量
#         total_energy = sum(value.sum() if isinstance(value, np.ndarray) else value for value in energy_consumption.values())
#         num_images = len(views)
#         avg_energy_per_image = total_energy / num_images if num_images > 0 else 0

#         # 计算平均渲染时间和FPS
#         mean_render_time_ms = sum(render_time_list) / len(render_time_list)
#         mean_render_time_seconds = mean_render_time_ms / 1000
#         fps = 1.0 / mean_render_time_seconds

#         print(f"Mean render time: {mean_render_time_ms:.2f} ms")
#         print(f"FPS: {fps:.2f}")
#         print("Energy consumption per component:")
#         print(energy_consumption)
#         print(f"Total energy consumption: {total_energy:.2f} joules")
#         print(f"Average energy consumption per image: {avg_energy_per_image:.2f} joules")

#         return {
#             "SSIM": torch.tensor(ssims).mean().item(),
#             "PSNR": torch.tensor(psnrs).mean().item(),
#             "LPIPS": torch.tensor(lpipss).mean().item(),
#         }


def run_hi(
    model_params: ModelParams,
    optim_params: OptimizationParams,
    pipeline_params: PipelineParams,
    comp_params: CompressionParams,
):
    gaussians = GaussianModel(
        model_params.sh_degree, quantization=not optim_params.not_quantization_aware
    )
    scene = Scene(
        model_params, gaussians, load_iteration=comp_params.load_iteration, shuffle=True
    )

    if comp_params.start_checkpoint:
        (checkpoint_params, first_iter) = torch.load(comp_params.start_checkpoint)
        gaussians.restore(checkpoint_params, optim_params)


    timings ={}

    # %%

    
    color_importance, gaussian_sensitivity = calc_importance(
        gaussians, scene, pipeline_params
    )

    start_time = time.time()
    color_contribution, dc_contribution, sh_contribution, cov_contribution, opacity_contribution = calc_importance2(
        gaussians, scene, pipeline_params
    )

    end_time = time.time()
    timings["sensitivity_calculation"] = end_time-start_time
    # %%
    print("vq compression..")
    with torch.no_grad():
        start_time = time.time()
        color_importance_n = color_importance.amax(-1)
        gaussian_importance_n = gaussian_sensitivity.amax(-1)

        print(f"初始高斯球的个数：{color_importance_n.shape[0]}")

        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=comp_params.color_importance_include,
            steps=int(comp_params.color_cluster_iterations),
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=comp_params.gaussian_importance_include,
            steps=int(comp_params.gaussian_cluster_iterations),
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )

        # compress_gaussians(
        #     gaussians,
        #     color_importance_n,
        #     gaussian_importance_n,
        #     color_compression_settings if not comp_params.not_compress_color else None,
        #     gaussian_compression_settings
        #     if not comp_params.not_compress_gaussians
        #     else None,
        #     comp_params.color_compress_non_dir,
        #     prune_threshold=comp_params.prune_threshold,
        # )

        compress_gaussians2(
            gaussians,
            color_contribution,
            dc_contribution,
            sh_contribution,
            cov_contribution,
            opacity_contribution,
            color_compression_settings if not comp_params.not_compress_color else None,
            gaussian_compression_settings
            if not comp_params.not_compress_gaussians
            else None,
            comp_params.color_compress_non_dir,
            prune_threshold=comp_params.prune_threshold,
        )
        
        end_time = time.time()
        timings["clustering"]=end_time-start_time

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)

    copyfile(
        path.join(model_params.model_path, "cfg_args"),
        path.join(comp_params.output_vq, "cfg_args"),
    )
    model_params.model_path = comp_params.output_vq

    with open(
        os.path.join(comp_params.output_vq, "cfg_args_comp"), "w"
    ) as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(comp_params))))

    iteration = scene.loaded_iter + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:

        print("评估微调前的模型...")
        pre_finetune_metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
        print("微调前的指标:", pre_finetune_metrics)

        start_time = time.time()
        finetune(
            scene,
            model_params,
            optim_params,
            comp_params,
            pipeline_params,
            testing_iterations=[
                -1
            ],
            debug_from=-1,
        )
        end_time = time.time()
        timings["finetune"]=end_time-start_time

        # %%
    # out_file = path.join(
    #     comp_params.output_vq,
    #     f"point_cloud/iteration_{iteration}/point_cloud.npz",
    # )

    out_file2 = path.join(
        comp_params.output_vq,
        f"point_cloud/iteration_{iteration}/point_cloud.ply",
    )

    start_time = time.time()
    # gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)

    gaussians.save_ply(out_file2)
    end_time = time.time()

    timings["encode"]=end_time-start_time
    timings["total"]=sum(timings.values())

    print("Timings data:")
    for key, value in timings.items():
        print(f"  {key}: {value:.4f} seconds")

    with open(f"{comp_params.output_vq}/times.json","w") as f:
        json.dump(timings,f)
    # file_size = os.path.getsize(out_file) / 1024**2

    file_size2 = os.path.getsize(out_file2) / 1024**2

    # print(f"saved vq finetuned model to {out_file}")

    # eval model
    print("evaluating...")
    metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
    # metrics["size"] = file_size

    metrics["size2"] = file_size2

    print(metrics)


    with open(f"{comp_params.output_vq}/results.json","w") as f:
        json.dump({f"ours_{iteration}":metrics},f,indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    args = get_combined_args(parser)

    if args.output_vq is None:
        args.output_vq = unique_output_folder()

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)

    run_hi(model_params, optim_params, pipeline_params, comp_params)
