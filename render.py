#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# import gc
# import torch
# from scene import Scene
# import os
# from tqdm import tqdm
# from os import makedirs
# from gaussian_renderer import render
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel


# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(
#             rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
#         )
#         torchvision.utils.save_image(
#             gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
#         )
#         gc.collect()
#         torch.cuda.empty_cache()


# def render_sets(
#     dataset: ModelParams,
#     iteration: int,
#     pipeline: PipelineParams,
#     skip_train: bool,
#     skip_test: bool,
# ):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree)
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,override_quantization=True)

#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         if not skip_train:
#             render_set(
#                 dataset.model_path,
#                 "train",
#                 scene.loaded_iter,
#                 scene.getTrainCameras(),
#                 gaussians,
#                 pipeline,
#                 background,
#             )

#         if not skip_test:
#             render_set(
#                 dataset.model_path,
#                 "test",
#                 scene.loaded_iter,
#                 scene.getTestCameras(),
#                 gaussians,
#                 pipeline,
#                 background,
#             )


# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(
#         model.extract(args),
#         args.iteration,
#         pipeline.extract(args),
#         args.skip_train,
#         args.skip_test,
#     )

# import gc
# import torch
# from scene import Scene
# import os
# from tqdm import tqdm
# from os import makedirs
# from gaussian_renderer import render
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
# import time
# import numpy as np

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     render_time_list = []  # 用于存储渲染时间
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         torch.cuda.synchronize()
#         start = time.time()
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         torch.cuda.synchronize()
#         end = time.time()

#         render_time_list.append((end - start) * 1000)  # 将渲染时间转换为毫秒
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(
#             rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
#         )
#         torchvision.utils.save_image(
#             gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
#         )
#         gc.collect()
#         torch.cuda.empty_cache()

#     # 计算并打印FPS
#     mean_time_ms = np.mean(render_time_list[5:])  # 忽略前5个样本计算平均时间
#     mean_time_seconds = mean_time_ms / 1000  # 转换为秒
#     fps = 1.0 / mean_time_seconds

#     print("Mean render time (excluding first 5 frames): {:.2f} ms".format(mean_time_ms))
#     print("FPS: {:.2f}".format(fps))

# def render_sets(
#     dataset: ModelParams,
#     iteration: int,
#     pipeline: PipelineParams,
#     skip_train: bool,
#     skip_test: bool,
# ):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree)
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, override_quantization=True)

#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         if not skip_train:
#             render_set(
#                 dataset.model_path,
#                 "train",
#                 scene.loaded_iter,
#                 scene.getTrainCameras(),
#                 gaussians,
#                 pipeline,
#                 background,
#             )

#         if not skip_test:
#             render_set(
#                 dataset.model_path,
#                 "test",
#                 scene.loaded_iter,
#                 scene.getTestCameras(),
#                 gaussians,
#                 pipeline,
#                 background,
#             )

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(
#         model.extract(args),
#         args.iteration,
#         pipeline.extract(args),
#         args.skip_train,
#         args.skip_test,
#     )


import numpy as np
import torch
import time
import pynvml
import pyRAPL
from tqdm import tqdm
import os
from torchvision.utils import save_image
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    # Initialize pynvml for GPU power measurement
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU (index 0)

    # Initialize pyRAPL for CPU power measurement
    pyRAPL.setup()
    cpu_meter = pyRAPL.Measurement("cpu_measurement")

    render_time_list = []
    total_render_time_seconds = 0
    gpu_power_samples = []
    cpu_power_samples = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Start CPU power measurement
        cpu_meter.begin()

        # Measure initial GPU power
        torch.cuda.synchronize()
        start_time = time.time()  # Start timer
        rendering = render(view, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize()
        end_time = time.time()  # End timer

        # Stop CPU power measurement
        cpu_meter.end()

        render_time_seconds = end_time - start_time
        total_render_time_seconds += render_time_seconds
        render_time_list.append(render_time_seconds * 1000)  # Convert to milliseconds

        # Measure GPU power usage (in Watts)
        gpu_power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert milliwatts to watts
        gpu_power_samples.append(gpu_power_draw)

        # Record CPU power usage from pyRAPL (convert to Watts)
        cpu_energy_microjoules = cpu_meter.result.pkg[0]  # Total package energy in microjoules
        cpu_power = cpu_energy_microjoules / (render_time_seconds * 1e6)  # Convert to watts
        cpu_power_samples.append(cpu_power)

        # Save the rendered image and ground truth
        gt = view.original_image[0:3, :, :]
        save_image(rendering, os.path.join(render_path, '{:05d}.png'.format(idx)))
        save_image(gt, os.path.join(gts_path, '{:05d}.png'.format(idx)))

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
    print(f"Total render time: {total_render_time_seconds:.2f} seconds")
    print(f"Mean render time per view: {mean_render_time_ms:.2f} ms")
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

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
# python render.py -m "/mnt/hy/output/playroom-hi-f/"
# python npz2ply.py "/mnt/hy/output/playroom-hi/point_cloud/iteration_30000/point_cloud.npz"
# python render.py -m "/mnt/hy/output/drjohnson-hi-cn/"
# python npz2ply.py "/mnt/hy/output/Leonard-hi-217/point_cloud/iteration_40000/point_cloud.npz"
# python render.py -m "/mnt/hy/output/bicycle-large/"
# python metrics.py -m "/mnt/hy/output/bicycle-large/"