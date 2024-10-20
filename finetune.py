# import os
# import torch
# from random import randint
# from utils.loss_utils import l1_loss,  ssim
# from gaussian_renderer import render
# from scene import Scene
# import uuid
# from tqdm import tqdm
# from utils.image_utils import psnr
# from argparse import  Namespace

# def finetune(scene: Scene, dataset, opt, comp, pipe, testing_iterations, debug_from):
#     prepare_output_and_logger(comp.output_vq, dataset)

#     first_iter = scene.loaded_iter
#     max_iter = first_iter + comp.finetune_iterations

#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#     iter_start = torch.cuda.Event(enable_timing=True)
#     iter_end = torch.cuda.Event(enable_timing=True)

#     scene.gaussians.training_setup(opt)
#     scene.gaussians.update_learning_rate(first_iter)

#     viewpoint_stack = None
#     ema_loss_for_log = 0.0
#     progress_bar = tqdm(range(first_iter, max_iter), desc="Training progress")
#     first_iter += 1
#     for iteration in range(first_iter, max_iter + 1):
#         iter_start.record()

#         # Pick a random Camera
#         if not viewpoint_stack:
#             viewpoint_stack = scene.getTrainCameras().copy()
#         viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

#         # Render
#         if (iteration - 1) == debug_from:
#             pipe.debug = True
#         render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background)
#         image, viewspace_point_tensor, visibility_filter, radii = (
#             render_pkg["render"],
#             render_pkg["viewspace_points"],
#             render_pkg["visibility_filter"],
#             render_pkg["radii"],
#         )

#         # Loss
#         gt_image = viewpoint_cam.original_image.cuda()
#         Ll1 = l1_loss(image, gt_image)
#         loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
#             1.0 - ssim(image, gt_image)
#         )
#         loss.backward()

#         iter_end.record()
#         scene.gaussians.update_learning_rate(iteration)

#         with torch.no_grad():
#             # Progress bar
#             ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
#             if iteration % 10 == 0:
#                 progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
#                 progress_bar.update(10)
#             if iteration == max_iter:
#                 progress_bar.close()

#             # Optimizer step
#             if iteration < max_iter:
#                 scene.gaussians.optimizer.step()
#                 scene.gaussians.optimizer.zero_grad()


# def prepare_output_and_logger(output_folder, args):
#     if not output_folder:
#         if os.getenv("OAR_JOB_ID"):
#             unique_str = os.getenv("OAR_JOB_ID")
#         else:
#             unique_str = str(uuid.uuid4())
#         output_folder = os.path.join("./output/", unique_str[0:10])

#     # Set up output folder
#     print("Output folder: {}".format(output_folder))
#     os.makedirs(output_folder, exist_ok=True)
#     with open(os.path.join(output_folder, "cfg_args"), "w") as cfg_log_f:
#         cfg_log_f.write(str(Namespace(**vars(args))))

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
import time  # 引入time模块

def finetune(scene: Scene, dataset, opt, comp, pipe, testing_iterations, debug_from):
    prepare_output_and_logger(comp.output_vq, dataset)

    first_iter = scene.loaded_iter
    max_iter = first_iter + comp.finetune_iterations

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene.gaussians.training_setup(opt)
    scene.gaussians.update_learning_rate(first_iter)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max_iter), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, max_iter + 1):
        # 开始计时
        iter_start_time = time.time()

        # Pick a random Camera
        camera_start_time = time.time()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        camera_end_time = time.time()

        # 渲染
        render_start_time = time.time()
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        render_end_time = time.time()

        # 损失计算
        loss_start_time = time.time()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        loss_end_time = time.time()

        # 学习率更新
        lr_update_start_time = time.time()
        scene.gaussians.update_learning_rate(iteration)
        lr_update_end_time = time.time()

        # 优化器步骤
        optimizer_start_time = time.time()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == max_iter:
                progress_bar.close()

            if iteration < max_iter:
                scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad()
        optimizer_end_time = time.time()

        # 每100次迭代打印一次时间信息
        if iteration % 100 == 0:
            iter_end_time = time.time()
            print(f"Iteration {iteration}/{max_iter}:")
            print(f"  - Camera Selection Time: {camera_end_time - camera_start_time:.4f} seconds")
            print(f"  - Rendering Time: {render_end_time - render_start_time:.4f} seconds")
            print(f"  - Loss Calculation Time: {loss_end_time - loss_start_time:.4f} seconds")
            print(f"  - Learning Rate Update Time: {lr_update_end_time - lr_update_start_time:.4f} seconds")
            print(f"  - Optimizer Step Time: {optimizer_end_time - optimizer_start_time:.4f} seconds")
            print(f"  - Total Iteration Time: {iter_end_time - iter_start_time:.4f} seconds")
            print("-" * 50)

def prepare_output_and_logger(output_folder, args):
    if not output_folder:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        output_folder = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(output_folder))
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
