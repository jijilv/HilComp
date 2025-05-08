import os
import subprocess

def main():
    # 设置要调用的脚本和参数
    script_path = "/home/kemove/github/hilcomp/compress.py"
    model_path = "/mnt/hy/data/3dgs_models/truck"
    data_device = "cuda"
    output_vq = "/mnt/hy/output/truck-vq-99/"
    source_path = "/mnt/hy/data/tandt/truck"  # 这里设置你的 source_path

    # script_path = "/home/kemove/github/c3dgs/compresshi.py"
    # model_path = "/mnt/hy/data/3dgs_models/hollywood"
    # data_device = "cuda"
    # output_vq = "/mnt/hy/output/hollywood-hi/"
    # source_path = "/mnt/hy/data/bungeenerf/hollywood"  # 这里设置你的 source_path
    # resolution = 1

    # script_path = "/home/kemove/github/c3dgs/compresshi.py"
    # model_path = "/home/kemove/data/3dgs_models/rubble"
    # data_device = "cuda"
    # output_vq = "/home/kemove/output/rubble-hi/"
    # source_path = "/home/kemove/data/meganerf/rubble"  # 这里设置你的 source_path


    # 准备命令
    command = [
        "python", script_path,
        "--model_path", model_path,
        "--data_device", data_device,
        "--output_vq", output_vq,
        "--source_path", source_path  # 添加 source_path 参数
    ]

    # 调用 compress.py 脚本
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")

if __name__ == "__main__":
    main()
