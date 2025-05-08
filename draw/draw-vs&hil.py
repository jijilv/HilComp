import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 Times New Roman
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小

# 数据
cb = [2**17, 180000, 2**18, 360000, 2**19]
cb_labels = [r'$2^{17}$', r'180000', r'$2^{18}$', r'360000', r'$2^{19}$']

# vq 数据
psnr_vq = [26.44, 26.47, 26.48, 26.47, 26.51]
mem_vq = [21.7, 22.2, 22.8, 23.5, 24.4]
time_vq = [39.8, 62.8, 88.3, 105.7, 137.2]
color_utility_vq = [4.74, 3.68, 2.52, 1.97, 1.27]
shape_utility_vq = [8.80, 8.35, 7.92, 7.54, 7.00]
ssim_vq = [0.873, 0.873, 0.874, 0.874, 0.874]
lpips_vq = [0.155, 0.155, 0.154, 0.154, 0.153]

# hilbert 数据
psnr_hilbert = [27.08, 27.16, 27.25, 27.34, 27.46]
mem_hilbert = [29.7, 31.5, 35.2, 39.9, 46.8]
time_hilbert = [12.0, 11.9, 11.8, 12.3, 13.0]
color_utility_hilbert = [100.0, 100.0, 100.0, 100.0, 100.0]  # 100%表示为1
shape_utility_hilbert = [100.0, 100.0, 100.0, 100.0, 100.0]  # 100%表示为1
ssim_hilbert = [0.895, 0.898, 0.902, 0.904, 0.906]
lpips_hilbert = [0.124, 0.122, 0.119, 0.117, 0.115]

# 创建图形
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# 设置颜色
color_vq = '#e44b47'  # 蓝色
color_hilbert = '#4c72b0'  # 红色

# 绘制 PSNR
axs[0, 0].plot(cb, psnr_vq, marker='o', label='vq', color=color_vq)
axs[0, 0].plot(cb, psnr_hilbert, marker='s', label='hilbert', color=color_hilbert)
axs[0, 0].set_title('Reconstruction Quality', fontsize=18)
axs[0, 0].set_xlabel('Codebook Size', fontsize=16)
axs[0, 0].set_ylabel('PSNR (dB)', fontsize=16)  # 修改单位为 dB
axs[0, 0].legend(fontsize=16)

axs[0, 1].plot(cb, ssim_vq, marker='o', label='vq', color=color_vq)
axs[0, 1].plot(cb, ssim_hilbert, marker='s', label='hilbert', color=color_hilbert)
axs[0, 1].set_title('Reconstruction Quality', fontsize=18)
axs[0, 1].set_xlabel('Codebook Size', fontsize=16)
axs[0, 1].set_ylabel('SSIM', fontsize=16)  # 修改单位为 dB
axs[0, 1].legend(fontsize=16)

axs[1, 0].plot(cb, lpips_vq, marker='o', label='vq', color=color_vq)
axs[1, 0].plot(cb, lpips_hilbert, marker='s', label='hilbert', color=color_hilbert)
axs[1, 0].set_title('Reconstruction Quality', fontsize=18)
axs[1, 0].set_xlabel('Codebook Size', fontsize=16)
axs[1, 0].set_ylabel('LPIPS', fontsize=16)  # 修改单位为 dB
axs[1, 0].legend(fontsize=16)
# 绘制 Mem (MB)
axs[1, 1].plot(cb, mem_vq, marker='o', label='vq', color=color_vq)
axs[1, 1].plot(cb, mem_hilbert, marker='s', label='hilbert', color=color_hilbert)
axs[1, 1].set_title('Memory Requirement', fontsize=18)
axs[1, 1].set_xlabel('Codebook Size', fontsize=16)
axs[1, 1].set_ylabel('Mem (MB)', fontsize=16)
axs[1, 1].legend(fontsize=16)

# 绘制 Time (min)
axs[2, 0].plot(cb, time_vq, marker='o', label='vq', color=color_vq)
axs[2, 0].plot(cb, time_hilbert, marker='s', label='hilbert', color=color_hilbert)
axs[2, 0].set_title('Time requirement', fontsize=18)
axs[2, 0].set_xlabel('Codebook Size', fontsize=16)
axs[2, 0].set_ylabel('Time (min)', fontsize=16)
axs[2, 0].legend(fontsize=16)

# 绘制 Utility (使用 color_utility_vq 和 color_utility_hilbert)
axs[2, 1].plot(cb, color_utility_vq, marker='o', label='color vq', color=color_vq, linestyle='-')
axs[2, 1].plot(cb, shape_utility_vq, marker='o', label='shape vq', color=color_vq, linestyle='--')
axs[2, 1].plot(cb, color_utility_hilbert, marker='s', label='color hilbert', color=color_hilbert, linestyle='-')
axs[2, 1].plot(cb, shape_utility_hilbert, marker='s', label='shape hilbert', color=color_hilbert, linestyle='--')
axs[2, 1].set_title('Codebook Utilization', fontsize=18)
axs[2, 1].set_xlabel('Codebook Size', fontsize=16)
axs[2, 1].set_ylabel('Utilization (%)', fontsize=16)
axs[2, 1].legend(fontsize=16)

# 设置横坐标显示为自定义标签，并均匀分布
for ax in axs.flat:
    ax.set_xticks(cb)
    ax.set_xticklabels(cb_labels, fontsize=16)

# 自动调整子图间距
plt.tight_layout()

# 保存图像
plt.savefig('comparison_plot2.png')

# 显示图形
plt.show()
