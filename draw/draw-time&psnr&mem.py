# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据：压缩时间和 PSNR 值
# compression_times = [706.9, 781.0, 935.0, 12701.1, 22543.2, 4563.9, 3738.8, 2199.5, 2639.3]  # 压缩时间（单位：秒）
# psnr_values = [27.25, 27.48, 26.96, 25.67, 27.34, 26.25, 25.81, 25.42, 27.73]  # PSNR值

# # 每种方法的内存占用 (MB)
# mem_values = [35.2, 47.1, 54.2, 36.0, 64.3, 81.5, 78.1, 185.0, 124.7]  # 示例数据

# # 9 种方法，每种方法对应一个不同的形状和颜色
# methods = ['Ours', 'Ours-large', 'C3DGS', 'RDO', 'MesonGS', 
#            'LightGaussian', 'Compact3DGS', 'SOG', 'EAGLES']
# markers = ['o', 'o', '^', '^', '^', '^', '^', '^', '^']  # 第一和第二个方法使用相同的形状
# colors = ['#8B0000', '#8B0000', '#4682B4', '#8B4513', '#6A5ACD', '#D2691E', '#708090', 'green', 'orange']

# # 创建散点图
# for i in range(9):
#     # 绘制数据点
#     plt.scatter(compression_times[i], psnr_values[i], 
#                 marker=markers[i],  # 对应的形状
#                 color=colors[i],    # 对应的颜色
#                 s=150)  # 点的大小为 150

#     # 直接在点旁边添加文字标签，位置可以根据需要调整（偏移量为 +0.1）
#     if i != 4 and i != 8 and i != 1 and i != 0 and i != 6:
#         plt.text(compression_times[i] + 0.1, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 0:
#         plt.text(compression_times[i] - 0.6, psnr_values[i] + 0.35, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 1: 
#         continue
#     elif i == 8:
#         plt.text(compression_times[i] - 600.0, psnr_values[i] - 0.2, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 4:
#         plt.text(compression_times[i] - 12000.0, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)
#     else:
#         plt.text(compression_times[i] - 2500.0, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)
        


# # 连接第一种和第二种方法的点
# plt.plot([compression_times[0], compression_times[1]], 
#          [psnr_values[0], psnr_values[1]], color='#8B0000', linestyle='-', lw=2)

# # 设置标题和轴标签
# plt.xlabel("Compression Time (seconds)", fontsize=16)  # 设置 x 轴标签字体大小
# plt.ylabel("PSNR (dB)", fontsize=16)  # 设置 y 轴标签字体大小

# # 去掉坐标轴上的尖尖刻度线
# plt.tick_params(axis='both', which='both', length=2, labelsize=16)  # 设置刻度线数字的字体大小为12

# # 禁用背景网格线（去掉灰色背景线）
# plt.grid(True)

# # 调整 x 轴比例尺：选择 log 或 linear
# plt.xscale('log')  # 或者 plt.xscale('linear')，如果你想要线性比例尺

# # 设置 x 轴的范围（如果需要）
# plt.xlim(500, 25000)  # 设置 x 轴的范围

# # 调整布局，去掉多余的空白
# plt.tight_layout()

# # 显示图例，设置图例在右侧
# handles = []
# for i in range(9):
#     # 创建每个图例项，将方法名称和内存占用数值一起显示，并且内存占用数值右对齐
#     label = f'{methods[i]} {str(mem_values[i]).rjust(5)}'  # 使用rjust(5)来右对齐内存数值
#     handles.append(plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=16, 
#                                label=label))

# # 添加图例，显示方法名称和内存占用
# plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title='Mem(MB)', fontsize=16, title_fontsize=16)


# # 保存图表到文件
# plt.savefig("time_vs_psnr_vs_mem2.png", dpi=300, bbox_inches='tight')

# # 显示图表
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 示例数据：压缩时间和 PSNR 值
compression_times = [706.9, 781.0, 935.0, 12701.1, 22543.2, 4563.9, 3738.8, 2199.5, 2639.3]  # 压缩时间（单位：秒）
compression_times_min = [t / 60 for t in compression_times]  # 将时间转换为分钟
psnr_values = [27.25, 27.48, 26.96, 25.67, 27.34, 26.25, 25.81, 25.42, 27.73]  # PSNR值

# 每种方法的内存占用 (MB)
mem_values = [35.2, 47.1, 54.2, 36.0, 64.3, 81.5, 78.1, 185.0, 124.7]  # 示例数据

# 9 种方法，每种方法对应一个不同的形状和颜色
methods = ['Ours', 'Ours-large', 'C3DGS', 'RDO', 'MesonGS', 
           'LightGaussian', 'Compact3DGS', 'SOG', 'EAGLES']
markers = ['o', 'o', '^', '^', '^', '^', '^', '^', '^']  # 第一和第二个方法使用相同的形状
colors = ['#8B0000', '#8B0000', '#4682B4', '#8B4513', '#6A5ACD', '#D2691E', '#708090', 'green', 'orange']

# 创建散点图
for i in range(9):
    # 绘制数据点
    plt.scatter(compression_times[i], psnr_values[i], 
                marker=markers[i],  # 对应的形状
                color=colors[i],    # 对应的颜色
                s=150)  # 点的大小为 150

    # 直接在点旁边添加文字标签，位置可以根据需要调整（偏移量为 +0.1）
    if i != 4 and i != 8 and i != 1 and i != 0 and i != 6:
        plt.text(compression_times[i] + 0.1, psnr_values[i] + 0.1, methods[i], 
                 color=colors[i], fontsize=18)
    elif i == 0:
        plt.text(compression_times[i] - 0.6, psnr_values[i] + 0.35, methods[i], 
                 color=colors[i], fontsize=18)
    elif i == 1: 
        continue
    elif i == 8:
        plt.text(compression_times[i] - 600.0, psnr_values[i] - 0.2, methods[i], 
                 color=colors[i], fontsize=18)
    elif i == 4:
        plt.text(compression_times[i] - 13000.0, psnr_values[i] + 0.1, methods[i], 
                 color=colors[i], fontsize=18)
    else:
        plt.text(compression_times[i] - 2500.0, psnr_values[i] + 0.1, methods[i], 
                 color=colors[i], fontsize=18)

# 连接第一种和第二种方法的点
plt.plot([compression_times[0], compression_times[1]], 
         [psnr_values[0], psnr_values[1]], color='#8B0000', linestyle='-', lw=2)

# 设置标题和轴标签
plt.xlabel("Compression Time (seconds)", fontsize=18)  # 设置 x 轴标签字体大小
plt.ylabel("PSNR (dB)", fontsize=18)  # 设置 y 轴标签字体大小

# 去掉坐标轴上的尖尖刻度线
plt.tick_params(axis='both', which='both', length=2, labelsize=18)  # 设置刻度线数字的字体大小为12

# 禁用背景网格线（去掉灰色背景线）
plt.grid(True)

# 调整 x 轴比例尺：选择 log 或 linear
plt.xscale('log')  # 或者 plt.xscale('linear')，如果你想要线性比例尺

# 设置 x 轴的范围（如果需要）
plt.xlim(500, 25000)  # 设置 x 轴的范围

# 调整布局，去掉多余的空白
plt.tight_layout()

# 添加图例，显示方法名称、内存占用和压缩时间
handles = []
for i in range(9):
    # 合并内存和时间信息显示在图例中
    label = f'{methods[i]} ({compression_times_min[i]:.1f}, {mem_values[i]:.1f})'
    handles.append(plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=14, label=label))

# 添加图例，显示方法名称、内存占用和压缩时间
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14, title='Time(minutes) & Mem(MB)', title_fontsize=14)

# 保存图表到文件
plt.savefig("time_vs_psnr_vs_mem7.png", dpi=300, bbox_inches='tight')

# 显示图表
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据：压缩时间和 PSNR 值
# compression_times = [706.9, 781.0, 935.0, 12701.1, 22543.2, 4563.9, 3738.8, 2199.5, 2639.3]  # 压缩时间（单位：秒）
# psnr_values = [27.25, 27.48, 26.96, 25.67, 27.34, 26.25, 25.81, 25.42, 27.73]  # PSNR值

# # 每种方法的内存占用 (MB)
# mem_values = [35.2, 47.1, 54.2, 36.0, 64.3, 81.5, 78.1, 185.0, 124.7]  # 示例数据

# # 9 种方法，每种方法对应一个不同的形状和颜色
# methods = ['Ours', 'Ours-large', 'C3DGS', 'RDO', 'MesonGS', 
#            'LightGaussian', 'Compact3DGS', 'SOG', 'EAGLES']
# markers = ['o', 'o', '^', '^', '^', '^', '^', '^', '^']  # 第一和第二个方法使用相同的形状
# colors = ['#8B0000', '#8B0000', '#4682B4', '#8B4513', '#6A5ACD', '#D2691E', '#708090', 'green', 'orange']

# # 阶段名称和阶段时间
# stage = ["Pruning", "Clustering", "QA Fine-tuning", "Entropy Coding"]
# stage_times = [13.40, 158.50, 531.48, 3.53]  # 阶段时间（单位：秒）
# stage_percentages = ["1.9%", "22.4%", "75.2%", "0.5%"]  # 阶段时间的百分比

# # 创建散点图
# for i in range(9):
#     # 绘制数据点
#     plt.scatter(compression_times[i], psnr_values[i], 
#                 marker=markers[i],  # 对应的形状
#                 color=colors[i],    # 对应的颜色
#                 s=150)  # 点的大小为 150

#     # 直接在点旁边添加文字标签，位置可以根据需要调整（偏移量为 +0.1）
#     if i != 4 and i != 8 and i != 1 and i != 0 and i != 6:
#         plt.text(compression_times[i] + 0.1, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 0:
#         plt.text(compression_times[i] - 0.6, psnr_values[i] + 0.35, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 1: 
#         continue
#     elif i == 8:
#         plt.text(compression_times[i] - 600.0, psnr_values[i] - 0.2, methods[i], 
#                  color=colors[i], fontsize=16)
#     elif i == 4:
#         plt.text(compression_times[i] - 12000.0, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)
#     else:
#         plt.text(compression_times[i] - 2500.0, psnr_values[i] + 0.1, methods[i], 
#                  color=colors[i], fontsize=16)

# # 连接第一种和第二种方法的点
# plt.plot([compression_times[0], compression_times[1]], 
#          [psnr_values[0], psnr_values[1]], color='#8B0000', linestyle='-', lw=2)

# # 设置标题和轴标签
# plt.xlabel("Compression Time (seconds)", fontsize=16)  # 设置 x 轴标签字体大小
# plt.ylabel("PSNR (dB)", fontsize=16)  # 设置 y 轴标签字体大小

# # 去掉坐标轴上的尖尖刻度线
# plt.tick_params(axis='both', which='both', length=2, labelsize=16)  # 设置刻度线数字的字体大小为12

# # 禁用背景网格线（去掉灰色背景线）
# plt.grid(True)

# # 调整 x 轴比例尺：选择 log 或 linear
# plt.xscale('log')  # 或者 plt.xscale('linear')，如果你想要线性比例尺

# # 设置 x 轴的范围（如果需要）
# plt.xlim(500, 25000)  # 设置 x 轴的范围

# # 调整布局，去掉多余的空白
# plt.tight_layout()

# # 创建内存占用图例
# mem_handles = []
# for i in range(9):
#     # 创建每个图例项，将方法名称和内存占用数值一起显示，并且内存占用数值右对齐
#     label = f'{methods[i]} {str(mem_values[i]).rjust(5)}'  # 使用rjust(5)来右对齐内存数值
#     mem_handles.append(plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=14, 
#                                   label=label))

# # 创建阶段时间占比的图例 (去掉点)
# stage_labels = [f'{stage[i]}: {stage_times[i]}s ({stage_percentages[i]})' for i in range(len(stage))]
# stage_handles = [plt.Line2D([0], [0], color='w', lw=0, label=stage_labels[i]) for i in range(len(stage))]

# # 获取当前图形
# ax = plt.gca()

# # 创建内存占用图例并添加到图形中
# mem_legend = ax.legend(handles=mem_handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=13, title='Mem(MB)', title_fontsize=13)
# ax.add_artist(mem_legend)

# # 创建阶段时间占比的图例并添加到图形中
# stage_legend = ax.legend(handles=stage_handles, loc='upper left', bbox_to_anchor=(1.05, 0.25), fontsize=13, title='Stage Time(seconds)', title_fontsize=13, handlelength=0)

# # 保存图表到文件
# plt.savefig("time_vs_psnr_vs_mem3.png", dpi=300, bbox_inches='tight')

# # 显示图表
# plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # 示例数据：压缩时间和 PSNR 值
# compression_times = [706.9, 781.0, 935.0, 12701.1, 22543.2, 4563.9, 3738.8, 2199.5, 2639.3]  # 压缩时间（单位：秒）
# psnr_values = [27.25, 27.48, 26.96, 25.67, 27.34, 26.25, 25.81, 25.42, 27.73]  # PSNR值

# # 每种方法的内存占用 (MB)
# mem_values = [35.2, 47.1, 54.2, 36.0, 64.3, 81.5, 78.1, 185.0, 124.7]  # 示例数据

# # 9 种方法，每种方法对应一个不同的形状和颜色
# methods = ['Ours', 'Ours-large', 'C3DGS', 'RDO', 'MesonGS', 
#            'LightGaussian', 'Compact3DGS', 'SOG', 'EAGLES']
# markers = ['o', 'o', '^', '^', '^', '^', '^', '^', '^']  # 第一和第二个方法使用相同的形状
# colors = ['#8B0000', '#8B0000', '#4682B4', '#8B4513', '#6A5ACD', '#D2691E', '#708090', 'green', 'orange']

# # 创建3D图
# fig = plt.figure(figsize=(12, 8))  # 增大图形的宽度和高度
# ax = fig.add_subplot(111, projection='3d')

# # 绘制每个点，调整点的大小和透明度
# for i in range(9):
#     ax.scatter(compression_times[i], mem_values[i], psnr_values[i],
#                marker=markers[i],  # 对应的形状
#                color=colors[i],    # 对应的颜色
#                s=80,  # 调整点的大小为80（减少点的密集感）
#                label=f'{methods[i]}')  # label用于图例

# # 连接第一个和第二个点
# ax.plot([compression_times[0], compression_times[1]], 
#         [mem_values[0], mem_values[1]], 
#         [psnr_values[0], psnr_values[1]], 
#         color='#8B0000', lw=2)  # 使用红色线段连接

# # 设置标题和轴标签
# ax.set_xlabel('Compression Time (seconds)', fontsize=12)
# ax.set_ylabel('Memory (MB)', fontsize=12)
# ax.set_zlabel('PSNR (dB)', fontsize=12, labelpad=20)  # 使用labelpad调整PSNR标签与坐标轴的距离

# # 设置刻度字体大小
# ax.tick_params(axis='both', which='both', labelsize=10)

# # 设置坐标轴范围，避免数据点过于集中
# ax.set_xlim(500, 25000)  # 压缩时间（X轴）范围
# ax.set_ylim(30, 200)     # 内存占用（Z轴）范围
# ax.set_zlim(25, 28)      # PSNR（Y轴）范围

# # 手动设置Y轴刻度，每40一个刻度
# yticks = np.arange(30, 201, 40)  # 从30到200，每隔40一个刻度
# ax.set_yticks(yticks)

# # 调整视角方向，使用俯仰角和方位角
# ax.view_init(elev=20, azim=45)  # elev: 俯仰角, azim: 方位角

# # 调整布局
# plt.tight_layout(pad=3.0)  # 增加边距，防止标签被裁剪

# # 手动设置边距，确保左边和右边标签都不会被裁剪
# plt.subplots_adjust(left=0.12, right=0.85)  # 增大右边距以显示PSNR标签

# # 添加图例，并将图例放置在图形的右侧
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)  # 图例右移

# # 保存图表到文件，不显示
# plt.savefig("time_vs_psnr_vs_mem_3d_with_adjusted_layout_zlabel_space.png", dpi=300, bbox_inches='tight')

# # 关闭显示图表
# plt.close()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # 示例数据：压缩时间和 PSNR 值
# compression_times = [706.9, 781.0, 935.0, 12701.1, 22543.2, 4563.9, 3738.8, 2199.5, 2639.3]  # 压缩时间（单位：秒）
# psnr_values = [27.25, 27.48, 26.96, 25.67, 27.34, 26.25, 25.81, 25.42, 27.73]  # PSNR值

# # 每种方法的内存占用 (MB)
# mem_values = [35.2, 47.1, 54.2, 36.0, 64.3, 81.5, 78.1, 185.0, 124.7]  # 示例数据

# # 9 种方法，每种方法对应一个不同的形状和颜色
# methods = ['Ours', 'Ours-large', 'C3DGS', 'RDO', 'MesonGS', 
#            'LightGaussian', 'Compact3DGS', 'SOG', 'EAGLES']
# markers = ['o', 'o', '^', '^', '^', '^', '^', '^', '^']  # 第一和第二个方法使用相同的形状
# colors = ['#8B0000', '#8B0000', '#4682B4', '#8B4513', '#6A5ACD', '#D2691E', '#708090', 'green', 'orange']

# # 创建3D图
# fig = plt.figure(figsize=(12, 8))  # 增大图形的宽度和高度
# ax = fig.add_subplot(111, projection='3d')

# # 绘制每个点，调整点的大小和透明度
# for i in range(9):
#     ax.scatter(compression_times[i], mem_values[i], psnr_values[i],
#                marker=markers[i],  # 对应的形状
#                color=colors[i],    # 对应的颜色
#                s=80,  # 调整点的大小为80（减少点的密集感）
#                label=f'{methods[i]}')  # label用于图例

# # 连接第一个和第二个点
# ax.plot([compression_times[0], compression_times[1]], 
#         [mem_values[0], mem_values[1]], 
#         [psnr_values[0], psnr_values[1]], 
#         color='#8B0000', lw=2)  # 使用红色线段连接

# # 设置标题和轴标签
# ax.set_xlabel('Compression Time (seconds)', fontsize=12)
# ax.set_ylabel('Memory (MB)', fontsize=12)
# ax.set_zlabel('PSNR (dB)', fontsize=12)

# # 设置刻度字体大小
# ax.tick_params(axis='both', which='both', labelsize=10)

# # 设置坐标轴范围，避免数据点过于集中
# ax.set_xlim(500, 25000)  # 压缩时间（X轴）范围
# ax.set_ylim(30, 200)     # 内存占用（Z轴）范围
# ax.set_zlim(25, 28)      # PSNR（Y轴）范围

# # 手动设置Y轴刻度，每40一个刻度
# yticks = np.arange(30, 201, 40)  # 从30到200，每隔40一个刻度
# ax.set_yticks(yticks)

# # 调整视角方向，使用俯仰角和方位角
# ax.view_init(elev=20, azim=45)  # elev: 俯仰角, azim: 方位角

# # 调整布局
# plt.tight_layout(pad=3.0)  # 增加边距，防止标签被裁剪

# # 手动设置左边距，确保左边标签不会被裁剪
# plt.subplots_adjust(left=0.12)  # 增大左边距

# # 添加图例，并将图例放置在图形的右侧
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)  # 图例右移

# # 保存图表到文件，不显示
# plt.savefig("time_vs_psnr_vs_mem_3d_with_adjusted_layout_left_margin.png", dpi=300, bbox_inches='tight')

# # 关闭显示图表
# plt.close()