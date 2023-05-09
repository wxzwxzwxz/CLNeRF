import numpy as np
import math
import matplotlib.pyplot as plt

# x = np.arange(1, 9, 1)
# x = np.arange(20, 26, 1)
# x = [560, 280, 186, 140, 112, 93, 80, 70]
x = [100, 50, 33, 25, 20, 16, 14, 12]
# x = [100, 90, 80, 70, 60, 50, 40, 30]
# x = [100, 90, 70, 50, 30, 10]

y_global = [32.207, 32.904, 32.424, 32.091, 31.775, 29.630, 30.634, 29.157]
y_local_bookshelf = [31.859, 32.722, 32.704, 32.135, 32.374, 31.228, 31.670, 30.486]
y_local_sofa1 = [32.576, 32.961, 32.315, 31.983, 31.606, 30.297, 30.448, 29.425]
y_local_sofa2 = [33.934, 34.791, 34.493, 34.445, 33.654, 33.355, 32.633, 32.738]
y_local_sofa3 = [31.855, 32.483, 32.500, 31.691, 31.879, 30.913, 30.964, 30.176]
y_local_table = [30.585, 30.612, 30.678, 29.767, 29.303, 28.395, 27.601, 28.077]
y_local = [32.162, 32.714, 32.538, 32.004, 31.763, 30.838, 30.663, 30.180]

# 优雅地创建Figure和Axes
fig, ax = plt.subplots()

# 优雅地添加基础类对象
style_dict = {
    'ours_task1':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#fdae61'),
    'ours_task2':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#d7191c'),
    'finetune_task1':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#abdda4'),
    'finetune_task2':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#2b83ba'),
    'ts1':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#fdae61'),
    'ts2':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#d7191c'),
    'ts4':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#abdda4'),
    'ts8':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#2b83ba')
}
# ax.plot(df.index, df['MC_Price'], **style_dict['MC_Price'])
# ax.plot(df.index, df['DT_Price'], **style_dict['DT_Price'])
# ax.plot(df.index, df['TT_Price'], **style_dict['TT_Price'])
# ax.plot(df.index, df['WT_Price'], **style_dict['WT_Price'])
# ax.plot(x, y_obama_gt, **style_dict['obama_gt'])


ax.plot(x, y_global, **style_dict['ts2'])
ax.plot(x, y_local, **style_dict['ts8'])
# ax.plot(x, y_local_sofa1, **style_dict['ts4'])
# ax.plot(x, y_local_sofa2, **style_dict['ts8'])

# plt.xticks(fontsize = 30) 
# plt.yticks(fontsize = 30) 
plt.xticks(fontsize = 10) 
plt.yticks(fontsize = 10) 
# ax.set_ylabel('PSNR', fontproperties='Times New Roman', fontsize = 30) # Y label
# ax.set_xlabel('Iterations', fontproperties='Times New Roman', fontsize = 30) # X label
ax.set_ylabel('PSNR', fontsize = 10) # Y label
ax.set_xlabel('Train images (%)', fontsize = 10) # X label

# 优雅地局部美化格式
fig.legend(('global view','local view'),frameon=False, loc='upper center', ncol=2, handlelength=4) # 图例

# ax.fill_between(df.index, df['MC_up'], df['MC_down'], alpha=0.15, linewidth=0, color='#fdae61') # 阴影
ax.grid(linestyle="--", alpha=0.2) # 网格线

plt.savefig('vis_split_psnr.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)