import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(10, 90, 10)
# x = np.arange(20, 26, 1)

algo = 'ours'
# algo = 'finetune'
task = '1'

y_task1_ours_new = [24.00, 23.94, 26.54, 25.85, 26.78, 26.45, 26.69, 26.96]
y_task1_ours_old = [31.02, 31.62, 32.51, 32.53, 32.58, 32.77, 32.54, 32.87]
# print(x.shape)

# plt.plot(x, y)
# plt.grid()
# plt.xlabel("offset")
# plt.ylabel("sync score")
# plt.savefig('test.jpg')

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

if algo == 'ours':
    ax.plot(x, y_task1_ours_new, **style_dict['ts4'])
    ax.plot(x, y_task1_ours_old, **style_dict['ts8'])
    # ax.plot(x, y_task1_ours_ts4, **style_dict['ts4'])
    # ax.plot(x, y_task1_ours_ts8, **style_dict['ts8'])

# plt.xlim(fontsize = 10) 
plt.ylim(10, 35) 

# plt.xticks(fontsize = 30) 
# plt.yticks(fontsize = 30) 
plt.xticks(fontsize = 20) 
plt.yticks(fontsize = 20) 
# ax.set_ylabel('PSNR', fontproperties='Times New Roman', fontsize = 30) # Y label
# ax.set_xlabel('Iterations', fontproperties='Times New Roman', fontsize = 30) # X label
ax.set_ylabel('PSNR', fontsize = 20) # Y label
ax.set_xlabel('Number of training images', fontsize = 20) # X label

# 优雅地局部美化格式
# fig.legend(('MC','DT','TT','WT'),frameon=False, loc='upper center',ncol=4,handlelength=4) # 图例
# fig.legend(('Ours','Finetune'),frameon=False, loc='upper center',ncol=2,handlelength=4) # 图例
# fig.legend(('ts1','ts2','ts4','ts8'),frameon=False, loc='upper center',ncol=4,handlelength=4) # 图例
fig.legend(('new task', 'old task'),frameon=False, loc='upper center',ncol=2,handlelength=4, fontsize = 20) # 图例

# ax.fill_between(df.index, df['MC_up'], df['MC_down'], alpha=0.15, linewidth=0, color='#fdae61') # 阴影
ax.grid(linestyle="--", alpha=0.2) # 网格线

# plt.savefig('task'+task+'_algo'+algo+'_delete.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('test.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)