import numpy as np
import math
import matplotlib.pyplot as plt

# x = np.arange(20, 26, 1)
x = np.arange(20, 40, 1)

task = 0

# Ours_add
y_task1_ours = [32.16202973, 31.98420949, 31.95580564, 31.88790592, 31.91339428, 32.00292784, 32.0413041, 31.98103804, 31.98405479, 32.0480213, 32.12686882, 32.05513911, 32.05953765, 32.15624991, 32.20096357, 32.13314461, 32.18109744, 32.24653354, 32.1876062, 32.20767549]
y_task2_ours = [26.99257707, 27.67303686, 27.97509091, 28.20223545, 28.42904744, 28.51303774, 28.58895715, 28.54548718, 28.5989298, 28.71013351, 28.70288751, 28.77829834, 28.8591554, 28.9029517, 28.84875218, 28.89601046, 28.99672431, 29.03009219, 28.98741571, 29.01328667]
# Ours_delete
# y_task1_ours = [31.10594383, 31.08608215, 31.08326109, 31.30377506, 31.28495348, 30.38291115]
# y_task2_ours = [32.14812871, 33.27734269, 33.85697301, 34.1768014, 34.53068625, 34.7605785, 34.91179841, 35.02422458, 35.22331962, 35.37841414, 35.41161356, 35.54810865, 35.51408768, 35.63484853, 35.70993766, 35.84327111, 35.89898126, 35.99809612, 36.00799314, 36.05912873]

# Finetune_add
y_task1_finetune = [27.42945853, 25.93193517, 25.20944341, 24.23806861, 23.81647596, 23.07500258, 22.89658582, 22.68049649, 22.35670791, 22.0399847, 21.83492911, 21.73827991, 21.20723274, 21.19288037, 21.23879115, 21.01406139, 20.8105164, 20.63009219, 20.65917378, 20.6087228]
y_task2_finetune = [26.9440483, 27.60447632, 27.93142853, 27.84056084, 28.18937685, 28.1875795, 28.22825464, 28.32550322, 28.31429046, 28.38314092, 28.33143232, 28.38453548, 28.26723684, 28.39072464, 28.33639673, 28.33761665, 28.17908062, 28.28570872, 28.32442102, 28.34578609]
# Finetune_delete
# y_task1_finetune = [26.20753562, 24.98176461, 24.22772292, 21.69167026, 21.52148834, 21.42790694]
# y_task2_finetune = [32.1221889, 33.28999694, 33.98719807, 34.3554131, 34.5846548, 34.87378894, 35.07392504, 35.1747966, 35.21727325, 35.46484458, 35.57682994, 35.68107407, 35.68467414, 35.79439248, 35.86729329, 35.96099931, 35.97512891, 36.09897477, 36.10076845, 36.17525466]

# plt.plot(x, y)
# plt.grid()
# plt.xlabel("offset")
# plt.ylabel("sync score")
# plt.savefig('test.jpg')

# 优雅地创建Figure和Axes
fig, ax = plt.subplots()

# 优雅地添加基础类对象
style_dict = {
    'MC_Price':dict(linestyle=':', marker='o',markersize=6,color='#fdae61'),
    'WT_Price':dict(linestyle='-',marker='*',markersize=6,color='#d7191c'),
    'DT_Price':dict(linestyle='--',marker='s',markersize=6,color='#abdda4'),
    'TT_Price':dict(linestyle='-.',marker='v',markersize=6,color='#2b83ba'),
    'obama_gt':dict(linestyle=':', marker='o',markersize=6,color='#fdae61'),
    'ours_task1':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#fdae61'),
    'ours_task2':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#d7191c'),
    'finetune_task1':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#abdda4'),
    'finetune_task2':dict(linestyle='--', linewidth=3, marker='s',markersize=10,color='#2b83ba')
}
# ax.plot(df.index, df['MC_Price'], **style_dict['MC_Price'])
# ax.plot(df.index, df['DT_Price'], **style_dict['DT_Price'])
# ax.plot(df.index, df['TT_Price'], **style_dict['TT_Price'])
# ax.plot(df.index, df['WT_Price'], **style_dict['WT_Price'])
# ax.plot(x, y_obama_gt, **style_dict['obama_gt'])

if task == 0:
    ax.plot(x, y_task1_ours[:x.shape[0]], **style_dict['ours_task1'])
    ax.plot(x, y_task1_finetune[:x.shape[0]], **style_dict['finetune_task1'])
    ax.plot(x, y_task2_ours[:x.shape[0]], **style_dict['ours_task2'])
    ax.plot(x, y_task2_finetune[:x.shape[0]], **style_dict['finetune_task2'])
elif task == 1:
    ax.plot(x, y_task1_ours, **style_dict['ours_task1'])
    ax.plot(x, y_task1_finetune, **style_dict['finetune_task1'])
elif task == 2:
    ax.plot(x, y_task2_ours, **style_dict['ours_task2'])
    ax.plot(x, y_task2_finetune, **style_dict['finetune_task2'])

# plt.xticks(fontsize = 30) 
# plt.yticks(fontsize = 30) 
plt.xticks(fontsize = 10) 
plt.yticks(fontsize = 10) 
# ax.set_ylabel('PSNR', fontproperties='Times New Roman', fontsize = 30) # Y label
# ax.set_xlabel('Iterations', fontproperties='Times New Roman', fontsize = 30) # X label
ax.set_ylabel('PSNR', fontsize = 10) # Y label
ax.set_xlabel('Iterations (w)', fontsize = 10) # X label

# 优雅地局部美化格式
# fig.legend(('MC','DT','TT','WT'),frameon=False, loc='upper center',ncol=4,handlelength=4) # 图例
# fig.legend(('Ours','Finetune'),frameon=False, loc='upper center',ncol=2,handlelength=4) # 图例

# ax.fill_between(df.index, df['MC_up'], df['MC_down'], alpha=0.15, linewidth=0, color='#fdae61') # 阴影
ax.grid(linestyle="--", alpha=0.2) # 网格线
# plt.savefig('offset_score_obama.jpg')

if task == 0:
    fig.legend(('Ours_task1','Finetune_task1','Ours_task2','Finetune_task2'),frameon=False, loc='upper center',ncol=4,handlelength=4) # 图例
    plt.savefig('task12_add.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)
elif task == 1:
    fig.legend(('Ours','Finetune'),frameon=False, loc='upper center',ncol=2,handlelength=4) # 图例
    plt.savefig('task1_add.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)
elif task == 2:
    fig.legend(('Ours','Finetune'),frameon=False, loc='upper center',ncol=2,handlelength=4) # 图例
    plt.savefig('task2_add.jpg', format='jpg', bbox_inches='tight', dpi=300, transparent=True)