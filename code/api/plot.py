#本代码用于读取成像的CSV文件，然后画出一个三维图来
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 让用户输入csv绝对路径
csv_file = r'C:\Users\wuyida\Desktop\0728最后一搏\scanning.csv'
df = pd.read_csv(csv_file)

# 获取坐标和厚度
x = df['x'].values
y = df['y'].values
th1 = df['thickness1'].values
th2 = df['thickness2'].values
th3 = df['thickness3'].values

# 计算累积厚度
z1 = th1
z2 = th1 + th2
z3 = th1 + th2 + th3

# 生成网格
X = x.reshape((120, 40)).T  # shape: (40, 120)
Y = y.reshape((120, 40)).T
Z1 = z1.reshape((120, 40)).T
Z2 = z2.reshape((120, 40)).T
Z3 = z3.reshape((120, 40)).T

# 生成本层厚度的网格
TH1 = th1.reshape((120, 40)).T
TH2 = th2.reshape((120, 40)).T
TH3 = th3.reshape((120, 40)).T

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# 每层用本层厚度做颜色映射
# 统一颜色映射范围：20um(深蓝)到50um(深红)
norm_unified = plt.Normalize(20, 50)
colors1 = plt.cm.jet(norm_unified(TH1))
colors2 = plt.cm.jet(norm_unified(TH2))
colors3 = plt.cm.jet(norm_unified(TH3))

# 绘制三层表面，使用更平滑的渲染
surf1 = ax.plot_surface(X, Y, Z1, facecolors=colors1, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.7)
surf2 = ax.plot_surface(X, Y, Z2, facecolors=colors2, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.7)
surf3 = ax.plot_surface(X, Y, Z3, facecolors=colors3, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.7)

# 设置xyz比例，z轴视觉高度缩小5倍
ax.set_box_aspect([X.max()-X.min(), Y.max()-Y.min(), (Z3.max()-0)/2])

# 设置z轴从0开始
ax.set_zlim(0, Z3.max())

# 设置标签，字体放大一倍，只有X轴增加labelpad让标签离坐标轴更远
ax.set_xlabel('X (mm)', fontsize=16, fontfamily='Times New Roman', labelpad=20)
ax.set_ylabel('Y (mm)', fontsize=16, fontfamily='Times New Roman')
ax.set_zlabel('Cumulative Thickness (μm)', fontsize=16, fontfamily='Times New Roman')

# 设置x轴和y轴刻度间隔 - 您可以手动修改这些值
x_ticks = np.arange(1, 121, 20)  # 修改这里的步长来改变x轴刻度间隔
y_ticks = np.arange(1, 41, 20)   # 修改这里的步长来改变y轴刻度间隔

# 确保刻度在数据范围内
x_ticks = x_ticks[x_ticks <= 120]
y_ticks = y_ticks[y_ticks <= 40]

# 手动设置x轴和y轴刻度标签 - 您可以手动修改这些标签
x_labels = [str(x) for x in x_ticks]  # 修改这里来自定义x轴标签
y_labels = [str(y) for y in y_ticks]  # 修改这里来自定义y轴标签

# 设置刻度位置和标签
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# 强制关闭自动刻度
ax.xaxis.set_major_locator(plt.FixedLocator(x_ticks))
ax.yaxis.set_major_locator(plt.FixedLocator(y_ticks))

# 设置刻度标签字体大小，字体放大一倍
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z', labelsize=12)

# 固定x轴和y轴范围 - 您可以手动修改这些值
ax.set_xlim(1, 120)  # 修改这里的范围
ax.set_ylim(1, 40)   # 修改这里的范围

# 颜色条（以第三层厚度为准）- 扩大颜色范围
mappable = plt.cm.ScalarMappable(cmap='jet', norm=norm_unified)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, location='left', pad=0.01)
# 设置颜色条标签字体大小
cbar.ax.tick_params(labelsize=12)
# 扩大颜色条范围
cbar.set_ticks(np.linspace(20, 50, 6))

plt.tight_layout()
plt.show() 