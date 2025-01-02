import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Haversine 公式计算两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 地球半径，单位为公里
    return c * r

file_path = r'F:\SM\interpolation\dataset\SMN-SDR_ground-data(03cm)_2018-2020\lat-lon coordinates of stations in SMN-SDR.xlsx'
df = pd.read_excel(file_path)
stations = df['Station']
longitudes = df['Longitude']
latitudes = df['Latitude']

n = len(stations)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix[i, j] = haversine(longitudes[i], latitudes[i], longitudes[j], latitudes[j])

# 应用高斯模型计算空间相关性
L = 50  # 相关性的尺度参数，可以根据实际情况调整
correlation_matrix = np.exp(- (distance_matrix / L)**2)

# 将相关性矩阵转换为DataFrame
correlation_df = pd.DataFrame(correlation_matrix, index=stations, columns=stations)
correlation_df.to_csv(r'result\correlation_matrix.csv')

# 设置字体以避免缺失警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制相关性矩阵图
plt.figure(figsize=(17,15))  # 调整图片大小
heatmap = sns.heatmap(
    correlation_df,
    annot=True,
    cmap='viridis',  # 选择其他颜色地图，如 'viridis''coolwarm', 'plasma', 'inferno'
    fmt=".1f", #显示2位小数
    annot_kws={"size": 16},#注释文本大小
    cbar_kws={"shrink": 1.0, "label": "correlation"}  # 给 colorbar 添加标题
)
# 设置 colorbar 标题字体大小
colorbar = heatmap.collections[0].colorbar
colorbar.ax.yaxis.label.set_size(20)
# 设置 colorbar 刻度字体大小
colorbar.ax.tick_params(labelsize=20)
plt.title('correlation of sites', fontsize=20)  # 设置标题字体大小

# 设置横纵坐标标签的字体大小
plt.xticks(fontsize=16)  # 设置横坐标标签字体大小
plt.yticks(fontsize=16)  # 设置纵坐标标签字体大小
# 设置横纵坐标标题及其字体大小
plt.xlabel('Sites', fontsize=18)  # 设置横坐标标题及字体大小
plt.ylabel('Sites', fontsize=18)  # 设置纵坐标标题及字体大小

plt.tight_layout(pad=1.0)  # 调整图片两边的间距
plt.show()
