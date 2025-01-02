import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 选择黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 假设你的数据框 df 已经加载并设置了 Measure_Times 为索引
df = pd.read_excel(r'F:\SM\interpolation\dataset\SMN-SDR_ground-data(03cm)_2018-2020\30min_2019-2020\station_all\L1.xlsx')
df['Measure_times'] = pd.to_datetime(df['Measure_times'])
df.set_index('Measure_times', inplace=True)

# 创建图形和坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制 SM、TS 和 DIFF 的线图
ax1.plot(df.index, df['SM'], label='SM', color='b')
ax1.plot(df.index, df['TS'], label='TS', color='g')
ax1.plot(df.index, df['DIFF'], label='DIFF', color='r')

# 设置 x 轴标签和 y 轴标签
ax1.set_xlabel('时间')
ax1.set_ylabel('SM 和 TS DIFF 值')
ax1.legend(loc='upper left')

# 创建第二个坐标轴用于倒立柱状图
ax2 = ax1.twinx()
ax2.bar(df.index, -df['PP'], width=0.01, color='orange', alpha=0.5, label='PP (倒立柱状图)')

# 设置时间格式化
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 显示年月
ax1.xaxis.set_major_locator(mdates.MonthLocator())  # 每个月为一个刻度

# 自动旋转日期标签
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 添加图例和标签
ax1.set_xlabel('时间')
ax1.set_ylabel('SM, TS, DIFF 值')
ax2.set_ylabel('PP 值 (倒立)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图形
plt.title('时序图: SM, TS DIFF 和倒立柱状图 PP')
plt.tight_layout()  # 自动调整布局
plt.show()