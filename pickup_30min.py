import pandas as pd
from datetime import timedelta

sites = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
         'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12',
         'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14']
combined_df = pd.DataFrame()

for i in sites:
    # 读取CSV文件
    df = pd.read_csv(f'F:\\SM\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\2019-2020\\{i}_SM_2019_2020.csv')
    if (df['Soil_VWC_03'] == 9999).any():
        # 将异常值改为0
        df.loc[df['Soil_VWC_03'] == 9999, 'Soil_VWC_03'] = 0

    # 确保Measure_Times列为datetime格式
    df['Measure_Times'] = pd.to_datetime(df['Measure_Times'])

    # 以30分钟的频率进行采样
    df_resampled = df.set_index('Measure_Times').resample('1D').mean().reset_index()

    if combined_df.empty:
        # 如果combined_df是空的，先添加Measure_Times列
        combined_df['Measure_Times'] = df_resampled['Measure_Times']

    combined_df[i] = df_resampled['Soil_VWC_03']

    # 保存结果到新的CSV文件
    df_resampled.to_csv(f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\1D_\\{i}_SM.csv',index=False)
# output_file_path = r'F:\SM\soil prediction\1d_data.csv'
# combined_df.to_csv(output_file_path, index=False)

