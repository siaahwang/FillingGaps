import pandas as pd
import numpy as np
sites = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
         'M1', 'M2', 'M3', 'M4', 'M5', 'M6','M7','M8', 'M9', 'M10', 'M11', 'M12',
         'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14']

for station in sites:
    sm_data = pd.read_csv(f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv')
    pp_data = pd.read_csv(f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_PP.csv')

    sm_data['Measure_Times'] = pd.to_datetime(sm_data['Measure_Times'])
    pp_data['Measure_Times'] = pd.to_datetime(pp_data['Measure_Times'])
    # 生成日期列表
    date_range = pd.date_range(start='2019-01-01', end='2020-12-30', freq='D')
    # 转换为列表
    date_list = date_range.tolist()

    result = []
    for date in date_list:
        start_time = date + pd.Timedelta(hours=6)
        end_time = date + pd.Timedelta(hours=30)
        pp_total = pp_data[(pp_data['Measure_Times'] >= start_time) & (pp_data['Measure_Times'] <= end_time)]['Prep'].sum()
        sm_end = (sm_data.loc[sm_data['Measure_Times'] == end_time, 'Soil_VWC_03']).values[0]
        sm_start = (sm_data.loc[sm_data['Measure_Times'] == start_time, 'Soil_VWC_03']).values[0]
        if pp_total ==0 and sm_end != 9999 and sm_start != 9999 :
            sm_diff = sm_end - sm_start
        else:
            sm_diff = None
        result.append([date,sm_diff])
    result_df = pd.DataFrame(result, columns=['date','sm_diff'])
    result_df.to_excel(f'result/sm_diff/{station}_diff.xlsx', index=False)
