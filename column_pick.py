import pandas as pd

# 定义站点列表
station_list = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
         'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12',
         'L2', 'L3', 'L4', 'L5', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14']

# 初始化一个空的 DataFrame 来存储合并的数据
combined_df = pd.DataFrame()

# 遍历每个站点
for i in [1]:
    for station in station_list:
    # 构造文件路径
        file_path = f'F:\\SM\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_TS.csv'
        df = pd.read_csv(file_path)

        column_name = df.columns[i]  # 获取第二列的列名
        combined_df[station] = df[column_name]  # 将列名改为站点名并添加到 combined_df 中

    # 保存合并后的 DataFrame 到新的 CSV 文件
    output_file_path = r'F:\\SM\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\total_TS.xlsx'
    combined_df.to_excel(output_file_path, index=False)

for station in station_list:
    combined_df = pd.DataFrame()
    file_path_1 = f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\1D_\\{station}_SM.csv'
    file_path_2 = f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\1D_\\{station}_PP.csv'
    file_path_3 = f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\1D_\\{station}_TS.csv'
    file_path_4 = f'F:\\SM\interpolation\\code\\result\\sm_diff\\{station}_diff.xlsx'
    df_SM = pd.read_csv(file_path_1)
    df_PP = pd.read_csv(file_path_2)
    df_TS = pd.read_csv(file_path_3)
    df_diff = pd.read_excel(file_path_4)
    combined_df['Measure_times'] = df_SM['Measure_Times']
    combined_df['SM'] = df_SM['Soil_VWC_03']
    combined_df['DIFF'] = df_diff['sm_diff']
    combined_df['TS'] = -df_TS['Soil_TEM_03']
    combined_df['PP'] = -df_PP['Prep']
    output_file_path = f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\1D_\\station_all\\{station}.xlsx'
    combined_df.to_excel(output_file_path, index=False)

