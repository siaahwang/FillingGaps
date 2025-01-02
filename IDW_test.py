import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 读取站点的地理坐标
stations_df = pd.read_excel(r'E:\SM\interpolation\dataset\SMN-SDR_ground-data(03cm)_2018-2020\lat-lon coordinates of stations in SMN-SDR.xlsx',
                            sheet_name='Sheet1')
stations_coords = stations_df.set_index('Station')[['Latitude', 'Longitude']]

# 站点列表和深度
stations1 = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
stations2 = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
             'M9', 'M10', 'M11', 'M12', 'L3', 'L5', 'L7', 'L13']
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# 读取 stations2 的数据
stations2_data_frames = {}
for station in stations2:
    file_path = f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv'
    df = pd.read_csv(file_path, parse_dates=['Measure_Times'], index_col='Measure_Times', na_values=9999)
    df.columns = ['Soil_VWC_03']
    stations2_data_frames[station] = df

# 读取 stations1 的数据
stations1_data_frames = {}
for depth in depths:
    depth_str = f'Soil_VWC_03_{depth}'
    for station in stations1:
        file_path = f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test1\\{station}_SM.csv'
        df = pd.read_csv(file_path, parse_dates=['Measure_Times'], index_col='Measure_Times')
        stations1_data_frames[(station, depth_str)] = df

# 合并所有站点的数据
all_time_indices = [df.index for df in stations2_data_frames.values()] + [df.index for df in stations1_data_frames.values()]
all_times = pd.Index(sorted(set(time for sublist in all_time_indices for time in sublist)))
merged_df = pd.DataFrame(index=all_times)

# 将 stations2 数据加入到合并的数据框中
for station, df in stations2_data_frames.items():
    df = df.reindex(all_times)
    merged_df[(station, 'Soil_VWC_03')] = df['Soil_VWC_03']

# 将 stations1 数据加入到合并的数据框中
for (station, depth_str), df in stations1_data_frames.items():
    df = df.reindex(all_times)
    merged_df[(station, depth_str)] = df[depth_str]

merged_df.columns = pd.MultiIndex.from_tuples(merged_df.columns)

def idw_interpolation(x, y, values, xi, yi, power=2):
    distances = np.sqrt((x - xi)**2 + (y - yi)**2)
    weights = 1 / distances**power
    weights[distances == 0] = 0  # 防止距离为零导致分母为零
    interpolated_value = np.sum(weights * values) / np.sum(weights)
    return interpolated_value

metrics = []

# 循环处理每个深度的数据
for station in stations1:
    for depth in depths:
        depth_str = f'Soil_VWC_03_{depth}'
        print(f'{station}_{depth}')

        # 读取当前站点的数据
        station_df = pd.read_csv(
            f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test1\\{station}_SM.csv',
            parse_dates=['Measure_Times'], index_col='Measure_Times', na_values=np.nan)

        # 找到当前深度的缺失值索引
        missing_indices = station_df[depth_str].isna()
        missing_times = station_df.index[missing_indices]

        y_true = []
        y_pred = []

        # 计算插补值并更新数据
        for time in missing_times:
            available_stations = [(s[0], s[1]) for s in merged_df.columns if s[0] != station and not pd.isna(merged_df.loc[time, s])]
            if len(available_stations) > 0:
                neighbor_coords = stations_coords.loc[[s[0] for s in available_stations]]
                neighbor_values = merged_df.loc[time, available_stations].values
                target_coords = stations_coords.loc[station]
                interpolated_value = idw_interpolation(
                    neighbor_coords['Latitude'].values,
                    neighbor_coords['Longitude'].values,
                    neighbor_values,
                    target_coords['Latitude'],
                    target_coords['Longitude'])

                # 更新插补值到数据框
                station_df.loc[time, depth_str] = interpolated_value

                # 收集真实值和插补值
                true_value = station_df.loc[time, 'Soil_VWC_03']  # 真实值列是 'Soil_VWC_03'
                if not np.isnan(true_value):
                    y_true.append(true_value)
                    y_pred.append(interpolated_value)

                # 保存插值后的数据
        station_df.to_csv(f'result/test1/IDW/{station}_SM_interpolated_{depth}.csv')

        # 计算指标
        if len(y_true) > 0 and len(y_pred) > 0:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

            print(f"Station {station}_{depth} MAE:", mae)
            print(f"Station {station}_{depth} RMSE:", rmse)
            print(f"Station {station}_{depth} R2:", r2)
            print(f"Station {station}_{depth} MAPE:", mape)

            metrics.append({'Station': station, 'Depth': depth, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape})

# 保存计算指标结果
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('result/test1/IDW/IDW_interpolation_metrics.csv', index=False)

