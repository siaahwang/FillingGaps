import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 站点列表
stations = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
            'M9', 'M10', 'M11', 'M12', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9',
            'L10', 'L11', 'L12', 'L13', 'L14']
#station_list = ['L3', 'L5', 'L7', 'L13', 'M3', 'M4', 'M7', 'M8']

# 读取站点的地理坐标
stations_df = pd.read_excel(r'E:\SM\interpolation\dataset\SMN-SDR_ground-data(03cm)_2018-2020\lat-lon coordinates of stations in SMN-SDR.xlsx',
                            sheet_name='Sheet1')
stations_coords = stations_df.set_index('Station')[['Latitude', 'Longitude']]

# 读取所有站点的数据
data_frames = {}
for station in stations:
    df = pd.read_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv',
                     parse_dates=['Measure_Times'], index_col='Measure_Times', na_values=9999)
    df.columns = ['Soil_VWC_03']
    data_frames[station] = df

all_time_indices = [df.index for df in data_frames.values()]
all_times = pd.Index(sorted(set(time for sublist in all_time_indices for time in sublist)))
merged_df = pd.DataFrame(index=all_times)

# 将每个站点的数据加入到合并的数据框中
for station, df in data_frames.items():
    df = df.reindex(all_times)
    merged_df[station] = df['Soil_VWC_03']

def idw_interpolation(x, y, values, xi, yi, power=2):
    distances = np.sqrt((x - xi)**2 + (y - yi)**2)
    weights = 1 / distances**power
    weights[distances == 0] = 0  # 防止距离为零导致分母为零
    interpolated_value = np.sum(weights * values) / np.sum(weights)
    return interpolated_value

# 分割数据集，20%作为测试集并计算指标
metrics = []
for station in stations:
    station_data = merged_df[[station]].dropna().reset_index()
    train_data, test_data = train_test_split(station_data, test_size=0.6, random_state=42)
    train_data.columns = ['Measure_Times', 'Soil_VWC_03']
    test_data.columns = ['Measure_Times', 'Soil_VWC_03']

    y_true = test_data['Soil_VWC_03'].values
    y_pred = []

    for index, row in test_data.iterrows():
        time = row['Measure_Times']
        available_stations = [s for s in merged_df.columns if s != station and not pd.isna(merged_df.loc[time, s])]
        if len(available_stations) > 0:
            neighbor_coords = stations_coords.loc[available_stations]
            neighbor_values = merged_df.loc[time, available_stations].values
            target_coords = stations_coords.loc[station]
            interpolated_value = idw_interpolation(
                neighbor_coords['Latitude'].values,
                neighbor_coords['Longitude'].values,
                neighbor_values,
                target_coords['Latitude'],
                target_coords['Longitude'])
            y_pred.append(interpolated_value)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics.append({'Station': station, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('result/IDW/interpolation_metrics.csv', index=False)

for metric in metrics:
    print(f'Station {metric["Station"]} - MAE: {metric["MAE"]:.4f}, RMSE: {metric["RMSE"]:.4f}, R2: {metric["R2"]:.4f}')

# 循环处理每个站点的数据
for station in stations:
    station_df = data_frames[station]
    for time in station_df.index:
        if pd.isna(station_df.loc[time, 'Soil_VWC_03']):
            available_stations = [s for s in merged_df.columns if s != station and not pd.isna(merged_df.loc[time, s])]
            if len(available_stations) > 0:
                neighbor_coords = stations_coords.loc[available_stations]
                neighbor_values = merged_df.loc[time, available_stations].values
                target_coords = stations_coords.loc[station]
                interpolated_value = idw_interpolation(neighbor_coords['Latitude'].values,
                                                       neighbor_coords['Longitude'].values,
                                                       neighbor_values,
                                                       target_coords['Latitude'],
                                                       target_coords['Longitude'])
                station_df.loc[time, 'Soil_VWC_03'] = interpolated_value
    data_frames[station] = station_df

# 合并所有站点的数据
for station, df in data_frames.items():
    merged_df[station] = df['Soil_VWC_03']
# 保存插值后的数据
for station in merged_df.columns:
    merged_df[[station]].dropna().to_csv(f'result/IDW/{station}_SM_IDW.csv')
