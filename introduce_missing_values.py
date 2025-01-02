
import pandas as pd
import numpy as np


def introduce_missing_values(data, column, missing_rate=0.1):
    """
    Introduce random missing values into a specified column in a DataFrame.

    Parameters:
    - data: pd.DataFrame, the input DataFrame.
    - column: str, the column name where missing values should be introduced.
    - missing_rate: float, the proportion of missing values to introduce (default is 0.1).

    Returns:
    - pd.DataFrame, the DataFrame with missing values introduced in the specified column.
    """
    data_with_missing = data.copy()
    n_missing = int(len(data_with_missing) * missing_rate)
    missing_indices = np.random.choice(data_with_missing.index, n_missing, replace=False)
    data_with_missing.loc[missing_indices, column] = np.nan
    return data_with_missing


station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
for station in station_list:
    data = pd.read_csv(
        f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv")
    df = pd.DataFrame(data)
    for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        missing_data = introduce_missing_values(df, 'Soil_VWC_03', missing_rate=i)
        df[f'Soil_VWC_03_{i}'] = missing_data['Soil_VWC_03']

    df.to_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test1\\{station}_SM.csv',
              index=False)

print("完成")

import pandas as pd
import numpy as np


def introduce_continuous_missing_values(data, column, missing_rate=0.1, block_size_range=(100, 2000)):
    """
    Introduce continuous missing values into a specified column in a DataFrame.

    Parameters:
    - data: pd.DataFrame, the input DataFrame.
    - column: str, the column name where missing values should be introduced.
    - missing_rate: float, the proportion of missing values to introduce (default is 0.1).
    - block_size_range: tuple, the range of block sizes for missing values (default is (100, 2000)).

    Returns:
    - pd.DataFrame, the DataFrame with continuous missing values introduced in the specified column.
    """
    data_with_missing = data.copy()
    total_length = len(data_with_missing)
    n_missing = int(total_length * missing_rate)

    all_indices = np.arange(total_length)

    # Introduce missing values in blocks
    while n_missing > 0 and len(all_indices) > 0:
        block_size = np.random.randint(block_size_range[0], block_size_range[1] + 1)
        block_size = min(block_size, n_missing)  # Ensure block size does not exceed remaining missing values

        if len(all_indices) <= block_size:
            # Use remaining all indices if the length is less than block_size
            block_size = len(all_indices)

        start_idx = np.random.choice(all_indices[:-block_size + 1], 1)[0]
        end_idx = start_idx + block_size

        if end_idx > total_length:
            end_idx = total_length

        data_with_missing.loc[start_idx:end_idx, column] = np.nan

        n_missing -= (end_idx - start_idx + 1)
        all_indices = np.setdiff1d(all_indices, np.arange(start_idx, end_idx + 1))

    return data_with_missing


station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
for station in station_list:
    data = pd.read_csv(
        f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv")
    df = pd.DataFrame(data)
    for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        # Calculate the required missing rate based on total data length
        missing_rate = i
        missing_data = introduce_continuous_missing_values(df, 'Soil_VWC_03', missing_rate=missing_rate,
                                                           block_size_range=(100, 2000))
        df[f'Soil_VWC_03_{i}'] = missing_data['Soil_VWC_03']

    df.to_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test2\\{station}_SM.csv',
              index=False)

print("完成")



import pandas as pd
import numpy as np

def introduce_random_missing_values(data_list, column, missing_rate=0.1):
    """
    Introduce random missing values at the same time points into a specified column in a list of DataFrames.

    Parameters:
    - data_list: list of pd.DataFrame, the list of input DataFrames.
    - column: str, the column name where missing values should be introduced.
    - missing_rate: float, the proportion of missing values to introduce (default is 0.1).

    Returns:
    - list of pd.DataFrame, the list of DataFrames with random missing values introduced at the same time points.
    """
    total_length = len(data_list[0])
    n_missing = int(total_length * missing_rate)

    missing_indices = np.random.choice(total_length, n_missing, replace=False)

    for data in data_list:
        data.loc[missing_indices, column] = np.nan

    return data_list


station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
data_dict = {station: pd.read_csv(
    f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv") for
             station in station_list}

for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    data_list = [data_dict[station].copy() for station in station_list]
    data_list = introduce_random_missing_values(data_list, 'Soil_VWC_03', missing_rate=i)
    for j, station in enumerate(station_list):
        data_dict[station][f'Soil_VWC_03_{i}'] = data_list[j]['Soil_VWC_03']
        data_dict[station].to_csv(
            f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test3\\{station}_SM.csv',
            index=False)

print("完成")



import pandas as pd
import numpy as np
import random

# 读取 CSV 文件
file_path = 'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\1_data.csv'
data = pd.read_csv(file_path)

# 参数设置
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
n_rows = data.shape[0]
n_stations = len(station_list)
missing_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


# 引入缺失值的函数
def introduce_missing_values(df, missing_rate):
    df_copy = df.copy()
    n_missing = int(n_rows * missing_rate)

    # 随机选择时间段和站点
    missing_indices = []
    while len(missing_indices) < n_missing:
        # 随机选择一个时间段长度在50到200行之间
        segment_length = random.randint(50, 200)
        start_index = random.randint(0, n_rows - segment_length)
        end_index = start_index + segment_length

        # 选择时间段内的所有行
        segment_indices = list(range(start_index, end_index))

        # 计算剩余所需缺失值数量
        remaining_missing = n_missing - len(missing_indices)
        if remaining_missing <= len(segment_indices):
            missing_indices.extend(random.sample(segment_indices, remaining_missing))
            break
        else:
            missing_indices.extend(segment_indices)

    # 引入缺失值
    missing_indices = list(set(missing_indices))  # 去除重复值
    for index in missing_indices:
        num_stations = random.randint(4, 10)
        stations_to_mask = random.sample(station_list, num_stations)
        for station in stations_to_mask:
            df_copy.loc[index, station] = np.nan

    return df_copy


# 处理每个缺失率并保存结果
for rate in missing_rates:
    df_missing = introduce_missing_values(data, rate)
    output_path = f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\data_missing_{int(rate * 100)}.csv'
    df_missing.to_csv(output_path, index=False)
    print(f'Saved file with missing rate {rate}: {output_path}')

# 定义站点列表
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
missing_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for station in station_list:
    df = pd.read_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv')
    for rate in missing_rates:
        df2 = pd.read_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\data_missing_{int(rate * 100)}.csv')
        df[f'Soil_VWC_03_{rate}'] = df2[f'{station}']
    df.to_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv',index=False)






