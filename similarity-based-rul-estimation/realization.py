from random import randrange

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# 数据预处理
table_header = {'id', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
                'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                'sensor_21'}

df = pd.read_table("data/train.txt", header=None, delim_whitespace=True)
df.tail()

id_list = df[0].unique()
degradation_data = []
for machine_id in id_list:
    degradation_data.append(df[df[0] == machine_id])


# K folds分组
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 划分数据集
folds = cross_validation_split(degradation_data, 5)
train_data = []
for j in range(len(folds) - 1):
    train_data.extend(folds[j])
validation_data = folds[-1]


def sensor_plot(train_data):
    idx = randrange(len(train_data))
    plt.plot(train_data[idx][1], train_data[idx][5],
             train_data[idx + 1][1], train_data[idx + 1][5],
             train_data[idx + 2][1], train_data[idx + 2][5])
    plt.xlabel('time')
    plt.ylabel('sensor_1')
    plt.show()

    plt.plot(train_data[idx][1], train_data[idx][6],
             train_data[idx + 1][1], train_data[idx + 1][6],
             train_data[idx + 2][1], train_data[idx + 2][6])
    plt.xlabel('time')
    plt.ylabel('sensor_2')
    plt.show()


# 绘制传感器折线图
sensor_plot(train_data)

# 聚类工作条件中心点
cluster_data = pd.concat(train_data)
kmeans = KMeans(6).fit(cluster_data.iloc[:, 2:5])
ax = plt.axes(projection='3d')
color = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#00FFFF', '#0000FF']
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2],
           c=color)
plt.show()

# 计算每个聚类的均值和标准差并进行标准化
train_data_normalized = []
for j in range(kmeans.n_clusters):
    cluster_idx = np.where(kmeans.labels_ == j)
    cluster = cluster_data.iloc[cluster_idx[0], 5:]

    cluster_mean = cluster.mean()
    cluster_std = cluster.std()
    normalized_tmp = cluster.apply(lambda x: (x - cluster_mean) / cluster_std, axis=1)
    train_data_normalized.append(
        pd.merge(cluster_data.iloc[cluster_idx[0], 0:5], normalized_tmp,
                 left_index=True, right_index=True))

# 重新按照机器ID分组
train_data_normalized = pd.concat(train_data_normalized)
train_data_normalized = train_data_normalized.fillna(0)
id_list = train_data_normalized[0].unique()
train_data = []
for machine_id in id_list:
    train_data.append(train_data_normalized[train_data_normalized[0] == machine_id].sort_values(by=1))

# 绘制标准化后的数据折线图
sensor_plot(train_data)

# 构建健康指标
health_condition = []
for data in train_data:
    rul = data.iloc[:, 1].max() - data.iloc[:, 1]
    health_condition.append(rul / rul.max())

plt.plot(train_data[0][1], health_condition[0],
         train_data[1][1], health_condition[1],
         train_data[2][1], health_condition[2],
         train_data[3][1], health_condition[3])
plt.xlabel('time')
plt.ylabel('health_condition')
plt.show()

# Sensor趋势分析
sensor_sorted_list = []
for i in range(len(train_data)):
    sensor_slope = np.zeros(21)
    for j in range(21):
        linear_regression = LinearRegression()
        array_x = train_data[i][j].values
        array_y = health_condition[i]
        linear_regression.fit(array_x.reshape(-1, 1), array_y)
        sensor_slope[j] = linear_regression.coef_
    # 降序排列斜率绝对值
    sensor_sorted_list.append(np.argsort(-np.abs(sensor_slope)))

plt.plot(train_data[0][1], train_data[0][sensor_sorted_list[0][0]], label='sensor_{}'.format(sensor_sorted_list[0][0]))
plt.plot(train_data[0][1], train_data[0][sensor_sorted_list[0][1]], label='sensor_{}'.format(sensor_sorted_list[0][1]))
plt.plot(train_data[0][1], train_data[0][sensor_sorted_list[0][2]], label='sensor_{}'.format(sensor_sorted_list[0][2]))
plt.xlabel('time')
plt.ylabel('sensor')
plt.legend(loc='best')
plt.show()

# 使用前八个sensor作为自变量拟合数据
linear_regression = LinearRegression()
array_x = train_data[0][sensor_sorted_list[0][0:8]]
array_y = health_condition[0]
linear_regression.fit(array_x, array_y)

plt.plot(train_data[0][1], health_condition[0])
plt.plot(train_data[0][1], linear_regression.predict(train_data[0][sensor_sorted_list[0][0:8]]))
plt.xlabel('time')
plt.ylabel('health_condition')
plt.show()
