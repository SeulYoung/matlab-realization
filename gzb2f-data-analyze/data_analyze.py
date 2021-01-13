import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def drop_constant_column(df):
    return df.loc[:, (df != df.iloc[0]).any()]


def max_min_temperature(row):
    time = row['时间']
    max = row[f'温度{row["最大温度测点序号"]}']
    min = row[f'温度{row["最小温度测点序号"]}']
    return pd.Series([time, max, min], index=['time', 'max temperature', 'min temperature'])


def temperature_analyze(filename):
    df = pd.read_table(filename, sep=',', header=0, encoding='utf-8')
    df = df.dropna(axis=1, how='all')
    temperature_data = df.apply(max_min_temperature, axis=1)
    temperature_data.plot(x='time')
    plt.xticks(rotation=-10, fontsize=8)
    plt.show()

    data_corr = df.iloc[:, 3:-5].corr()
    # data_corr = df.loc[:, '温度1':'温度24'].corr()
    print(data_corr)

    df = drop_constant_column(df)
    pca = PCA(n_components=3)
    x_reduced = pca.fit_transform(df.iloc[:, 3:-5])
    ax = plt.axes(projection='3d')
    ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2])
    plt.show()
    print(pca.explained_variance_ratio_)


# temperature_analyze('GZB2F/GZB-ss0211-0219-unit2.txt')
# temperature_analyze('GZB2F/GZB-tbs0211-0219-unit2.txt')
# temperature_analyze('GZB2F/GZB-tgb0211-0219-unit2.txt')
# temperature_analyze('GZB2F/GZB-ugb0211-0219-unit2.txt')


def plot_3d(filename):
    df = pd.read_table(filename, sep=',', header=0, encoding='utf-8')
    df = df.dropna(axis=1, how='all')
    df = drop_constant_column(df)
    df['日期'] = df['日期'] + df['时间']
    df = df.drop(['时间'], axis=1)

    ax = plt.axes(projection='3d')
    # ax.plot(range(df.shape[0]), df['2F上导摆度+X总振值'], df['2F上导摆度+Y总振值'], label='2F上导摆度')
    # ax.plot(range(df.shape[0]), df['2F水导摆度+X总振值'], df['2F水导摆度+Y总振值'], label='2F水导摆度')
    ax.plot(range(df.shape[0]), df['2F上机架水平振动+X总振值'], df['2F上机架水平振动+Y总振值'], label='2F上机架水平振动')
    ax.plot(range(df.shape[0]), df['2F上机架垂直振动+X总振值'], df['2F上机架垂直振动+Y总振值'], label='2F上机架垂直振动')
    ax.set_xlabel('数据序列')
    ax.set_ylabel('振动+X总振值')
    ax.set_zlabel('振动+Y总振值')
    ax.legend()
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot(range(df.shape[0]), df['2F定子铁芯径向振动总振值'], df['2F定子铁芯切向振动总振值'], label='定子铁芯总振值')
    ax.set_xlabel('数据序列')
    ax.set_ylabel('径向总振值')
    ax.set_zlabel('切向总振值')
    ax.legend()
    plt.show()


# plot_3d('GZB2F/GZB-disp0211-0219-unit2.txt')
# plot_3d('GZB2F/GZB-vib0211-0219-unit2.txt')


def power_columns_linear_regression(filename):
    df = pd.read_table(filename, sep=',', header=0, encoding='utf-8')
    df = df.dropna(axis=1, how='all')
    df['日期'] = df['日期'] + df['时间']
    df = df.drop(['时间'], axis=1)

    linear_regression = LinearRegression()
    # array_x = df.loc[:, '励磁电流(A)':'2F涡壳进口水压峰峰值']
    array_x = df.loc[:, '励磁电流(A)':'2F水导摆度+Y总振值']
    array_y = df['出力(MW)']
    linear_regression.fit(array_x, array_y)

    plt.plot(array_y, label='Real')
    plt.plot(linear_regression.predict(array_x), label='Predict')
    plt.xlabel('数据序列')
    plt.ylabel('出力(MW)')
    plt.legend()
    plt.show()
    print(linear_regression.coef_)


# power_columns_linear_regression('GZB2F/GZB-fluc0211-0219-unit2.txt')
# power_columns_linear_regression('GZB2F/GZB-disp0211-0219-unit2.txt')


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


x = [5, 7, 11, 15, 16, 17, 18]
y = [25, 18, 17, 9, 8, 5, 8]

nstd = 2
ax = plt.subplot(111)

cov = np.cov(x, y)
vals, vecs = eigsorted(cov)
tmp = np.arctan2(*vecs[:, 0][::-1])
theta = np.degrees(tmp)
w, h = 2 * nstd * np.sqrt(vals)
ell = Ellipse(xy=(np.mean(x), np.mean(y)),
              width=w, height=h,
              angle=theta, color='black')
ell.set_facecolor('none')
ax.add_artist(ell)
plt.scatter(x, y)
plt.show()

# 3 dim
ax = plt.subplot(111)

df = pd.read_table('GZB2F/GZB-fluc0211-0219-unit2.txt', sep=',', header=0, encoding='utf-8')
df = df.dropna(axis=1, how='all')
data = df.loc[:, '励磁电流(A)':'2F导接关腔压力']
mean = data.mean()
cov = data.cov()

vals, vecs = eigsorted(cov)
tmp = np.arctan2(*vecs[:, 0][::-1])
theta = np.degrees(tmp)
w, h = 2 * nstd * np.sqrt(vals)
ell = Ellipse(xyz=(mean[0], mean[1], mean[2]),
              width=w, height=h,
              angle=theta, color='black')
ell.set_facecolor('none')
ax.add_artist(ell)
plt.scatter(data)
plt.show()
