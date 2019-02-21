# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# series 数据的归一化处理
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())
    # return (series - series.mean()) / series.var()


def neural_network_model(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu'))
    # Add another:
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    # Add a softmax layer with 3 output units:
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))
    return model


path = '/Users/alex/Downloads/guotai/Data_m_hs300_trade_v701.csv'
df = pd.read_csv(path)
df = df.dropna(axis=0)
# 选择训练数据x
x = df[['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
       'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock']]
# 选取涨跌幅作为训练的标签
y = np.array(df.ths_chg_ratio_nd_stock_x)

for data_index in x:
    x[[data_index]] = min_max_scale(x[[data_index]])

# 设定y的标签，0为震荡，1为上涨，2为下跌
for i in range(len(y)):
    if abs(y[i]) < 20:
        y[i] = 0
    else:
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = 2
# 转变成对应的one-hot形式
# y = to_categorical(y, 3)
x = np.array(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# train model
# model = neural_network_model(x_train, y_train, x_test, y_test)
# acc = model.evaluate(x_test, y_test, batch_size=128)
# print(acc)

clf = DecisionTreeClassifier(max_depth=6)
clf.fit(x_train, y_train)
z = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
# a = len(z)
# b = 0
# for i in range(a):
#     if z[i] != y_test[i]:
#         b += 1
# print((a-b)/a)

# plt.scatter(x_test[:, 1], x_test[:, 4], c=np.squeeze(y_test), s=3)
index = np.linspace(0,6,6)
plt.bar(index, clf.feature_importances_, color='rgby',tick_label=['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
       'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock'])
