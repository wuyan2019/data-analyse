# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# series 数据的归一化处理
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())
    # return (series - series.mean()) / series.var()


class DataProcess:
    def __init__(self):
        self.path = 'Data_m_hs300_trade_v701.csv'
        self.df = pd.read_csv(self.path).dropna(axis=0)
        # 选择训练数据x
        self.x = self.df[['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
                          'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock']]
        self.x['ths_win_days_m_stock'] = self.x['ths_win_days_m_stock'].fillna(self.x['ths_win_days_m_stock'].median)

    def data_select(self, one_hot=True):
        # 数据归一化
        for data_index in self.x:
            self.x[[data_index]] = min_max_scale(self.x[[data_index]])
        x = np.array(self.x)
        # 选取涨跌幅作为训练的标签
        y = np.array(self.df['ths_chg_ratio_nd_stock_x'])
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
        if one_hot:
            # y = to_categorical(y, 3)
            # y = np.array(pd.get_dummies(y))
            y = y.reshape(len(y), 1)
            enc = OneHotEncoder(sparse=False)
            y = enc.fit_transform(y)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

        return train_x, test_x, train_y, test_y

