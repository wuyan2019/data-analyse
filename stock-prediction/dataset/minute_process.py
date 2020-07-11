import pandas as pd
import matplotlib.pyplot as plt


def profit_hist(num, name):
    profit_list = []
    for i in range(num, len(df['ClosePrice'])):
        profit_list.append(((df['ClosePrice'][i] - df['ClosePrice'][i - num]) / df['ClosePrice'][i - num]) * 10000)
    middle_list = num * [0]
    middle_list.extend(profit_list)
    df['{}'.format(name)] = middle_list
    df[name].plot(kind='hist', xlim=(-100, 100), bins=90)
    plt.show()


df = pd.read_csv('600030.csv')
df = df.iloc[:5300]
profit_hist(2, 'threedays_prorate')
