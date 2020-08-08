import pandas as pd
import datetime
from math import log


# input file name: ./input/stockPrices_raw.json
# output file name: ./input/stockReturns.json
# json structure: crawl daily price data from yahoo finance
#          term (short/mid/long)
#         /         |         \
#   ticker A   ticker B   ticker C
#      /   \      /   \      /   \
#  date1 date2 date1 date2 date1 date2
#
# Note: short return: adjClose/open - 1
#       mid return: adjClose/adjClose(7 days ago) - 1
#       long return: adjClose/adjClose(28 days ago) - 1


def main():
    data = pd.read_csv('stock_price.csv')
    data['next_open'] = data.groupby('code')['open'].shift(1)
    data['next_open'] = data.groupby('code')['open'].shift(-1)
    data['pre_close7'] = data.groupby('code')['open'].shift(7)
    data['pre_close28'] = data.groupby('code')['open'].shift(28)