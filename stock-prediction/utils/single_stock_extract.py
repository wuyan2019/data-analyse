import pandas as pd
from glob import glob
import zipfile
from datetime import datetime, timedelta

stock_num = 600031
start_time = '20180101'
end_time = '20180301'
stock_file = '..\\dataset\\{}.csv'
dir_path = 'C:\\Users\\moka\\Documents\\XshgShareCandlestickPerMinuteMergedZipByDay' \
           '\\{}\\{}\\{}.zip'


def get_stock_num(stock_num, start_time, end_time):
    dir_zip = []
    result_stock = pd.DataFrame()
    #  提取开始和结束日期间的所有日期
    date_list = []
    begin_date = datetime.strptime(start_time, r"%Y%m%d")
    end_date = datetime.strptime(end_time, r"%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime(r"%Y%m%d")
        date_list.append(date_str)
        # 日期加法days=1 months=1等等
        begin_date += timedelta(days=1)
    for i in (date_list):
        dir_zip.extend(glob(dir_path.format(i[0:4], i[0:6], i)))
        #  遍历解压zip文件
    for i in dir_zip:
        with zipfile.ZipFile(i, 'r') as z:
            csv_name = i[-12:-4]
            f = z.open('{}.csv'.format(csv_name))
            data = pd.read_csv(f, index_col=0)
            get_stock = data[data['InstrumentId'] == stock_num]
            result_stock = pd.concat([result_stock, get_stock])
    return result_stock


if __name__ == '__main__':
    a = get_stock_num(stock_num, start_time, end_time)
    a.to_csv(stock_file.format(stock_num))
