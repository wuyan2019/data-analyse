import jqdatasdk as jq


def get_price(begin, end):
    jq.auth('', '')
    get_security = jq.get_all_securities(types=['stock'], date=None)
    sec_list = get_security.index.to_list()
    df_price = jq.get_price(sec_list, start_date=begin, end_date=end, frequency='daily', fields=['open', 'close', 'high',
                                                                                           'low', 'volume',
                                                                                           'pre_close'],
                      skip_paused=False, fq='pre', panel=True)
    df_price.to_csv('stock_price.csv', index=False)


if __name__ == "__main__":
    get_price('2018-01-01', '2018-03-01')
