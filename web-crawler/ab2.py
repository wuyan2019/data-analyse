from optparse import OptionParser
from multiprocessing.pool import Pool as ProcessPool
import requests
import re

from multiprocessing.dummy import Pool as ThreadPool

"""
python3 ab2.py -u localhost:5000 -c 4 -n 20 -T text/html;charset=utf-8
python3 ab2.py -u 127.0.0.1:5000/json -c 4 -n 20 -p '{"name": "asd"}' -T application/json;charset=utf-8
"""

parser = OptionParser()
parser.add_option("-u", "--url", dest="url",
                  help="url")
parser.add_option("-c", "--concurrency",
                  dest="concurrency", default='1',
                  help="Number of multiple requests to make")
parser.add_option("-C", "--Cookies",
                  dest="cookies",
                  help="cookies")
parser.add_option("-H", "--headers",
                  dest="headers",
                  help="Add Arbitrary header line, eg. 'Accept-Encoding: gzip'")
parser.add_option("-n", "--number",
                  dest="number", default='1',
                  help="Number of requests to perform")
parser.add_option("-s", "--ssl",
                  dest="ssl", default='F',
                  help="ssl")
parser.add_option("-S", "--ssl_verify",
                  dest="ssl_verify", default='F',
                  help="Do not show confidence estimators and warnings")
parser.add_option("-t", "--timelimit",
                  dest="time_limit", default='0',
                  help="Seconds to max. wait for responses")
parser.add_option("-p", "--postfile",
                  dest="data", default='',
                  help="File containing data to POST")
parser.add_option("-X", "--proxy",
                  dest="proxy", default='',
                  help="Proxyserver and port number to use")
parser.add_option("-T", "--contenttype",
                  dest="content_type", default='',
                  help="Content-type header for POSTing")
(options, args) = parser.parse_args()
print(u'参数列表')
print(options, args)
# url = options.url
#
# concurrency = options.concurrency
#
# cookies = options.cookies
#
# headers = options.headers
#
# number = options.number

options.ssl = 'https://' if options.ssl == 'T' else 'http://'

options.ssl_verify = True if options.ssl_verify == 'T' else False

options.time_limit = int(options.time_limit) if options.time_limit.isalnum() else 0

options.concurrency = int(options.concurrency) if options.concurrency.isalnum() else 1

options.number = int(options.number) if options.number.isalnum() else 1

# 优化缩减代码
options_dict = options.__dict__

# post_file = options.post_file
#
# proxy = options.proxy


class ApacheBenchmark(object):

    total_time = 0

    def __init__(self, **kwargs):
        # 优化缩减代码
        self.__dict__.update(kwargs)
        self.timeout = self.time_limit/self.concurrency/self.number if self.time_limit else None
        # self.url = url
        # self.cookies = cookies
        # self.headers = headers
        # self.number = number
        # self.ssl = ssl
        # self.r_type = r_type
        # self.data = data
        # 将字符串变成可使用的dict
        self.__get_cookies()
        self.__get_headers()
        self.__get_proxy()

    # @classmethod
    # def init_dict(cls, **kwargs):
    #     self.__dict__.update(kwargs)

    def __get_headers(self):
        if not self.headers:
            self.headers = {}
            return
        headers_dict = ''
        # for h in self.headers:
        headers_dict += re.sub(r'(.*?): (.*?)$', r'"\1": "\2",\n', self.headers)
        headers_dict = eval('{'+headers_dict+'}')
        self.headers = headers_dict

    def __get_cookies(self):
        if not self.cookies:
            self.cookies = {}
            return
        cookies = re.sub(r'(.*?)=(.*?);', r'"\1": "\2",\n', self.cookies + ';')
        cookies_dict = eval('{'+cookies+'}')
        cookies_dict = {k.strip(): v.strip() for k, v in cookies_dict.items()}
        self.cookies = cookies_dict

    def __get_data(self):
        if not self.data:
            self.data = {}
            return
        data = re.sub(r'(.*?)=(.*?);', r'"\1": "\2",\n', self.data + ';')
        data = eval('{'+data+'}')
        data_dict = {k.strip(): v.strip() for k, v in data.items()}
        self.data = data_dict

    def __get_proxy(self):
        if not self.proxy:
            self.proxy = {}
            return
        self.proxy = {self.ssl[:-3]: self.ssl+self.proxy}

    def run_request(self, thread_name):
        print(u'第{}个线程启动'.format(thread_name))
        if self.total_time and ApacheBenchmark.total_time >= self.total_time:
            return

        r_type = 'post' if self.data else 'get'
        data = self.data if r_type == 'post' else None
        self.headers.update({'Content-Type': self.content_type})
        print(self.__dict__)
        r = requests.request(r_type,
                             self.ssl + self.url,
                             cookies=self.cookies,
                             headers=self.headers,
                             verify=self.ssl_verify,
                             data=data,
                             proxies=self.proxy,
                             timeout=self.timeout)

        if r.status_code == 200:
            status = u'成功'
        else:
            status = u'失败，responded_code：{}'.format(r.status_code)
        ApacheBenchmark.total_time += r.elapsed.microseconds
        print(u'返回请求状态：{:10}，目前消耗总时间{:20}'.format(status, ApacheBenchmark.total_time))
        return r.elapsed.microseconds

    def run(self, pool_name):
        print(u'第{}个进程启动'.format(pool_name))
        thread_pool = ThreadPool(processes=self.number)
        thread_pool.map(self.run_request, range(self.number))
        thread_pool.close()
        thread_pool.join()

        # 方案二
        # while self.number <= 0:
        #     self.number -= 1
        #     self.run_request(self.number)
        #     # 总时间限定
        #     if self.total_time and ApacheBenchmark.total_time >= self.total_time:
        #         break

    def star(self):
        process_pool = ProcessPool(processes=self.concurrency)
        process_pool.map(self.run, range(self.concurrency))
        process_pool.close()
        process_pool.join()


def main(**kwargs):

    print(kwargs)
    ab = ApacheBenchmark(**kwargs)
    ab.star()


if __name__ == '__main__':
    # main(url=url, concurrency=concurrency, cookies=cookies,
    #      headers=headers, number=number, ssl=ssl, ssl_verify=ssl_verify,
    #      time_limit=time_limit, post_file=post_file, proxy=proxy,
    #      )
    main(**options_dict)