import sys
import urllib.request
import threading
import queue
import time
import logging
import glog
from optparse import OptionParser

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)

# # 再创建一个handler，用于输出到控制台
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
# logger.addHandler(ch)


class ThreadPool(object):
    def __init__(self, urlpth, req_number, thread_num):
        self.work_queue = queue.Queue()
        self.threads = []
        self.__init_work_queue(req_number, urlpth)
        self.__init_thread_pool(thread_num)

    """
        initialize threads
    """

    def __init_thread_pool(self, thread_num):
        for i in range(thread_num):
            self.threads.append(MyThread(self.work_queue))

    """
        initialize work queue
    """

    def __init_work_queue(self, req_number, urlpth):
        for i in range(req_number):
            self.add_job(do_job, urlpth)

    """
        add a job to the queue
    """

    def add_job(self, func, args):
        self.work_queue.put((func, args))

    """
        wait for all the threads to be completed
    """

    def wait_all_complete(self):
        for item in self.threads:
            if item.isAlive():
                item.join()


class MyThread(threading.Thread):
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.start()
        self.time_ = []

    def run(self):
        while True:
            try:
                t1 = time.time()
                do, args = self.work_queue.get(block=False)
                do(args)
                logger.info('do job')
                self.work_queue.task_done()  # notify the completement of the job
                t2 = time.time()
                self.time_.appand(t2 - t1)
            except:
                break
        print(self.time_)



ERROR_NUM = 0


def do_job(args):
    try:
        html = urllib.request.urlopen(args)
        logger.info(html.length)
    except Exception as e:
        logger.error(e)
        global ERROR_NUM, time_
        ERROR_NUM += 1



def parse():
    """parse the args"""
    parser = OptionParser(
        description="The scripte is used to simulate apache benchmark(sending requests and testing the server)")
    parser.add_option("-n", "--number", dest="num_of_req", action="store", help="Number of requests you want to send",
                      default=1)
    parser.add_option("-c", "--concurrent", dest="con_req", action="store",
                      help="Number of concurrent requests you set", default=1)
    parser.add_option("-u", "--url", dest="urlpth", action="store", help="The url of server you want to send to")
    (options, args) = parser.parse_args()
    return options


def main():
    """main function"""
    start = time.time()
    options = parse()

    if not options.urlpth:
        print('Need to specify the parameter option "-u"!')
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)

    tp = ThreadPool(options.urlpth, int(options.num_of_req), int(options.con_req))
    tp.wait_all_complete()
    end = time.time()

    logger.info("===============================================")
    logger.info("URL: {}".format(options.urlpth))
    logger.info("Total Requests Number: {}".format(options.num_of_req))
    logger.info("Concurrent Requests Number: {}".format(options.con_req))
    logger.info("Total Time Cost(seconds): {}".format((end - start)))
    logger.info("Average Time Per Request: {}".format((end - start) / int(options.num_of_req)))
    logger.info("Average Requests Number Per Second: {}".format(int(options.num_of_req) / (end - start)))
    logger.info("Total Error Number: {}".format(ERROR_NUM))

    """
    Request per second=Complete requests/Time taken for tests
    Time per request,
    
    """


if __name__ == '__main__':
    main()
    # print(time_)
