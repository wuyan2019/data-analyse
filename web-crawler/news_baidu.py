import re
import requests
import time


# 定义函数进行爬取
def baidu_news(company,page):
    num=(page-1)*10
    url=('https://www.baidu.com/s?ie=utf-8&cl=2&medium=0&rtt=1&bsst=1&rsv_dl=news_t_sk&tn=news&word='
         +company+'&rsv_sug3=5&rsv_sug4=215&rsv_sug1=3&rsv_sug2=0&inputT=803&x_bfe_rqs=03E80&x_bfe_tjscore=0.499279&tngroupname=organic_news&pn='
         +str(num))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome'
                             '/72.0.3626.121 Safari/537.36'}
    res = requests.get(url, headers=headers).text
    # 爬取新闻标题、日期作者、网页想相关信息
    p_href = '<h3 class="c-title">.*?<a href="(.*?)"'
    p_title = '<h3 class="c-title">.*?>(.*?)</a>'
    p_info = '<p class="c-author">(.*?)</p>'
    info = re.findall(p_info, res, re.S)
    href = re.findall(p_href, res, re.S)
    title = re.findall(p_title, res, re.S)
    # 对相关字段进行处理，去除多余字段，调整格式
    source = []
    date = []
    for i in range(len(title)):
        title[i] = title[i].strip()
        title[i] = re.sub('<.*?>','',title[i])
        source.append(info[i].split('&nbsp;&nbsp;')[0])
        date.append(info[i].split('&nbsp;&nbsp;')[1])
        source[i] = source[i].strip()
        date[i] = date[i].strip()

    # 将爬取的文件储存为txt格式于根目录下，方便进行查看
    file=open('新闻舆情爬取.txt','a')
    file.write(company+'舆情爬取完毕'+'\n'+'\n')
    for i in range(len(title)):
        file.write(str(i + 1) + '.' + title[i] + '(' + date[i] + '-' + source[i] + ')'+'\n')
        file.write(href[i]+'\n')
    file.write('-----------------------------------'+'\n'+'\n')
    file.close()


# 不间断进行爬取bat的百度新闻，可以对公司和页码进行赋值
while True:
    companys=['阿里巴巴','百度','腾讯']
    for company in companys:
        for i in range(3):
            try:
                baidu_news(company,i+1)
                print(company+'百度新闻第{}页爬取成功'.format(i+1))
            except:
                print(company + '百度新闻第{}页爬取失败'.format(i+1))
    time.sleep(10800)








