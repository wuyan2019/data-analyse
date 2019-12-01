import re
import requests

def get_github(pages):
    for a in (0,pages):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome'
                                 '/72.0.3626.121 Safari/537.36'}
        url = 'https://github.com/antirez/redis/issues?pages='+str(a)
        res = requests.get(url, headers=headers).text
        get_num = '<div class="float-left col-8 lh-condensed p-2">.*?"issue_(.*?)_link"'
        num = re.findall(get_num, res, re.S)
        for j in num:
            url1 = 'https://github.com/antirez/redis/issues/' + str(j)
            res1 = requests.get(url1, headers=headers).text
            get_title = '<h1 class="gh-header-title f1 mr-0 flex-auto break-word">.*?>(.*?)</span>'
            get_body = '<td class="d-block comment-body markdown-body  js-comment-body">(.*?)</td>'
            title = re.findall(get_title, res1, re.S)
            body = re.findall(get_body, res1, re.S)
            for i in range(len(body)):
                title[0]=title[0].strip()
                body[i] = re.sub('<.*?>', '', body[i])
                body[i] = body[i].strip()
                body[i] = re.sub('\n', '', body[i])

            # 将爬取的文件储存为txt格式于根目录下，方便进行查看
            file = open('github.txt', 'a')
            file.write(title[0] + '#' + j+ '\n')
            for i in range(len(body)):
                file.write(str(i + 1) + '.'+ body[i] + '\n')
            file.write('-----------------------------------' + '\n' + '\n')
            file.close()




get_github(2)





