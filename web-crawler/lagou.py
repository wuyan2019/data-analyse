import requests
import pandas
import time



# 定义存储的dataframe名称
def getGoalData(data):
    for i in range(15):#每页默认15个职位
        info={
            'positionName': data[i]['positionName'],    #职位简称
            'companyShortName': data[i]['companyShortName'],    #平台简称
            'salary': data[i]['salary'],    #职位薪水
            'createTime': data[i]['createTime'],    #发布时间
            'companyId':data[i]['companyId'],   #公司ID
            'companyFullName':data[i]['companyFullName'],   #公司全称
            'companySize': data[i]['companySize'],  #公司规模
            'financeStage': data[i]['financeStage'],    #融资情况
            'industryField': data[i]['industryField'],  #所在行业
            'education': data[i]['education'],  #教育背景
            'district': data[i]['district'],    #公司所在区域
            'businessZones':data[i]['businessZones']    #区域详细地
        }
        data[i]=info
    return data


# 定义存储文件的名称
def saveData(data):
    table = pandas.DataFrame(data)
    table.to_csv('LaGou1.csv', index=False, mode='a+')


# 主函数，爬取拉勾信息
def main(pages):
    # 拼装header信息
    header = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Referer': 'https://www.lagou.com/jobs/list_python?px=default&city=%E5%85%A8%E5%9B%BD',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
        'Host': 'www.lagou.com' }

    # 模拟请求的url
    url1 = 'https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98?labelWords=&fromSearch=true&suginput='
    url='https://www.lagou.com/jobs/positionAjax.json?city=%E4%B8%8A%E6%B5%B7&needAddtionalResult=false'


# 拉勾网有反爬虫机制，直接进行爬取会报错，因此需要进行一个session操作来储存cookie
    for page in range(1, pages):
        form = {
            'first': 'false',
            'pn': page,
            'kd': '数据挖掘'
        }

        s = requests.Session()  # 建立session
        s.get(url=url1, headers=header, timeout=3)
        cookie = s.cookies  # 获取cookie
        respon = s.post(url=url, headers=header, data=form, cookies=cookie, timeout=3)
        time.sleep(7)
        result = respon.json()
        data = result['content']['positionResult']['result']  # 返回结果在preview中的具体返回值
        data_goal = getGoalData(data)
        saveData(data_goal)
    return data


# 爬取25页
data = main(26)


