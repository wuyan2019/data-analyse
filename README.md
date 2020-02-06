# data-analyse
## 目录
├── README.md   
├── management-after-loan  
│   ├── DebtCollectionModel.py  
│   ├── prosperLoanData_chargedoff.csv  
│   └── 贷后管理&风险模型纲要.ipynb  
├── shujufenxi.py  
├── stock-prediction  
│   ├── GOOG-year.csv  
│   ├── data_preprocess.py  
│   ├── decision_tree.py  
│   ├── lstm.py  
│   ├── main.py  
│   ├── nerual_network.py  
│   ├── series_to_supervised.py  
│   └── torch_lstm.py  
└── web-crawler  
    ├── get_image.py  
    ├── github.py  
    ├── lagou.py  
    ├── leetcode.py  
    └── news_baidu.py   
## 股票分析
通过搭建随机森林和神经网络模型对股票的上涨下跌进行预测，
随机森林模型还使用网格搜索进行优化。基于pytorch的时间序列模型也在逐步添加中。

## 爬虫学习
1.通过对百度图片的源码进行分析，爬取图片，并使用多线程下载。  
2.爬取拉勾网的职位信息并整理，使用session方法解决反爬问题。  
3.爬取百度新闻的内容。  

## 贷后模型
对他人搭建的贷后模型进行分析和学习的内容。
