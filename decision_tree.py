import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class RandomForests:
    # init部分用来调整参数
    def __init__(self, max_depth=6, min_samples_split=10, min_impurity_decrease=0,
                 n_estimators=20, random_state=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_estimators = n_estimators
        self.random_state = random_state

    def train_model(self, x_train, x_test, y_train, y_test):
        # 训练决策树模型
        clf = DecisionTreeClassifier(max_depth=self.max_depth,
                                     min_impurity_decrease=self.min_impurity_decrease,
                                     min_samples_split=self.min_samples_split)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        accuracy = clf.score(x_test, y_test)
        print('The score of decision tree is', accuracy)

        # plt.scatter(x_test[:, 1], x_test[:, 4], c=np.squeeze(y_test), s=3)
        # 画出个模型的影响情况
        # index = np.linspace(0, 6, 6)
        # plt.bar(index, clf.feature_importances_, color='rgby',
        #         tick_label=['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
        #                     'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock'])
        # numFeatures = ['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
        #                'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock']

        # mydata = pd.read_csv("Data_m_hs300_trade_v701.csv", header=0)
        # trainData, testData = train_test_split(mydata, test_size=0.4)
        # X, y = trainData[numFeatures], trainData['ths_chg_ratio_nd_stock_x']

        param_test1 = {'n_estimators': range(10, 80, 5)}
        gsearch1 = GridSearchCV(
            estimator=RandomForestRegressor(min_samples_split=50, min_samples_leaf=10, max_depth=8, max_features='sqrt',
                                            random_state=10),
            param_grid=param_test1, scoring='neg_mean_squared_error', cv=5)
        gsearch1.fit(x_train, y_train)
        gsearch1.best_params_, gsearch1.best_score_
        best_n_estimators = gsearch1.best_params_['n_estimators']

        gsearch1.best_params_

        param_test2 = {'max_depth': range(3, 21), 'min_samples_split': range(10, 100, 10)}
        gsearch2 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=10, max_features='sqrt',
                                            random_state=10, oob_score=True),
            param_grid=param_test2, scoring='neg_mean_squared_error', cv=5)
        gsearch2.fit(x_train, y_train)
        gsearch2.best_params_, gsearch2.best_score_
        best_max_depth = gsearch2.best_params_['max_depth']
        best_min_sample_split = gsearch2.best_params_['min_samples_split']

        param_test3 = {'min_samples_split': range(50, 201, 10), 'min_samples_leaf': range(1, 20, 2)}
        gsearch3 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                            max_features='sqrt', random_state=10, oob_score=True),
            param_grid=param_test3, scoring='neg_mean_squared_error', cv=5)
        gsearch3.fit(x_train, y_train)
        gsearch3.best_params_, gsearch3.best_score_
        best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
        best_min_samples_split = gsearch3.best_params_['min_samples_split']

        # numOfFeatures = len(x_train)
        # mostSelectedFeatures = numOfFeatures // 2
        param_test4 = {'max_features': range(3, 6)}
        gsearch4 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                            min_samples_leaf=best_min_samples_leaf,
                                            min_samples_split=best_min_samples_split, random_state=10, oob_score=True),
            param_grid=param_test4, scoring='neg_mean_squared_error', cv=5)
        gsearch4.fit(x_train, y_train)
        gsearch4.best_params_, gsearch4.best_score_
        best_max_features = gsearch4.best_params_['max_features']

        print(gsearch1.best_params_)
        print('最佳深度: %d' % best_n_estimators)

        print(gsearch2.best_params_)
        print(gsearch3.best_params_)
        print(gsearch4.best_params_)

        print('最佳深度: %d' % best_max_depth)
        print('最小叶结点样本：%d' % best_min_samples_leaf)
        print('最小样本分割： %d' % best_min_samples_split)
        print('最大特征: %.2f' % best_max_features)






        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=best_n_estimators, oob_score=True,
                                       max_depth=best_max_depth, n_jobs=4, max_features=best_max_features,
                                       random_state=self.random_state)
        model.fit(x_train, y_train)
        print("The score of random forests is", model.oob_score_)

        # 训练Adaboosting模型
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth),
                                     n_estimators=self.n_estimators)
        ada_clf.fit(x_train, y_train)
        print("The score of adaboost is", ada_clf.score(x_test, y_test))
