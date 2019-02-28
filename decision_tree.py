import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


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
        print('The score of decision is', accuracy)

        # plt.scatter(x_test[:, 1], x_test[:, 4], c=np.squeeze(y_test), s=3)
        # 画出个模型的影响情况
        index = np.linspace(0, 6, 6)
        plt.bar(index, clf.feature_importances_, color='rgby',
                tick_label=['ths_vol_m_stock', 'ths_avg_turnover_rate_m_stock', 'ths_avg_price_m_stock',
                            'ths_win_days_m_stock', 'ths_swingm_stock', 'ths_relative_chg_ratio_m_stock'])

        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=self.n_estimators, oob_score=True,
                                       max_depth=self.max_depth, n_jobs=4, max_features=None,
                                       random_state=self.random_state)
        model.fit(x_train, y_train)
        print("The score of random forests is", model.oob_score_)

        # 训练Adaboosting模型
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth),
                                     n_estimators=self.n_estimators)
        ada_clf.fit(x_train, y_train)
        print("The score of adaboost is", ada_clf.score(x_test, y_test))



