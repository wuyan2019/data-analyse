# !/usr/bin/env python
from utils.data_preprocess import DataProcess
from nerual_network import parameter_search

if __name__ == '__main__':
    stock_data = DataProcess()
    x_train, x_test, y_train, y_test = stock_data.data_select()
    p1 = parameter_search(x_train, y_train)
    print(p1)
    # # train nerual networt model
    # model = NNModel(epoch=300)
    # nn_model = model.neural_network_model(x_train, y_train, x_test, y_test)
    # acc = nn_model.evaluate(x_test, y_test, batch_size=64)
    # # 输出准确率
    # print(acc[1])
    # if acc > 0.765:
    #     nn_model.save('nn_model.h5')
    # stock_data = DataProcess()
    # x_train, x_test, y_train, y_test = stock_data.data_select(one_hot=False)
    # rf = RandomForests()
    # rf.train_model(x_train, x_test, y_train, y_test)
