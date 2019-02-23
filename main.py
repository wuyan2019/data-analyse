# !/usr/bin/env python
from data_preprocess import DataProcess
from nerual_network import NNModel


if __name__ == '__main__':
    stock_data = DataProcess()
    x_train, x_test, y_train, y_test = stock_data.data_select()
    # # train nerual networt model
    # model = NNModel(epoch=300)
    # nn_model = model.neural_network_model(x_train, y_train, x_test, y_test)
    # acc = nn_model.evaluate(x_test, y_test, batch_size=64)
    # # 输出准确率
    # print(acc[1])
    # if acc > 0.765:
    #     nn_model.save('nn_model.h5')
