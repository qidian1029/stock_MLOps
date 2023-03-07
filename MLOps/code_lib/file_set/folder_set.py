import os


def folder_set(exper_path):
    if os.path.exists(exper_path)==False:  #判断实验数据存储路径是否存在
        os.mkdir(exper_path)
    exper_path = exper_path + '/' + experiment_name #修改实验数据存储文件夹路径
    if os.path.exists(exper_path)==False:
        os.mkdir(exper_path) #创建当前实验数据存储文件夹
        dataset_path = exper_path + '/' + 'dataset'
        os.mkdir(dataset_path) #创建存储dataset数据文件夹
        model_path = exper_path + '/' + 'model'
        os.mkdir(model_path) #创建存储模型相关数据文件夹
        predict_path = exper_path + '/' + 'predict'
        os.mkdir(predict_path) #创建模型预测结果文件夹
        backtest_path = exper_path + '/' + 'backtest'
        os.mkdir(backtest_path) #创建回测结果文件夹