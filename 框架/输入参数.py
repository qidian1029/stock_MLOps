#定义输入参数格式


#市场参数
#股票池设置
market = "text"
benchmark = "SH600000"

#数据范围
train_segments = {
                "train": ("2020-01-01", "2020-05-31"),
                "valid": ("2020-06-01", "2020-06-30"),
                "test": ("2020-07-01", "2020-09-30"),
            }
data_handler_config = {
    "start_time": "2020-01-01",
    "end_time": "2020-09-30",
    "fit_start_time": "2020-01-01",
    "fit_end_time": "2020-05-31",
    "instruments": market,
    'label':['Ref($open, -2) / Ref($open, -1) - 1']
}


#模型参数
train_num = 4#训练次数
model_type=['LightGBM','LSTM','TRA',"TCN"]
model_label = [
    'Ref($open, -2) / Ref($open, -1) - 1',
    'Ref($open, -2) / Ref($open, -1) - 1',
    'Ref($open, -2) / Ref($open, -1) - 1',
    'Ref($open, -2) / Ref($open, -1) - 1',
    ]

#因子库设置
factor = "Alpha158"
factor_path = "qlib.contrib.data.handler"


#回测参数
topk = 10
n_drop = 3
backtest_start_time = "2020-07-01"
backtest_end_time = "2020-09-30"



import model_hyper_parameter
train_dict={}
for i in range(train_num):
    train_dict[str(i)]= {
        "model_Number": i,#训练编号
        "model_type":model_type[i],
        "model_parameter":model_hyper_parameter.get_hyper_parameter(model_type[i]),
        "model_label":model_label[i],
    }

for i in range(train_num):
    #print(train_dict[str(i+1)])
    print(train_dict[str(i)]["model_label"])


