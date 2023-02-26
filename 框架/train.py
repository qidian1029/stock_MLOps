#训练函数
#导入包
import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

#自建库
import time_parameter
import backtest_parameter
import model_hyper_parameter


#数据位置
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


class Experiment():
    def __init__(self,market,model,time,):
        self.market = market,
        self.model = model,
        self.time = time
    
    def data_set(self,task):

        pass

    def train():

        pass

    def backtest():

        pass
    

market ={
    "market":"text",
    "benchmarket":"SH600000"
}
model = {
    model ：,
    model_path : ,
    model_kwargs ; ,
}

#股票池设置
market = market['market']
benchmark = market['benchmarket']
#模型设置
model = "LGBModel"
model_path = "qlib.contrib.model.gbdt"
#超参数
model_kwargs = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        }
#因子库设置
factor = "Alpha158"
factor_path = "qlib.contrib.data.handler"
#训练时间设置
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
#回测参数
topk = 10
n_drop = 3
backtest_start_time = "2020-07-01"
backtest_end_time = "2020-09-30"


#导入包
import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    sys.path.append(str(scripts_dir))
    from get_data import GetData
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)
#任务配置
task = {
    "model": {
        "class": model,
        "module_path": model_path,
        "kwargs": model_kwargs,
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": factor,
                "module_path": factor_path,
                "kwargs": data_handler_config,
            },
            "segments": train_segments,
        },
    },
}
# model initiaiton
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# start exp to train model
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id


# prediction, backtest & analysis
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": topk,
            "n_drop": n_drop,
        },
    },
    "backtest": {
        "start_time":backtest_start_time,
        "end_time": backtest_end_time,
        "account": 100000000,
        "benchmark": benchmark,
        #"verbose":True,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}
# backtest and analysis
with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")
    # prediction
    recorder = R.get_recorder() #记录类 记录了训练的时间 id 
    ba_rid = recorder.id #获取记录id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()#生成交易方向
    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
