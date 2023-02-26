#获取内置模型超参数设置
'''
新增模型时，需要在此文件增添模型超参数基础设置值
'''
#模型类型有：

#输入为模型类型，输出为模型超参数设置
def get_hyper_parameter(model_type):
    #字典类型存储模型设置值
    model_hyper_parameter = {
        #长短期神经网络
        "LSTM":{
            "class": "LSTM",
            "module_path": "qlib.contrib.model.pytorch_lstm_ts",
            "kwargs":{
                "d_feat": 20,
         	    "hidden_size": 64,
                "num_layers": 2,
     	        "dropout": 0.0,
    	        "n_epochs": 200,
                "lr": 1e-3,
        	    "early_stop": 10,
       	        "batch_size": 800,
    	        "metric": "loss",
    	        "loss": "mse",
                "n_jobs": 20,
    	        "GPU": 0,
            },                
        },
        #线性神经网络
        "Linear":{
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs":{
                "estimator": "ols",
            },                
        },
        #时序路由选择器
        "TRA":{
            "class": "TRAModel",
            "module_path": "qlib.contrib.model.pytorch_tra",
            "kwargs":{
                "tra_config": "*tra_config",
                "model_config": "*model_config",
                "model_type": "RNN",
                "lr": 1e-3,
                "n_epochs": 100,
                "max_steps_per_epoch":None,
                "logdir": "output/Alpha158",
                "early_stop": 20,
                "seed": 0,
                "lamb": 1.0,
  	            "rho": 0.99,
                "alpha": 0.5,
	            "transport_method": "router",
	            "memory_mode": "*memory_mode",
                "eval_train": False,
  	            "eval_test": True,
      	        "pretrain": True,
      	        "init_state":None,
                "freeze_model": False,
  	            "freeze_predictors": False,
            },                
        },
        #LightGBM梯度升降机
        "LightGBM":{
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs":{
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },                
        },
        #TCN时序卷积网络
        "TCN":{
            "class": "TCN",
            "module_path": "qlib.contrib.model.pytorch_tcn_ts",
            "kwargs":{
                "d_feat": 20,
                "num_layers": 5,
                "n_chans": 32,
                "kernel_size": 7,
                "dropout": 0.5,
                "n_epochs": 200,
                "lr": 1e-4,
                "early_stop": 20,
                "batch_size": 2000,
                "metric": "loss",
                "loss": "mse",
                "optimizer": "adam",
                "n_jobs": 20,
                "GPU": 0,
            },                
        },
        
    }
    return model_hyper_parameter[model_type]

