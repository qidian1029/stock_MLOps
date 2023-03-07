# 模型初始化，超参数设置
def load_lightGBM():
    from qlib.contrib.model.gbdt import LGBModel
    config = {
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
    model = LGBModel(**config) # model
    return model