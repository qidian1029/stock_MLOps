#获取默认回测参数
class Backtest_parameter():
    def __init__(self,backtest_name,model,dataset,topk,n_drop,):
        self.backtest_name = backtest_name
        self.start_time = "2017-01-01"
        self.end_time = "2020-08-01"
    def get_parameter(self):
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
                "start_time":self.start_time,
                "end_time": self.end_time,
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
        return port_analysis_config
        
    
