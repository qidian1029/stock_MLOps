from qlib.contrib.data.handler import Alpha158


#dataset 设置

def load_handler():
    alpha158()


def alpha158(start_time,end_time,valid_end,market):
    def load_alpha158_handler():
        config = {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": start_time,
            "fit_end_time": valid_end,
            "instruments": market,
        }
        return Alpha158(**config)
    
    load_alpha158_handler()