#获取默认时间参数

class train_time():
    def __init__(self,train_name):
        self.train_name = train_name
        self.train_start = "2008-01-01"
        self.train_end = "2014-12-31"
        self.valid_start = "2015-01-01"
        self.valid_end = "2016-12-31"
        self.test_start ="2017-01-01"
        self.test_end = "2020-08-01"
    def get_train_time(self):
        train_segments = {
                "train": (self.train_start, self.train_end),
                "valid": (self.valid_start, self.valid_end),
                "test": (self.test_start, self.test_end),
            }
        return train_segments








class exercise_time():
    def __init__(self,train_name):
        self.train_name = train_name
        self.train_start = "2020-01-01"
        self.train_end = "2020-05-31"
        self.valid_start = "2020-06-01"
        self.valid_end = "2020-06-30"
        self.test_start ="2020-07-01"
        self.test_end = "2020-09-30"
    def get_train_time(self):
        train_segments = {
                "train": (self.train_start, self.train_end),
                "valid": (self.valid_start, self.valid_end),
                "test": (self.test_start, self.test_end),
            }
        return train_segments

