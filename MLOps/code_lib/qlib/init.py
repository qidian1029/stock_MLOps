#初始化
def qlib_init():
    import qlib 
    from qlib.config import REG_CN
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

