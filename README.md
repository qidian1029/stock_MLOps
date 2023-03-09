https://github.com/qidian1029/stock_MLOps.git
`stock_MLOps` 是一个基于量化交易实验模型的 MLOps 项目，在Qlib量化框架下进行扩充，专为想要将ML模型部署到生产环境的数据科学家或算法工程师而构建的，量化交易ML模型可重用、高性能且易于部署，适应于相关的学术研究和生产。

旨在提供一款标准化，易操作的MLOps，使得量化交易模型研究更加便捷和功能全面。

## requirements
包名            版本号
python          3.8
pyqlib           
numpy           
pandas          
scipy           
request         
pytorch         
cython          
cmake           
scikit-learn      
transformers    
catboost        
xgboost         
lightgbm        
H2O             



## 需求实现

模型的研发更新快速				
数据持续更新，需要对模型持续监测和更新				
使用者知识层次不同，需要保证对开发环节和生产环境进行良好的封装，提高开发和维护的效率				
保密性：量化交易的策略、训练数据对外保密				
版本管理：旧模型可能过一段时间又可用了				
多模型并行：多个模型可能同时有效				


## 数据流
### 框架输入
#### 股票数据
结构化股票量价数据
量化交易相关文本数据
#### 模型数据
模型选择
参数数据

### 框架输出
预测数据
数据可视化图表


## 工具及功能
具体步骤	        流程步骤            主流工具                  qlib支持     对比工具
MLOps的工作流程	    开发环境	        VSCode and Jupyter
全流程活动	        远程调试	        VSCode remote debugger
	               代码管理	           Git                                    H2O
数据准备            数据打标	        Label Studio                           Micorsoft Azure
	               数据管理	           DVC
特征工程	        元数据管理	        MLflow
模型开发	        模型管理	        MLflow                                 H2O
	               超参数优化          NNI
	               实验跟踪	           MLflow
	               测试	               Locust
模型部署	        部署	            Seldon Core                            H2O
	                分布式训练	        Neu.ro
模型监控	        监测	            Prometheus + Grafana
	                理解	            Seldon Alibi
	                管道编排	        Neu.ro
	                资源编排	        Neu.ro
	                访问控制编排	    Neu.ro
工具安装及配置
Label Studio    
H2O             github.com/h2oai
SAS             github.com/sassoftware
Neu.ro          https://neu.ro/mlops/

### 量化交易模型
#### 结构化数据模型
LSTM
模型位置：

模型输入数据结构：

模型调用方式：


#### NLP模型
Astock
模型位置：

模型输入数据结构：

模型调用方式：


## 工具使用及说明
H2O
功能：

存储位置：

参数设置：

调用方式：

MLflow
功能：

存储位置：

参数设置：

调用方式：


## 数据存储位置
### 数据流存储
存储位置：

数据格式：

数据含义：

### 模型存储位置
存储位置：

数据格式：

数据含义：

### 预测及图表位置
存储位置：

数据格式：

数据含义：