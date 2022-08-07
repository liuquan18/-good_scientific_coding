# 不可能的任务：用深度学习模型预测NAO每日指数



# Predicting daily North Atlantic Oscillation daily index using a seq2seq model

## 介绍

由于北极地区比热带温度低，所以同一位势，北极地区比南部低（Geopotential height）。但是，如果我们仅关注北大西洋部分的距平数据（anomaly，去除时间平均，因此表示相对于平均值的变化），会发现，位势高度异常有时候是北方高，有时候是南面高。低的北方场（图1左蓝色部分）经常与高的南方场（图1左红色部分）同时出现，反之亦然。这种不同空间上的两点相互关联的现象被称为是遥相关（Teleconnections）。如果我们定义一个指数，用来描述这中空间模式的符号和强度，我们就得到了NAO指数（图1右）。NAO对欧洲乃至全球的气象气候有重要的影响。提高NAO指数预测精度，将极大促进天气预报精度和气候变化研究。但作为气候系统的重要内部变异（internal variability) 之一，NAO被认为是混沌的和难以预测的过程驱动。因此预测NAO指数是一个极具挑战的任务。

北大西洋涛动（North Atlantic Oscillation, NAO）

<img src="/Users/liuquan/Documents/wechat/deep_learning/NAO_pattern.png" alt="image-20220730142457746" style="zoom:50%;" />

图1. 500hpa位势高度表示的北大西洋涛动的空间模式（左）和时间序列（右）。

尽管如此，预测NAO指数的努力从未停止。比如近期Met Office Seasonal Prediction System (GloSea5) (Nick Dunstone, Doug Smith, and Adam Scaife, et al., 2016) 展示了利用物理模型预测下一年NAO冬季指数的效果。该研究表明，四个指数对预测NAO指数有重要指导意义，分别是: the El Niño–Southern Oscillation (ENSO) in the tropical Pacific; the Atlantic SST tripole pattern (AST) that has been linked to NAO variations in early winter; the sea-ice coverage (SIC) in the Kara Sea region; and the stratospheric polar vortex strength (SPVS) via which many different drivers can act.

<img src="/Users/liuquan/Documents/wechat/deep_learning/img2_fourindex.png" alt="image-20220730154934589" style="zoom:50%;" />

Dunstone, Nick, et al. "Skilful predictions of the winter North Atlantic Oscillation one year ahead." *Nature Geoscience* 9.11 (2016): 809-814.

本项目利用上述四个指数，来预测下一年的NAO指数。相比于上述研究中预测冬季（季度平均）指数，本项目预测冬季每日指数，当然，这几乎是不可能的任务。

# 数据和方法

The data used in this project comes from MPI-Grand Ensemble. In historical run, there are totally 100 ensembles, providing a big dataset to train a deep learning model. The five indexes are firstly calculated: NAO is represented as the principle components of EOF analysis over 500hpa geopotential height. ENSO and AST is calculated as the field mean of SST over tropical Pacific and North Atlantic. SIC is the evolution of spatial averaged Kara Sea ice. Since no daily output of MPI-GE over 50hpa is available, the SPVS index is calculated over 200hpa. The data pre-processor is not include in this blog.

<img src="/Users/liuquan/Documents/wechat/deep_learning/img3data.png" alt="image-20220730155425611" style="zoom:50%;" />

Fig.3 The MPI-GE data 

The project is based on Seq2seq model, The encoder model is a simple LSTM model, taking four independent variables as inputs, the decoder is a simple LSTM model plus fully connected layer. Three experiments are implemented. 

1. The first experiment uses the LSTM as an encoder, and another LSTM as the decoder. In decoder, the NAO index of this year is also inputted. Use the MSE as the loss function.

   <img src="/Users/liuquan/Library/Application Support/typora-user-images/image-20220730155723189.png" alt="image-20220730155723189" style="zoom:50%;" />

   Fig.4 work flow of Seq2seq model in this project

2. The second experiment uses the same frame as the first experiment, but a costumed loss function to optimise the temporal variability.

3. The third experiment is the same as the second, but the input to decoder changes from NAO index of this year, to the several spectrums (11 in this project) of NAO index of this year. such spectrums are gotten from Singular Spectrum analysis (SSA). 

# 代码

## 1. Import

```python
##########################Load Libraries  ####################################
import pandas as pd
import numpy as np
# import dask.dataframe as dd

import matplotlib.pyplot as plt
# import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing, metrics
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta 
from tqdm.notebook import tqdm_notebook as tqdm
from torch.autograd import Variable
import random 
import os
from matplotlib.pyplot import figure
from fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import torch.optim as optim

%matplotlib inline

from torch.utils.data import TensorDataset, DataLoader
import scipy
```

## 2. Data Pre-process

### Load data

The data used in this project is alread pre-process in levante, including the normalization. Here the data is imported by numpy.

The time series data is already pre-precess into sequences, and saved as .npy files.

> shape of **X**: 15400,120,4     ➡ [ENSO,AST,SIC,SPVS]
>
> shape of **Y**: 15400,120,1     ➡ NAO of next year
>
> shape of **Z**: 15400,120,1     ➡ NAO of this year
>
> shape of **ZS**: 15400,120,11 ➡ SSA components of NAO in this year (will be used in experiment 3.)

```python
X = np.load("/content/drive/MyDrive/Deeplearning/X.npy")
Y = np.load("/content/drive/MyDrive/Deeplearning/Y.npy")
Z = np.load("/content/drive/MyDrive/Deeplearning/Z.npy")
ZS = np.load("/content/drive/MyDrive/Deeplearning/ZS.npy")
```

```python
X = np.float32(X)
Y = np.float32(Y)
Z = np.float32(Z)
ZS = np.float32(ZS)
```

简单查看一下：
```python
fig,axes = plt.subplots(1,2,figsize = (12,3),sharey = True)
axes[0].plot(X[0],)
# axes[1].plot(Y[0],label = "nao of next year")
axes[1].plot(Z[0],label = "nao of this year",color ='#ff7f0e' )

axes[0].legend(['enso','tripole','sic','spvs'])
axes[1].legend()
axes[0].set_xlabel("days")
axes[1].set_xlabel("days")

axes[0].set_ylabel("anomaly index")
```

![img](/Users/liuquan/Documents/wechat/deep_learning/img5 index.png)

再看一下我们experiment3要用到的不同频率的数据，数据处理方法可以参考之前推送中的SSA分解。

![spectrum](/Users/liuquan/Documents/wechat/deep_learning/spectrum.png)

### Change the data range to (-1,1)

尽管数据已经做过标准化，但是数据范围是（0，1），我们这里将其转化为（-1，1）

```python
X = 2*X-1
Y = 2*Y-1
Z = 2*Z-1
ZS = 2*ZS-1
```

### Train data and test data

选择用cpu or gpu，google colab是可以用GPU的。

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is:",device)
```

将numpy array转化为tensor，建议参考上文分享的notebook。

```python
X = torch.from_numpy(X).to(device)
Y = torch.from_numpy(Y).to(device)
Z = torch.from_numpy(Z).to(device)
ZS = torch.from_numpy(ZS).to(device)
```

深度学习训练一般不会一次性把所有的数据送进内存，这样一来内存吃紧，二来也影响训练。因此会将数据分解为长度一样的batches，每一次batch送进去之后计算loss，更新参数。此处batch_size是超参（超级参数，指模型中不会通过训练更新的参数）之一。
```python
def loader_dataset(x,y,z,batch_size = 2000):
  	"""
  	transform tensors to data loader. 
    """
    full_dataset = TensorDataset(x, y, z)  # combine the inputs and outputs into a PyTorch Dataset object
    
    # size
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.2*len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    
    # split
    train_dataset, valid_dataset,test_dataset = torch.utils.data.random_split(full_dataset, [train_size,valid_size, test_size])
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    # data loader
    train_loader = DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)
    return train_loader,valid_loader,test_loader
```

```python
train_loader,valid_loader,test_loader = loader_dataset(X,Y,Z)
```

# 3. Define Model

## Encoder

基本没有做什么改变，很容易就可以找着代码。

```python
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim,num_layers ):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = embedding_dim
        self.num_layers = num_layers
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=num_layers,
          batch_first=True,
          dropout = 0.1
        )
        self.output_layer1 = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
      """
      simple process for LSTM.
      **Arguments**
      	*x* the input data.
      **Return**
      	*x,hidden,cell*
      """

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))
              
        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))
        x = self.output_layer1(x)
        x = self.tanh(x)

        return x,hidde, cell 
```

## Decoder

```python
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1,num_layers=3,z_len=12):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features =  input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=input_dim,
          num_layers=num_layers,
          batch_first=True,
          dropout = 0.1
        )

        self.tanh = nn.Tanh()
        self.output_layer1 = nn.Linear(self.hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(1,affine=False)
        self.output_layer2 = nn.Linear(self.hidden_dim,1)

    def forward(self, x,input_hidden,input_cell,z_len):
       
        x = x.reshape((-1,1,z_len )) # z_len = 1 for exp1 and 2, 12 for exp3
        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

        x = self.output_layer1(x)
        x = self.tanh(x)

        return x, hidden_n, cell_n
```

