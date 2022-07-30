# 不可能的任务：用深度学习模型预测NAO日指数

深度学习的话题似已不如前两年火了。得益于网上五花八门的教程，每个人都能讲五分钟深度学习。**但深度学习的门槛已经降到和EOF分解这样的统计方法一样高了么？**

一方面，深度学习被过分标榜为一种独特的算法，使有心尝试的人总过分评估使用深度学习算法解决自己问题的必要性。另一方面，深度学习算法框架pytroch, tensorflow等给人一种新的计算机语言的即视感。

就使用而言，基础深度学习模型的数学过程并不复杂。当然，深度学习模型需要调参，不会像传统算法那样，给一个输入进去，就有一个固定的输出结果，需要大量的尝试。至于深度学习框架，Pytorch和Numpy的初学成本差不多。相比于网上零星的教程，本文建议想应用这类算法的同学接受系统的训练，比如上课，尤其是需要交作业那种，使深度学习算法像其他所有常用算法一样，成为一个你可以想用就能用的东西。

这学期选修了一门《Practical Deep Learning with Climate Data》的课，算是终于了了之前一直上理论课，却不敢实操的遗憾。理论部分很简单，所以我很少去上课。但这门课最大的价值在于练习，每节课后都会有一个notebook，实话说，作业量非常大。但这也是这门课最大的价值所在。练习甚至包含一个入门python的notebook。刚开始的时候，所有练习是不允许用pytorch的，所以手敲函数会让你对模型的训练过程有更深刻的理解。到后面模型变复杂之后，开始使用pytroch自带的函数，跟着做完这些作业，pytorch 无非就是一个python的包而已。

因为是第一次开课，我不好意思直接把作业直接放到GitHub开源。我将课件和代码上传到了谷歌云盘：https://drive.google.com/drive/folders/1gLsVDVIEdq-21RBXwKJEq5yUZW0g8kNE?usp=sharing 因为notebook直接可以在colab里运行，所以比较推荐谷歌云盘。当然如果因为网络原因无法获取，也可以在这里下载：https://owncloud.gwdg.de/index.php/s/mYPlh7rYnhypveM 

> **NOTE**: 该资源（包括notebook和PDF）著作权归David Greenberg，该分享链接仅用于学习交流。任何人未经David Greenberg本人允许，不得将该资源用于任何商业用途。

下面分享我的结课project，利用seq2seq model预测NAO daily index。

# Predicting daily North Atlantic Oscillation daily index using a seq2seq model

## 介绍

由于北极地区比热带温度低，所以同一位势，北极地区比南部低（Geopotential height）。但是，如果我们仅关注北大西洋部分的距平数据（anomaly，去除时间平均，因此表示相对于平均值的变化），会发现，位势高度异常有时候是北方高，有时候是南面高。低的北方场（图1左蓝色部分）经常与高的南方场（图1左红色部分）同时出现，反之亦然。这种不同空间上的两点相互关联的现象被称为是遥相关（Teleconnections）。如果我们定义一个指数，用来描述这中空间模式的符号和强度，我们就得到了NAO指数（图1右）。NAO对欧洲乃至全球的气象气候有重要的影响。提高NAO指数预测精度，将极大促进天气预报精度和气候变化研究。但作为气候系统的重要内部变异（internal variability) 之一，NAO被认为是混沌的和难以预测的过程驱动。因此预测NAO指数是一个极具挑战的任务。

北大西洋涛动（North Atlantic Oscillation, NAO）

<img src="/Users/liuquan/Library/Application Support/typora-user-images/image-20220730142457746.png" alt="image-20220730142457746" style="zoom:50%;" />

图1. 500hpa位势高度表示的北大西洋涛动的空间模式（左）和时间序列（右）。

尽管如此，预测NAO指数的努力从未停止。比如近期Met Office Seasonal Prediction System (GloSea5) (Nick Dunstone, Doug Smith, and Adam Scaife, et al., 2016) 展示了利用物理模型预测下一年NAO冬季指数的效果。该研究表明，四个指数对预测NAO指数有重要指导意义，分别是: the El Niño–Southern Oscillation (ENSO) in the tropical Pacific; the Atlantic SST tripole pattern (AST) that has been linked to NAO variations in early winter; the sea-ice coverage (SIC) in the Kara Sea region; and the stratospheric polar vortex strength (SPVS) via which many different drivers can act.

<img src="/Users/liuquan/Library/Application Support/typora-user-images/image-20220730153527743.png" alt="image-20220730153527743" style="zoom:50%;" />

本项目利用上述四个指数，来预测下一年的NAO指数。相比于上述研究中预测冬季（季度平均）指数，本项目预测冬季每日指数，当然，这几乎是不可能的任务。

# 数据和方法

The data used in this project comes from MPI-Grand Ensemble. In historical run, there are totally 100 ensembles, providing a big dataset to train a deep learning model. The five indexes are firstly calculated: NAO is represented as the principle components of EOF analysis over 500hpa geopotential height. ENSO and AST is calculated as the field mean of SST over tropical Pacific and North Atlantic. SIC is the evolution of spatial averaged Kara Sea ice. Since no daily output of MPI-GE over 50hpa is available, the SPVS index is calculated over 200hpa. 

The project is based on Seq2seq model, The encoder model is a simple LSTM model, taking four independent variables as inputs, the decoder is a simple LSTM model plus fully connected layer, taking the last hidden layer of encoder model and the NAO index of this year as input.



Three experiments are implemented:

\1. The first experiment uses the MSE loss function.

\2. The second experiment uses a customed loss function to optimize the temporal variability.

\3. The third experiment is the same as the second, but the input to decoder changes from NAO index of this year, to the several spectrums (11 in this project) of NAO index of this year. such spectrums are gotten from Singular Spectrum analysis (SSA). 



![pastedGraphic.png](blob:file:///18709b26-831a-47ec-85b4-fb93ed62e86f)

Fig. 1 work flow of Seq2seq model in this project
