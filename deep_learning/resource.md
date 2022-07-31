# Practical Deep Learning with Climate Data

深度学习的话题似已不如前两年火了。得益于网上五花八门的教程，每个人都能讲五分钟深度学习。**但深度学习的门槛已经降到和EOF分解这样的统计方法一样了么？**

一方面，深度学习被过分标榜为一种独特的算法，使有心尝试的人总过分评估使用深度学习算法解决自己问题的必要性。另一方面，深度学习算法框架pytroch, tensorflow等给人一种新的计算机语言的即视感。

就使用而言，基础深度学习模型的数学过程并不复杂。当然，深度学习模型需要调参，不会像传统算法那样，给一个输入进去，就有一个固定的输出结果，需要大量的尝试。至于深度学习框架，Pytorch和Numpy的初学成本差不多。相比于网上零星的教程，本文建议想应用这类算法的同学接受系统的训练，比如上课，尤其是需要交作业那种，使深度学习算法像其他所有常用算法一样，成为一个你可以想用就能用的东西。

这学期选修了一门《Practical Deep Learning with Climate Data》的课，算是终于了了之前一直上理论课，却不敢实操的遗憾。理论部分很简单，所以我很少去上课。但这门课最大的价值在于练习，每节课后都会有一个notebook，实话说，作业量非常大。但这也是这门课最大的价值所在。练习甚至包含一个入门python的notebook。刚开始的时候，所有练习是不允许用pytorch的，所以手敲函数会让你对模型的训练过程有更深刻的理解。到后面模型变复杂之后，开始使用pytroch自带的函数，跟着做完这些作业，pytorch 无非就是一个python的包而已。

-----

因为是第一次开课，我不好意思直接把作业直接放到GitHub开源。我将课件和代码上传到了谷歌云盘：https://drive.google.com/drive/folders/1gLsVDVIEdq-21RBXwKJEq5yUZW0g8kNE?usp=sharing 因为notebook直接可以在colab里运行，所以比较推荐谷歌云盘。当然如果因为网络原因无法获取，也可以在这里下载：https://owncloud.gwdg.de/index.php/s/mYPlh7rYnhypveM 

Exercise 主要包含以下模块：

<img src="/Users/liuquan/Documents/wechat/deep_learning/img-code.png" alt="image-20220731162425150" style="zoom:50%;" />

> **NOTE**: 该资源（包括notebook和PDF）著作权归David Greenberg，该分享链接仅用于学习交流。任何人未经David Greenberg本人允许，不得将该资源用于任何商业用途。

----

notebook中老师通过markdown尽量详细描述需要做的步骤，有些重要步骤会有一些代码提示，比如下面这样：

<img src="/Users/liuquan/Documents/wechat/deep_learning/img_ex.png" alt="image-20220731172746374" style="zoom:50%;" />

练习循序渐进，后面的练习，可以部分拷贝之前练习中的代码，连续做的很好。总之，是入门深度学习的完美材料。

做完上面的练习，会发现实际上pytorch非常灵活，而且文档非常完善。用到的时候只需查询所用函数（比如pytroch.conv2d) 每一个形参的意义和size（比如pytroch.conv2d的第一个参数表示输入该函数的数据的通道数，比如一次性输入前七天的SST就是in_channel=7），函数输入数据的size和每一维度的意义（比如pytroch.conv2d要求输入的数据的size依此为（batch_size, input_channels, Height, Width))，以及输出数据的size和每一维度的意义，以方便将该输出正确的喂给下一个函数。

最后结课的时候，每个人需要做一个project，虽然选课的同学大部分都只是研一的同学，每一个project都很让我惊讶。除了一些扩展上面练习04- Convolutional Networks的优秀的项目之外，有一个小组想到利用auto-encoder模型扩展数据的ensemble size。我不太了解ensemble的正确翻译应该是什么，就是在模型中，人们发现，仅对初始条件做一些微小的振荡，然后给它们同样的forcing，结果会变得非常不同，所以我们现实世界中观测到的数据，实际上，只是无数可能性中恰好出现的一种而已。当我们观察到比如温度有上升趋势的时候，如何证明是因为外界强迫导致的温度上升，而不是只是恰好出现在这一realization中的温度上升而已。但是如果我们有非常多的ensemble，将所有ensemble的数据一起分析，就会发现尽管每个ensemble的温度变化都不一样，但是每一个的长期趋势都在上升，这样，ensemble- mean就可以代表外界强迫的信号。尽管我们可以利用模型输出多个ensembles，但是还是有人会说，模型模型，怎么能证明每个ensemble的结果都是对的呢，毕竟现实里就只有一个ensembe，所以构建观测数据的ensemble就变成了一个大项目。而这个小组的想法是，不同ensemble，其空间均值和方差应该相同，而深度学习模型中有一个叫auto- encoder的模型，就是输入一个场，输出另一个场，使其具有和输入接近的mean和std。两者不谋而合，就是说，用auto-encoder可以生产已有数据的big ensemble。

至于我自己的项目，是打算用seq2seq模型预测NAO每日指数。这种项目也就在课程结课项目中做一做，现实的科研中是不敢这么选题的，因为NAO指数预测非常困难，何况是每日指数。汇报结束之后，老师也是直呼这在当下是不可能的任务。但是我喜欢这样的尝试，在我们的日常科研中，尤其是作为科研民工博士，为了产出，不敢做很大胆的尝试，但是有些做起来也挺有意思。所以我之后的公众号，也会尝试更新一些不保证成功，也不一定有用的小项目。