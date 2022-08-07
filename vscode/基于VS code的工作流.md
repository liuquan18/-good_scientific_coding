# 基于VS code的工作流

> 在之前的推送中介绍了**Good Scientific coding**的workshop，最近收到邮件，当时workshop的录屏已经上传，有兴趣的同学可以尝试观看。YouTube video recording: https://youtu.be/x3swaMSCcYk

我们大部分人都已经慢慢有了自己的工作方式，比如主要用哪个软件编程，也有了自己的工作流，编程，出图，保存，展示等。但是如果回头看，这些方式实际上是在最开始接触科研时选择的工具上慢慢发展的。就是说，并不代表一定是高效的方式。

本文主要分享自己最近的编程体验。主要包括：

> - 使用vs code作为主力编程软件。
> - 使用GitHub同步代码和控制版本。
> - 使用Latex保存图件和笔记，并展示讨论。
> - 使用outlook的日历和task工具控制工作进度。
> - 使用safari浏览器的sidebar优化网页管理。

我记得我在研一的时候选修过一门《python 空间数据分析》的课，老师主要介绍gdal。现在想来当时老师也是奇葩地很，他说他见过的国外研究人员，都直接选择在**命令行**里敲python。所以他在课上推崇我们在命令行里敲`python`, 然后开始`import`，完事了还得`exit()`。现在仍然记得自己当时Linux和python代码混在一起分不清楚的恐惧。后来也在其他地方见过什么评论说大神都用文本编辑器敲代码之类的逼话。

我都不想加限定条件，正经敲代码，选择合适的工具（不包括命令行和文本编辑器）是第一步。因为Markdown和python代码的完美结合，以及逐行运行输出的特点，**jupyter** notebook成了大部分科研人员的首选，之前的推送中已经指出过这个工具的诸多不便。也许另外一个经常听说的软件叫**anaconda**，但是anaconda本身并不是一个敲代码的地方，对我来说，安装它就是为了用`conda`, 我用conda也就干两件事，`conda create`新建一个环境，`conda install`新安装一个包。前者在你电脑上新建一个全新的python环境，这样在这个环境里安装什么包都不会影响其他环境，后者相比于`pip`安装，优势在于会自动安装该包依赖的其他包。当然windows上anaconda会绑定安装一个叫**Spyder**的IDE，也有人用它编程。这里就提到了**IDE**，全称叫**integrated development environment** (IDE)，也可以看作是用于编程的一类软件的总称。如果我们排除text editor，IDE有时候也被看作是notebook的对立面。但实际上在好些IDE内部也是可以使用notebook的。

Python编程常听说的IDE有**pycharm**和**Visual Studio Code (vs code)**，前者专门用于python开发，后者算是一个集成的平台，各种语言都很方便。pycharm功能非常全面，但是也非常复杂，比如常见的Find功能  `ctrl+F`这种操作，在里面也能玩出花来。而vs code介绍的第一句就包括*light weight*，轻量化的软件，做简单真是一件非常难的事。所以本文推荐的IDE是vs code.

正如开头说的，熟悉的方式并不是高效的方式，我的方式当然也不是你高效的方式。因此有必要交代一下我的硬件环境。首先我需要处理大量数据，而这些数据存储在我们所（租用）的超算上(DKRZ, German Climate Computing Center德语首字母，说是租用，实际上DKRZ和马普气象所的初代所长都是Klaus Hasselmann, 2021年诺贝尔物理学奖得主，我们更多将DKRZ看作是一个部门)，同时处理这些数据也完全超出了本地算力的能力。其次，我用的是MacBook，之前主用Linux，希望可以更加专注。但是Linux和其他设备的同步问题一直没有解决，所以就转到MacBook，MacBook的触控板多桌面切换简直好用。还有一些软件支持，比如gwdg提供的在线markdown和latex工具等，可以很容易实现多设备同步。

将**vs code作为主力开发工具**，

