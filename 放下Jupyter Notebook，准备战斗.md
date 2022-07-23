# 放下Jupyter Notebook，准备战斗

从今天起：

> 开始使用GitHub；
>
> 开始使用git；
>
> 减少使用Jupyer notebook.

这是我jupyter notebook某个文件夹现在的样子：

<img src="/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/Good_Scientif_Coding/Image1.png" alt="image-20220722231350478" style="zoom:50%;" />

两个周前，因为在一个新方向苦苦探索了好久没有任何突破，我打算接着三个月前的工作继续做。但是当我回去的时候，人都麻了。首先我因为每周都需要跟导师汇报，所以每张图都有草稿，但是，我找不到出这个图的代码了。然后就是如上图所示，我跟所有看这个图的人一样，不知道哪个版本是我需要的。还有就是，即便找着了代码，有些代码已经不是很能理解了。里面各种报错信息也是满满的，当时无所谓，因为有时候只是只是一些测试报错，运行的时候跳过即可。考虑到我还只是在读博的开始阶段，这简直就是灾难。

曾经的我，因为发现了notebook这种代码+markdown的模式，激动不已，感觉科研已经站在风口，随时准备起飞。就在两个周前，看到同事们都喜欢用IDE，还很是不解。但现在，notebook似乎成了一个阻碍。首先Notebook是很方便的工具，但是也存在很多问题，比如无法调用另一notebook中的函数，这就导致我需要不停地duplicate。其他问题包括：

<img src="/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/Good_Scientif_Coding/Img2.png" alt="image-20220723100903860" style="zoom:50%;" />



以上的slides来自我这周参加的所里组织的**Good Scientific Code**的workshop。

以下是该workshop的简介：

>Scientific code is notorious for being hard to read and navigate, difficult to reproduce, and badly documented. One reason leading to this situation is that curricula that traditionally train scientists do not explicitly treat writing good code, and during the scientific life there is little time for the individual to practice this on their own. In this intensive two-day-block-workshop we will change that and teach you all you need to know to write code that is **Clear, Easy to understand, Well-documented, Reproducible, Testable, Reliable, Reusable, Extendable, and Generic**.

简单理解，就是教科研工作者，如何用软件工程的思维，提高工作效率。这实在是久旱甘霖。这个workshop目前slides在GitHub上已经开源，后续视频也会上线。

**GitHub 网址**：https://github.com/JuliaDynamics/GoodScientificCodeWorkshop.git



The workshop is divided into the following six blocks:

- **Version control**: retraceable code history using git
- **Clear code**: write code that is easy to understand and reason for
- **Software developing paradigms**: write your code like a software developer
- **Collaboration & publishing code**: modern team-based development on GitHub
- **Documenting software**: documentation that conveys information efficiently and intuitively
- **Scientific project reproducibility**: publish reproducible papers

## 

感兴趣的同学可以直接去看slides，实际上，他把slides当成字幕的。后期录屏也会发布。所以我不会复述这个workshop讲了什么，下面我罗列一些我仍然印象深刻的点：

首先是version control with git。解决的痛点就是duplication。像我上面的jupyter notebook，文件名很像的文件中，有很大一部分是我打算尝试一些新的东西，但是，不想把原代码毁掉。所以就复制一个，然后修改这个复制版本。git的理想是，一个repository中有一个多个 branch，其中只有一个main branch。main branch可以看作是你的确定版本，如果不想这个版本被修改，就新建一个branch，然后在这个branch中做修改，等到觉得没有问题了，再并入main branch。当然，这只是我浅显的理解。git的教程网上很多，以下是一个简单的开始：

-------

> 首先，想把一个文件夹变成一个可以追踪历史修改的repository，只需在command line中`cd`到这个文件夹，然后`git init`.当然，用之前，需要确保git已经安装。

> 在git中，一个文件有三种状态：modified，staged，committed。在上面建好init之后的文件夹中做一些修改，比如新建README.txt，随便敲一些内容，这时候，可以用`git status`查看文件的状态，这时候文件应该是属于modified状态，需要用`git add README.txt`将状态转变味staged。然后用`git commit -m "created the README.txt"`将文件状态转变为committed，并用message tag这次修改。这时候可以用`git log` 查看修改历史。此外，git还有一些修改commit的命令，如`git amend`, `git reset`等，可以自行查看。

> 当然有些文件我们不想被git追踪，否则就会有10w+修改，可以将其添加到.gitignore文件中。

> 接下来就是我们前面提到的branch，现在我们默认是在main branch中，新建一个branch `git branch name_of_the_branch`. 然后你想要跳转到这个new branch：`git checkout name_of_the_branch`. 当然这两步可以合并为`git checkout -b name_of_the_branch`. 在这个new branch中做的修改，将仅保留在这个branch中，当然这个修改可以merge到其他branch（当然包括main branch）中。例如使用`git rebase the-name-of-the-branch-you-want-to-merge-from`. 

上文提到的命令行对于理解git非常帮助，实际上，如果你使用IDE，git已经集成到里面了。比如vs code，在左边的栏中有git的logo，点击就可以使用。也有专门的git软件。

<img src="/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/Good_Scientif_Coding/Image3.png" alt="Screenshot 2022-07-23 at 11.41.27" style="zoom:50%;" />

但愿上面简单的命令可以让你对git提起兴趣。提到git就不得不说GitHub.今天仅是我正式接触GitHub的第三天。所以不敢多讲。但是可以接着上面对git的简单介绍，再用一个简单的例子介绍两个怎么联系起来。

> 上文我们用`git init`将一个文件夹转换成一个repository。当然你的工作可以直接从GitHub某个云端的repository开始。首先你需要fork一个Origin repository，就是说，将这个repository拷贝为您个人的repository，然后通过`git clone some-git-code-address` cope这个repository到本地。code address可以在下图中看到。
>
> <img src="/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/Good_Scientif_Coding/img5.png" alt="image-20220723120311746" style="zoom:50%;" />

> 现在`cd`到这个新clone的文件夹，就可以直接调用上文提到的git命令，添加新branch，做一些修改等。
>
> 修改完成之后，通过`git push`可以直接将修改同步到GitHub。

GitHub一个重要的操作是**pull request**。我的理解是，当你对这个repository做了一些修改之后，你也想对community做一些贡献，所以发起一个pull request，请求这个repository的实际发起人接受你的修改。这个网址可以做一个简单的小练习：https://docs.github.com/en/get-started/quickstart/contributing-to-projects



让我印象深刻的点不止是git，实际上git只是开始，GitHub只是其中的一个block。比如给变量和函数等正确命名，使每一个函数尽量简单，所有的代码块都应该是1-level，代码应该区分source code （src）和scripts，GitHub公开等。都使我受益匪浅。感兴趣又恰好有这方面烦恼的人，建议去GitHub看一看slides。

