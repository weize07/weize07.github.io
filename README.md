1. 对于任意一个客观存在的物体，我们为什么要用向量去表示？

   例如，每个人有很多种属性——年龄、身高、体重、发量、房产套数、代码行数……

   如果我们把每一种属性都认为是一个坐标轴，

   那么，"我”这个人，此时此刻就可以表达为：[30, 180, 180, 1800000000000, 0, 10086]，它是一个什么？没错，就是向量，一个高维空间的向量。 所以，我认为，我是一个高维的向量。

   有了这种向量表示的一个个物体（"样本”，例如一群人），以及我们关于每个物体在某一方面的结论（“标签”，例如是否已婚）。机器学习的那些各式各样的模型才可以进行“学习”——

   决策树说，我，擅长写if else。比如，按照房产套数切一刀，2套以上的择偶成功率提高30%，而按照代码行数来切一刀，好像完全没有区别……

   SVM说，我，先把你这个向量通过歪七扭八的方式映射到一个我自己都不知道在哪儿的高维空间，在这个歪七扭八的映射之后，你会和一些意想不到人划为了一类（geek），发现你们心灵的距离原来是如此的贴近，而和别的群体相隔千里:  kernel_geek(你, geek1) = 0.001，而geek的择偶成功率……对不起打扰了。

   神经网络（万能函数拟合器）说，给我一堆向量X，输出y和足够多的隐层节点，我能拟合世界——你看，这个世界就是： f1(f2(f3(……fn(X)))) = y。想知道我为什么work？听过《洋葱》么？你需要一层一层地剥开我的心。

   当然，有些物体用向量表达是否合理，还未可知。比如图像的原始数据。虽然，目前常见的图像算法，经过N层卷积之后，还是会提取相对抽象的向量表达。

   扯远了，首先需要知道的第一点，最常见的情况下，我们用向量去描述物体，无论它是客观的还是抽象的。

   

2. 自然语言表示的一些背景

   首先，自然语言的目的，一般是表达"意思"，或者说是传递信息。当然，"信息"和"意思"这两个词本身就太过宽泛，只能进行一些粗浅的讨论。

   字符串本身有"意思"吗？考虑下面这个例子：

   "我喜欢猫”，想象一个小孩，没有见过猫，你把这句话给他看，能完整地传递信息吗？

   也许他会问

   "猫是什么？"

   "一种毛茸茸的动物。"

   如果小孩也不知道"毛茸茸" "动物"是什么意思，那么解释将会非常困难。

   再举一个例子，想象你走进了一间只有泰语书的图书馆，这些书没有任何的插图，只有文字。把他们都看一遍，你能学到什么？

   所以，我想表达的意思是，**字符本身并没有任何的含义，除非它和现实世界中的物体产生了联系，或者通过和其他字符的联系间接地联系到世界**。这也是所谓的Knowledge Grounding。

   

   当然，我们并不是一定要从最基本的东西一点一点教计算机。另一种方法是，通过某种方式先建立一套"世界规则"，这个就是符号逻辑推理系统和知识图谱等领域正在做的事情，而这些，存在着**稀疏性**（世界那么大，我们整理不过来）和**语义解析**（semantic parsing）等困难的问题，在此不进一步展开。

   ![img](https://cdn-images-1.medium.com/max/1200/1*9lcDx9SLPpdF4F0iS868CA.png)

    									    Meaning representation:  𝑚𝑎𝑥(𝚙𝚛𝚒𝚖𝚎𝚜 *∩(−∞; 𝟣𝟢))

   上图是符号逻辑表示自然语言的一个例子

   需要：

    (i) 把自然语言映射到关于数学知识的知识库 

   (ii) 把知识片段整合到一起以"推理"得到答案 

   至此，我们大概引出了在自然语言中，"表示"所要发挥的作用：

   （1）把自然语言和世界（知识库）、观察、动作相关联

   （2）能够支持推理（注意，不一定是符号逻辑），能够使知识片段之间产生关联（例如，说到蔡徐坤就想到WCUBA）。

   嗯，好大的两句废话，哈哈哈哈。

   （3）无歧义性

   例如，"苹果"可以是用来吃的，也可以是用来给吃的拍照的，我们需要根据上下文（甚至视觉输入）决定它的含义，例如：

   "看，他在吃苹果！”（请别跟我杠，说你见过人吃手机……）

   "他上月花1w块买了个苹果，现在卖8000了哈哈哈😂"

   

   ![img](http://blog.openkg.cn/wp-content/uploads/2018/08/word-image-186.jpeg)

   甚至，如上图所示，有些词的"意思"本身，也是随着时间迁移在不断变化的。

   

3. 自然语言的向量表示

   阶段性总结一下，上面我们提到了两点：

   （1）我们常常用向量 "表示" 物体

   （2）自然语言需要有合理的表示

   那么很自然地，就引出了 "用向量来表示自然语言中的单词" 的需求。

   为此，人们尝试过：

   （1）one-hot编码

   例如，假如我只关心这个世界上的四个字："噼里啪啦"，那么我可以将它们表示为：

   噼：[1,0,0,0]

   里：[0,1,0,0]

   啪：[0,0,1,0]

   啦：[0,0,0,1]

   乍一看，还行哈？

   其实呢，too simple, sometimes naive.

   simple并有没有错，我们一直以来都在追寻简单而有效的理论或者方法。

   然而，上面这种one-hot有两个问题，显得很naive：

   **问题一：维度爆炸及稀疏问题**

   在one-hot的设定下，每个词都是一个独立的维度，那么我们日常生活中有多少个词呢？光英语四级就4000了……这就意味着我们的维度数会是一个很庞大的数字。另外，有一些词并不多见（比如你能读出饕餮我就嘿嘿嘿），所以意味着在它这个维度上，数据是稀疏的。

   **问题二：词之间的距离全部一样**

   词与词的关系是不同的，例如：（1）**哺乳动物**和**猫星人** （2）**哺乳动物**和**航天飞机**，这两对你觉得哪个关联性更大呢？

   那么，我们看看，在之前的例子中，词向量之间两两正交，意味着，他们之间的距离没有任何区别，甚至，两两之间压根一点关系都没有。

   (2) count-based编码

   ![img](http://blog.openkg.cn/wp-content/uploads/2018/08/word-image-169.jpeg)

   就是说，我发现，stars和shining一起出现了38次，和bright一起出现了45次……

   这种方法，一定程度上解决了词与词之间相似性的问题，但是还不够好。

   而且，维度，依然是高的，某些维度上，数据依然是稀疏的。

   （3）word2vec

   在word2vec的方法中，词被映射为了**低维、稠密、实数值**的向量。

   那么，怎么做映射呢，假设我们的词汇量是V，想要映射为N(N<<V)维向量，那么我们通过学习如下的一个扁平的网络结构：一层编码（W），一层解码（W'）。

   ![alt txt](https://nlpoverview.com/img/CBOW.png)

   中间这个**比较小的N，常见的有100、300、500等，就是低维**；

   那么，怎么捕获词与词之间的相关性呢？

   如果熟悉神经网络会知道，只要给了输入输出，冷酷无情的BP算法能够荡平一切，找到崎岖空间里的（局部）最优点，逼近出一个还不错的映射函数。

   所以，这个负责映射单词的网络，应该怎么训练呢？

   word2vec给出的方案是，

   ![diagrams](https://skymind.ai/images/wiki/word2vec_diagrams.png)

   （1）CBOW：W'WSum(word_nearby) ~= word，即句子中单词t附近几个单词t-2、t-1、t+1、t+2求和之后经过网络的映射，输出应该约等于单词t。

   即，任务内容是根据一定范围内的上下文预测中间的单词

   为什么有效呢？举个例子，比如，I like to eat ___ because it's juicy， 中间预测apple、orange、watermelon是不是都还行？ 中间预测个dog、cat好像就有点奇怪了？所以说，意思接近的单词在某些句式中具有一定的相互替代性，也就是说，**如果某两个单词常出现的上下文比较类似，那么它经过word2vec表示之后就比较接近**。

   甚至，我们发现，网络学好之后，映射得到的词向量有一些奇妙的特性，举个例子：

   ​									**King-Man+Woman~=Queen**

   

   ![alt txt](https://nlpoverview.com/img/distributional.png)

   

   ​							**国家和对应的首都之间的距离好像都差不多？**

   ![img](http://blog.openkg.cn/wp-content/uploads/2018/08/word-image-181.jpeg)

   ​				**单词和它对应的"父类型"单词距离接近，且自发地聚簇。**

   ![img](http://blog.openkg.cn/wp-content/uploads/2018/08/word-image-182.jpeg)

   （2）Skip-gram，原理差不多，用单词t预测上下文。

   如果我们在大批量的文本上（例如wikipedia）训练了word2vec嵌入模型，则该网络就能够比较大概率上拟合出一个能够体现词与词相关性的映射函数。

   

   这里再提一下分布式表示的概念，意思就是，词由多个维度的信息组合而成

   那么，one-hot并不是分布式表示，count-based是离散分布式表示，而word2vec是**实值连续分布式表示**

   从左至右，表示越来越稠密，而可解释性越来越低（word2vec的每个维度是什么？）。

   

4. 疑问

   写到这里，我对自己之前的论调又有了不同的看法。

   上面说到，"**字符本身并没有任何的含义，除非它和现实世界中的物体产生了联系，或者通过和其他字符的联系间接地联系到世界**。" 真的是这样吗？

   事实上，诸如word2vec的词嵌入方法的输入仅仅是文本而已。模型阅读了海量的纯文本，通过建立具有预测上下文的能力，学习一个能够将单词嵌入低维稠密空间的模型。且在这个空间中，相似的词之间居然开始形成团簇了。

   这就好比，跟小孩说：

   "我喜欢养猫"，"我喜欢养狗"，"我喜欢养植物''……  即使不去弄明白猫、狗、植物分别是什么，至少能够知道，它们之间在一些场景下有关联。

   所以，**即使是大量的纯文本，也是有信息的，因为纯文本中的词的组合并不是真正随机分布的，句子有特定的语法结构，而常说的话，也有很大的统计学偏差（bias）**。

