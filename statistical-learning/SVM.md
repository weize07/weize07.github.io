### 线性可分支持向量机

对于线性可分的训练集，感知机是找到任意一个分离超平面将正负样本分开即可。支持向量机的思路是，不仅将正负样本分开，而且距离超平面最近的样本，要尽量远离超平面。

分离超平面：$\vec w \cdot \vec x+b=0$ 

判别函数：$ y=sign(\vec w \cdot \vec x+b) $



#### 问题定义

（1）样本和分离超平面的函数距离  $y_i(\vec w \cdot \vec x+b) $ 

如果$\vec w$和b的等比例扩大, 分离超平面并没有变化，而函数距离却等比例扩大了

（2）样本和分离超平面的几何距离 $ \frac{1}{||\vec w||_2}y_i(\vec w \cdot \vec x+b) $

用 $ \frac{1}{||\vec w||_2} $ 做归一化，消除等比例扩大对距离的影响



令$\gamma=\frac{1}{||\vec w||_2}y_i(\vec w \cdot \vec x+b) ,  \hat\gamma=y_i(\vec w \cdot \vec x+b) $

所以，支持向量机的优化目标为使得最小的几何距离最大：
$$
max_{\vec w,b}\gamma\\
s.t. \frac{1}{||\vec w||_2}y_i(\vec w \cdot \vec x_i+b) \geq \gamma, i=1,2,...,N
$$
即：
$$
max_{\vec w,b}\frac{1}{||\vec w||_2}\hat\gamma\\
s.t. y_i(\vec w \cdot \vec x_i+b) \geq \hat\gamma, i=1,2,...,N
$$


如果等比例地调整$\vec w,b$ 为$\lambda\vec w, \lambda b$， 那么$\hat \lambda$变成$\lambda \hat \gamma$，所以，$\hat \gamma$ 的大小对于上式没有影响，不妨设为1; 同时，$max_{\vec w,b}\frac{1}{||\vec w||_2}\hat\gamma\\$ 等同于 $min_{\vec w,b}\frac{1}{2}||\vec w||_2^2\hat\gamma\\$  ，因此，优化目标变为：
$$
min_{\vec w,b}\frac{1}{2}||\vec w||_2^2\\
s.t. y_i(\vec w \cdot \vec x_i+b) - 1\geq 0, i=1,2,...,N
$$


这是一个凸二次规划问题，如果求得该问题的最优解$ \vec w^*和 b^*$， 即可得到判别函数$y(\vec x)=sign(\vec w^*\cdot \vec x+b^*)$。



#### 对偶问题

svm对偶问题的提出主要有两个目的：

（1）引入样本间内积，从而引出核函数

（2）对偶问题往往更容易求解



构造拉格朗日函数：
$$
L(\vec w,b,\vec \alpha) = \frac{1}{2}||w||_2^2-\sum\alpha_i[y_i(\vec w \cdot \vec x_i+b) - 1]
$$
原问题是：
$$
min_{\vec w, b}max_{\vec \alpha}L(\vec w, b, \vec \alpha)
$$


根据拉格朗日对偶性，对偶问题是：
$$
max_{\vec \alpha}min_{\vec w, b}L(\vec w, b, \vec \alpha)
$$


先求极小问题：
$$
min_{\vec w, b}L(\vec w, b, \vec \alpha) \\
\frac{\partial L}{\partial \vec w} = \vec w - \sum\alpha_iy_i\vec x_i = 0 \\
\frac{\partial L}{\partial b} = -\sum\alpha_iy_i = 0 \\
\therefore \vec w = \sum\alpha_iy_i\vec x_i, \sum\alpha_iy_i = 0
$$
把上式得到 $\vec w, b$ 代入, 
$$
L(\vec \alpha) = \frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j-\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i \\
=-\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i
$$
再对 $L$ 求极大：
$$
max_{\vec \alpha}L(\vec \alpha) \\
s.t. \sum_i\alpha_iy_i = 0 \\
\alpha_i \geq 0, i=1,2,...,N
$$
该问题的最优解$\vec\alpha^*$，应当满足KKT条件：
$$
\vec w^* = \sum\alpha_i^*y_i\vec x_i \\
\sum\alpha_i^*y_i = 0  \\
\alpha_i^*[y_i(\vec w \cdot \vec x_i+b) - 1] = 0, i=1,2,...,N \\
y_i(\vec w \cdot \vec x_i+b) - 1\geq 0, i=1,2,...,N \\
\alpha_i^* \geq 0
$$
如果 $\vec a^*=0$， 那么$ \vec w^*=0 $， 这样会导致分离超平面对于样本没有任何区分能力。因此一定有分量$\alpha_i^* \neq 0$, 那么其对应的 $y_i(\vec w^* \cdot \vec x_i+b^*)-1 = 0$, 那么 $b^*=y_i^*-\vec w^* \cdot \vec x_i$



### 线性支持向量机

上述的线性可分支持向量机，无法处理数据集线性不可分的情况，为此，我们为每个样本引入松弛变量，并对松弛变量加以惩罚。



#### 问题定义

$$
min_{\vec w,b}\frac{1}{2}||\vec w||_2^2 + C\sum_i \varepsilon_i\\
s.t. y_i(\vec w \cdot \vec x_i+b) \geq 1-\varepsilon_i, i=1,2,...,N\\
\varepsilon_i \geq 0, i=1,2,...,N
$$



其中$\varepsilon_i$是松弛变量；$C$ 是惩罚系数，用来调和**间隔最大化**和**误分类惩罚**对于优化目标的重要程度。



#### 对偶问题

构造拉格朗日函数：
$$
L(\vec w, b, \vec \varepsilon, \vec \alpha, \vec \mu) =
\frac{1}{2}||\vec w||_2^2 + C\sum_i \varepsilon_i - 
\sum_i\alpha_i[y_i(\vec w \cdot \vec x_i+b)-1+\varepsilon_i] - 
\sum_i\mu_i\varepsilon_i \\
s.t. \alpha_i \geq 0, \mu_i \geq 0, \varepsilon_i \geq 0,  i=1,2,...,N
$$


原始问题：
$$
min_{\vec w, b, \vec \varepsilon}max_{\vec \alpha, \vec u}L(\vec w, b, \vec \varepsilon, \vec \alpha, \vec \mu)
$$
对偶问题：
$$
max_{\vec \alpha, \vec u}min_{\vec w, b, \vec \varepsilon}L(\vec w, b, \vec \varepsilon, \vec \alpha, \vec \mu)
$$
先对$\vec w, b, \vec \varepsilon$ 求极小：
$$
\nabla_{\vec w}L = \vec w-\sum_i\alpha_iy_i\vec x_i = 0 \Rightarrow \vec w = \sum_i\alpha_iy_i\vec x_i\\
\nabla_{b}L = -\sum_i\alpha_iy_i = 0 \\
\nabla_{\varepsilon_i} = C - \alpha_i - \mu_i = 0
$$
将上面式子代入$L$, 得到：
$$
L(\vec w, b, \vec \varepsilon, \vec \alpha, \vec \mu) =
\frac{1}{2}||\vec w||_2^2 + C\sum_i \varepsilon_i - 
\sum_i\alpha_i[y_i(\vec w \cdot \vec x_i+b)-1+\varepsilon_i] - 
\sum_i\mu_i\varepsilon_i \\
= -\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i
$$
再对 $\vec \alpha, \vec \mu$ 求极大：
$$
max_{\vec \alpha, \vec \mu}-\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i \\
等价于: min_{\vec \alpha}\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i\\
s.t. C \geq \alpha_i \geq 0 \\
\sum_i\alpha_iy_i = 0 \\
$$
最优解$\vec w^*, b^*, \vec \varepsilon^*, \alpha^*, \vec \mu^*$ 应满足KKT条件：
$$
\vec w = \sum_i\alpha_iy_i\vec x_i \\
\sum_i\alpha_iy_i = 0 \\
C-\alpha_i-\mu_i = 0 \\
C \geq \alpha_i \geq 0 \\
\varepsilon_i \geq 0 \\
\mu_i \geq 0 \\
\mu_i\varepsilon_i = 0 \\
\alpha_i[y_i(\vec w \cdot \vec x_i+b)-1+\varepsilon_i] = 0 \\
y_i(\vec w \cdot \vec x_i+b)-1+\varepsilon_i \geq 0 \\
i = 1,2,...,N
$$
如果$\vec \alpha = \vec 0$, 那么 $\vec w=\vec 0$， 模型将不具备分类能力，没有意义。

如果$\alpha_i = C, i=1,2,...,N $, 那么$\sum_iy_i = 0$,  这是强加的约束，不符合实际情况。

所以存在$0<\alpha_j<C$， 因此$\mu_j > 0, \varepsilon_j=0$,  那么
$$
b^* = y_j-\vec w^*\cdot \vec x_j\\
=y_i-\sum_i\alpha_i^*y_i(\vec x_i \cdot \vec x_j)
$$
那么分离超平面是：$\sum_i\alpha_i^*y_i(\vec x_i \cdot \vec x) + b^* = 0$

决策函数为：$f(\vec x) = sign(\sum_i\alpha_i^*y_i(\vec x_i \cdot \vec x) + b^*)$



#### 支持向量

对偶问题的最优解$\vec \alpha^*=(\alpha_1^*, \alpha_2^*,...,\alpha_N^*)^T$中，大于0的分量 $\alpha_i^*$   对应的样本点$(\vec x_i, y_i)$  的特征向量$\vec x_i$ 被称为支持向量。

其中，如果$0<\alpha_i^* < C$, 那么$\mu_i > 0, \varepsilon_i=0$, 所以 $\vec x_i$ 在支持向量边界上。

如果$\alpha_i^*=C$， 那么$\varepsilon_i$ 为任意正数，如果

* 如果$0<\varepsilon_i<1$， 则$\vec x_i$ 在支持向量边界和分离超平面之间，分类正确。
* 若 $\varepsilon_i=1$，则支持向量落在分离超平面上。
* 若 $\varepsilon_i>1$，则支持向量落在分离超平面另一侧，分类错误。



#### 核函数

截至目前，支持向量机只能用于解决线性分类问题，如果我们想要解决非线性问题，就需要用到核函数了。

注意到之前的最优化问题：
$$
max_{\vec \alpha, \vec \mu}-\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i \\
等价于: min_{\vec \alpha}\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j \vec x_i \cdot \vec x_j+\sum_i\alpha_i\\
s.t. C \geq \alpha_i \geq 0 \\
\sum_i\alpha_iy_i = 0 \\
$$
如果我们用非线性变换$h(\vec x_i)$代替 $\vec x_i$, 那么内积$\vec x_i \cdot \vec x_j$ 就成了$h(\vec x_i) \cdot h(\vec x_j)$。

举个例子，假设$\vec x = (x_1,x_2)^T, h(\vec x)=x_1^2+x_2^2$ ， 那么对于以原点为圆心，任一半径的圆上的点，将在$h(\vec x)$的映射下变得没有区别。两两之间的内积也变得没有区别，跟原始空间里的内积已经不一样了。

当然，一般不会显式定义变换函数，而是用一个核函数$K(\vec x_i, \vec x_j)$来代替内积$\vec x_i \cdot \vec x_j$。

核函数隐式地完成非线性变换，再在变换后的空间中求得两点的内积。

常用的核函数有：

**多项式核函数:** $K(\vec x_i, \vec x_j)=(\vec x_i\cdot\vec x_j + 1)^p $

**高斯e核函数：**$K(\vec x_i, \vec x_j)=exp(-\frac{||\vec x_i-\vec x_j||^2}{2\sigma^2}) $



用核函数代替了样本内积之后，最优化问题变成了：
$$
max_{\vec \alpha, \vec \mu}-\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j K(\vec x_i , \vec x_j)+\sum_i\alpha_i \\
等价于: min_{\vec \alpha}\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j K(\vec x_i , \vec x_j)+\sum_i\alpha_i\\
s.t. C \geq \alpha_i \geq 0 \\
\sum_i\alpha_iy_i = 0 \\
$$
决策函数变成：$f(\vec x) = sign(\sum_i\alpha_i^*y_iK(\vec x_i , \vec x) + b^*)$



#### SMO算法

1. 支持向量机的学习问题可以形式化为求解凸二次规划问题。这样的凸二次规划问题具有全局最优解，并且有多种算法可以用于这一问题的求解。

   当训练样本容量非常大时，这些算法往往非常低效。而序列最小最优化(`sequential minimal optimization:SMO`）算法可以高效求解。

2. `SMO`算法的思路：

   - 若所有变量都满足条件，则最优化问题的解就得到了。

   - 否则，选择两个变量的同时固定其他所有变量，针对这两个变量构建一个二次规划子问题。

     - 这个二次规划子问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。

     - 更重要的是，这个二次规划子问题可以通过解析的方法求解。

     - 此时子问题有两个变量，至少存在一个变量不满足约束条件（否则就是所有变量满足条件了）。

       假设其中一个是违反约束最严重的那个，另一个由约束等式自动确定： 。

3. `SMO` 算法将原始问题不断地分解为子问题并且对子问题求解，进而达到求解原问题的目的。

   整个 `SMO` 算法包括两部分：

   - 求解两个变量二次规划的解析方法。
   - 选择变量的启发式方法。