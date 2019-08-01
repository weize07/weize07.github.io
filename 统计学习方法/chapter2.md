### 感知机

#### 问题定义

给定数据集 $D= \{(\vec{x_1}, y_1), (\vec{x_2}, y_2),..., (\vec{x_N}, y_N)\}, \vec{x_i} \in \mathcal{X} \subseteq \mathbb{R}^n, y_i \in \mathcal{Y} = \{+1,-1\}$ , 其中$ \vec x_i $ 是特征向量，$ y_i  $ 是标签，表示二分类问题中的正、负两类样本。

需要找出一个分离超平面 $ f(\vec x) =\vec w \cdot \vec x + b$，使得对于任意的$x_i, y_i, 满足 y_i(\vec w \cdot \vec x_i+b) > 0$

最终的判别函数是 $sign(\vec w \cdot \vec x + b)$



#### 求解方法

首先定义损失函数如下：

如果$ (\vec x_i, y_i)  $ 被误分类，则 $ y_i $ 和 $ (\vec w \cdot \vec x_i+b)$ 异号，且我们用误分类点距离超平面的距离 $ -\frac{1}{||w||}(\vec w \cdot \vec x_i+b) $ 来表示分类错误的程度（损失），因此定义单个数据点上的损失如下：

$$ L(\vec x_i, y_i)=-\frac{1}{||w||}y_i(\vec w \cdot \vec x_i+b) $$

所有数据点上的累计损失为：$\mathcal{L}(D)=\sum_{i\in\mathcal{M}} L(\vec x_i, y_i), \mathcal{M}:误分类点集合$

因为最终的优化目标是$ \mathcal{L}(D)=0 $, 所以，系数$ \frac{1}{||w||} $ 可以略去，变成：$$ L(\vec x_i, y_i)=-y_i(\vec w \cdot \vec x_i+b) $$

如果用随机梯度下降法，对于每一个误分类点首先计算梯度：

$ \frac{\partial L}{\partial{\vec w}} = -y_ix_i $

$ \frac{\partial L}{\partial b} = -y_i$

并按照学习率 $ \eta $ 更新梯度:

$ \vec w = \vec w+ \eta y_ix_i$

$ b = b+ \eta y_i$



#### 算法描述

- 选取初始值 $\vec w_0, b_0$, 设置学习率$ \eta \in(0,1] $ 。

- 在训练集中选取数据$ (\vec x_i, y_i) $,若 $ y_i(\vec w \cdot \vec x_i+b)<0 $ 则更新 $ \vec w, b $:

  $ \vec w \leftarrow \vec w+ \eta y_ix_i$

  $ b \leftarrow b+ \eta y_i$

- 在训练集中重复选取数据来更新 $ \vec w, b $  直到训练集中没有误分类点。



#### 算法收敛性

为了方便，首先将偏置b并入权重向量 $ \vec w $,  计作 $ \hat w = (\vec w^T, b)^T$, 同样，$ \vec x $ 增加常数1，计作 $\hat x = (\vec x^T, 1)^T $。

（1）对于线性可分的数据集，存在一个分离超平面$ \hat w_{opt} \cdot \vec x=0 $，其中$ ||\hat w_{opt}||=1 $将所有样本全部正确分开，即$ \forall i \in D, y_i(\hat w_{opt} \cdot \vec x_i) > 0 $, 那么，显然存在$ \gamma>0, 使得 y_i(\hat w_{opt} \cdot \vec x_i) \ge \gamma$ 

（2）令$ R = max_i||\hat x_i||$, 下面要证明：感知机误分类次数k满足不等式 $ k \le (\frac{R}{\gamma})^2$

首先证明 $ \hat w_k \cdot \hat w_{opt} \ge \eta k \gamma $ :
$$
\begin{equation}
\begin{aligned}
\hat w_k \cdot \hat w_{opt} &= (\hat w_{k-1} + \eta y_i \hat x_i)\cdot \hat w_{opt}\\ 
&=\hat w_{k-1} \cdot \hat w_{opt} + \eta y_i\hat w_{opt} \hat x_i\\
&\ge \hat w_{k-1} \cdot \hat w_{opt} + \eta\gamma\\
&\ge ...\\
&\ge k\eta\gamma
\end{aligned}
\end{equation}
$$
再证明 $ ||\hat w_k||^2 \le k \eta^2  R^2$ :
$$
\begin{equation}
\begin{aligned}
||\hat w_k||^2 &= ||\hat w_{k-1}||^2 + 2\eta y_i \hat w_{k-1} \hat x_i + \eta^2||\hat x_i||^2\\ 
&\le ||\hat w_{k-1}||^2 + \eta^2R^2 \\
&\le ... \\
&\le k\eta^2R^2
\end{aligned}
\end{equation}
$$
所以，$ \eta k \gamma \le \hat w_k \cdot \hat w_{opt} \le ||\hat w_k||\cdot||\hat w_{opt}|| \le \sqrt k\eta R $

所以,  $ k \le (\frac{R}{\gamma})^2 $

