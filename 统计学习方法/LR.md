### 逻辑回归

#### 问题描述

考虑二分类问题，给定数据集 $ \mathbb D=\{(\vec x_1, y_1),(\vec x_2, y_2),…,(\vec x_N, y_N)\}, \vec x_i \in \mathbb R^n, y_i\in\{0,1\} $

对于回归问题，我们可以用一个线性函数 $f(x)=\vec w\cdot\vec x+b$ 来对数据集进行拟合；

但是，对于二分类问题，$ y_i \in \{0,1\} $, 然而，上述函数$ f(x) \in (-\infin,+\infin) $， 因此我们需要把f(x)的区间映射到 

{0, 1} 。

一种可行的方程是：
$$
g(x)=
\left\{
	\begin{array}{**lr**}
	1, f(x)>0\\
	0.5, f(x)=0\\
	0, f(x)<0
	\end{array}
\right.
$$
但是，该方程是不可导的，因此不适合用梯度下降等方式进行参数更新。

换个思路，如果我们希望建模$ p(y=1|z), z=\vec w\cdot\vec x+b $， 需要怎么做呢？

考虑sigmoid函数$ \sigma(z)=\frac{1}{1+e^{-z}} $, 其中
$$
limit_{z \to +\infin}\sigma(z) = 1\\
limit_{z \to -\infin}\sigma(z) = 0\\
\sigma(0)=0.5
$$
sigmoid图形：

![](./images/sigmoid.png)

所以，sigmoid起到的作用就是，把$ \vec w \cdot \vec x+b $ 从$ (-\infin,+\infin) $ 映射到了(0,1)区间。

我们可以把$\sigma(\vec w \cdot \vec x_i+b)$ 的值理解为，该样本$ \vec x_i $  属于正样本的概率；

而$ 1-\sigma(\vec w \cdot \vec x_i+b) $ 是该样本属于负样本的概率。

那么，我们的学习目标就变成了，对于训练集 $\mathbb D $ 中样本正负标签出现的概率尽可能大，即似然函数：
$$
Likelihood = \prod_{i=1}^{N}(\sigma(\vec w \cdot \vec x_i+b))^{y_i}(1-\sigma(\vec w \cdot \vec x_i+b))^{1-y_i}
$$
为了更方便计算，通常取对数似然：
$$
\begin{equation}
\begin{aligned}
L(\vec w, b) &= log(Likelihood) \\
						 &= \sum_{i=1}^{N}(y_ilog\frac{\sigma(\vec w \cdot \vec x_i+b)}{1-\sigma(\vec w \cdot \vec x_i+b)} + log(1-\sigma(\vec w \cdot \vec x_i+b)) \\
						 &= \sum_{i=1}^{N}(-y_i(\vec w \cdot \vec x_i+b)+(\vec w \cdot \vec x_i+b)-log(1+exp(\vec w\cdot \vec x_i+b)))
						 
\end{aligned}
\end{equation}
$$
使用梯度下降法更新参数$ \vec w, b $。
$$
\begin{equation}
\begin{aligned}
\nabla_{w_i} &= \sum_{i=1}^{N}(-y_i\vec x_i+\vec x_i-(1-\sigma(\vec w \cdot \vec x_i+b))\vec x_i)\\
							&= \sum_{i=1}^{N}(\sigma(\vec w\cdot \vec x_i+b)-y_i)\vec x_i\\
\nabla_b &= \sum_{i=1}^{N}(1-y_i-(1-\sigma(\vec w \cdot \vec x_i+b)))\\
					&= \sum_{i=1}^{N}(\sigma(\vec w\cdot \vec x_i+b)-y_i)
\end{aligned}
\end{equation}
$$
每轮迭代：

$ \vec w_i \leftarrow \vec w_i + \nabla_{w_i} $

$ b \leftarrow b + \nabla_b $