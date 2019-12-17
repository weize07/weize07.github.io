### EM算法核心

在一般的问题中，我们通常定义由参数$\theta$ 决定的概率密度函数$P(\bold x;\bold \theta)$, 然后在数据集$\bold X=\{\bold x_1,\bold x_2,...,\bold x_N\}$上求极大对数似然：
$$
\mathop{\arg\max}_{\bold \theta} log(\prod_i P(\bold x_i;\theta))\\
=\mathop{\arg\max}_{\bold \theta}\sum_ilog(P(\bold x_i;\theta))
$$
但是在有些问题中，存在不可观测的隐变量$\bold z$，问题就变成了
$$
\begin{align*}
\mathop{\arg\max}_{\bold \theta}\sum_ilog(\sum_{z_i}P(\bold x_i,\bold z_i;\theta)) 
\end{align*}
$$
这个问题我们无法用极大似然去求解，那么就需要用一些技巧了。

先引入一个未知的分布$Q(z_i),其中\sum_iQ(z_i) = 1$, 将上式改写为：
$$
\mathop{\arg\max}_{\theta}\sum_ilog(\sum_{z_i}P(\bold x_i,\bold z_i;\theta)) \\
= \mathop{\arg\max}_{\theta}\sum_ilog(\sum_{z_i}Q(z_i)\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})\\
\geq \mathop{\arg\max}_{\theta}\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})
$$
其中,  $\geq$ 是由Jensen不等式推导得到：对于上凸函数$f(x), f(E(x)) \geq E(f(x))$。

对于上式，仅在$\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)} = C$ 的情况下， 可以使等号成立。

而因为 $\sum_iQ(z_i) = 1$, 所以：
$$
Q(z_i) = \frac{P(\bold x_i,\bold z_i;\theta)}{\sum_{z_i}P(\bold x_i,\bold z_i;\theta)}\\
=P(\bold z_i|\bold x_i;\theta)
$$
所以，当$Q(\bold z_i)=P(\bold z_i|\bold x_i;\theta)$ 时，$\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})$ 就是原问题 $\sum_ilog(\sum_{z_i}P(\bold x_i,\bold z_i;\theta))$ 的一个下界，且此时两者相等。那么如果我们能使得下界$\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})$ 变大，那么原问题一定也会变大。



