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

先引入一个未知的分布$Q(z_i),其中\sum_{z_i}Q(z_i) = 1$, 将上式改写为：
$$
\mathop{\arg\max}_{\theta}\sum_ilog(\sum_{z_i}P(\bold x_i,\bold z_i;\theta)) \\
= \mathop{\arg\max}_{\theta}\sum_ilog(\sum_{z_i}Q(z_i)\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})\\
\geq \mathop{\arg\max}_{\theta}\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})
$$
其中,  $\geq$ 是由Jensen不等式推导得到：对于上凸函数$f(x), f(E(x)) \geq E(f(x))$。

对于上式，仅在$\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)} = C$ 的情况下， 可以使等号成立。

而因为 $\sum_{z_i}Q(z_i) = 1$, 所以：
$$
Q(z_i) = \frac{P(\bold x_i,\bold z_i;\theta)}{\sum_{z_i}P(\bold x_i,\bold z_i;\theta)}\\
=P(\bold z_i|\bold x_i;\theta)
$$
所以，当$Q(\bold z_i)=P(\bold z_i|\bold x_i;\theta)$ 时，$\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})$ 就是原问题 $\sum_ilog(\sum_{z_i}P(\bold x_i,\bold z_i;\theta))$ 的一个下界，且此时两者相等。那么如果我们能使得下界$\sum_i\sum_{z_i}Q(z_i)log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)})$ 变大，那么原问题一定也会变大。

EM算法拆分成两步：

1. E步，给定$ \theta^{(t)} $ 的情况下, 求得分布$ Q(z_i)^{(t)}=P(\bold z_i|\bold x_i;\theta^{(t)}) $使得下界等于原问题。

   并且基于新的分布$ Q(z_i)^{(t)} $求得$ log(P(\bold x_i, \bold z_i;\theta)) $的期望:
   $$
   \sum_i\sum_{z_i}Q(z_i)^{(t)}log(\frac{P(\bold x_i,\bold z_i;\theta)}{Q(z_i)^{(t)}})\\
   =\sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold x_i,\bold z_i;\theta))-const
   $$

2. M步，对上述问题求极大，即
   $$
   \theta^{(t+1)}=\mathop{\arg\max}_{\theta}\sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold x_i,\bold z_i;\theta))
   $$

初始化的时候可以对随机获得$\theta^{(0)}$, 重复E-M步直至收敛。



### EM算法收敛性证明

要说明算法收敛，只需证明:
$$
\sum_ilog(P(\bold x_i;\theta^{(t+1)})) >=\sum_ilog(P(\bold x_i;\theta^{(t)}))
$$
首先，我们对式子做一下变形：
$$
\sum_ilog(P(\bold x_i;\theta))\\ 
= \sum_i\sum_{z_i}Q(\bold z_i)^{(t)} log(P(\bold x_i;\theta))\\
= \sum_i\sum_{z_i}Q(\bold z_i)^{(t)} log(\frac{P(\bold x_i,\bold z_i;\theta)}
{P(\bold z_i|\bold x_i;\theta)})\\
= \sum_i\sum_{z_i}Q(\bold z_i)^{(t)} log(P(\bold x_i,\bold z_i;\theta) - 
\sum_i\sum_{z_i}Q(\bold z_i)^{(t)} log(P(\bold z_i|\bold x_i;\theta))\\
$$
于是令：
$$
L(\theta,\theta^{(t)}) = \sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold x_i,\bold z_i;\theta))\\
H(\theta,\theta^{(t)}) = \sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold z_i|\bold x_i;\theta))\\
$$
因此：
$$
\because \sum_ilog(P(\bold x_i;\theta))= L(\theta, \theta^{(t)}) - H(\theta, \theta^{(t)})\\
\therefore \sum_ilog(P(\bold x_i;\theta^{(t+1)})) - \sum_ilog(P(\bold x_i;\theta^{(t)}))\\
= [L(\theta^{(t+1)}, \theta^{(t)}) - L(\theta^{(t)}, \theta^{(t)})] - [H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})]
$$
由于$\theta^{(t+1)}$使得$L(\theta, \theta^{(t)})$ 极大，所以:
$$
L(\theta^{(t+1)}, \theta^{(t)}) - L(\theta^{(t)}, \theta^{(t)}) \ge 0
$$
而:
$$
H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)}) \\
= \sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold z_i|\bold x_i;\theta^{(t+1)}))
  - \sum_i\sum_{z_i}Q(z_i)^{(t)}log(P(\bold z_i|\bold x_i;\theta^{(t)})) \\
= \sum_i\sum_{z_i}Q(z_i)^{(t)}log(\frac{P(\bold z_i|\bold x_i;\theta^{(t+1)})}{P(\bold z_i|\bold x_i;\theta^{(t)})}) \\
\le \sum_ilog(\sum_{z_i}Q(z_i)^{(t)}\frac{P(\bold z_i|\bold x_i;\theta^{(t+1)})}{P(\bold z_i|\bold x_i;\theta^{(t)})}) \\
= \sum_ilog(\sum_{z_i}Q(z_i)^{(t)}\frac{P(\bold z_i|\bold x_i;\theta^{(t+1)})}{Q(z_i)^{(t)}}) \\ 
= 0
$$
所以：
$$
[L(\theta^{(t+1)}, \theta^{(t)}) - L(\theta^{(t)}, \theta^{(t)})] - [H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})] \ge 0
$$
所以EM算法会收敛。但是需要注意的是，并不保证收敛到全局的极大值点。



### Kmeans算法的EM解释

Kmeans算法是一个经典的无监督聚类算法，它的执行过程如下：

1. 选择K个初始中心点$\{\bold u_1,\bold u_2,...,\bold u_K\}$
2. 将每个数据点$\bold x_i \in D$ 划归到离它最近的中心点$ \bold u_k $所代表的类别
3. 固定$\bold x_i$所属的分类，对每一个类别k，令其中心点$\bold u_k=\frac{1}{|D_k|}\sum_{i \in D_k}\bold x_i$
4. 重复2、3步直至收敛



翻译成EM版本：
$$
参数\theta:\{\bold u_1,\bold u_2,...,\bold u_K\}\\
隐变量：z_i \in \{1,2,...,K\}\\
联合概率分布： P(\bold x_i,z_i;\theta) \propto 
\left\{
\begin{aligned}
-exp(||\bold x_i-\bold u_{z_i}||^2), ||\bold x_i-\bold u_{z_i}||^2 = min_{1 \le k \le K}||\bold x_i-\bold u_{k}||^2 \\
0, ||\bold x_i-\bold u_{z_i}||^2 > min_{1 \le k \le K}||\bold x_i-\bold u_{k}||^2
\end{aligned}
\right. \\
z_i的条件概率分布：P(z_i|\bold x_i;\theta) \propto 
\left\{
\begin{aligned}
1, ||\bold x_i-\bold u_{z_i}||^2 = min_{1 \le k \le K}||\bold x_i-\bold u_{k}||^2 \\
0, ||\bold x_i-\bold u_{z_i}||^2 > min_{1 \le k \le K}||\bold x_i-\bold u_{k}||^2
\end{aligned}
\right. \\
$$

1. 选定初始值$\theta^{(0)}$
2. 计算$P(z_i|\bold x_i;\theta^{(t)})$, $\bold x_i$ 属于条件概率最大的分类k, 即 $z_i = argmin_{1 \le k \le K}||\bold x_i-\bold u_{k}||^2 $

3. 固定$z_i$,  求解最优化问题
   $$
   \theta^{(t+1)}=\mathop{\arg\max}_{\theta}\sum_i\sum_{z_i}P(z_i|\bold x_i;\theta^{(t)})log(P(\bold x_i,\bold z_i;\theta)) \\
   = \mathop{\arg\max}_{\theta}\sum_{k}\sum_{i, z_i=k}log(-exp(||\bold x_i-\bold u_{z_i}||^2))\\
   \frac{\partial L}{\partial \bold u_k} = \frac {\partial\sum_{i, zi=k}log(-exp(||\bold x_i-\bold u_{k}||^2))}{\partial \bold u_k}\\
   =\sum_{i,zi=k}2(\bold x_i-\bold u_k)\\
   令\frac{\partial L}{\partial \bold u_k} = 0，得\bold u_k = \frac{1}{|D_k|}\sum_{i \in D_k}\bold x_i
   $$

4. 重复3、4步骤直至收敛