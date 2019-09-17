### 支持向量机

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

（1）引入样本间两两内积，从而引出核函数

（2）对偶问题往往更容易求解



