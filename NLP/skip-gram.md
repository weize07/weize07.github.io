Skip-gram是训练词嵌入的一种常见方法。主要思路是用一段话的中间某一个词，去预测它的上下文词，每一个上下文词是一个多分类问题，通过不断调整输入矩阵$W_{V \times N}$和输出矩阵$W'_{N \times V}$的权重（V是词表大小，N是希望学到的词向量的维度）来提升上下文预测准确度。

最终学到的输入矩阵$W_{V \times N}$即是词嵌入的查询表（例如词表中的第i个单词，对应的词向量就是$W_{V \times N}$的第i个行向量的转置）。

#### 模型结构

![](./images/skip-gram.jpeg)

输入$\vec x_k$是某一个中心词的one-hot编码，而输出层经过softmax后转化成每一个单词的概率。

所以，给定输入向量$\vec x$的情况下，输出为：
$$
y = softmax(W'^T_{N \times V}(W^T_{V \times N}\vec x))
$$
假设中心词是词表中的第i个，那么隐藏层输出为： $\vec h = W_i$

输出向量y的第j位为$y_j=\vec h \cdot W'_j = W_i \cdot W'_j$, 损失函数使用负对数似然：
$$
L = -log P(w_{O,1},...,w_{O,C}|w_I)\\
= -log \prod_{c=1}^C P(w_{O,c}|w_I)\\
= -log \prod_{c=1}^C \frac{exp(y_{c,j})}{\sum_{v=1}^Vexp(y_v)}\\
= -\sum_{c=1}^{C}log \frac{exp(y_{c,j})}{\sum_{v=1}^Vexp(y_v)}
$$
损失函数的梯度：
$$
\nabla W'_j = \frac{\partial L}{\partial W'_j} \\
= -\sum_{c=1}^{C}\frac{\partial L}{\partial y_{c,j}}\frac{\partial y_{c,j}}{\partial W'_j} \\
= -\sum_{c=1}^{C}(1-y_{c,j})y_{c,j}\vec h\\

\nabla W = \frac{\partial L}{\partial W} \\
= -\sum_{c=1}^{C}\frac{\partial L}{\partial y_{c,j}}\frac{\partial y_{c,j}}{\partial \vec h}\frac{\partial \vec h}{\partial W} \\
= -\sum_{c=1}^{C}(1-y_{c,j})\vec x{W'}_{j}^{T}
$$

参数更新：

$W' \leftarrow W'-\nabla W'$

$W \leftarrow W-\nabla W$



