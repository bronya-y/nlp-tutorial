# Word2Vec

> [word2Vec](https://blog.csdn.net/vincent_duan/article/details/117967110)

## 思路

- **词嵌入** ： 不使用One-Hot而是将词的索引都投影到一个较短的词向量上面

### CBOW

- 通过上下文预测中心词
- 也就是拥有相似/相同上下文的词可能是近义词

#### 前向过程

- 首先上下文词语one-hot编码
- 编码后都W1投影得到1*N的向量
- 取平均得到1*N的向量
- W2投影到1*V的向量（V的大小是词的数量）
- 嗯，softMax最大的就是预测词了

----

- 肯定有疑惑
  - 这不是来学习词嵌入的吗
  - 怎么是预测中心词的
  - 其实要的是W1矩阵
    - 这个能够将ont-Hot转1*N
  - 为什么有效呢？
    - 因为前提是--上下文相同的词--就是相似的词
    - （猜的---经过训练，同样上下文输出的词概率是一致的）

----

### skip-gram模型

- 通过中心词扩展预测上下文
- 做法，输入中心词，使得预测出来的上下文对应的词的概率更大，输出还是个长度为V（词数量）的向量，softMax后与目标求损失即可。
- 我永远喜欢爱莉希雅
- [我， 永远， 喜欢， 爱莉希雅]
- 那么输入永远的时候，要使得[我， 喜欢]的概率大

---

#### 前向过程

- 首先将【永远】表示为one-hot
- 构建参数矩阵 
  - 中心词矩阵 V*N
  - 周围词矩阵 N*V
- One-Hot编码通过W1投影到1*N的向量
  - 中心词向量
- 用这个乘周围词矩阵
  - 也就是和其他的词做内积
  - 得到1*V的向量
- 做softMax
- 得到1*V的向量概率表示
  - 将向量概率中相邻的最大化

---

## 总结（新）

##### CBOW

- 不是直接转One-hot而是转为词向量，代表了一定的语义信息
- 可以用单张图分类-->无监督的对比学习InstDisc方法来做吗

---

- 压缩编码
  - 也可以利用cycle Gan来做吧

---

##### SKIP-GRAM

- 两组词向量？？？
- 有用吗，还是只是为了好写

---


